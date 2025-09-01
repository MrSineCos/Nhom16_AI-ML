# backend/recommender_service.py
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from surprise import Reader, Dataset, SVD
from tqdm import tqdm

class RecommenderService:
    def __init__(self):
        print("Khởi tạo RecommenderService...")
        self._load_data()
        self._build_models()
        self._load_dqn_agent()
        print("RecommenderService đã sẵn sàng.")

    def _load_data(self):
        print("Đang tải và xử lý dữ liệu...")
        movies_df = pd.read_csv(os.path.join('data/The Movies Dataset', 'movies_metadata.csv'), low_memory=False)
        self.ratings_df = pd.read_csv(os.path.join('data/The Movies Dataset', 'ratings_small.csv'))

        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        movies_df.dropna(subset=['id'], inplace=True)
        movies_df['id'] = movies_df['id'].astype('int')
        movies_df['overview'] = movies_df['overview'].fillna('')
        movie_ids_in_ratings = self.ratings_df['movieId'].unique()
        self.small_movies_df = movies_df[movies_df['id'].isin(movie_ids_in_ratings)].copy()
        self.small_movies_df.drop_duplicates(subset=['id'], inplace=True)
        self.small_movies_df.reset_index(drop=True, inplace=True)
        self.ratings_df = self.ratings_df.sort_values('timestamp').reset_index(drop=True)

    def _build_models(self):
        print("Đang xây dựng các mô hình nền...")
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.small_movies_df['overview'])
        self.cosine_sim_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.movie_id_to_idx = pd.Series(self.small_movies_df.index, index=self.small_movies_df['id'])
        
        # SVD
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        svd_model.fit(trainset)
        anti_testset = trainset.build_anti_testset()
        svd_predictions = svd_model.test(anti_testset)
        self.svd_recs_precomputed = defaultdict(list)
        for uid, iid, _, est, _ in svd_predictions:
            self.svd_recs_precomputed[uid].append((iid, est))
        for uid, user_ratings in self.svd_recs_precomputed.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)

        # Popularity
        movie_rating_counts = self.ratings_df['movieId'].value_counts()
        self.popular_movies_df = self.small_movies_df[self.small_movies_df['id'].isin(movie_rating_counts.index)][['id', 'title']].copy()
        self.popular_movies_df['rating_count'] = self.popular_movies_df['id'].map(movie_rating_counts)
        self.popular_movies_df = self.popular_movies_df.sort_values('rating_count', ascending=False)

    def _load_dqn_agent(self):
        print("Đang tải mô hình DQN...")
        self.dqn_agent = self._build_dqn_model(self.tfidf_matrix.shape[1], 200)
        self.dqn_agent.load_weights('data/dqn_movie_rec_weights.weights.h5')

    def _build_dqn_model(self, state_size, action_size):
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam())
        return model

    # Các hàm đề xuất chính
    def get_most_popular_movies(self, num_recs=10):
        return self.popular_movies_df.head(num_recs).to_dict('records')

    def get_content_based_recommendations(self, movie_id, num_recs=10):
        if movie_id not in self.movie_id_to_idx:
            return None
        idx = self.movie_id_to_idx[movie_id]
        sim_scores = sorted(list(enumerate(self.cosine_sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:num_recs+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.small_movies_df.iloc[movie_indices][['id', 'title']].to_dict('records')

    def get_recommendations_for_user(self, user_id, num_recs=10):
        user_history = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_history.empty:
            return []

        interaction_count = len(user_history)
        INTERACTION_THRESHOLD = 5
        
        watched_movies = user_history['movieId'].values
        svd_recs = [mid for mid, score in self.svd_recs_precomputed.get(user_id, []) if mid not in watched_movies]
        content_recs = []
        last_liked_movie = user_history[user_history['rating'] >= 3.5].tail(1)
        if not last_liked_movie.empty:
            last_movie_id = last_liked_movie['movieId'].iloc[0]
            if last_movie_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[last_movie_id]
                sim_scores = sorted(list(enumerate(self.cosine_sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:150+1]
                content_movie_ids = self.small_movies_df['id'].iloc([i[0] for i in sim_scores]).tolist()
                content_recs = [mid for mid in content_movie_ids if mid not in watched_movies]
        
        candidate_movies = list(dict.fromkeys(svd_recs[:150] + content_recs))[:200]

        if not candidate_movies:
            return []

        if interaction_count <= INTERACTION_THRESHOLD:
            final_recs_ids = candidate_movies[:num_recs]
        else:
            def create_state(history_df):
                high_rated_movies = history_df[history_df['rating'] >= 3.5].tail(5)
                if high_rated_movies.empty: return np.zeros(self.tfidf_matrix.shape[1])
                movie_indices = self.movie_id_to_idx.reindex(high_rated_movies['movieId']).dropna().astype(int)
                state_vectors = self.tfidf_matrix[movie_indices]
                return np.array(state_vectors.mean(axis=0)).flatten()

            state = create_state(user_history)
            state = np.reshape(state, [1, self.tfidf_matrix.shape[1]])
            q_values = self.dqn_agent.predict(state, verbose=0)[0]
            
            candidate_q_values = [(candidate_movies[i], q_values[i]) for i in range(len(candidate_movies))]
            sorted_recommendations = sorted(candidate_q_values, key=lambda x: x[1], reverse=True)
            final_recs_ids = [mid for mid, q in sorted_recommendations[:num_recs]]

        rec_df = self.small_movies_df[self.small_movies_df['id'].isin(final_recs_ids)][['id', 'title']]
        rec_df['sort_order'] = rec_df['id'].apply(lambda x: final_recs_ids.index(x))
        return rec_df.sort_values('sort_order').drop('sort_order', axis=1).to_dict('records')

# Khởi tạo một đối tượng duy nhất để app FastAPI sử dụng
recommender = RecommenderService()