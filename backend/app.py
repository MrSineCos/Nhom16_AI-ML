# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

# Import đối tượng recommender đã được khởi tạo sẵn
from recommender_service import recommender

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Movie Recommendation Backend is running!"}

@app.get("/api/popular_movies")
def get_popular_movies_api():
    return recommender.get_most_popular_movies(num_recs=10)

@app.get("/api/similar_movies/{movie_id}")
def get_similar_movies_api(movie_id: int):
    recs = recommender.get_content_based_recommendations(movie_id, num_recs=10)
    if not recs:
        raise HTTPException(status_code=404, detail="Movie not found or no similar movies.")
    return recs

@app.get("/api/recommendations/user/{user_id}")
def get_user_recommendations_api(user_id: int):
    recs = recommender.get_recommendations_for_user(user_id)
    if not recs:
        raise HTTPException(status_code=404, detail="User not found or no recommendations.")
    return recs