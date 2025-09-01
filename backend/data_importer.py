# backend/data_importer.py
import pandas as pd
import psycopg2
import os

DB_HOST = 'database'
DB_PORT = 5432
DB_NAME = 'movie_db'  # <-- SỬA LẠI THÀNH 'movie_db'
DB_USER = 'postgres'
DB_PASSWORD = '12345'

DATA_FOLDER = '../data/The Movies Dataset'
MOVIE_METADATA_PATH = os.path.join(DATA_FOLDER, 'movies_metadata.csv')

def create_tables(cur):
    print("Đang tạo bảng...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id BIGINT PRIMARY KEY,
            title VARCHAR(255),
            overview TEXT,
            poster_path VARCHAR(255),
            release_date VARCHAR(50)
        );
    """)

def import_movies_data(conn, cur):
    print("Đang đọc và tiền xử lý dữ liệu...")
    movies_df = pd.read_csv(MOVIE_METADATA_PATH, low_memory=False)
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
    movies_df.dropna(subset=['id'], inplace=True)
    movies_df['id'] = movies_df['id'].astype('int')
    movies_df['overview'] = movies_df['overview'].fillna('')

    print("Đang chèn dữ liệu vào bảng movies...")
    for index, row in movies_df.iterrows():
        try:
            cur.execute("""
                INSERT INTO movies (id, title, overview, poster_path, release_date)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (
                row['id'], row['title'], row['overview'],
                row['poster_path'] if 'poster_path' in row else None,
                row['release_date'] if 'release_date' in row else None
            ))
        except Exception as e:
            print(f"Lỗi khi chèn dòng {index}: {e}")
            conn.rollback()
    conn.commit()
    print("Hoàn tất nhập dữ liệu.")

if __name__ == '__main__':
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        cur = conn.cursor()
        create_tables(cur)
        import_movies_data(conn, cur)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Lỗi kết nối hoặc xử lý database: {e}")