# frontend/app.py
import streamlit as st
import requests
import os

# Lấy URL của backend từ biến môi trường
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(layout="wide")

def fetch_movies(endpoint):
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}")
        response.raise_for_status() # Kiểm tra lỗi HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi kết nối tới backend: {e}")
        return None

def main_page():
    st.title("Hệ thống đề xuất phim")

    # --- Đề xuất phim phổ biến ---
    st.header("🍿 Phim Phổ Biến")
    popular_movies = fetch_movies("/api/popular_movies")
    if popular_movies:
        cols = st.columns(5)
        for i, movie in enumerate(popular_movies):
            with cols[i % 5]:
                st.subheader(movie.get("title", "Không rõ"))
                # TODO: Lấy poster và hiển thị
                st.write(f"ID: {movie['id']}")

    # --- Đề xuất cá nhân hóa ---
    st.header("👤 Phim Dành Cho Bạn")
    user_id = st.text_input("Nhập User ID để nhận đề xuất:", value="1")
    if user_id:
        recommendations = fetch_movies(f"/api/recommendations/user/{user_id}")
        if recommendations:
            cols = st.columns(5)
            for i, movie in enumerate(recommendations):
                with cols[i % 5]:
                    st.subheader(movie.get("title", "Không rõ"))
                    st.write(f"ID: {movie['id']}")
        else:
            st.info(f"Không có đề xuất cho User ID {user_id}. Vui lòng thử User ID khác.")

def similar_movies_page():
    st.title("Tìm kiếm phim tương tự")
    movie_id = st.text_input("Nhập ID phim:", value="862")
    if movie_id:
        st.subheader(f"Phim tương tự với ID: {movie_id}")
        similar_movies = fetch_movies(f"/api/similar_movies/{movie_id}")
        if similar_movies:
            cols = st.columns(5)
            for i, movie in enumerate(similar_movies):
                with cols[i % 5]:
                    st.subheader(movie.get("title", "Không rõ"))
                    st.write(f"ID: {movie['id']}")
        else:
            st.warning("Không tìm thấy phim này hoặc không có phim tương tự.")

# --- Điều hướng trang ---
page = st.sidebar.radio("Chọn trang:", ["Trang chủ", "Tìm kiếm tương tự"])
if page == "Trang chủ":
    main_page()
elif page == "Tìm kiếm tương tự":
    similar_movies_page()
