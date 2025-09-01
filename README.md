# Movie Recommendation System - Nhóm 16

## Tổng quan hệ thống

Hệ thống đề xuất phim này bao gồm 3 thành phần chính:

1. **Notebook huấn luyện mô hình (`Movie_Recommendation_System.ipynb`)**
   - Chạy trên Google Colab để tận dụng tài nguyên GPU miễn phí.
   - Thực hiện các bước:
     - Tiền xử lý dữ liệu phim và đánh giá.
     - Xây dựng các mô hình lọc (Content-Based, SVD, DQN).
     - Huấn luyện mô hình DQN và lưu trọng số vào file `dqn_movie_rec_weights.weights.h5`.
     - Triển khai các API thử nghiệm bằng Flask (chỉ dùng cho demo hoặc kiểm thử nhanh trên Colab).

2. **Backend (FastAPI)**
   - Thư mục: `backend/`
   - Đọc dữ liệu, tải trọng số mô hình đã huấn luyện từ notebook.
   - Cung cấp các API phục vụ frontend:
     - Đề xuất phim phổ biến: `/api/popular_movies`
     - Đề xuất cá nhân hóa cho user: `/api/recommendations/user/{user_id}`
     - Tìm phim tương tự: `/api/similar_movies/{movie_id}`
   - Có thể chạy độc lập hoặc bằng Docker.

3. **Frontend (Streamlit)**
   - Thư mục: `frontend/`
   - Giao diện web đơn giản cho phép người dùng:
     - Xem phim phổ biến.
     - Nhận đề xuất cá nhân hóa theo user.
     - Tìm kiếm phim tương tự.
   - Giao tiếp với backend qua các API.

## Quy trình hoạt động

1. **Huấn luyện mô hình trên Colab**
   - Mở file `Movie_Recommendation_System.ipynb` trên Google Colab.
   - Chạy toàn bộ notebook để huấn luyện và lưu file trọng số `dqn_movie_rec_weights.weights.h5` vào thư mục dự án.

2. **Khởi động backend**
   - Đảm bảo đã có file trọng số và dữ liệu cần thiết.
   - Cài đặt các thư viện cần thiết từ `backend/requirements.txt`.
   - Chạy backend bằng lệnh:
     ```
     uvicorn app:app --host 0.0.0.0 --port 8000
     ```
   - Hoặc sử dụng Docker.

3. **Khởi động frontend**
   - Cài đặt các thư viện từ `frontend/requirements.txt`.
   - Chạy frontend bằng lệnh:
     ```
     streamlit run app.py
     ```

4. **Sử dụng hệ thống**
   - Truy cập giao diện web của frontend.
   - Nhận đề xuất phim dựa trên dữ liệu và mô hình đã huấn luyện.

## Lưu ý

- File notebook chỉ dùng để huấn luyện và thử nghiệm, không dùng để chạy API chính thức.
- Đảm bảo các file dữ liệu và trọng số đã được đặt đúng vị trí theo cấu