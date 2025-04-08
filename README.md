<h1>Tomato Leaf Disease Detection Project 🍅</h1>

<h2>## 📌 Giới thiệu</h2>

Dự án này nhằm xây dựng một hệ thống nhận diện bệnh trên lá cà chua từ ảnh chụp thông qua các thuật toán học máy truyền thống. Hệ thống có khả năng:
- Tiền xử lý ảnh và trích xuất đặc trưng bằng HOG.
- Áp dụng giảm chiều với PCA.
- Huấn luyện và đánh giá nhiều mô hình học máy như SVM, Random Forest, KNN,...
- Dự đoán ảnh mới thông qua mô hình đã lưu.


<h2>## 🗂 Cấu trúc thư mục</h2>

📂 project_ai/

│── 📂 data/                # Thư mục chứa dữ liệu (hình ảnh, CSV, JSON,...)

│── 📂 models/              # Lưu trữ mô hình đã huấn luyện

│── 📂 notebooks/           # Chứa các file Jupyter Notebook (nếu cần)

│── 📂 src/                 # Chứa mã nguồn chính của dự án

│   │── 📜 train.py         # Huấn luyện mô hình

│   │── 📜 preprocess.py    # Tiền xử lý dữ liệu

│── 📂 results/             # Lưu trữ kết quả đầu ra

│── 📜 requirements.txt     # Danh sách thư viện cần cài đặt

│── 📜 main.py              # Chương trình chính chạy AI

│── 📜 README.md            # Mô tả dự án, cách chạy

│── 📜 .gitignore           # Loại trừ các file không cần thiết khi đẩy lên GitHub

<h2>## 🔧 Yêu cầu môi trường</h2>

- Python >= 3.8

- Thư viện:

  - `numpy`

  - `scikit-learn`

  - `scikit-image`

  - `joblib`

  - `Pillow`

<h2>📊 Kết quả mô hình (Accuracy)</h2>

Logistic Regression	0.4073

Decision Tree	0.3293

SVM	0.5689

Random Forest	0.4160

KNN	0.5050

<h2>DataSet</h2>
Path download Dataset : https://www.kaggle.com/datasets/ashishmotwani/tomato?resource=download

Các bệnh của cây cà chua trong tập dữ liệu này bao gồm:
- Bacterial_spot  
- Early_blight  
- healthy  
- Late_blight  
- Leaf_Mold  
- powdery_mildew  
- Septoria_leaf_spot  
- Spider_mites Two-spotted_spider_mite  
- Target_Spot  
- Tomato_mosaic_virus  
- Tomato_Yellow_Leaf_Curl_Virus

<h2>🌟 Ví dụ minh họa</h2>
Input ảnh: ảnh lá cà chua chụp thực tế

→ Tiền xử lý

→ Trích xuất đặc trưng HOG

→ Giảm chiều PCA

→ Dự đoán: Tomato_Yellow_Leaf_Curl_Virus


<h2>📬 Liên hệ</h2>
Email: dhhuongdhlt1@gmail.com 
