import gradio as gr
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import color
import joblib
import os
from sklearn.decomposition import PCA

# Tải các mô hình đã huấn luyện
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "KNN": joblib.load("models/knn.pkl")
}

def preprocess_image(image):
    # Đảm bảo ảnh có 3 kênh màu (RGB)
    image = image.convert('RGB')
    
    # Resize ảnh về kích thước 112x112
    image = image.resize((112, 112))
    
    # Chuyển ảnh sang ảnh xám để trích xuất đặc trưng HOG
    image_gray = color.rgb2gray(image)
    
    # Trích xuất đặc trưng HOG: tạo vector đặc trưng 1D
    features = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False).reshape(1, -1)
    
    # Tải scaler để chuẩn hóa đặc trưng
    scaler = joblib.load("models/scaler.pkl")
    features_scaled = scaler.transform(features)

    # Tải mô hình PCA và giảm chiều đặc trưng
    pca = joblib.load("models/pca.pkl")
    features_pca = pca.transform(features_scaled)

    return features_pca


# Hàm dự đoán
def predict(image, model_name):
    # Tiền xử lý ảnh
    features = preprocess_image(image)
    
    # Lấy mô hình được chọn
    model = models[model_name]
    
    # Dự đoán
    prediction = model.predict(features)[0]
    disease = ""
    i = 0
    for label in os.listdir("data/train"):
        if i == prediction:
            disease = label
            break
        i += 1
    
    # Tính xác suất (nếu mô hình hỗ trợ)
    try:
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100  # Lấy xác suất cao nhất
        return f"Bệnh dự đoán: {disease} ({confidence:.2f}% tin cậy)"
    except AttributeError:
        # Nếu mô hình không hỗ trợ predict_proba (như SVM mặc định)
        return f"Bệnh dự đoán: {disease} (Không có xác suất)"

# Tạo giao diện Gradio
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Tải ảnh lá cà chua lên"),
        gr.Dropdown(
            choices=list(models.keys()),
            label="Chọn mô hình",
            value="Random Forest"  # Mô hình mặc định
        )
    ],
    outputs=gr.Textbox(label="Kết quả dự đoán"),
    title="Tomato Leaf Disease Detection",
    description="Tải lên ảnh lá cà chua để phát hiện bệnh lý bằng các mô hình học máy.",
)

# Chạy giao diện
interface.launch()