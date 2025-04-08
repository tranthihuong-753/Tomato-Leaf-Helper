import os
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image, ImageFilter

# Đọc dữ liệu từ dataset
def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label) 
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            # Resize image 
            img = img.resize((112, 112))
            images.append(np.array(img))
            labels.append(label)
            """ # Nếu label là "powdery_mildew", tạo thêm ảnh biến đổi
            if label == "powdery_mildew":
                # Xoay ảnh 90 độ và làm mờ ảnh
                img_rotated = img.rotate(30)  # Xoay ảnh 30 độ
                img_blurred = img.filter(ImageFilter.GaussianBlur(radius=2))  # Làm mờ ảnh với GaussianBlur                   
                # Thêm ảnh gốc và ảnh đã biến đổi vào danh sách images và labels
                images.append(np.array(img))
                labels.append(label)
                images.append(np.array(img_rotated))
                labels.append(label)
                images.append(np.array(img_blurred))
                labels.append(label)
            else:
                # Nếu không phải "powdery_mildew", chỉ thêm ảnh gốc
                images.append(np.array(img))
                labels.append(label) """
    return np.array(images), np.array(labels)

# Hàm trích xuất đặc trưng HOG
def extract_hog_features(images):
    print("Extracting HOG features...")
    hog_features = np.array([hog(rgb2gray(img), pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False) for img in images])
    print("Shape đặc trưng HOG:", hog_features.shape)
    return hog_features

# Lưu dữ liệu đã xử lý
def save_processed_data(x, y, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "x.npy"), x)
    np.save(os.path.join(output_dir, "y.npy"), y)

def preprocess_pipeline(data_dir="../data"):
    x_train, y_train = load_data(os.path.join(data_dir, "train"))
    x_valid, y_valid = load_data(os.path.join(data_dir, "valid"))

    # Bước 3: Trích xuất đặc trưng HOG
    x_train = extract_hog_features(x_train)        
    x_valid = extract_hog_features(x_valid)

    # chuẩn hóa (scaled)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    joblib.dump(scaler, "../models/scaler.pkl")
    x_valid = scaler.transform(x_valid)

    # Bước 4: Giảm chiều dữ liệu bằng PCA
    pca = PCA(n_components=300)
    x_train = pca.fit_transform(x_train)
    joblib.dump(pca, "../models/pca.pkl")
    x_valid = pca.transform(x_valid)

    # Dùng label do có 1 số thuật toán yêu cầu đầu ra là vecto 1 chiều 
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_valid_encoded = le.transform(y_valid)

    # Bước 5: Lưu dữ liệu
    save_processed_data(x_train, y_train_encoded, "../data/processed/train")
    save_processed_data(x_valid, y_valid_encoded, "../data/processed/valid")
    
    return x_train, x_valid, y_train, y_valid

preprocess_pipeline()
