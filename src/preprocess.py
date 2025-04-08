import os
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ dataset
def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label) 
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img = Image.open(img_path).convert('RGB')
                # Resize image 
                img = img.resize((112, 112))
                images.append(np.array(img))
                labels.append(label)
    return np.array(images), np.array(labels)

# Chia cho 255 để đưa giá trị pixel về khoảng [0, 1]
def normalized_images(images):
    print("Normalizing images...")
    images_normalized = images / 255.0  
    return images_normalized

# Hàm trích xuất đặc trưng HOG
def extract_hog_features(images):
    print("Extracting HOG features...")
    hog_features = np.array([hog(rgb2gray(img), pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False) for img in images])
    print("Shape đặc trưng HOG:", hog_features.shape)
    #np.save("../results/hog_features.npy", hog_features)
    return hog_features

# Giảm chiều dữ liệu bằng PCA
def reduce_dimensionality(features, n=300):
    print("Reducing dimensionality with PCA...")
    pca = PCA(n_components=n)
    features_reduced = pca.fit_transform(features)
    return features_reduced

# Lưu dữ liệu đã xử lý
def save_processed_data(x, y, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "x.npy"), x)
    np.save(os.path.join(output_dir, "y.npy"), y)

def preprocess_pipeline(data_dir="../data"):
    x_train, y_train = load_data(os.path.join(data_dir, "train"))
    x_valid, y_valid = load_data(os.path.join(data_dir, "valid"))

    # Bước 2: Chia cho 255
    x_train = normalized_images(x_train)
    x_valid = normalized_images(x_valid)

    # Bước 3: Trích xuất đặc trưng HOG
    x_train = extract_hog_features(x_train)
    x_valid = extract_hog_features(x_valid)

    """ Extracting HOG features...
Shape đặc trưng HOG: (6745, 441)
Extracting HOG features...
Shape đặc trưng HOG: (1776, 441) """

    # Bước 4: Giảm chiều dữ liệu bằng PCA
    x_train = reduce_dimensionality(x_train)
    x_valid = reduce_dimensionality(x_valid)

    # Dùng label do có 1 số thuật toán yêu cầu đầu ra là vecto 1 chiều 
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_valid_encoded = le.transform(y_valid)

    # Bước 5: Lưu dữ liệu
    save_processed_data(x_train, y_train_encoded, "../data/processed/train")
    save_processed_data(x_valid, y_valid_encoded, "../data/processed/valid")
    
    return x_train, x_valid, y_train, y_valid

#preprocess_pipeline()
