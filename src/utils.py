

def plot_class_distribution(y, class_names):
    """Vẽ biểu đồ phân phối lớp"""
    plt.figure()
    counts = np.bincount(y)
    sns.barplot(x=class_names, y=counts, palette="viridis")
    plt.title("Phân phối lớp trong tập dữ liệu")
    plt.xlabel("Lớp")
    plt.ylabel("Số lượng mẫu")
    plt.close()
