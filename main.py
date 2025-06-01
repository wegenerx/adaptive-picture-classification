import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shutil

# Create target folder
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Extract image features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use SIFT to extract features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is None:
        return None
    return descriptors

# Main program
def classify_images(source_folder, result_folder, num_clusters=5):
    # Create result folder
    create_folder(result_folder)

    # Extract features from all images
    image_paths = []
    features = []
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(source_folder, filename)
            descriptors = extract_features(image_path)
            if descriptors is not None:
                image_paths.append(image_path)
                features.append(descriptors)

    if not features:
        print("No valid images or features found!")
        return

    # Vectorize features
    all_features = np.vstack(features)
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(all_features)

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(reduced_features)

    # Classify images into different folders
    for i, image_path in enumerate(image_paths):
        cluster_label = kmeans.labels_[i]
        cluster_folder = os.path.join(result_folder, f"cluster_{cluster_label}")
        create_folder(cluster_folder)
        shutil.copy(image_path, cluster_folder)

    print(f"Image classification completed. Images are divided into {num_clusters} categories and saved in {result_folder}.")

if __name__ == "__main__":
    source_folder = "goal"  # Source folder path
    result_folder = "result"  # Result folder path
    num_clusters = 10  # Number of categories, can be adjusted as needed
    classify_images(source_folder, result_folder, num_clusters)