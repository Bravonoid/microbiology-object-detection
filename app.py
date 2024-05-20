# Streamlit app for Microbiology Object Detection

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


from PIL import Image

# Title
st.title("Microbiology Object Detection using Image Processing")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app is a simple demo of Microbiology Object Detection using Image Processing. You can upload an image and the app will detect the objects in the image."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Filter size
st.sidebar.subheader("Filter Size")
filter_size = st.sidebar.slider("Filter Size", 1, 250, 30)


def detect(image):
    image = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Fill holes
    filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Remove small objects
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small objects
    filtered = np.zeros_like(filled)
    for contour in contours:
        if cv2.contourArea(contour) > filter_size:
            cv2.drawContours(filtered, [contour], -1, 255, -1)

    # Obtain contours
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return image, contours


def extract_features(image, contours):
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Define GLCM properties
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    # Define distances and angles
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Initialize feature matrix
    features = []

    # Extract features for each bounding box
    for x, y, w, h in bounding_boxes:
        # Crop the object
        obj = image[y : y + h, x : x + w]

        # Convert the object to grayscale
        gray = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)

        # Compute GLCM
        glcm = graycomatrix(gray, distances, angles, symmetric=True, normed=True)

        # Compute GLCM properties
        props = [graycoprops(glcm, prop)[0, 0] for prop in properties]

        # Append to feature matrix
        features.append(props)

    # Convert to numpy array
    features = np.array(features)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features


def cluster(features, n_clusters=2):
    # Cluster the features
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    labels = kmeans.labels_

    return labels


def visualize(image, contours, labels):
    # Visualize the objects
    vis = image.copy()

    # Different colors for each label
    colors = np.random.randint(0, 255, (labels.max() + 1, 3))

    # Draw bounding boxes
    for contour, label in zip(contours, labels):
        x, y, w, h = cv2.boundingRect(contour)
        color = colors[label].tolist()
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

    return vis


def crop(image, contours):
    # Crop objects from the image
    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        obj = image[y : y + h, x : x + w]
        objects.append(obj)

    return objects


if uploaded_image is not None:
    # Read the image
    img = Image.open(uploaded_image)
    img = np.array(img)

    # Display the image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detect objects
    result, contours = detect(img)

    # Extract features
    features = extract_features(img, contours)

    # Cluster size
    st.sidebar.subheader("Cluster Size")
    n_clusters = st.sidebar.slider("Cluster Size", 1, len(features), 2)

    # Check if n_clusters is valid
    n_clusters = min(n_clusters, len(features))

    # Cluster the features
    labels = cluster(features, n_clusters=n_clusters)

    # Visualize the result
    result = visualize(result, contours, labels)

    # Crop the objects
    objects = crop(result, contours)

    # Display the result
    st.image(result, caption="Detected Objects", use_column_width=True)

    # Display objects as per row per cluster
    st.subheader("Detected Objects")
    cluster_objects = []
    for i in range(n_clusters):
        cluster_objects.append(
            [obj for obj, label in zip(objects, labels) if label == i]
        )

    for i, objects in enumerate(cluster_objects):
        st.write(f"Cluster {i + 1}")
        # Maximum display 10 objects per cluster in a row
        n_objects = min(len(objects), 10)
        n_rows = (n_objects - 1) // 5 + 1
        fig, axs = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
        axs = axs.ravel()
        for j in range(n_objects):
            axs[j].imshow(objects[j])
            axs[j].axis("off")
        for j in range(n_objects, n_rows * 5):
            axs[j].axis("off")
        st.pyplot(fig)


else:
    st.info("Please upload an image.")
