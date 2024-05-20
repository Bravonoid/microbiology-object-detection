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

    clustered_objects = []
    for i, obj in enumerate(objects):
        clustered_objects.append((obj, labels[i]))

    # Plot 1 cluster examples in a grid plot
    st.subheader("Clustered Object Examples")

    # Sort objects by cluster
    clustered_objects.sort(key=lambda x: x[1])

    # Count the number of objects in each cluster
    cluster_counts = [0] * n_clusters
    for _, label in clustered_objects:
        cluster_counts[label] += 1

    # Only show 1 object per cluster
    clustered_objects = [
        clustered_objects[i]
        for i in range(len(clustered_objects))
        if i == 0 or clustered_objects[i][1] != clustered_objects[i - 1][1]
    ]

    # Displaying 1 object per cluster and the cluster count
    cols = st.columns(8)
    for i, (obj, label) in enumerate(clustered_objects):
        cols[i % 8].write(f"Cluster {label+1}")
        cols[i % 8].image(
            obj,
            caption=f"({cluster_counts[label]} objects)",
            use_column_width=True,
        )
        if i % 8 == 7:
            cols = st.columns(8)

else:
    st.info("Please upload an image.")
