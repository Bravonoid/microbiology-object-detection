# Streamlit app for Microbiology Object Detection

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
st.write("Select the amount of pixels to filter out small objects.")
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

    # Find bounding boxes
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Draw bounding boxes
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, contours


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

    # Display the result
    st.image(result, caption="Detected Objects", use_column_width=True)

    # Display the number of objects detected
    st.subheader(f"Number of objects detected: {len(contours)}")

    # Crop objects
    objects = crop(img, contours)

    # Display the cropped objects in a grid
    cols = st.columns(10)
    for i, obj in enumerate(objects):
        cols[i % 10].image(obj, caption=f"Object {i+1}", use_column_width=True)

else:
    st.info("Please upload an image.")
