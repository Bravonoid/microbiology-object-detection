import streamlit as st
import numpy as np
import tempfile
import cv2
from roboflow import Roboflow
from PIL import Image
import io
import av

# Title
st.title("Microbiology Object Detection using Image Processing")

# Confidence level
st.sidebar.subheader("Confidence Level")
st.sidebar.info("Set the confidence level for object detection.")
confidence = st.sidebar.slider("Confidence Level", 1, 100, 20)

# Input type
st.sidebar.subheader("Input Type")
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

# Model
rf = Roboflow(api_key=st.secrets["roboflow"]["apikey"])
project = rf.workspace().project("histogram-equalization-u3ofv")
model = project.version(1).model

# Color
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


# Biomass calculation
def calculate_biomass(area):
    # Biomass = area * 6.836/0.00001 / 2 * 0.5
    return area * 6.836 / 0.00001 / 2 * 0.5


if input_type == "Image":
    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    detected_objects = []

    if uploaded_image is not None:
        # Read the image
        image = Image.open(uploaded_image)
        image = np.array(image)

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect objects in the image
        prediction = model.predict(image, confidence=confidence, overlap=30).json()

        for p in prediction["predictions"]:
            x_center, y_center, w, h = (
                int(p["x"]),
                int(p["y"]),
                int(p["width"]),
                int(p["height"]),
            )
            x1, y1, x2, y2 = (
                x_center - w // 2,
                y_center - h // 2,
                x_center + w // 2,
                y_center + h // 2,
            )

            # Calculate area
            area = w * h

            # Calculate biomass
            biomass = calculate_biomass(area)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[p["class_id"]], 2)

            # Draw label with border
            cv2.putText(
                image,
                f'{p["class"]} ({p["confidence"]*100:.2f}%)',
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

            # Draw biomass
            cv2.putText(
                image,
                f"Biomass: {biomass:.2f} g",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

            # Append to detected objects
            detected_objects.append(
                {
                    "class": p["class"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "biomass": biomass,
                    "confidence": p["confidence"],
                    "color": f"rgb{colors[p['class_id']]}",
                    "class_id": p["class_id"],
                }
            )

        # Display the detected objects
        st.image(image, caption="Detected Objects", use_column_width=True)

    else:
        st.info("Please upload an image.")

elif input_type == "Video":
    # Upload video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    # Add loading spinner
    if uploaded_video is not None:
        with st.spinner("Analyzing video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            # Read the video
            video = cv2.VideoCapture(tfile.name)

            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))

            output_memory_file = io.BytesIO()  # Create BytesIO "in memory file".

            output = av.open(
                output_memory_file, "w", format="mp4"
            )  # Open "in memory file" as MP4 video output
            stream = output.add_stream(
                "h264", str(fps)
            )  # Add H.264 video stream to the MP4 container, with framerate = fps.
            stream.width = width  # Set frame width
            stream.height = height  # Set frame height
            # stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
            stream.pix_fmt = (
                "yuv420p"  # Select yuv420p pixel format for wider compatibility.
            )
            stream.options = {
                "crf": "17"
            }  # Select low crf for high quality (the price is larger file size).

            # Create a video writer using x264 codec
            # fourcc = cv2.VideoWriter_fourcc(*"H264")
            # out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                # Detect objects in the frame
                prediction = model.predict(
                    frame, confidence=confidence, overlap=30
                ).json()

                for p in prediction["predictions"]:
                    x_center, y_center, w, h = (
                        int(p["x"]),
                        int(p["y"]),
                        int(p["width"]),
                        int(p["height"]),
                    )
                    x1, y1, x2, y2 = (
                        x_center - w // 2,
                        y_center - h // 2,
                        x_center + w // 2,
                        y_center + h // 2,
                    )

                    # Calculate area
                    area = w * h

                    # Calculate biomass
                    biomass = calculate_biomass(area)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[p["class_id"]], 2)

                    # Draw label with border
                    cv2.putText(
                        frame,
                        f'{p["class"]} ({p["confidence"]*100:.2f}%)',
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                    )

                    # Draw biomass
                    cv2.putText(
                        frame,
                        f"Biomass: {biomass:.2f} g",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                    )

                # Write the frame to the output video
                frame = av.VideoFrame.from_ndarray(
                    frame, format="bgr24"
                )  # Convert image from NumPy Array to frame.
                packet = stream.encode(frame)  # Encode video frame
                output.mux(
                    packet
                )  # "Mux" the encoded frame (add the encoded frame to MP4 file).

            # Flush the encoder
            packet = stream.encode(None)
            output.mux(packet)
            output.close()

            output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
            # video_bytes = output_memory_file.read()  # Convert BytesIO to bytes array
            # st.video(video_bytes)
            st.video(
                output_memory_file
            )  # Streamlit supports BytesIO object - we don't have to convert it to bytes array.

    else:
        st.info("Please upload a video.")
