import streamlit as st
import cv2
import os
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip
import tempfile
import time
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch


st.title('Deep Learning Case Study')


# smoke detection
def process_smoke_detection(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    model_smoke = YOLO('/home/codezeros/Desktop/combined/models/best.pt')

    frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = model_smoke(frame_rgb)
        label = "smoke"

        for result in results:
            for box in result.boxes.xyxy:
                conf = result.boxes.conf[0]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)
        frames.append(frame)

    cap.release()
    out.release()


# emotion detection
def process_emotion_detection(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    model_emotion = load_model("/home/codezeros/Desktop/combined/models/emotion_model.h5")

    frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_region_resized = cv2.resize(face_region, (48, 48))
            face_region_resized = np.expand_dims(face_region_resized, axis=0)
            face_region_resized = np.expand_dims(face_region_resized, axis=-1)
            face_region_resized = face_region_resized.astype('float32') / 255

            result = model_emotion.predict(face_region_resized)
            predicted_emotion_index = np.argmax(result)
            predicted_emotion_label = label_dict[predicted_emotion_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        out.write(frame)
        frames.append(frame)

    cap.release()
    out.release()

#object detection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def process_object_detection(input_video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Define output video parameters
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_path = "/home/codezeros/Desktop/combined/output"
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Perform object detection
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        # Draw bounding boxes and labels on the frame
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            # Draw rectangle
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            # Add label text
            cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Write the frame to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()
    

# Function to convert video format
def convert_to_mp4(input_file, output_file):
    try:
        video = VideoFileClip(input_file)
        video.write_videofile(output_file, codec='libx264', audio_codec='aac')
        video.close()
    except Exception as e:
        st.error(f"Error occurred while converting video: {e}")

#mnist classification
def preprocess_mnist_image(image_path):
    try:
        uploaded_image = Image.open(image_path).convert('L')  # Convert image to grayscale
        uploaded_image = uploaded_image.resize((28, 28))  # Resize image to match MNIST input
        image_array = np.array(uploaded_image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        return None

def perform_mnist_classification(image_path):
    model = load_model('/home/codezeros/Desktop/combined/models/mnist.h5')
    processed_image = preprocess_mnist_image(image_path)
    if processed_image is None:
        return None, None, None
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    probability = np.max(prediction) * 100  # Convert probability to percentage
    return predicted_digit, probability, processed_image[0]

# hand written alphabet 
def perform_az_handwritten_recognition(image_path):
    model = load_model("/home/codezeros/Desktop/combined/models/model_hand.h5")
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality downsampling
    image_array = np.array(image).astype('float32') / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension

    # Predict the class
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = chr(predicted_class_index + 65)  # Convert index to uppercase alphabet

    return predicted_class, image_array  # Return the image array for further use


#fruit image classification        
def perform_fruit_classification(image_path):
    
    model_path = "/home/codezeros/Desktop/combined/models/fruit_model.h5"
    model = tf.keras.models.load_model(model_path)

    data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

    
    img = img_preprocessing.load_img(image_path, target_size=(128, 128))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = data_cat[predicted_class_index]
    probability = prediction[0][predicted_class_index] * 100

    return predicted_class, probability, img

# cat and dogs classification
def perform_cat_dog_classification(image_path):
   
    model = load_model('/home/codezeros/Desktop/combined/models/cat_dogs.h5')

    def preprocess_image(img_path):
        img = Image.open(img_path)
        img = img.resize((100, 100))  # Resize the image to the desired dimensions
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        return img_array
    
    def predict_image_class(img_path):
        preprocessed_img = preprocess_image(img_path)
        pred = model.predict(preprocessed_img)
        if pred[0][0] > 0.5:
            return 'Dog', pred[0][0]
        else:
            return 'Cat', 1 - pred[0][0]  

    predicted_class, probability = predict_image_class(image_path)

    img = image.load_img(image_path, target_size=(100, 100))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted class: {predicted_class} (Confidence: {probability:.2f})')
    plt.show()
    return predicted_class, probability, img


# Function to perform sentiment analysis
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Get the polarity scores for the text
    scores = sid.polarity_scores(text)
    
    # Determine sentiment based on the compound score
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def process_sentiment_analysis(text):
    sentiment = analyze_sentiment(text)
    return sentiment


# text to text
def process_text_to_text():
    return


#dropdown list 
selected_model = st.selectbox("Select Model", ["Smoke Detection", "Emotion Detection","Object Detection", "A-Z hand written Alphabet Recognition","Fruit Image Classification","Mnist Classification","Cat and Dog Classification","Sentiment Analysis", "Text to Text"])

uploads_dir = 'uploads'

os.makedirs(uploads_dir, exist_ok=True)

# video file upload
uploaded_file_video = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file_video is not None: 
    video_path = None
    image_path = None

    if uploaded_file_video is not None:
        video_path = os.path.join(uploads_dir, uploaded_file_video.name)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file_video.read())

                
if st.button("Process Video"):  
    if video_path is not None:
        start_time = time.time()
        output_video_path = os.path.join(uploads_dir, "output_video.mp4")

        if selected_model == "Smoke Detection":
            process_smoke_detection(video_path, output_video_path)
        elif selected_model == "Emotion Detection":
            process_emotion_detection(video_path, output_video_path)
        elif selected_model == "Object Detection":
            process_object_detection(video_path, output_video_path)
     
        st.markdown("---")
        st.subheader("Output Video")

        output_video_path_conv = os.path.join(uploads_dir, "new_output_video.mp4")
        convert_to_mp4(output_video_path, output_video_path_conv)
        end_time = time.time()
        print("Total time consumed:", end_time-start_time)

        st.video(output_video_path_conv)
        st.session_state['video_bytes'] = open(output_video_path, 'rb').read()
        st.write(f"Output video saved at: {output_video_path}")

        if os.path.exists(output_video_path):
            convert_to_mp4(output_video_path, output_video_path_conv)
        else:
            print("The output video file does not exist. Please check the path:", output_video_path)    

        if 'video_bytes' in st.session_state:
            st.markdown("---")
            st.download_button(label="Download output video",
                                data=st.session_state['video_bytes'],
                                file_name="output_video.mp4",
                                mime="video/mp4")



# Image file upload

uploaded_file_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file_image is not None:  
        image_path = os.path.join(uploads_dir, uploaded_file_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file_image.read())
        

if st.button("Process Image"):
    if image_path is not None:
        image = Image.open(image_path)
        if image is None:
            st.error("Failed to load image. Check the file path and file itself.")
        else:
            st.success("Image loaded successfully.")
            st.image(image, caption="Uploaded Image", use_column_width=True)

    if selected_model == "MNIST Classification":
        st.write("Attempting to process MNIST classification...")  # Debugging output
        try:
            predicted_digit, probability, processed_image = perform_mnist_classification(image_path)
            if predicted_digit is not None:
                st.write(f"Predicted digit: {predicted_digit} with a probability of {probability:.2f}%")
                processed_image_display = (processed_image * 255).astype(np.uint8)
                st.image(processed_image_display, caption="Processed Image for MNIST", use_column_width=True, output_format="PNG")
            else:
                st.error("Failed to predict the digit.")
        except Exception as e:
            st.error(f"An error occurred during MNIST classification: {e}")

    elif selected_model == "Fruit Image Classification":
        predicted_class, probability, processed_image = perform_fruit_classification(image_path)
        st.image(processed_image, caption=f"Predicted class: {predicted_class} (Confidence: {probability:.2f}%)", use_column_width=True)

    elif selected_model == "A-Z hand written Alphabet Recognition":
        predicted_class, image_array = perform_az_handwritten_recognition(image_path)
        processed_image = Image.fromarray((image_array[0, :, :, 0] * 255).astype('uint8'), 'L')  # Convert numpy array to PIL Image
        st.image(processed_image, caption=f"Predicted Alphabet: {predicted_class}", use_column_width=True)

    elif selected_model == "Cat and Dog Classification":
        
        predicted_class, probability, processed_image = perform_cat_dog_classification(image_path)
        processed_image.save("processed_image_cat_dog.jpg")
        st.image(processed_image, caption=f"Predicted class: {predicted_class} (Confidence: {probability:.2f}%)", use_column_width=True)


# Text data input
text_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze Text"):
    if text_input.strip() == "":
        st.warning("Please enter some text for sentiment analysis.")
    else:
        sentiment = process_sentiment_analysis(text_input)
        st.write(f"Sentiment: {sentiment}")

        
