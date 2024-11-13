# import the rquired libraries.
import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
# classifier =load_model('D:/BHUVI/GUVI/Projects/Final_Project/Emotion Detection from Uploaded Images/model.keras')
classifier =load_model('./model.keras')

# load weights into new model
# classifier.load_weights("D:/BHUVI/GUVI/Projects/Final_Project/Emotion Detection from Uploaded Images/model.weights.h5")
classifier.load_weights("./model.weights.h5")

# Load face using OpenCV
try:
    # face_cascade = cv2.CascadeClassifier('D:/BHUVI/GUVI/Projects/Final_Project/Emotion Detection from Uploaded Images/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # st.write("face cascade path set successfully | " + cv2.data.haarcascades)
except Exception:
    st.write("Error loading cascade classifiers")

def main():
    # Face Emotion Detection Application #
    st.title('''Emotion Detection by Uploaded Image Application 
    😠🤮😨😀😐😔😮''')
    activiteis = ["Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Bhuvaneswaran R
            [LinkedIn](https://www.linkedin.com/in/b-h-u-v-a-n-e-s-w-a-r-a-n-r-1b1b59188/)""")

    # Emotion Detection.
    if choice == "Emotion Detection":
        st.subheader('''Get ready with all the emotions you can express. ''')
        emotion = ""

        # Upload image
        uploaded_file = st.file_uploader("Upload an image of the emotion", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:

            # Read the uploaded image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if img is None:
                st.write("Failed to decode the image. Please upload a valid image.")
                st.stop()  # Stops further execution if the image can't be decoded
            else:
                st.write("Image uploaded successfully.")

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                st.write("No faces detected in the image. Use a different image")
            
            # Draw rectangles around the faces and predict the emotion
            for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(
                    x + w, y + h), color=(0, 255, 255), thickness=2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = emotion_labels[maxindex]
                    output = str(finalout)
                label_position = (x, y-10)
                emotion = output
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the image back to RGB (OpenCV uses BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the image with Streamlit
            st.image(img_rgb, caption="Detected Emotion from the uploaded image - " + emotion, use_column_width=True)

    # About.
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achived through openCV.
                                    </h4>
                                    </div>
                                    </br>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)
    else:
        pass

if __name__ == "__main__":
    main()