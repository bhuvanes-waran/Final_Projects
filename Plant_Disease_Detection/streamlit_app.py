# import the rquired libraries.
import random
import os
import numpy as np
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# Load model.
# classifier = load_model('D:/BHUVI/GUVI/Projects/Final_Project/Plant Disease Detection from Images/model.keras')
classifier = load_model('./cnn_model.keras')

class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def main():
    # Plant Disease Detection Application #
    st.title('''Plant Disease Detection by Uploaded Image Application''')
    activiteis = ["Plant Disease Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Bhuvaneswaran R
            [LinkedIn](https://www.linkedin.com/in/b-h-u-v-a-n-e-s-w-a-r-a-n-r-1b1b59188/)""")

    # Plant Disease Detection.
    if choice == "Plant Disease Detection":
        st.subheader('''Get ready with all the images. ''')
        disease = ""

        # Upload image
        uploaded_file = st.file_uploader("Upload an image of the diseased plant leaf", type=["jpg", "png", "jpeg"])

        true_labels = []
        predicted_labels = []
        images = []

        if uploaded_file is not None:
            # Load and preprocess the uploaded image
            image = Image.open(uploaded_file)
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = classifier.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Extract the label (name of image without extension)
            true_label = uploaded_file.name.split('.')[0]

            # Get the predicted class label
            predicted_label = class_labels[predicted_class]

            # Append results for display
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            images.append(image)

        # Randomly select three images if there are enough uploaded
        num_images = len(images)
        selected_indices = random.sample(range(num_images), min(3, num_images))

        # Show selected images
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(selected_indices):
            plt.subplot(1, 3, i + 1)
            plt.imshow(images[idx])
            plt.title(f'True: {true_labels[idx]}\nPredicted: {predicted_labels[idx]}')
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(plt)

        # plt.tight_layout()
        # st.pyplot(plt)
        

    # About.
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts plant disease using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                    </h4>
                                    </div>
                                    </br>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)
    else:
        pass

if __name__ == "__main__":
    main()