import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# TF MODEL PREDICTION
def model_prediction(test_image_path):
    cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = cnn.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Callback function for 'Predict' button
def predict_image():
    test_image_path = filedialog.askopenfilename(title="Choose an Image")
    if test_image_path:
        image = Image.open(test_image_path)
        image.thumbnail((300, 300))  # Resizing image for display
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo
        result_index = model_prediction(test_image_path)
        result_label.config(text="Model predicts it's a {}".format(class_name[result_index]))

# Main Tkinter window
root = tk.Tk()
root.title("Plant Disease Recognition System")

# Sidebar
sidebar = tk.Frame(root, width=200, bg='lightgrey', padx=10, pady=10)
sidebar.grid(row=0, column=0, sticky='ns')
tk.Label(sidebar, text="Dashboard", font=("Helvetica", 16), bg='lightgrey').pack(pady=20)
tk.Button(sidebar, text="Home", command=lambda: change_page("Home")).pack()
tk.Button(sidebar, text="About", command=lambda: change_page("About")).pack()
tk.Button(sidebar, text="Disease Recognition", command=lambda: change_page("Disease Recognition")).pack()

# Content area
content = tk.Frame(root, padx=20, pady=20)
content.grid(row=0, column=1, sticky='nsew')

# Home page
def show_home_page():
    content.grid_forget()
    content.grid(row=0, column=1, sticky='nsew')
    home_text = """
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    How It Works:
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    Our best features:
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    Get Started:
    Click on the Disease Recognition page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    About Us:
    Learn more about the project, our team, and our goals on the About page.
    """
    tk.Label(content, text=home_text, font=("Helvetica", 14)).pack()

# About page
def show_about_page():
    content.grid_forget()
    content.grid(row=0, column=1, sticky='nsew')
    about_text = """
    About Dataset:
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    Content:
    1. train (37940 images)
    2. test (33 images)
    3. validation (17572 images)
    """
    tk.Label(content, text=about_text, font=("Helvetica", 14)).pack()

# Disease Recognition page
def show_disease_recognition_page():
    content.grid_forget()
    content.grid(row=0, column=1, sticky='nsew')
    tk.Label(content, text="Disease Recognition", font=("Helvetica", 16)).pack()
    browse_button = tk.Button(content, text="Choose an Image", command=predict_image)
    browse_button.pack(pady=10)
    global image_label
    image_label = tk.Label(content)
    image_label.pack(pady=10)
    global result_label
    result_label = tk.Label(content, font=("Helvetica", 14))
    result_label.pack(pady=10)
    
# Function to change page based on sidebar selection
def change_page(page_name):
    if page_name == "Home":
        show_home_page()
    elif page_name == "About":
        show_about_page()
    elif page_name == "Disease Recognition":
        show_disease_recognition_page()

# Load class names for predictions
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

# Show the home page by default
