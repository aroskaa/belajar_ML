import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def classify_new_image():
    # Hide the main tkinter window
    Tk().withdraw()

    # Open file dialog to select an image file
    image_path = askopenfilename()

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Load the trained model
    model = load_model("my_model.keras")

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to the corresponding label
    labels = ["rock", "paper", "scissors"]
    predicted_label = labels[predicted_class]

    print(f"Gambar tersebut diklasifikasikan sebagai: {predicted_label}")


# Call the function
classify_new_image()
