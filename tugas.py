import os, shutil, zipfile, sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = "tmp/rockpaperscissors.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("tmp")
zip_ref.close()

# Define directories
base_dir = "tmp/rockpaperscissors"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "val")

# Create directories if they don't exist
for dir in [train_dir, validation_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)


# Function to move files
def move_files(file_names, source_dir, target_dir):
    for file_name in file_names:
        shutil.move(
            os.path.join(source_dir, file_name), os.path.join(target_dir, file_name)
        )


# Function to process images
def process_images(image_dir, train_dir, validation_dir):
    all_files = os.listdir(image_dir)
    train_files, validation_files = train_test_split(all_files, test_size=0.4)

    # Create directories if they don't exist
    for dir in [train_dir, validation_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    move_files(train_files, image_dir, train_dir)
    move_files(validation_files, image_dir, validation_dir)


# Process each category of images
for category in ["rock", "paper", "scissors"]:
    process_images(
        os.path.join(base_dir, category),
        os.path.join(train_dir, category),
        os.path.join(validation_dir, category),
    )

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
)


# Define the model
def train_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(150, 150, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=25,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=5,
        verbose=2,
    )

    model.save("my_model.keras")


# untuk train model
try:
    train_model()
except Exception as e:
    print(f"An error occurred: {e}")
