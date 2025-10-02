import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN
import pickle
#u can use any best fit model which ever gives the good accuracy.
# Path setup
dataset_dir = "dataset"  # Directory containing subfolders with images for each person
names_and_usns_path = os.path.join("data", "names_and_usns.pkl")  # Save the label mapping
processed_dir = os.path.join("processed_faces")  # Directory to store cropped faces
model_path = os.path.join("data", "Trainer_CNN.h5")  # Path to save the trained model
os.makedirs(processed_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)

def extract_faces_with_mtcnn(image_path, detector, image_size=(100, 100)):
    """
    Detects and extracts faces using MTCNN from a given image.
    Resizes faces to the specified size.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN
    detections = detector.detect_faces(image_rgb)

    faces = []
    for detection in detections:
        if detection['confidence'] > 0.9:  # Confidence threshold
            x, y, width, height = detection['box']
            x, y = max(0, x), max(0, y)  # Ensure box is within bounds
            face = image_rgb[y:y + height, x:x + width]
            face_resized = cv2.resize(face, image_size)
            faces.append(face_resized)

    return faces

def process_dataset_with_mtcnn(dataset_dir, detector, image_size=(100, 100)):
    """
    Processes the dataset directory, detects faces using MTCNN, and saves cropped faces.
    """
    label_map = {}  # Dictionary to map folder names to labels
    current_label = 0
    processed_images = []

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label_map[folder] = current_label
        folder_save_path = os.path.join(processed_dir, str(current_label))
        os.makedirs(folder_save_path, exist_ok=True)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if not image_name.lower().endswith(('jpg', 'jpeg', 'png')):
                continue

            faces = extract_faces_with_mtcnn(image_path, detector, image_size)
            for i, face in enumerate(faces):
                save_path = os.path.join(folder_save_path, f"{os.path.splitext(image_name)[0]}_face_{i}.png")
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                cv2.imwrite(save_path, face_bgr)
                processed_images.append((save_path, current_label))

        current_label += 1

    return label_map, processed_images

def create_and_train_model(processed_images, num_classes, image_size=(100, 100)):
    """
    Creates a CNN model, trains it on the processed faces, and returns the trained model.
    """
    # Prepare the data for training
    faces = []
    labels = []
    for image_path, label in processed_images:
        face = cv2.imread(image_path)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face, image_size)
        faces.append(face_resized)
        labels.append(label)

    # Convert to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)

    # Normalize the images
    faces = faces.astype('float32') / 255.0

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes)

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(faces, labels, epochs=10, batch_size=32, validation_split=0.2)

    return model

# Initialize MTCNN face detector
mtcnn_detector = MTCNN()

# Process the dataset
print("Processing dataset using MTCNN...")
label_map, processed_images = process_dataset_with_mtcnn(dataset_dir, mtcnn_detector)

# Save the label map to a file
with open(names_and_usns_path, 'wb') as f:
    pickle.dump(list(label_map.items()), f)
print(f"Label mapping saved to {names_and_usns_path}.")

# Train the model
num_classes = len(label_map)
print(f"Training model with {num_classes} classes...")
model = create_and_train_model(processed_images, num_classes)

# Save the trained model
model.save(model_path)
print(f"Model saved to {model_path}.")

# Summary
if len(label_map) > 0:
    print("Processing Summary:")
    for folder, label in label_map.items():
        print(f"Label {label}: {folder}")
else:
    print("No data found for processing! Make sure the dataset directory is populated.")

