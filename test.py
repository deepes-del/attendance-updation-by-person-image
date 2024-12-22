import os
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import winsound  # For beep sound
import csv
from mtcnn import MTCNN
from scipy.spatial import distance as dist

# Eye aspect ratio calculation (for blink detection)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Blink detection function
def detect_blink(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

# Paths to data files
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "data", "Trainer_CNN.h5")
names_and_usns_path = os.path.join(script_dir, "data", "names_and_usns.pkl")
attendance_dir = os.path.join(script_dir, "attendance")
os.makedirs(attendance_dir, exist_ok=True)

# Load names and USNs
with open(names_and_usns_path, 'rb') as f:
    names_and_usns = pickle.load(f)

# Load the trained CNN model
model = load_model(model_path)

# Initialize video capture and MTCNN detector
video_capture = cv2.VideoCapture(0)
mtcnn_detector = MTCNN()

# Blink detection and liveness parameters
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
blink_counter = 0
liveness_confirmed = False

# Attendance set
attendance = set()  # To store recorded attendance for the session

# Recognition loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = mtcnn_detector.detect_faces(frame_rgb)

    for detection in detections:
        if detection['confidence'] > 0.9:  # Confidence threshold
            x, y, width, height = detection['box']
            x, y = max(0, x), max(0, y)  # Ensure box is within bounds
            face_image = frame_rgb[y:y + height, x:x + width]

            # Resize face_image to the size used during training (100x100)
            face_image_resized = cv2.resize(face_image, (100, 100)) / 255.0
            face_image_expanded = np.expand_dims(face_image_resized, axis=(0, -1))

            # Perform face recognition
            predictions = model.predict(face_image_expanded)
            label = np.argmax(predictions)
            confidence = predictions[0][label]

            if confidence < 0.80:  # Unknown person
                print("Unknown person detected!")
                # Record attendance for unknown person
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                full_name, usn = "Unknown", "Unknown"
                if (full_name, usn) not in attendance:
                    attendance.add((full_name, usn, current_time))
                    winsound.Beep(1000, 500)
                    print(f"Attendance recorded for Unknown person")

                    # Save attendance to CSV
                    date = datetime.now().strftime('%Y-%m-%d')
                    file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["Name", "USN", "Time"])
                        writer.writerow([full_name, usn, current_time])
                        print(f"Attendance saved for Unknown person.")
            else:
                full_name, usn = names_and_usns[label]
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if (full_name, usn) not in attendance:
                    attendance.add((full_name, usn, current_time))
                    winsound.Beep(1000, 500)
                    print(f"Attendance recorded for {full_name}")

                    # Save attendance to CSV
                    date = datetime.now().strftime('%Y-%m-%d')
                    file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["Name", "USN", "Time"])
                        writer.writerow([full_name, usn, current_time])
                        print(f"Attendance saved for {full_name}.")

            # Exit after recording the first face (known or unknown)
            break

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit manually (if no face detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If attendance has been recorded, exit the loop
    if len(attendance) > 0:
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
