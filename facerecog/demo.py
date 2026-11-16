#& ".\.venv\Scripts\python.exe" ".\facerecog\demo.py"

import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import csv
from PIL import Image
from datetime import datetime

# ---------------------------------
# PROJECT PATHS
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_faces")

CASCADE_PATH = os.path.join(
    BASE_DIR, "haarcascade_frontalface_alt2.xml"
)

CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier.xml")
LOG_PATH = os.path.join(BASE_DIR, "log.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")

# ---------------------------------
# EMAIL CONFIGURATION
# ---------------------------------

load_dotenv(os.path.join(BASE_DIR, ".env"))

EMAIL = os.getenv("EMAIL_ADDRESS")
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

# Family member ID and name
PERSON_NAMES = {
    1: "Vinit",
    2: "Khushi",
    3: "Tanisha"
}

SAMPLE_COUNT = 20


# ---------------------------------
# 1. GENERATE DATASET
# ---------------------------------

def generate_dataset(user_id, sample_count=SAMPLE_COUNT):

    face_classifier = cv2.CascadeClassifier(CASCADE_PATH)

    if face_classifier.empty():
        print("Error: Haar Cascade file not loaded!")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not available!")
        return

    img_id = 0

    print(f"Capturing face samples for User {user_id}")
    print("Look at the camera and move your face slowly.")
    print("Press ENTER to stop.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame not available.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y + h, x:x + w]

            face = cv2.resize(face, (200, 200))

            img_id += 1

            file_path = os.path.join(
                DATA_DIR,
                f"user.{user_id}.{img_id}.jpg"
            )

            cv2.imwrite(file_path, face)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Samples: {img_id}/{sample_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2              
            )

            if img_id >= sample_count:
                break

        cv2.imshow("Dataset Collection", frame)

        if cv2.waitKey(1) == 13 or img_id >= sample_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Dataset collection completed: {img_id} images")


# ---------------------------------
# 2. TRAIN CLASSIFIER
# ---------------------------------

def train_classifier():

    if not os.path.exists(DATA_DIR):
        print("Data folder not found!")
        return

    faces = []
    ids = []

    for image_name in os.listdir(DATA_DIR):

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(DATA_DIR, image_name)

        try:
            image = Image.open(image_path).convert("L")
            image_np = np.array(image, dtype="uint8")

            # Filename format: user.ID.SAMPLE.jpg
            parts = image_name.split(".")

            if len(parts) < 4 or parts[0] != "user":
                print("Skipping invalid filename:", image_name)
                continue

            user_id = int(parts[1])

            faces.append(image_np)
            ids.append(user_id)

        except Exception as error:
            print("Could not read:", image_name, error)

    if len(faces) == 0:
        print("No face images found. Collect dataset first.")
        return

    ids = np.array(ids, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(faces, ids)

    recognizer.write(CLASSIFIER_PATH)

    print("Training completed successfully!")
    print("Total images:", len(faces))
    print("Model saved as classifier.xml")


# ---------------------------------
# 3. LOG RECOGNITION
# ---------------------------------

def log_recognition(user_id):

    file_exists = os.path.exists(LOG_PATH)


    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_PATH, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists or os.path.getsize(LOG_PATH) == 0:
            writer.writerow(["User ID", "Name", "Date and Time"])

        name = PERSON_NAMES.get(user_id, "Unknown ID")

        writer.writerow([user_id, name, now])

    print(f"Logged: {name} at {now}")

def mark_attendance(user_id):
    
     name = PERSON_NAMES.get(user_id, "Unknown")
     today = datetime.now().strftime("%Y-%m-%d")
     current_time = datetime.now().strftime("%H:%M:%S")

     file_exists = os.path.exists(ATTENDANCE_PATH)

     if file_exists:
        with open(ATTENDANCE_PATH, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)

            for row in reader:
                if len(row) >= 3 and row[0] == str(user_id) and row[2] == today:
                    return

     with open(ATTENDANCE_PATH, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(ATTENDANCE_PATH) == 0:
            writer.writerow(["User ID", "Name", "Date", "Time", "Status"])

        writer.writerow([user_id, name, today, current_time, "Present"])

     print(f"Attendance Marked: {name}")


# ---------------------------------
# SEND UNKNOWN FACE EMAIL ALERT
# ---------------------------------

def send_unknown_alert(image_path):

    if not EMAIL or not APP_PASSWORD:
        print("Email configuration missing!")
        return

    msg = EmailMessage()
    msg["Subject"] = "Unknown Face Detected!"
    msg["From"] = EMAIL
    msg["To"] = EMAIL

    msg.set_content(
        "Alert!\n\n"
        "An unknown face was detected by your Face Recognition System.\n"
        "The captured image is attached."
    )

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        msg.add_attachment(
            image_data,
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path)
        )

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(EMAIL, APP_PASSWORD)
            smtp.send_message(msg)

        print("Unknown face email alert sent!")

    except Exception as error:
        print("Email sending failed:", repr(error))


# ---------------------------------
# 4. SAVE UNKNOWN FACE
# ---------------------------------

def save_unknown_face(face_img):

    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"unknown_{now}.jpg"

    image_path = os.path.join(UNKNOWN_DIR, filename)

    cv2.imwrite(image_path, face_img)

    print("Unknown face saved:", image_path)

    return image_path


# ---------------------------------
# 5. RECOGNIZE FACE
# ---------------------------------

def recognize_face():

    if not os.path.exists(CLASSIFIER_PATH):
        print("Classifier model not found!")
        print("First collect images and train the classifier.")
        return

    if os.path.getsize(CLASSIFIER_PATH) == 0:
        print("Classifier file is empty!")
        print("Please train the classifier first.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        print("Haar Cascade not loaded!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        recognizer.read(CLASSIFIER_PATH)
    except Exception as error:
        print("Could not load trained model:", error)
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not available!")
        return

    print("Face Recognition Started!")
    print("Press ENTER to stop.")

    last_unknown_alert = None
    unknown_save_count = 0
    MAX_UNKNOWN_IMAGES = 3

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame failed!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y + h, x:x + w]

            face = cv2.resize(face, (200, 200))
            # face = cv2.equalizeHist(face)

            user_id, confidence = recognizer.predict(face)
            print("Predicted ID:", user_id, "| Confidence:", confidence)

            if confidence < 80:

                name = PERSON_NAMES.get(
                    user_id, f"User {user_id}"
                )

                label = f"{name} ({confidence:.1f})"

                color = (0, 255, 0)

                log_recognition(user_id)
                mark_attendance(user_id)

            else:
                label = "Unknown"
                color = (0, 0, 255)

                now = datetime.now()

                if (
                    unknown_save_count < MAX_UNKNOWN_IMAGES
                    and (
                        last_unknown_alert is None
                        or (now - last_unknown_alert).total_seconds() >= 10
                    )
                ):
                 image_path = save_unknown_face(frame[y:y + h, x:x + w])

                 send_unknown_alert(image_path)

                 last_unknown_alert = now
                 unknown_save_count += 1       

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Face Recognition Stopped!")


# ---------------------------------
# 6. MAIN MENU
# ---------------------------------

if __name__ == "__main__":

    while True:

        print("\n========== FACE RECOGNITION SYSTEM ==========")
        print("1. Generate Family Dataset")
        print("2. Train Classifier")
        print("3. Recognize Face")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":

            try:
                user_id = int(input("Enter family member ID: "))

                if user_id not in PERSON_NAMES:
                    print("ID not found in PERSON_NAMES.")
                    continue

                generate_dataset(user_id)

            except ValueError:
                print("Please enter a valid numeric ID.")

        elif choice == "2":

            train_classifier()

        elif choice == "3":

            recognize_face()

        elif choice == "0":

            print("Program closed.")
            break

        else:

            print("Invalid choice. Please try again.")
