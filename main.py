import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import cv2
import dlib
import shutil
import numpy as np
from keras.models import load_model


label_dict = {
    0: "happy",
    1: "sad",
    2: "neutral"
}

model = load_model("model_optimal.keras")
detector = dlib.get_frontal_face_detector()



def refresh_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)


def process_image(file_path):
    global label_dict
    global model
    global detector

    # Load the image using OpenCV
    img = cv2.imread(file_path)

    # Convert the image to grayscale (dlib works better on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for i, face in enumerate(faces):
        # Get the coordinates of the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extract the face region
        face_img = gray[y:y+h, x:x+w]
        
        # Resize the face to 48x48 pixels
        face_img_resized = cv2.resize(face_img, (48, 48))

        img = np.array(face_img_resized)
        
        # Normalize the pixel values (if required by the model)
        face_img_normalized = face_img_resized / 255.0
        
        # Reshape the image to match the input shape of the model (1, 48, 48, 1)
        face_img_reshaped = face_img_normalized.reshape(1, 48, 48, 1)
        
        # Pass the face through the model
        prediction = model.predict(face_img_reshaped)
        
        if len(prediction) > 0:
            result = list(prediction[0]) # [0.87, 0.67, 0.341]
            img_index = result.index(max(result))
            print(label_dict[img_index])
            
            if label_dict[img_index] == "happy":
                target_directory = "SMILE"
                os.makedirs(target_directory, exist_ok=True)
                destination_file = os.path.join(target_directory, os.path.basename(file_path))
                shutil.copy(file_path, destination_file)
            else:
                target_directory = "NOT SMILE"
                os.makedirs(target_directory, exist_ok=True)
                destination_file = os.path.join(target_directory, os.path.basename(file_path))
                shutil.copy(file_path, destination_file)


if __name__ == "__main__":
    
    refresh_directories(["SMILE", "NOT SMILE"])

    directory_path = "images"

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(file_path)
        else:
            print(f"Skipping non-image file: {filename}")