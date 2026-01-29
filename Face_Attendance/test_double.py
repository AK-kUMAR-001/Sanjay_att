import cv2
import face_recognition
import os
import pickle
import random
from face_recognizer import FaceRecognizer
from face_recognizer_pt import FaceRecognizerPT
import numpy as np

DATASET_DIR = "dataset"

def test_predictions_double():
    print("Initializing models...")
    rec_dlib = FaceRecognizer()
    rec_pt = FaceRecognizerPT()
    
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found.")
        return

    print("\n=== STARTING DOUBLE ACCURACY TEST ===")
    
    for student_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, student_folder)
        if not os.path.isdir(folder_path):
            continue

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        # Test 1 random image per student
        test_images = random.sample(images, 1)
        
        print(f"\nTesting Student: {student_folder}")
        
        for img_name in test_images:
            img_path = os.path.join(folder_path, img_name)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None: continue
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces (using Dlib as primary detector)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                print(f"  [FAIL] No face detected in {img_name}")
                continue
                
            # Take first face
            face_encoding = face_encodings[0]
            top, right, bottom, left = face_locations[0]
            face_img = rgb_image[top:bottom, left:right]
            
            # 1. Dlib
            name_dlib, id_dlib = rec_dlib.recognize_face(face_encoding)
            
            # 2. PyTorch
            name_pt, id_pt, dist_pt = rec_pt.recognize_face(face_img)
            
            # Result
            status = "UNKNOWN"
            if name_dlib != "Unknown" and name_pt != "Unknown":
                if id_dlib == id_pt:
                    status = f"VERIFIED ({name_dlib})"
                else:
                    status = f"CONFLICT ({name_dlib} vs {name_pt})"
            elif name_dlib != "Unknown":
                status = f"DLIB ONLY ({name_dlib})"
            elif name_pt != "Unknown":
                status = f"PT ONLY ({name_pt})"
            
            print(f"  {img_name} -> {status}")
            print(f"     Details: Dlib={name_dlib}, PT={name_pt} (Dist: {dist_pt:.3f})")

    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_predictions_double()
