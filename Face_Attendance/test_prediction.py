import cv2
import face_recognition
import os
import pickle
import random

# Load encodings
ENCODINGS_FILE = "encodings.pkl"
DATASET_DIR = "dataset"

def load_encodings():
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None

def test_predictions():
    data = load_encodings()
    if not data:
        return

    known_encodings = data["encodings"]
    known_names = data["names"]
    known_ids = data["ids"]

    print("=== STARTING RANDOM PREDICTION TEST ===")
    
    # Iterate through each student folder
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found.")
        return

    for student_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, student_folder)
        if not os.path.isdir(folder_path):
            continue

        # Get all images
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        # Pick 2 random images
        test_images = random.sample(images, min(2, len(images)))
        
        print(f"\nTesting Student: {student_folder}")
        
        for img_name in test_images:
            img_path = os.path.join(folder_path, img_name)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [WARN] Could not read {img_name}")
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                print(f"  [FAIL] No face detected in {img_name}")
                continue
                
            # Recognize
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                
                best_match_index = -1
                if len(face_distances) > 0:
                    best_match_index = list(face_distances).index(min(face_distances))
                
                if best_match_index != -1 and matches[best_match_index]:
                    predicted_name = known_names[best_match_index]
                    predicted_id = known_ids[best_match_index]
                    result = "PASS" if str(predicted_id) in student_folder else "FAIL"
                    print(f"  [{result}] {img_name} -> Predicted: {predicted_name} ({predicted_id})")
                else:
                    print(f"  [FAIL] {img_name} -> Unknown")

    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_predictions()
