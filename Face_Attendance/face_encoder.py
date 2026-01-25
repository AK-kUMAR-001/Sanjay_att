import os
import pickle
import face_recognition
import cv2

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
ENCODINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encodings.pkl")

def generate_encodings():
    known_encodings = []
    known_names = []
    known_ids = []

    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} directory not found.")
        return

    print("Starting face encoding...")
    
    # Loop through student folders
    for student_folder in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student_folder)
        
        if not os.path.isdir(student_path):
            continue
            
        # Parse Folder name: 101_Rahul -> ID: 101, Name: Rahul
        try:
            student_id, student_name = student_folder.split('_', 1)
        except ValueError:
            print(f"Skipping folder {student_folder}: Invalid format. Use ID_Name.")
            continue
            
        print(f"Processing {student_name} ({student_id})...")
        
        # Process images
        for filename in os.listdir(student_path):
            image_path = os.path.join(student_path, filename)
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes = face_recognition.face_locations(rgb_image, model="hog")
            # num_jitters=10 acts like "echos" to improve accuracy by resampling the face multiple times
            encodings = face_recognition.face_encodings(rgb_image, boxes, num_jitters=10)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student_name)
                known_ids.append(student_id)
            else:
                print(f"No face found in {filename}")

    # Save encodings
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "ids": known_ids
    }
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Encodings saved to {ENCODINGS_FILE}")

if __name__ == "__main__":
    generate_encodings()
