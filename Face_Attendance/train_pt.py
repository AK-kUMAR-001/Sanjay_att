import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np

# Settings
DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'encodings_pt.pkl' # Saving as pickle for consistency with dictionary structure

def generate_encodings_pt():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize MTCNN for face detection (better than dlib for alignment)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # Initialize Inception Resnet V1 (FaceNet)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Prepare dataset
    # We need a custom loader or just iterate folders manually to keep track of IDs
    
    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found.")
        return

    known_encodings = []
    known_names = []
    known_ids = []

    print("Starting face encoding with PyTorch (FaceNet)...")

    # Walk through dataset
    for student_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, student_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract ID and Name from folder name (e.g., "101_sanjay")
        try:
            student_id, student_name = student_folder.split('_', 1)
        except ValueError:
            print(f"Skipping folder {student_folder}: Invalid format (expected ID_Name)")
            continue

        print(f"Processing {student_name} ({student_id})...")
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # MTCNN expects PIL image or numpy array
                from PIL import Image
                img = Image.open(img_path)
                
                # Get cropped face directly from MTCNN
                # This returns a tensor of shape (3, 160, 160)
                img_cropped = mtcnn(img)

                if img_cropped is not None:
                    # Calculate embedding
                    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
                    
                    # Detach from graph and convert to numpy
                    embedding_numpy = img_embedding.detach().cpu().numpy()[0]
                    
                    known_encodings.append(embedding_numpy)
                    known_names.append(student_name)
                    known_ids.append(student_id)
                else:
                    print(f"  No face found in {img_file}")
            
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")

    # Save encodings
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "ids": known_ids
    }

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\nEncodings saved to {ENCODINGS_FILE}")
    print(f"Total faces encoded: {len(known_encodings)}")

if __name__ == "__main__":
    generate_encodings_pt()
