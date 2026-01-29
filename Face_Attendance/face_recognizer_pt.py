import torch
from facenet_pytorch import InceptionResnetV1
import pickle
import numpy as np
import os
from PIL import Image

# Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings_pt.pkl")
THRESHOLD = 0.6 # Distance threshold for FaceNet (lower is stricter)

class FaceRecognizerPT:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.load_encodings()

    def load_encodings(self):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.known_ids = data["ids"]
            print(f"PyTorch Encodings loaded successfully. ({len(self.known_encodings)} faces)")
        except FileNotFoundError:
            print("Error: encodings_pt.pkl not found. Run train_pt.py first.")

    def recognize_face(self, face_image_np):
        """
        face_image_np: Numpy array of the cropped face (RGB)
        """
        if not self.known_encodings:
            return "Unknown", None

        try:
            # Preprocess image for FaceNet
            img = Image.fromarray(face_image_np)
            img = img.resize((160, 160))
            
            # Convert to tensor and normalize (FaceNet expects whitened image if not using mtcnn.extract)
            # Standard normalization for InceptionResnetV1 in facenet-pytorch
            img_tensor = torch.tensor(np.array(img)).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5 # Normalize to [-1, 1]
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Get embedding
            embedding = self.resnet(img_tensor).detach().cpu().numpy()[0]

            # Compare with known encodings (Euclidean distance)
            # Calculate distances to all known encodings
            distances = np.linalg.norm(self.known_encodings - embedding, axis=1)
            
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist < THRESHOLD:
                name = self.known_names[min_dist_idx]
                student_id = self.known_ids[min_dist_idx]
                return name, student_id, min_dist
            else:
                return "Unknown", None, min_dist

        except Exception as e:
            print(f"Error in PT recognition: {e}")
            return "Error", None, 0.0
