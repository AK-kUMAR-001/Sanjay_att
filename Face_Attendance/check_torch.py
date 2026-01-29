import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("FaceNet-PyTorch imported successfully")
except Exception as e:
    print(f"FaceNet import failed: {e}")
