import os
import cv2
import torch
import numpy as np
from collections import deque
import torchvision.transforms as transforms

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define image transformation for webcam frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

model = None 

# Load the model from file if available; otherwise download and save it
def load_model():
    global model
    model_path = 'cifar10_resnet56.pth'

    if os.path.isfile(model_path):
        print(f"Found existing model at '{model_path}'. Loading...")
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=False)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Downloading pretrained model...")
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to '{model_path}'.")

    model.to(device)
    model.eval()

# Return the most common prediction from the last N predictions (Avoids jittery labels)
def smooth(predictions):
    if not predictions:
        return None
    counts = np.bincount(predictions)
    return np.argmax(counts)

def classify():
    # Open default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam classification started. (Focus the window and press 'q' to quit.)")
    cv2.namedWindow('CIFAR-10 Webcam Classifier')

    # Keep track of last 20 predictions for smoothing
    prediction_history = deque(maxlen=20)

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        img_tensor = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            # Get predicted class
            prediction = output.argmax(dim=1).item()
            # Add prediction to history
            prediction_history.append(prediction)

        # Smooth prediction
        smoothed_prediction = smooth(prediction_history)
        label = CIFAR10_CLASSES[smoothed_prediction] if smoothed_prediction is not None else "Predicting..."

        # Draw label
        cv2.putText(
            frame,
            f"Prediction: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display the frame
        cv2.imshow('CIFAR-10 Webcam Classifier', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    load_model()
    classify()
