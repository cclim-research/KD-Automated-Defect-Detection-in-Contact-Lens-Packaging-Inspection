import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize model
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(r'D:\Adrian Leong\Coding\ResNet101_V2B64lr00001.pth', map_location=device))
model = model.to(device)
model.eval()

classes = ["PASS", "INVERTED", "NONE", "TEAR", "OUTPOSITION", "DOUBLE", "SEAL"]

def preprocess_and_save(image_path, save_dir):
    """Preprocess image and save transformed version"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process and save image
    image = Image.open(image_path).convert('RGB')
    transformed_tensor = transform(image)
    pil_image = transforms.ToPILImage()(transformed_tensor)
    
    # Save with same filename stem
    save_path = os.path.join(save_dir, f"{Path(image_path).stem}_preprocessed.jpg")
    pil_image.save(save_path)
    return save_path

def predict_from_preprocessed(preprocessed_path):
    """Predict from already preprocessed image"""
    # Load preprocessed image
    image = Image.open(preprocessed_path).convert('RGB')

    # Create a figure to display both image and prediction
    plt.figure(figsize=(8, 4))
    
    # Show the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    # Convert to tensor (already normalized)
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = F.softmax(output[0], dim=0)
    top_prob, top_catid = torch.max(probabilities, 0)
    # Show prediction results
    plt.subplot(1, 2, 2)
    text = f'Prediction: {classes[top_catid.item()]}\nConfidence: {top_prob.item():.2%}'
    plt.text(0.1, 0.5, text, fontsize=12, va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return classes[top_catid.item()], top_prob.item()

# Example usage
raw_image_path = r"D:\Adrian Leong\Coding\Dataset\New Dataset\DOUBLE\Cam0_Recoverable35_SFL-06 (multiple lenses detected)Img.bmp"
save_directory = r"D:\Adrian Leong\Coding\Preprocessed_Images"

# 1. Preprocess and save
preprocessed_path = preprocess_and_save(raw_image_path, save_directory)

# 2. Classify from saved preprocessed image
predicted_class, confidence = predict_from_preprocessed(preprocessed_path)
print(f'Predicted: {predicted_class} with confidence: {confidence:.2%}')