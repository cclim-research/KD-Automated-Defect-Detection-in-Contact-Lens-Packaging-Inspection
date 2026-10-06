import torch
from torchvision import models, transforms
from vit_pytorch import ViT
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import os
import matplotlib.pyplot as plt
classes = ["PASS", "INVERTED", "NONE", "TEAR", "OUTPOSITION", "DOUBLE", "SEAL"]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ViT(
    image_size=256,          
    patch_size=32,
    num_classes=7,  
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1)

#ResNEt
#model = models.resnet101(pretrained=False)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 7)  # Match the number of classes

#DenseNet
#model = models.densenet121(pretrained=False)
#num_ftrs = model.classifier.in_features
#model.classifier = torch.nn.Linear(num_ftrs, len(classes)) 

# Load weights (NOT the entire model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(r"D:\Adrian Leong\Coding\VanillaViT_V2B64lr00001.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ["PASS", "INVERTED", "NONE", "TEAR", "OUTPOSITION", "DOUBLE", "SEAL"]

def preprocess_and_save(image_path, save_dir):
    """Preprocess image and save transformed version"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process and save image
    image = Image.open(image_path).convert('RGB')
     #used if want to display image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1) 
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    transformed_tensor = transform(image)
    pil_image = transforms.ToPILImage()(transformed_tensor)
    #this section is scallable
    
    # Save with same filename stem
    save_path = os.path.join(save_dir, f"{Path(image_path).stem}_preprocessed.jpg")
    pil_image.save(save_path)
    return save_path

def predict_image(image_path):
    image = Image.open(preprocessed_path).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = F.softmax(output[0], dim=0)
    top_prob, top_catid = torch.max(probabilities, 0)

    plt.subplot(1, 2, 2)
    text = f'Prediction: {classes[top_catid.item()]}\nConfidence: {top_prob.item():.2%}'
    plt.text(0.1, 0.5, text, fontsize=12, va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return classes[top_catid.item()], top_prob.item()

# Test
raw_image_path = r"D:\Adrian Leong\Coding\Dataset\New Dataset\DOUBLE\Cam1_Recoverable333_SFL-06 (multiple lenses detected)Img.bmp"
save_directory = r"D:\Adrian Leong\Coding\Preprocessed_Images"
preprocessed_path = preprocess_and_save(raw_image_path, save_directory)
#image_path = r"D:\Adrian Leong\Coding\Preprocessed_Images\Cam0_Recoverable35_SFL-06 (multiple lenses detected)Img_preprocessed.jpg"
predicted_class, confidence = predict_image(preprocessed_path)
print(f'Predicted: {predicted_class} with confidence: {confidence:.2%}')

#"D:\Adrian Leong\Coding\Dataset\New Dataset\DOUBLE\Cam1_Recoverable333_SFL-06 (multiple lenses detected)Img.bmp" double
#"D:\Adrian Leong\Coding\Dataset\New Dataset\INVERTED\Cam2_Recoverable221_SFL-05 (no lens detected)Img.bmp" inverted
#"D:\Adrian Leong\Coding\Dataset\New Dataset\NONE\Cam1_Recoverable1_SFL-05 (no lens detected)Img.2025-04-22T15-37-50-738.bmp" none
#"D:\Adrian Leong\Coding\Dataset\New Dataset\PASS\Good.Img. (208).bmp"
#"D:\Adrian Leong\Coding\Dataset\New Dataset\OUTPOSITION\Cam0_Recoverable301_SFL-13 (no lens detected)Img.bmp" out of position
#"D:\Adrian Leong\Coding\Dataset\New Dataset\SEAL\Cam1_Recoverable5_SSAE-01 (debris in outer seal area)Img.bmp" lens in seal
#"D:\Adrian Leong\Coding\Dataset\New Dataset\TEAR\Cam0_Recoverable186_STIP-01 (Torn In Pack)Img.bmp" tear