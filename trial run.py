import torch
from torchvision import models, transforms
from PIL import Image
import json
from vit_pytorch import ViT
import torch.nn as nn
import os
import matplotlib.pyplot as plt
classes = ["PASS", "INVERTED", "NONE", "TEAR", "OUTPOSITION", "DOUBLE", "SEAL"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViT(
    image_size=224,          
    patch_size=32,
    num_classes=7,  
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1)


#model = models.densenet121(pretrained=False)
#num_ftrs = model.classifier.in_features
#model.classifier = torch.nn.Linear(num_ftrs, len(classes)) 

#model = models.resnet101(pretrained=False)
#num_ftrs = model.fc.in_features #resnet function
#model.fc = nn.Linear(num_ftrs, 7)  # resnet

model.load_state_dict(torch.load('D:\Adrian Leong\Coding\VanillaViT_V3B32lr00001.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing pipeline matching training transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],  
        #std=[0.229, 0.224, 0.225])
    
])



def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img).unsqueeze(0).to(device)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1) 
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')


    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    predicted_class = classes[pred_idx.item()]
    plt.subplot(1, 2, 2)
    text = f'Prediction: {classes[pred_idx.item()]}\nConfidence: {confidence.item():.2%}'
    plt.text(0.1, 0.5, text, fontsize=12, va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return predicted_class, confidence.item()

if __name__ == "__main__":
    test_image_path = r"D:\Adrian Leong\Coding\Dataset\second dataset\NONE\Cam0_Recoverable6_SFL-05 (no lens detected)Img.2025-04-21T17-29-23-697.bmp"  # Change to your test image path
    pred_class, conf = predict_image(test_image_path)
    print(f"Predicted class: {pred_class} with confidence {conf:.4f}")

#"D:\Adrian Leong\Coding\Dataset\second dataset\DOUBLE\Cam0_Recoverable74_SFL-06 (multiple lenses detected)Img.bmp" double
#"D:\Adrian Leong\Coding\Dataset\second dataset\INVERTED\Cam0_Recoverable143_SFL-05 (no lens detected)Img.bmp" inverted
#"D:\Adrian Leong\Coding\Dataset\second dataset\NONE\Cam0_Recoverable6_SFL-05 (no lens detected)Img.2025-04-21T17-29-23-697.bmp" none
#"D:\Adrian Leong\Coding\Dataset\second dataset\PASS\Good.Img. (80).bmp"
#"D:\Adrian Leong\Coding\Dataset\second dataset\OUTPOSITION\Cam0_Recoverable198_SFL-07 (out of position)Img.bmp" out of position
#"D:\Adrian Leong\Coding\Dataset\second dataset\SEAL\Cam1_Recoverable37_SSAE-01 (debris in outer seal area)Img.bmp" lens in seal
#"D:\Adrian Leong\Coding\Dataset\second dataset\TEAR\Cam0_Recoverable185_STIP-01 (Torn In Pack)Img.bmp" tear