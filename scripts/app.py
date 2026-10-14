# 1. Imports and class names setup
import gradio as gr
import os
import torch
from torchvision import transforms
from model import HybridModel
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image
import numpy as np
import spaces  

# Setup class names
with open("class_names.txt", "r") as f:
    class_names = [label.strip() for label in f.readlines()]

# 2. Model and transform preparation

# Correct Spatial Transform for Evaluation (No Augmentations)
eval_transform_spatial = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Frequency domain preprocessing function from your training dataset
def to_frequency_domain(img_pil: Image.Image) -> torch.Tensor:
    # Resize
    resize = transforms.Resize((224, 224))
    img_resized = resize(img_pil)
    
    # Convert PIL Image to numpy array -> torch Tensor -> permute -> normalize [0,1]
    img_np = np.array(img_resized)       
    img_tensor = torch.from_numpy(img_np).float() 
    img_tensor = img_tensor.permute(2, 0, 1) 
    img_tensor = img_tensor / 255.0          

    # Apply FFT, shift zero frequency to center, get magnitude
    img_fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    img_fft_shifted = torch.fft.fftshift(img_fft, dim=(-2, -1))
    img_mag = torch.abs(img_fft_shifted) 

    # Average magnitude across channels
    img_mag_avg = img_mag.mean(dim=0, keepdim=True) 

    # Apply log compression
    img_freq = torch.log1p(img_mag_avg) 

    # Apply standard normalization for frequency branch
    frequency_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    img_freq = frequency_normalize(img_freq)

    return img_freq.float()

# Setup device to utilize ZeroGPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = HybridModel(feature_dim=256, fusion_hidden_dim=128, lambda_grl=1.0)

# Load saved weights
checkpoint = torch.load("hybrid_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device) 
model.eval()

# 3. predict function
@spaces.GPU  
def predict(img) -> Tuple[Dict, float]:
    start_time = timer()
    
    # Transform input image using the corrected spatial and frequency pipelines
    spatial_img = eval_transform_spatial(img).unsqueeze(0).to(device)
    freq_img = to_frequency_domain(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.inference_mode():
        class_output, _ = model(spatial_img, freq_img)
        # Binary classification (sigmoid instead of softmax)
        pred_prob = torch.sigmoid(class_output).squeeze().item()
        
    # Create pred label and prob dict (Fixed syntax error)
    pred_dict = {
        class_names[1]: float(pred_prob),
        class_names[0]: float(1.0 - pred_prob)
    }
    
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    return pred_dict, pred_time

# 4. Gradio app interface
title = "Fake Media Detector"
description = "A Hybrid Spatial-Frequency neural network to classify Real vs AI-Generated images."
article = "Created for detecting synthetic media using dual-branch feature extraction."

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples/")]

# Create the Gradio demo
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

if __name__ == "__main__":
    demo.launch()
