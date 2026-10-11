# 1. Imports and class names setup
import gradio as gr
import os
import torch
from torchvision import transforms
from model import HybridModel
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image

# Setup class names
with open("class_names.txt", "r") as f:
    class_names = [label.strip() for label in f.readlines()]

# 2. Model and transform preparation
# Setup transforms
transform_spatial = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_frequency = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize model
model = HybridModel(feature_dim=256, fusion_hidden_dim=128, lambda_grl=1.0)

# Load saved weights
model.load_state_dict(torch.load("hybrid_model.pth", map_location=torch.device("cpu")))
model.eval()

# 3. predict function
def predict(img) -> Tuple[Dict, float]:
    # start a timer
    start_time = timer()

    # transform input image for spatial and frequency branches
    spatial_img = transform_spatial(img).unsqueeze(0)
    freq_img = transform_frequency(img).unsqueeze(0)

    # put model into eval mode, make prediction
    model.eval()
    with torch.inference_mode():
        class_output, _ = model(spatial_img, freq_img)
        
        # Binary classification (sigmoid instead of softmax)
        pred_prob = torch.sigmoid(class_output).squeeze().item()

    # create pred label and prob dict
    # class_names[1] is Fake/AI, class_names[0] is Real/Human
    pred_dict = {
        class_names[1]: float(pred_prob),
        class_names[0]: float(1.0 - pred_prob)
    }

    # calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)

    # return pred dict and pred time
    return pred_dict, pred_time

# 4. Gradio app interface
# Create title, description, and article
title = "Fake Media Detector "
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

# Launch the demooo!
if __name__ == "__main__":
    demo.launch()
