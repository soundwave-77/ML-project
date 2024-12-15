import numpy as np
import pandas as pd
import json
import pickle

import torch
import clip
from PIL import Image

# Load the CLIP model and preprocessing once at the module level for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load the DINO v2 model and preprocessing
try:
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    dinov2_model.to(device)
    dinov2_model.eval()
except Exception as e:
    print(f"Error loading DINO v2 model: {e}")
    dinov2_model = None

# Define DINO v2 preprocessing (adjust based on the model's requirements)
def dinov2_preprocess(image: Image.Image):
    # Example preprocessing steps; adjust as needed
    transform = torch.nn.Sequential(
        torch.nn.Resize((480, 480)),
        torch.nn.ToTensor(),
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    )
    return transform(image).unsqueeze(0).to(device)

def get_dinov2_embeddings(image_path: str):
    """
    Returns the image embedding for the given image path using DINO v2.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Normalized image embedding.
    """
    if dinov2_model is None:
        print("DINO v2 model is not loaded.")
        return None

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = dinov2_preprocess(image)

        # Generate image embeddings
        with torch.no_grad():
            embeddings = dinov2_model(image_input)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
