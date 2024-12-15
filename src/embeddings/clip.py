import torch
import clip
from PIL import Image

# Load the CLIP model and preprocessing once at the module level for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_clip_image_embeddings(image_path: str):
    """
    Returns the image embedding for the given image path using CLIP.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Normalized image embedding.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Generate image embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def get_clip_text_embeddings(text: str):
    """
    Returns the text embedding for the given text using CLIP.

    Args:
        text (str): Input text.

    Returns:
        numpy.ndarray: Normalized text embedding.
    """
    try:
        # Tokenize and encode the text
        text_input = clip.tokenize([text]).to(device)

        # Generate text embeddings
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return None
