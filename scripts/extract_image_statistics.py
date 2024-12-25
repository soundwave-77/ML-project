import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image


# Detect GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# RGB -> HSV
def rgb_to_hsv(img_tensor):
    r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
    max_val, _ = torch.max(img_tensor, dim=0)
    min_val, _ = torch.min(img_tensor, dim=0)
    delta = max_val - min_val

    hue = torch.zeros_like(max_val)
    mask_r = (max_val == r) & (delta != 0)
    mask_g = (max_val == g) & (delta != 0)
    mask_b = (max_val == b) & (delta != 0)

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    hue /= 6

    saturation = torch.zeros_like(max_val)
    saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]
    value = max_val
    return torch.stack((hue, saturation, value))


# Average RGB value
def calculate_mean_rgb(img_tensor):
    return img_tensor.mean(dim=(1, 2))


# Whiteness (proportion of light pixels)
def calculate_whiteness(img_tensor):
    img_gray = img_tensor.mean(dim=0)
    whiteness = (img_gray > 0.8).float().mean().item()
    return whiteness


# Dullness (low saturation)
def calculate_dullness(img_tensor):
    hsv = rgb_to_hsv(img_tensor)
    dullness = 1 - hsv[1].mean().item()
    return dullness


# Brightness
def calculate_brightness(img_tensor):
    hsv = rgb_to_hsv(img_tensor)  # RGB -> HSV
    brightness = hsv[2].mean().item()
    return brightness


# Contrast
def calculate_contrast(img_tensor):
    hsv = rgb_to_hsv(img_tensor)  # RGB -> HSV
    contrast = hsv[2].max().item() - hsv[2].min().item()
    return contrast


# Function to process a single image
def process_image(image_path, image_array):
    # try:
    if image_array is None:
        print('reading from file')
        img = Image.open(image_path).convert('RGB')
    else:
        print('reading from array')
        img = image_array

    img_tensor = transform(img).to(device)

    mean_rgb = calculate_mean_rgb(img_tensor)
    whiteness = calculate_whiteness(img_tensor)
    dullness = calculate_dullness(img_tensor)
    brightness = calculate_brightness(img_tensor)
    contrast = calculate_contrast(img_tensor)

    # clean_file_name = Path(image_path).stem

    return {
        "image": os.path.basename(image_path) if image_path else None,
        "mean_r": mean_rgb[0].item(),
        "mean_g": mean_rgb[1].item(),
        "mean_b": mean_rgb[2].item(),
        "whiteness": whiteness,
        "dullness": dullness,
        "brightness": brightness,
        "contrast": contrast
    }
    # except Exception as e:
    #     print(f"Error processing image {image_path}: {e}")
    #     return None
    

# Example usage
if __name__ == "__main__":
    # Path to the image
    image_path = "image.jpg"

    # Process the image and get statistics
    stats = process_image(image_path)

    # Print the results
    if stats:
        print(stats)