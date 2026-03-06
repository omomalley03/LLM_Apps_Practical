import os
import torch
from PIL import Image
from transformers import CLIPProcessor
import numpy as np

def main():
    # Initialize CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Process images 1.jpg to 4.jpg (though we only have 1.jpg, 2.jpg in the directory)
    image_files = []
    for i in range(1, 5):
        filename = f"{i}.jpg"
        if os.path.exists(filename):
            image_files.append(filename)
    
    print(f"Found {len(image_files)} images to process: {image_files}")
    
    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Load original image
        original_image = Image.open(image_file).convert('RGB')
        print(f"Original image size: {original_image.size}")
        
        # Process image with CLIP processor
        processed = processor(images=original_image, return_tensors="pt")
        pixel_values = processed['pixel_values']
        
        # Convert back to PIL Image for saving
        # The processor normalizes the image, so we need to denormalize
        # CLIP uses ImageNet normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        
        # Denormalize
        denormalized = pixel_values * std + mean
        
        # Clamp to [0, 1] and convert to [0, 255]
        denormalized = torch.clamp(denormalized, 0, 1) * 255
        
        # Convert to numpy and rearrange dimensions
        image_array = denormalized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Convert to PIL Image and save
        processed_image = Image.fromarray(image_array)
        
        # Save the processed image
        base_name = os.path.splitext(image_file)[0]
        output_path = f"output/{base_name}_processed.jpg"
        processed_image.save(output_path)
        
        print(f"Saved processed image to: {output_path}")
        print(f"Processed image size: {processed_image.size}")
        print("-" * 50)

if __name__ == "__main__":
    main()
