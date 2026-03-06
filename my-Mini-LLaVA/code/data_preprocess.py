import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

from tqdm import tqdm
class VLMDataset(Dataset):
    """
    Unified dataset for both alignment and instruction tuning stages.
    """
    def __init__(self, json_file, images_dir, tokenizer, processor, stage, max_length=512, check_img=False):
        """
        Args:
            json_file: Path to the JSON file containing data, or list of paths for multiple files.
            images_dir: Directory containing the images.
            tokenizer: Tokenizer for the LLM.
            processor: Processor for the vision encoder.
            stage: 'alignment' or 'instruction'.
            max_length: Maximum length of the tokenized text.
        """
        self.data = []
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.stage = stage
        self.max_length = max_length
        # Handle both single file and multiple files
        json_files = json_file if isinstance(json_file, list) else [json_file]
        
        for json_path in json_files:
            print(f"Loading data from: {json_path}")
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
            
            # Filter out items with missing image files if images_dir is provided
            if images_dir:
                for item in tqdm(raw_data, desc=f"Loading {stage} data from {os.path.basename(json_path)}"):
                    if 'image' in item:
                        
                        if self.stage == 'instruction' and not item['image'].startswith('COCO_train2014_'):
                            item['image'] = f"COCO_train2014_{item['image']}"
                        if check_img:
                            image_path = os.path.join(self.images_dir, item['image'])
                            try:
                                image = Image.open(image_path).convert('RGB')
                            except FileNotFoundError:
                                print(f"Warning: Image file not found {image_path}, skipping this item.")
                                continue
                            except Exception as e:
                                print(f"Error processing image {image_path}: {e}")
                                print("skipping this item.")
                                continue
                        self.data.append(item)
                    
                    else:
                        print(f"Warning: Image file not found for item {item.get('id', 'unknown')}, skipping this item.")
            else: # If no images_dir, assume it's for text-only or images are handled differently
                self.data.extend(raw_data)
        
        print(f"Total loaded samples: {len(self.data)}")
            


        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        pixel_values = None
        if self.images_dir and 'image' in item:
            # For instruct stage, add COCO_train2014_ prefix if not already present
            image_name = item['image']
            
            
            image_path = os.path.join(self.images_dir, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                processed_image = self.processor(images=image, return_tensors="pt")
                pixel_values = processed_image.pixel_values.squeeze(0)
            except FileNotFoundError:
                print(f"Warning: Image file not found {image_path}, skipping image for this item.")
                # Create a dummy pixel_values tensor if image is not found but expected
                # This assumes processor has a known image size, e.g., CLIP (224,224)
                # You might need to adjust this based on your vision model
                img_size = self.processor.feature_extractor.size
                pixel_values = torch.zeros((3, img_size, img_size))
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                img_size = self.processor.feature_extractor.size
                pixel_values = torch.zeros((3, img_size, img_size))

        # Both stages use the same data format: conversations
        # LLaVA format: item['conversations'] is a list of dicts
        # [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        conversations = item['conversations']
        human_msg = conversations[0]['value']
        assistant_msg = conversations[1]['value']
        
        # Remove <image> placeholder from human_msg if it exists, as we add it systematically
        human_msg = human_msg.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
        
        # Format: <image> USER: {instruction} ASSISTANT: {response}
        instruction_text = f"<image> USER: {human_msg} ASSISTANT: "
        full_text = f"{instruction_text}{assistant_msg}"
        text = full_text

        encoded_text = self.tokenizer(text, 
                                      padding="max_length", 
                                      truncation=True, 
                                      max_length=self.max_length, 
                                      return_tensors="pt")
        
        input_ids = encoded_text.input_ids.squeeze(0)
        attention_mask = encoded_text.attention_mask.squeeze(0)
        labels = input_ids.clone()

        # Label masking: mask out the instruction part (<image> USER: ... ASSISTANT: )
        # Only the assistant's response should be predicted
        instruction_encoded = self.tokenizer(instruction_text, add_special_tokens=False)
        instruction_length = len(instruction_encoded.input_ids)
        labels[:instruction_length] = -100
        
        # If pixel_values is still None (e.g. text-only data or image error), create a dummy one
        if pixel_values is None:
            img_size = self.processor.feature_extractor.size if hasattr(self.processor, 'feature_extractor') else 224
            pixel_values = torch.zeros((3, img_size, img_size))
            
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
