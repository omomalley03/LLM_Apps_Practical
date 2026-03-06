import json
import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict

# Import VQA evaluation tools
from vqaEval import VQAEval

class VQA:
    """Helper class to wrap ground truth data for VQAEval"""
    def __init__(self, annotations):
        self.qa = {ann['question_id']: ann for ann in annotations}
        self.getQuesIds = self.qa.keys

class VQARes:
    """Helper class to wrap results for VQAEval"""
    def __init__(self, results):
        self.qa = {res['question_id']: res for res in results}
        self.getQuesIds = self.qa.keys

class OKVQADatasetVLM(Dataset):
    """Dataset class for batch processing of OK-VQA data"""
    
    def __init__(self, okvqa_data, images_dir, processor, tokenizer, prompt):
        self.data = okvqa_data
        self.images_dir = images_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt = prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_filename = f"COCO_val2014_{item['image_id']:012d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            # Process image but don't add batch dimension yet
            processed_image = self.processor(images=image, return_tensors="pt")
            pixel_values = processed_image.pixel_values.squeeze(0)  # Remove batch dim
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Create dummy image tensor
            pixel_values = torch.zeros(3, 224, 224)
        
        # Prepare prompt
        question = item['question']
        formatted_prompt = f"<image> USER: {self.prompt} \n {question}  ASSISTANT:"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': inputs.input_ids.squeeze(0),  # Remove batch dim
            'attention_mask': inputs.attention_mask.squeeze(0),  # Remove batch dim
            'formatted_prompt': formatted_prompt,
            'item_data': item
        }
    def custom_collate_fn(self, batch):
        """Custom collate function to handle variable length sequences"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # Pad input_ids and attention_mask to same length
        max_length = max([item['input_ids'].size(0) for item in batch])
        
        input_ids = []
        attention_mask = []
        
        for item in batch:
            # Pad sequences
            seq_len = item['input_ids'].size(0)
            if seq_len < max_length:
                padding_length = max_length - seq_len
                padded_input_ids = torch.cat([
                    torch.full((padding_length,), item['input_ids'][-1]),  # Pad with last token
                    item['input_ids'], 
                ])
                padded_attention_mask = torch.cat([
                    torch.zeros(padding_length, dtype=torch.long),
                    item['attention_mask'],
                ])
            else:
                padded_input_ids = item['input_ids']
                padded_attention_mask = item['attention_mask']
                
            input_ids.append(padded_input_ids)
            attention_mask.append(padded_attention_mask)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'formatted_prompts': [item['formatted_prompt'] for item in batch],
            'item_data': [item['item_data'] for item in batch]
        }
        
class OKVQADatasetHF(Dataset):
    """Dataset class for batch processing of OK-VQA data"""
    
    def __init__(self, okvqa_data, images_dir, prompt):
        self.data = okvqa_data
        self.images_dir = images_dir
        self.prompt = prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_filename = f"COCO_val2014_{item['image_id']:012d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        
        # Prepare prompt
        question = item['question']
        formatted_prompt = f"{self.prompt} \n {question}"
        
        return {
            'image': image_path,
            'text': formatted_prompt,
            'item_data': item
        }
    def custom_collate_fn(self, batch):
        # custom collate function to handle batch processing

        return {
            'text': [item['text'] for item in batch],
            'image': [item['image'] for item in batch],
            'item_data': [item['item_data'] for item in batch]
        }

def load_okvqa_data(questions_file, annotations_file=None):
    """
    Load OK-VQA questions and annotations
    
    Args:
        questions_file: Path to OK-VQA questions JSON file
        annotations_file: Path to OK-VQA annotations JSON file (optional for test set)
    
    Returns:
        List of question-answer pairs
    """
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Create question lookup
    questions = {q['question_id']: q for q in questions_data['questions']}
    
    data = []

    # Load annotations for validation/train set
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    for ann in annotations_data['annotations']:
        question_id = ann['question_id']
        if question_id in questions:
            question = questions[question_id]
            
            # Extract all answers (OK-VQA has multiple annotators)
            answers = [answer['answer'] for answer in ann['answers']]
            
            data.append({
                'question_id': question_id,
                'image_id': question['image_id'],
                'question': question['question'],
                'answers': answers,
                'answer_type': ann.get('answer_type', 'other'),
                'question_type': ann.get('question_type', 'other')
            })
    
    return data


def calculate_vqa_accuracy(predicted_answer, ground_truth_answers, question_id, question_type, answer_type):
    """
    Calculate VQA accuracy using the official VQAEval script.
    """
    # Create dummy VQA and VQARes objects
    
    # Ground truth
    gt_annotations = [{
        'answers': [{'answer': ans} for ans in ground_truth_answers],
        'question_type': question_type,
        'answer_type': answer_type,
        'question_id': question_id
    }]
    vqa = VQA(gt_annotations)

    # Predicted result
    pred_results = [{
        'answer': predicted_answer,
        'question_id': question_id
    }]
    vqa_res = VQARes(pred_results)

    # Initialize and run evaluation
    vqa_eval = VQAEval(vqa, vqa_res, n=2)
    vqa_eval.evaluate(quesIds=[question_id])
    
    # Return overall accuracy for this single question
    return vqa_eval.accuracy['overall'] / 100.0


def process_single_result(response, item, results,
                        accuracy_by_question_type, accuracy_by_answer_type, sample_count, args):
    """Process a single result and update statistics"""
    # Extract answer (first sentence or first few words)
    predicted_answer = response.split('.')[0].split('\n')[0].strip()
    
    # Calculate accuracy if ground truth is available
    accuracy = 0.0
    if item['answers']:
        accuracy = calculate_vqa_accuracy(
            predicted_answer, 
            item['answers'],
            item['question_id'],
            item['question_type'],
            item['answer_type']
        )
        
        # Track by question/answer type
        if item['question_type']:
            accuracy_by_question_type[item['question_type']].append(accuracy)
        if item['answer_type']:
            accuracy_by_answer_type[item['answer_type']].append(accuracy)
    
    # Store result
    result = {
        'question_id': item['question_id'],
        'image_id': item['image_id'],
        'question': item['question'],
        'predicted_answer': predicted_answer,
        'full_response': response,
        'ground_truth_answers': item['answers'],
        'accuracy': accuracy,
        'question_type': item['question_type'],
        'answer_type': item['answer_type']
    }
    results.append(result)
    
    
    return accuracy if item['answers'] else None