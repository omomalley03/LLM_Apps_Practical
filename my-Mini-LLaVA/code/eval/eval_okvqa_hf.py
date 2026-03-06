#!/usr/bin/env python3
"""
OK-VQA Evaluation Script for Vision Language Model with Batch Processing
"""

import argparse
import torch
import json
import os
import sys
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPProcessor
from torch.utils.data import Dataset, DataLoader

from utils_okvqa import load_okvqa_data, OKVQADatasetHF, process_single_result
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(args, device="cuda"):
    # Load model
    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"  # Ensure left padding for generation

    return model, processor

def evaluate_okvqa(args, model, processor=None, device="cuda"):
    """
    Main evaluation function for OK-VQA with batch processing
    """

    print("Loading OK-VQA evaluation data...")

    # Load OK-VQA data
    okvqa_data = load_okvqa_data(args.okvqa_questions, args.okvqa_annotations)
    if args.max_samples > 0:
        okvqa_data = okvqa_data[:args.max_samples]
    print(f"Loaded {len(okvqa_data)} questions")

    # Results storage
    results = []
    correct_answers = 0
    total_questions = 0

    # Track accuracy by question type and answer type
    accuracy_by_question_type = defaultdict(list)
    accuracy_by_answer_type = defaultdict(list)

    # Use batch processing
    print(f"Creating dataset with batch size {args.batch_size}...")
    dataset = OKVQADatasetHF(okvqa_data, args.images_dir, args.prompt)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.custom_collate_fn,  # Custom collate function for batch processing
        shuffle=False, 
        num_workers=args.num_workers
    )

    print("Starting batch evaluation...")

    sample_count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Process batch
        try:
            responses = process_batch(model, processor, batch, device, args)
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

        # Process results for each item in the batch
        for i, (response, item) in enumerate(zip(responses, batch['item_data'])):
            sample_count += 1
            accuracy = process_single_result(response, item, results,
                                             accuracy_by_question_type, accuracy_by_answer_type,
                                             sample_count, args)
            if accuracy is not None:
                correct_answers += accuracy
                total_questions += 1

        # Print batch progress
        if (batch_idx + 1) % 10 == 0:
            current_accuracy = correct_answers / \
                total_questions if total_questions > 0 else 0.0
            print(
                f"Processed {batch_idx + 1} batches, current accuracy: {current_accuracy:.4f}")

    # Calculate overall statistics
    overall_accuracy = correct_answers / \
        total_questions if total_questions > 0 else 0.0

    # Calculate accuracy by category
    category_stats = {}

    if accuracy_by_question_type:
        category_stats['by_question_type'] = {}
        for qtype, accuracies in accuracy_by_question_type.items():
            category_stats['by_question_type'][qtype] = {
                'accuracy': sum(accuracies) / len(accuracies),
                'count': len(accuracies)
            }

    if accuracy_by_answer_type:
        category_stats['by_answer_type'] = {}
        for atype, accuracies in accuracy_by_answer_type.items():
            category_stats['by_answer_type'][atype] = {
                'accuracy': sum(accuracies) / len(accuracies),
                'count': len(accuracies)
            }

    # Print results summary
    print(f"\n{'='*50}")
    print("OK-VQA EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total Questions Evaluated: {total_questions}")
    print(
        f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    if category_stats.get('by_question_type'):
        print(f"\nAccuracy by Question Type:")
        for qtype, stats in category_stats['by_question_type'].items():
            print(
                f"  {qtype}: {stats['accuracy']:.4f} ({stats['count']} questions)")

    if category_stats.get('by_answer_type'):
        print(f"\nAccuracy by Answer Type:")
        for atype, stats in category_stats['by_answer_type'].items():
            print(
                f"  {atype}: {stats['accuracy']:.4f} ({stats['count']} questions)")

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual results
    results_file = os.path.join(args.output_dir, "okvqa_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary statistics
    summary = {
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'category_statistics': category_stats,
        'evaluation_args': vars(args)
    }

    summary_file = os.path.join(args.output_dir, "okvqa_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary statistics saved to: {summary_file}")

    return overall_accuracy





def process_batch(model, processor, batch, device, args):
    """Process a batch of samples through the model"""
    messages = []
    for text, image in zip(batch['text'], batch['image']):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        messages.append(message)

    texts = [
            processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            ) for message in messages
        ]
        
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Generate responses
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    output_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
                output_trimmed, skip_special_tokens=True
            )
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on OK-VQA dataset")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Directory containing the fine-tuned model")

    # Data arguments
    parser.add_argument("--okvqa_questions", type=str, required=True,
                        help="Path to OK-VQA questions JSON file")
    parser.add_argument("--okvqa_annotations", type=str,
                        help="Path to OK-VQA annotations JSON file (optional for test set)")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing COCO validation images")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling (low for consistent answers)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")

    # Evaluation arguments
    parser.add_argument("--prompt", type=str,
                        default="Your answer should be in one word.",
                        help="Prompt template for questions")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate (-1 for all)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./okvqa_evaluation",
                        help="Directory to save evaluation results")

    # Batch processing arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing multiple samples simultaneously")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of worker processes for data loading")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.okvqa_questions):
        raise ValueError(
            f"OK-VQA questions file not found: {args.okvqa_questions}")
    if not os.path.exists(args.images_dir):
        raise ValueError(f"Images directory not found: {args.images_dir}")

    # Run evaluation
    model, processor = load_model(args)
    evaluate_okvqa(args, model, processor)
