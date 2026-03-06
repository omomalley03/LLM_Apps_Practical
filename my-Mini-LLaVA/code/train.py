import os
import argparse
import torch
from transformers import Trainer, TrainingArguments, CLIPProcessor, AutoTokenizer
from model import VisionLanguageModel
from data_preprocess import VLMDataset # Unified dataset
class CustomVLMTrainer(Trainer):
    """
    Custom Trainer that uses the model's custom save_pretrained method
    to avoid safetensors issues with shared tensors.
    """
    def _save(self, output_dir: str, state_dict=None):
        """Override the _save method to use our custom save_pretrained."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the model's custom save_pretrained method
        self.model.save_pretrained(output_dir)
        

def main(args):
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
        
    
    # Initialize tokenizer and processor
    # For stage 1, we init tokenizer from llm_name. For stage 2, from model_dir.
    tokenizer_path = args.llm_name if args.stage == 'alignment' else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor = CLIPProcessor.from_pretrained(args.vision_encoder_name)

    # Add special tokens if not already present (important for both stages)
    special_tokens = {"additional_special_tokens": ["<image>"]}
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use tokenizer.eos_token
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)

    # Create or load the model
    if args.stage == 'alignment':
        model = VisionLanguageModel(
            vision_encoder_name=args.vision_encoder_name,
            llm_name=args.llm_name
        )
        model.llm.resize_token_embeddings(len(tokenizer)) # Resize after adding special tokens
        # Freeze vision encoder and LLM for Stage 1
        model.freeze_vision_encoder()
        model.freeze_llm()
        model.unfreeze_mapping_network() # Ensure mapping network is trainable
        print("Initialized model for Stage 1 (Alignment). Vision encoder and LLM are frozen.")
    elif args.stage == 'instruction':
        model = VisionLanguageModel.from_pretrained(args.model_dir)
        # Freeze vision encoder, unfreeze LLM and mapping network for Stage 2
        model.freeze_vision_encoder()
        model.unfreeze_llm() # This should unfreeze LLM and mapping network if logic is correct in model.py
        model.unfreeze_mapping_network()
        print(f"Loaded model from {args.model_dir} for Stage 2 (Instruction). Vision encoder frozen, LLM and mapping network unfrozen.")
    else:
        raise ValueError(f"Invalid stage: {args.stage}")

    # Create the dataset
    train_dataset = VLMDataset(
        json_file=args.train_json,
        images_dir=args.images_dir,
        tokenizer=tokenizer,
        processor=processor,
        stage=args.stage,
        max_length=args.max_length,
        check_img=args.check_img
    )

    # Define the training arguments
    
    # If max_steps is -1, use epochs; otherwise use max_steps
    if args.max_steps == -1:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=True,
            dataloader_num_workers=args.dataloader_num_workers,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            run_name=args.run_name if args.run_name else os.path.basename(args.output_dir),
            seed=seed,  # Set seed for reproducibility
            data_seed=seed,  # Set data seed for reproducibility
        )
    else:
        # If max_steps is specified, override num_train_epochs
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=True,
            dataloader_num_workers=args.dataloader_num_workers,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            run_name=args.run_name if args.run_name else os.path.basename(args.output_dir),
            seed=seed,  # Set seed for reproducibility
            data_seed=seed,  # Set data seed for reproducibility
        )

    # Define the data collator
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Initialize the trainer
    trainer = CustomVLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer, # Use processing_class instead of deprecated tokenizer
    )

    # Debug: Check which parameters require gradients
    print("\n=== Parameter gradient status ===")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✓ {name}: requires_grad=True, shape={param.shape}")
        else:
            print(f"✗ {name}: requires_grad=False, shape={param.shape}")
    
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! All parameters are frozen.")

    # Train the model
    print(f"Starting training for stage: {args.stage}")
    trainer.train()

    # Save the model and tokenizer
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision-Language Model")
    
    # Arguments for both stages
    parser.add_argument("--stage", type=str, required=True, choices=['alignment', 'instruction'],
                        help="Training stage: 'alignment' or 'instruction'.")
    parser.add_argument("--train_json", type=str, nargs='+', required=True, 
                        help="Path(s) to the JSON file(s) containing training data. Can accept multiple files.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model checkpoints and final model.")
    parser.add_argument("--vision_encoder_name", type=str, default="openai/clip-vit-large-patch14",
                        help="Vision encoder ID from Hugging Face Model Hub.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of the tokenized text.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before updating model parameters.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total number of training steps. If -1, will train for the number of epochs specified.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="Weight decay for training.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Ratio of warmup steps to total training steps.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between saving checkpoints.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to save.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for data loading.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for the training run (used for WandB). If not provided, uses output_dir basename.")

    # Stage-specific arguments
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                        help="LLM ID from Hugging Face Model Hub (used for Stage 1 initialization).")
    parser.add_argument("--model_dir", type=str,
                        help="Directory containing the pre-aligned model (used for Stage 2 initialization).")
    parser.add_argument("--check_img", action="store_true",
                        help="Check if image files exist in the images directory during data loading.")
    args = parser.parse_args()

    # Validate arguments based on stage
    if args.stage == 'instruction' and not args.model_dir:
        parser.error("--model_dir is required for instruction stage.")
    if args.stage == 'alignment' and not args.output_dir:
         parser.error("--output_dir is required for alignment stage to save the model.")
    if args.stage == 'instruction' and not args.output_dir:
        parser.error("--output_dir is required for instruction stage to save the model.")
        

    main(args)
