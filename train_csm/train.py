import argparse
import os
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sesame CSM model using Unsloth.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the prepared dataset directory containing shards.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to save the trained model/checkpoints.")
    parser.add_argument("--max_steps", type=int, default=60, help="Max training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--model_id", type=str, default="unsloth/csm-1b", help="Model ID to load.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_id}")
    # Load model and tokenizer using Unsloth
    # max_seq_length and dtype are usually inferred or set to defaults if not provided, 
    # but for CSM we might want to be careful. The notebook uses defaults in the snippet I saw, 
    # or specific config. Let's use standard loading.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_id,
        max_seq_length = 2048, # From notebook context section
        dtype = None, # Auto detection
        load_in_4bit = False, # Disable 4bit quantization for macOS compatibility
    )

    # Configure LoRA adapters
    # The notebook output showed "Trainable parameters = 29M", which implies LoRA.
    # We need to apply LoRA adapters to the model.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Standard LoRA rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = args.seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    print(f"Loading dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    shards = sorted(dataset_path.glob("shard_*.arrow"))
    
    if not shards:
        raise ValueError(f"No shards found in {args.dataset_path}")
        
    loaded_shards = []
    for shard in shards:
        try:
            ds = load_from_disk(str(shard))
            loaded_shards.append(ds)
        except Exception as e:
            print(f"Error loading shard {shard}: {e}")
            
    if not loaded_shards:
        raise ValueError("Failed to load any datasets.")
        
    train_dataset = concatenate_datasets(loaded_shards)
    print(f"Total training samples: {len(train_dataset)}")

    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = 5,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = args.seed,
        output_dir = args.output_dir,
        report_to = "none",
    )

    print("Starting training...")
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args = training_args,
    )
    
    trainer_stats = trainer.train()
    
    print("Training complete.")
    print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds")
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Also save to GGUF if needed? Notebook mentions saving. 
    # For now, standard pretrained save is enough.

if __name__ == "__main__":
    main()
