import argparse
import torch
from unsloth import FastLanguageModel
from generation import AudioGenerator
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio using Sesame CSM model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory or Hugging Face model ID.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the example folder containing segments/script.txt and audio context.")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Path to save generated audio.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    # Load model and tokenizer
    # Note: For inference with unsloth trained models, we usually load adapters onto base.
    # If model_path is a local directory with adapters, unsloth handles it.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = args.load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    
    print(f"Initializing generator with input from {args.input_dir}...")
    generator = AudioGenerator(args.input_dir, tokenizer, args.output_dir)
    
    print("Generating audio...")
    generator.generate(model, step_name="inference_sample")
    
    print("Done.")

if __name__ == "__main__":
    main()
