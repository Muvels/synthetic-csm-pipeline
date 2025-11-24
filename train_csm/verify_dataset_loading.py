import argparse
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="unsloth/csm-1b")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    shards = sorted(dataset_path.glob("shard_*.arrow"))
    
    if not shards:
        print("No shards found.")
        return

    loaded_shards = [load_from_disk(str(s)) for s in shards]
    train_dataset = concatenate_datasets(loaded_shards)
    print(f"Total samples: {len(train_dataset)}")

    # Define the transform (copied from train.py logic)
    def preprocess_batch(batch):
        processed_batch = []
        batch_size = len(batch["conversation"])
        
        for i in range(batch_size):
            conversation = batch["conversation"][i]
            processed_conversation = []
            valid_sample = True
            
            for msg in conversation:
                new_content = []
                for item in msg["content"]:
                    if item["type"] == "audio":
                        try:
                            audio_path = item["path"]
                            target_sr = 24000 
                            data, sr = sf.read(audio_path)
                            # Simple check, real code has resampling
                            new_content.append({"type": "audio", "path": data, "sampling_rate": target_sr})
                        except Exception as e:
                            print(f"Failed to load {audio_path}: {e}")
                            valid_sample = False
                            break
                    else:
                        new_content.append(item)
                
                if not valid_sample:
                    break
                processed_conversation.append({"role": msg["role"], "content": new_content})
            
            if valid_sample:
                processed_batch.append(processed_conversation)
            else:
                processed_batch.append([])

        try:
            model_inputs = tokenizer.apply_chat_template(
                processed_batch,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                text_kwargs={
                    "padding": "max_length",
                    "max_length": 2048,
                    "pad_to_multiple_of": 8,
                    "padding_side": "right",
                },
            )
            return model_inputs
        except Exception as e:
            print(f"Error in apply_chat_template: {e}")
            return {"input_ids": [], "attention_mask": [], "labels": []}

    train_dataset.set_transform(preprocess_batch)

    print("Verifying first 5 samples...")
    for i in range(5):
        try:
            sample = train_dataset[i]
            print(f"Sample {i}: input_ids shape {len(sample['input_ids'])}")
            if len(sample['input_ids']) == 0:
                print("Empty sample (error?)")
        except Exception as e:
            print(f"Error accessing sample {i}: {e}")

    print("Verification complete.")

if __name__ == "__main__":
    main()
