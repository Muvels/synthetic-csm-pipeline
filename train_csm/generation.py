import torch
import soundfile as sf
import re
import wandb
from pathlib import Path
from transformers import TrainerCallback

class AudioGenerator:
    def __init__(self, example_path, processor, output_dir):
        self.example_path = Path(example_path)
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.example_data = self._load_example()

    def _load_example(self):
        segments_dir = self.example_path / "segments"
        script_path = segments_dir / "script.txt"
        if not script_path.exists():
             script_path = segments_dir / "vibevoice-podcast-script.txt"
        
        if not script_path.exists():
            print(f"Warning: Could not find script.txt in {segments_dir}")
            return None

        turns = []
        with open(script_path, "r") as f:
            for line in f:
                m = re.match(r"\[(\d+)\]:\s*(.*)", line.strip())
                if m:
                    turns.append({"speaker": m.group(1), "text": m.group(2)})
        
        if not turns:
            return None

        target_turn = turns[-1]
        history = turns[:-1]
        
        wav_files = sorted(segments_dir.glob("*.wav"))
        
        conversation = []
        for i, turn in enumerate(history):
            if i < len(wav_files):
                wav_path = str(wav_files[i])
                conversation.append({
                    "role": turn["speaker"],
                    "content": [
                        {"type": "text", "text": turn["text"]},
                        {"type": "audio", "path": wav_path}
                    ]
                })
        
        conversation.append({
            "role": target_turn["speaker"],
            "content": [
                {"type": "text", "text": target_turn["text"]},
            ]
        })
        
        return conversation

    def generate(self, model, step_name="generated", use_wandb=False, global_step=None):
        if not self.example_data:
            print("No example data loaded, skipping generation.")
            return

        print(f"\nGenerating audio sample: {step_name}...")
        try:
            inputs = self.processor.apply_chat_template(
                self.example_data,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                )
                
            audio_data = outputs.cpu().numpy().squeeze()
            
            filename = self.output_dir / f"{step_name}.wav"
            sf.write(filename, audio_data, 24000)
            print(f"Saved generated audio to {filename}")
            
            if use_wandb and wandb.run is not None:
                wandb.log({
                    "generated_audio": wandb.Audio(str(filename), caption=f"Step {global_step if global_step else step_name}"),
                    "global_step": global_step if global_step is not None else 0
                })
                
        except Exception as e:
            print(f"Error generating audio: {e}")

class AudioGenerationCallback(TrainerCallback):
    def __init__(self, example_path, gen_every, processor, use_wandb, output_dir):
        self.gen_every = gen_every
        self.use_wandb = use_wandb
        # Create a generator instance
        # We output to a subfolder "generated_samples" inside the training output dir
        gen_output_dir = Path(output_dir) / "generated_samples"
        self.generator = AudioGenerator(example_path, processor, gen_output_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.gen_every == 0 and state.global_step > 0:
            self.generator.generate(
                kwargs["model"], 
                step_name=f"step_{state.global_step}", 
                use_wandb=self.use_wandb, 
                global_step=state.global_step
            )
