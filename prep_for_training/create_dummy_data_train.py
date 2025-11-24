import os
import numpy as np
import soundfile as sf
from pathlib import Path

def create_dummy_dataset(root_dir):
    root = Path(root_dir)
    segments_dir = root / "en" / "test_cat" / "test_uuid" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    # Create script
    script_content = """[1]: Hello world.
[2]: Hi there.
[1]: How are you?
"""
    with open(segments_dir / "vibevoice-podcast-script.txt", "w") as f:
        f.write(script_content)
        
    # Create dummy wavs (1 second sine waves)
    sr = 24000
    t = np.linspace(0, 1, sr)
    audio = np.sin(2 * np.pi * 440 * t)
    
    sf.write(segments_dir / "001_speaker1.wav", audio, sr)
    sf.write(segments_dir / "002_speaker2.wav", audio, sr)
    sf.write(segments_dir / "003_speaker1.wav", audio, sr)
    
    print(f"Created dummy dataset at {root}")

if __name__ == "__main__":
    create_dummy_dataset("test_dataset_dummy_train")
