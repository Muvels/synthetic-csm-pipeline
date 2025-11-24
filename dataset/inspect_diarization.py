import torch
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

# Create dummy waveform
waveform = torch.rand(1, 16000 * 5) # 5 seconds
sample_rate = 16000

print("Running pipeline...")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

print(f"Output type: {type(output)}")
print(f"Output dir: {dir(output)}")

if hasattr(output, 'annotation'):
    print(f"Annotation type: {type(output.annotation)}")
