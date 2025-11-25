import os
import re
import glob

def rename_files(root_dir):
    # Regex to match various formats:
    # segment_0001_spk1.wav
    # segment_1_speaker1.wav
    # 0001_speaker1.wav
    # 1_spk1.wav
    # Group 1: segment_id (digits)
    # Group 2: speaker_id (digits)
    pattern = re.compile(r"^(?:segment_)?(\d+)_(?:spk|speaker)(\d+)\.wav$")
    
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                segment_id_str = match.group(1)
                speaker_id_str = match.group(2)
                
                segment_id = int(segment_id_str)
                speaker_id = int(speaker_id_str)
                
                # Desired format: {id:03d}_speaker{n}.wav
                new_filename = f"{segment_id:03d}_speaker{speaker_id}.wav"
                
                if filename == new_filename:
                    continue
                
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                
                if os.path.exists(new_path):
                    print(f"Skipping {filename} -> {new_filename} (Target exists)")
                else:
                    # print(f"Renaming {filename} -> {new_filename}")
                    os.rename(old_path, new_path)
                    count += 1
    
    print(f"Renamed {count} files.")

if __name__ == "__main__":
    root_dir = "synthetic-dataset"
    if os.path.exists(root_dir):
        rename_files(root_dir)
    else:
        print(f"Directory {root_dir} not found.")
