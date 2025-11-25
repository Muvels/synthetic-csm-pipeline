import argparse
import os
import glob
import json
import gc
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Inference, Model
from pyannote.core import Segment
from pydub import AudioSegment
from tqdm import tqdm
from langdetect import detect
import parakeet_mlx
import torch

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

OUTPUT_SAMPLE_RATE = 24000  # Hz
EMBEDDING_MIN_DURATION = 0.5  # seconds: min segment length for speaker embedding
GLOBAL_SIMILARITY_THRESHOLD = 0.75  # cosine similarity threshold for clustering

META_DIR_NAME = "_meta"  # under output folder
EMBEDDINGS_SUBDIR = "embeddings"
DIARIZATION_SUBDIR = "diarization"
GLOBAL_MAPPING_FILENAME = "global_speakers.json"

# -------------------------------------------------------------------
# Environment / HF token
# -------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set.")
    print("Please set HF_TOKEN in your environment or .env file.")
    raise SystemExit(1)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def base_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_audio_files(input_dir: str) -> List[str]:
    """
    Scan input_dir for audio files and return a sorted list of paths.
    Excludes any files inside 'segments' folders.
    """
    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]
    files: List[str] = []

    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    # Exclude already-produced segments
    files = [f for f in files if "segments" not in f]
    return sorted(files)


def get_diarization_path(meta_dir: str, file_basename: str) -> str:
    return os.path.join(
        meta_dir, DIARIZATION_SUBDIR, f"{file_basename}_diarization.json"
    )


def get_embeddings_path(meta_dir: str, file_basename: str) -> str:
    return os.path.join(
        meta_dir, EMBEDDINGS_SUBDIR, f"{file_basename}_embeddings.npz"
    )


def file_has_diarization_and_embeddings(meta_dir: str, file_basename: str) -> bool:
    return (
        os.path.exists(get_diarization_path(meta_dir, file_basename))
        and os.path.exists(get_embeddings_path(meta_dir, file_basename))
    )


def free_torch_memory() -> None:
    """Best-effort cleanup of Torch / device memory."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def load_diarization_models():
    """
    Load:
      - pyannote speaker diarization pipeline
      - pyannote speaker embedding model
    """
    print("Loading diarization models...")

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device for diarization: {device}")

    # Diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    diarization_pipeline.to(device)

    # Speaker embedding model
    embedding_model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=HF_TOKEN,
    )
    inference = Inference(embedding_model, window="whole")
    inference.to(device)

    return diarization_pipeline, inference, device


def load_parakeet_model():
    """
    Load Parakeet transcription model, configured for lower memory usage.
    """
    print("Loading Parakeet model...")
    model = parakeet_mlx.from_pretrained(
        "mlx-community/parakeet-tdt-0.6b-v3"
    )

    # Optional: use local attention when available (better for long audio, less memory)
    try:
        model.encoder.set_attention_model(
            "rel_pos_local_attn",  # local attention reduces intermediate memory usage
            (256, 256),
        )
    except Exception:
        # Older versions may not expose this, it's fine to ignore.
        pass

    return model


# -------------------------------------------------------------------
# Stage 1: diarization + per-file embeddings (saved to disk)
# -------------------------------------------------------------------

def diarize_and_save_embeddings(
    file_path: str,
    diarization_pipeline,
    inference: Inference,
    device,
    meta_dir: str,
) -> None:
    """
    For a single audio file:
      - Run diarization.
      - Save diarization as JSON (segments with local speaker labels).
      - Compute one embedding per local speaker (longest segment).
      - Save embeddings as NPZ.
    """
    file_basename = base_name(file_path)

    if file_has_diarization_and_embeddings(meta_dir, file_basename):
        # Already done, resumable
        print(f"Stage1: Skipping (already done): {file_path}")
        return

    print(f"Stage1: Diarizing & embedding {file_path}...")

    # Prepare meta directories
    diar_dir = os.path.join(meta_dir, DIARIZATION_SUBDIR)
    emb_dir = os.path.join(meta_dir, EMBEDDINGS_SUBDIR)
    ensure_dir(diar_dir)
    ensure_dir(emb_dir)

    # Run diarization directly on the audio file
    # Fix for torchcodec/soundfile error: load audio in-memory using pydub
    # Load audio with pydub (handles m4a via ffmpeg)
    audio = AudioSegment.from_file(file_path)
    
    # Convert to mono if needed (pyannote usually handles multi-channel, but mono is safer/standard)
    # However, pyannote expects (channel, time). Let's keep channels if possible, or just mix down.
    # For simplicity and robustness, let's convert to mono and shape as (1, time).
    audio = audio.set_channels(1)
    
    # Get raw samples as float32
    # pydub samples are int16 (usually), we need float32 in [-1, 1]
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1, 1]
    if audio.sample_width == 2: # 16-bit
        samples /= 32768.0
    elif audio.sample_width == 4: # 32-bit
        samples /= 2147483648.0
    
    # Convert to tensor (1, time)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    sample_rate = audio.frame_rate
    
    # Calculate exact duration in seconds to prevent out-of-bounds errors
    # Subtract a tiny epsilon to ensure we are strictly within bounds for pyannote
    audio_duration = (waveform.shape[1] / sample_rate) - 0.01
    
    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    # Handle DiarizeOutput wrapper if present (common when passing dict input)
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    # ---- Save diarization to JSON ----
    diarization_path = get_diarization_path(meta_dir, file_basename)
    
    # 1. Collect all turns and stats
    all_turns = []
    speaker_durations: Dict[str, float] = {}
    speaker_segments: Dict[str, List[Segment]] = {} # speaker -> list of Segments

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        orig_speaker = str(speaker)
        
        # Clamp segment to audio duration
        t_start = float(turn.start)
        t_end = min(float(turn.end), audio_duration)
        
        if t_end <= t_start:
            continue
            
        dur = t_end - t_start
        
        speaker_durations[orig_speaker] = speaker_durations.get(orig_speaker, 0.0) + dur
        
        # Store turn info
        all_turns.append({
            "start": t_start,
            "end": t_end,
            "speaker": orig_speaker
        })
        
        # Store segment for embedding extraction
        clamped_turn = Segment(t_start, t_end)
        speaker_segments.setdefault(orig_speaker, []).append(clamped_turn)

    # 2. Identify top 2 speakers
    # Sort by duration descending
    sorted_speakers = sorted(speaker_durations.keys(), key=lambda s: speaker_durations[s], reverse=True)
    
    if not sorted_speakers:
        print(f"Stage1: No speech segments found in '{file_basename}', skipping.")
        return

    canonical_speakers = sorted_speakers[:2]
    minor_speakers = sorted_speakers[2:]
    
    # 3. Compute embeddings for ALL detected speakers
    speaker_embeddings: Dict[str, np.ndarray] = {}
    
    for spk in sorted_speakers:
        segments = speaker_segments.get(spk, [])
        
        # Pick the longest segment above min duration
        longest: Optional[Segment] = None
        longest_duration = 0.0

        for seg in segments:
            dur = float(seg.end - seg.start)
            if dur < EMBEDDING_MIN_DURATION:
                continue
            if dur > longest_duration:
                longest_duration = dur
                longest = seg

        # Fallback: if none above threshold, take the actual longest segment
        if longest is None and segments:
            longest = max(segments, key=lambda s: float(s.end - s.start))
            longest_duration = float(longest.end - longest.start)

        if longest is None or longest_duration <= 0.0:
            continue

        # Extract embedding
        # Double check clamping just in case (though we clamped on creation)
        start_t = float(longest.start)
        end_t = min(float(longest.end), audio_duration)
        
        excerpt = Segment(start_t, end_t)
        try:
            emb = inference.crop({"waveform": waveform, "sample_rate": sample_rate}, excerpt)
            emb_np = np.asarray(emb, dtype=np.float32).squeeze()
            speaker_embeddings[spk] = emb_np
        except Exception as e:
            print(f"Warning: Failed to extract embedding for speaker {spk}: {e}")

    # 4. Build Mapping
    speaker_map: Dict[str, str] = {}
    
    # Identity mapping for canonicals
    for spk in canonical_speakers:
        speaker_map[spk] = spk
        
    # Similarity mapping for minors
    for spk in minor_speakers:
        emb_s = speaker_embeddings.get(spk)
        
        # Default fallback: map to the first canonical speaker (usually the most dominant)
        best_match = canonical_speakers[0]
        
        if emb_s is not None:
            best_sim = -2.0 # Cosine sim is [-1, 1]
            
            # Normalize source embedding
            norm_s = np.linalg.norm(emb_s)
            if norm_s > 1e-12:
                emb_s_norm = emb_s / norm_s
            else:
                emb_s_norm = emb_s

            for cand in canonical_speakers:
                emb_c = speaker_embeddings.get(cand)
                if emb_c is None:
                    continue
                
                # Normalize candidate embedding
                norm_c = np.linalg.norm(emb_c)
                if norm_c > 1e-12:
                    emb_c_norm = emb_c / norm_c
                else:
                    emb_c_norm = emb_c
                
                sim = np.dot(emb_s_norm, emb_c_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_match = cand
        
        speaker_map[spk] = best_match

    # 5. Build final diarization list
    diar_segments: List[Dict[str, float]] = []
    for turn in all_turns:
        mapped_speaker = speaker_map.get(turn["speaker"], canonical_speakers[0])
        diar_segments.append({
            "start": turn["start"],
            "end": turn["end"],
            "speaker": mapped_speaker
        })

    diar_json = {
        "sample_rate": 16000,  # pyannote internally resamples to 16kHz
        "segments": diar_segments,
    }

    with open(diarization_path, "w", encoding="utf-8") as f:
        json.dump(diar_json, f, indent=2)

    # 6. Save Embeddings (only canonicals)
    final_embeddings: List[np.ndarray] = []
    final_labels: List[str] = []
    
    for spk in canonical_speakers:
        if spk in speaker_embeddings:
            final_embeddings.append(speaker_embeddings[spk])
            final_labels.append(str(spk))

    if final_embeddings:
        emb_array = np.stack(final_embeddings, axis=0)
        local_labels_array = np.array(final_labels, dtype=object)

        emb_path = get_embeddings_path(meta_dir, file_basename)
        np.savez_compressed(
            emb_path,
            embeddings=emb_array,
            local_speakers=local_labels_array,
        )

        print(
            f"Stage1: Saved diarization + {emb_array.shape[0]} embeddings for '{file_basename}'"
        )
    else:
        print(f"Stage1: No valid embeddings saved for '{file_basename}'")


# -------------------------------------------------------------------
# Stage 2: global speaker clustering
# -------------------------------------------------------------------

def cluster_global_speakers(meta_dir: str, similarity_threshold: float) -> Dict[Tuple[str, str], int]:
    """
    Load all per-file embeddings from disk and cluster them into global speakers.

    Simple online threshold-based clustering.
    """
    emb_dir = os.path.join(meta_dir, EMBEDDINGS_SUBDIR)
    if not os.path.exists(emb_dir):
        print("Stage2: No embeddings directory found, nothing to cluster.")
        return {}

    npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    if not npz_files:
        print("Stage2: No embedding files found, nothing to cluster.")
        return {}

    all_embeddings: List[np.ndarray] = []
    all_keys: List[Tuple[str, str]] = []

    for path in npz_files:
        filename = os.path.basename(path)
        file_basename = filename
        if filename.endswith("_embeddings.npz"):
            file_basename = filename[: -len("_embeddings.npz")]

        data = np.load(path, allow_pickle=True)
        embs = data.get("embeddings")
        local_speakers = data.get("local_speakers")

        if embs is None or local_speakers is None:
            continue

        if embs.size == 0 or len(local_speakers) == 0:
            continue

        for emb, local_speaker in zip(embs, local_speakers):
            all_embeddings.append(np.asarray(emb, dtype=np.float32))
            all_keys.append((file_basename, str(local_speaker)))

    if not all_embeddings:
        print("Stage2: No non-empty embeddings to cluster.")
        return {}

    # Stack and L2-normalize embeddings
    all_embeddings_arr = np.stack(all_embeddings, axis=0)
    norms = np.linalg.norm(all_embeddings_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_embeddings_arr = all_embeddings_arr / norms

    # We hard-limit the number of global speakers to TWO.
    max_global_speakers = 2

    cluster_centers: List[np.ndarray] = []
    cluster_counts: List[int] = []
    mapping: Dict[Tuple[str, str], int] = {}

    next_global_id = 1

    for emb, key in zip(all_embeddings_arr, all_keys):
        if not cluster_centers:
            cluster_centers.append(emb.copy())
            cluster_counts.append(1)
            mapping[key] = next_global_id
            next_global_id += 1
            continue

        centers = np.stack(cluster_centers, axis=0)
        sims = centers @ emb  # cosine similarity

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        # If we don't yet have two clusters, we can still create a new one
        # based on the similarity threshold. Once we have two, we ALWAYS
        # assign to the closest existing cluster (no new clusters).
        if len(cluster_centers) < max_global_speakers:
            if best_sim >= similarity_threshold:
                # assign to existing cluster and update center via running mean
                count = cluster_counts[best_idx]
                new_center = (cluster_centers[best_idx] * count + emb) / (count + 1)
                new_center = new_center / max(np.linalg.norm(new_center), 1e-12)
                cluster_centers[best_idx] = new_center
                cluster_counts[best_idx] = count + 1
                global_id = best_idx + 1  # 1-based
            else:
                cluster_centers.append(emb.copy())
                cluster_counts.append(1)
                global_id = next_global_id
                next_global_id += 1
        else:
            # We already have the maximum number of global speakers.
            # Force assignment to the closest existing cluster regardless of threshold.
            count = cluster_counts[best_idx]
            new_center = (cluster_centers[best_idx] * count + emb) / (count + 1)
            new_center = new_center / max(np.linalg.norm(new_center), 1e-12)
            cluster_centers[best_idx] = new_center
            cluster_counts[best_idx] = count + 1
            global_id = best_idx + 1  # 1-based

        mapping[key] = global_id

    print(f"Stage2: Created {len(cluster_centers)} global speakers.")
    return mapping


def save_global_mapping(meta_dir: str, mapping: Dict[Tuple[str, str], int]) -> None:
    mapping_path = os.path.join(meta_dir, GLOBAL_MAPPING_FILENAME)
    ensure_dir(meta_dir)

    serializable = []
    for (file_basename, local_speaker), global_id in mapping.items():
        serializable.append(
            {
                "file": file_basename,
                "local_speaker": local_speaker,
                "global_speaker": int(global_id),
            }
        )

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"mappings": serializable}, f, indent=2)

    print(f"Stage2: Saved global speaker mapping to '{mapping_path}'")


def load_global_mapping(meta_dir: str) -> Optional[Dict[Tuple[str, str], int]]:
    mapping_path = os.path.join(meta_dir, GLOBAL_MAPPING_FILENAME)
    if not os.path.exists(mapping_path):
        return None

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[Tuple[str, str], int] = {}
    for entry in data.get("mappings", []):
        key = (entry["file"], entry["local_speaker"])
        mapping[key] = int(entry["global_speaker"])

    print(f"Stage2: Loaded existing global speaker mapping from '{mapping_path}'")
    return mapping


# -------------------------------------------------------------------
# Stage 3: transcription + segment cutting
# -------------------------------------------------------------------

def is_file_already_transcribed(file_path: str, output_base_dir: str) -> bool:
    """
    Check if the given file has already been processed into a script.
    """
    basename = base_name(file_path)
    pattern = os.path.join(
        output_base_dir,
        "*",          # any lang
        basename,
        "segments",
        "vibevoice-podcast-script.txt",
    )
    matches = glob.glob(pattern)
    return len(matches) > 0


def load_diarization_segments(meta_dir: str, file_basename: str):
    """
    Load diarization JSON and return list of segments:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    path = get_diarization_path(meta_dir, file_basename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Diarization file not found for '{file_basename}': {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def find_local_speaker_for_span(
    start: float,
    end: float,
    diar_segments,
):
    """
    Given a time span [start, end] and diarization segments,
    find which local speaker is active.
    """
    if not diar_segments:
        return None

    mid = 0.5 * (start + end)

    # 1) Try midpoint
    candidates = [
        seg
        for seg in diar_segments
        if seg["start"] <= mid < seg["end"]
    ]
    if candidates:
        return candidates[0]["speaker"]

    # 2) Fallback: majority overlap across the span
    best_speaker = None
    best_overlap = 0.0

    for seg in diar_segments:
        s = float(seg["start"])
        e = float(seg["end"])
        overlap = min(end, e) - max(start, s)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def transcribe_and_generate_segments(
    file_path: str,
    parakeet_model,
    global_mapping: Dict[Tuple[str, str], int],
    meta_dir: str,
    output_base_dir: str,
):
    """
    For a single audio file:
      - Load diarization segments.
      - Transcribe with Parakeet (in chunks).
      - Detect language.
      - For each sentence, determine local speaker from diarization,
        map to global speaker ID.
      - Cut audio segments and save.
      - Write vibevoice-podcast-script.txt.
    """
    if is_file_already_transcribed(file_path, output_base_dir):
        print(f"Stage3: Skipping (already transcribed): {file_path}")
        return

    print(f"Stage3: Transcribing + cutting segments for {file_path}...")

    file_basename = base_name(file_path)

    # Load diarization segments
    diar_segments = load_diarization_segments(meta_dir, file_basename)

    # Transcribe with chunking to keep memory use bounded
    result = parakeet_model.transcribe(
        file_path,
        chunk_duration=60.0 * 2.0,   # 2 minutes
        overlap_duration=15.0,
    )

    # Detect language
    full_text = " ".join([s.text for s in result.sentences])
    try:
        lang = detect(full_text) if full_text.strip() else "unknown"
    except Exception:
        lang = "unknown"

    # Prepare output paths
    output_dir = os.path.join(output_base_dir, lang, file_basename, "segments")
    ensure_dir(output_dir)

    script_path = os.path.join(output_dir, "vibevoice-podcast-script.txt")
    script_lines: List[str] = []

    # Load audio once for slicing
    audio = AudioSegment.from_file(file_path)

    for i, sentence in enumerate(result.sentences):
        start_sec = float(sentence.start)
        end_sec = float(sentence.end)
        text = sentence.text.strip()

        if not text:
            continue
        if end_sec <= start_sec:
            continue

        # Find local speaker label from diarization
        local_speaker = find_local_speaker_for_span(
            start_sec, end_sec, diar_segments
        )

        # Map to global speaker id
        global_speaker_id = 1  # fallback
        if local_speaker is not None:
            key = (file_basename, str(local_speaker))
            global_speaker_id = global_mapping.get(key, 1)

        # Add line to script
        script_lines.append(f"[{global_speaker_id}]: {text}")

        # Export audio segment
        start_ms = max(0, int(start_sec * 1000))
        end_ms = max(start_ms + 1, int(end_sec * 1000))
        segment_audio = audio[start_ms:end_ms]

        segment_filename = f"{i+1:03d}_speaker{global_speaker_id}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        segment_audio.export(segment_path, format="wav")

    # Write script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines))

    print(f"Stage3: Finished '{file_basename}', script at {script_path}")


# -------------------------------------------------------------------
# Internal helper: run Stage 3 for a single file in a subprocess
# -------------------------------------------------------------------

def run_stage3_single_file(file_path: str, output_dir: str):
    """
    INTERNAL ENTRYPOINT for subprocess:
    - loads Parakeet
    - loads existing global mapping
    - transcribes a single file
    - exits
    """
    meta_dir = os.path.join(output_dir, META_DIR_NAME)
    ensure_dir(meta_dir)

    global_mapping = load_global_mapping(meta_dir) or {}

    parakeet_model = load_parakeet_model()
    try:
        transcribe_and_generate_segments(
            file_path,
            parakeet_model,
            global_mapping,
            meta_dir,
            output_dir,
        )
    finally:
        parakeet_model = None
        free_torch_memory()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process podcasts: diarization, global speaker clustering, transcription."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=".",
        help="Directory containing input audio files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Directory where output and metadata will be stored.",
    )
    parser.add_argument(
        "--recluster",
        action="store_true",
        help="Force recomputation of global speaker mapping (Stage 2).",
    )

    # INTERNAL: used only when this script is invoked for a single Stage3 file
    parser.add_argument(
        "--_internal_stage3_file",
        type=str,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # ---------- INTERNAL SINGLE-FILE STAGE 3 MODE ----------
    if args._internal_stage3_file:
        # Only run Stage 3 for this one file and exit.
        run_stage3_single_file(args._internal_stage3_file, args.output)
        return

    # ---------- NORMAL MULTI-STAGE PIPELINE ----------

    # Collect files
    files = get_audio_files(args.input)
    if not files:
        print(f"No audio files found in '{args.input}'. Nothing to do.")
        return

    meta_dir = os.path.join(args.output, META_DIR_NAME)
    ensure_dir(meta_dir)

    # ---------------- Stage 1 ----------------
    print("\n=== Stage 1: Diarization & embeddings ===")

    need_stage1 = any(
        not file_has_diarization_and_embeddings(meta_dir, base_name(f))
        for f in files
    )

    stage1_did_work = False

    diarization_pipeline = None
    inference = None
    device = None

    if need_stage1:
        diarization_pipeline, inference, device = load_diarization_models()

        for file_path in tqdm(files, desc="Stage 1: files"):
            already_exists = file_has_diarization_and_embeddings(
                meta_dir, base_name(file_path)
            )
            diarize_and_save_embeddings(
                file_path,
                diarization_pipeline,
                inference,
                device,
                meta_dir,
            )
            now_exists = file_has_diarization_and_embeddings(
                meta_dir, base_name(file_path)
            )
            if now_exists and not already_exists:
                stage1_did_work = True
    else:
        print("Stage1: All diarization & embeddings already exist. Skipping Stage 1.")

    # Explicitly unload diarization models before moving on
    if diarization_pipeline is not None or inference is not None:
        print("Stage1: Releasing diarization models from memory...")
        diarization_pipeline = None
        inference = None
        device = None
        free_torch_memory()

    # ---------------- Stage 2 ----------------
    print("\n=== Stage 2: Global speaker clustering ===")

    existing_mapping = load_global_mapping(meta_dir)

    if args.recluster or stage1_did_work or not existing_mapping:
        print("Stage2: Computing new global speaker mapping...")
        global_mapping = cluster_global_speakers(
            meta_dir, similarity_threshold=GLOBAL_SIMILARITY_THRESHOLD
        )
        save_global_mapping(meta_dir, global_mapping)
    else:
        global_mapping = existing_mapping if existing_mapping is not None else {}

    # ---------------- Stage 3 ----------------
    print("\n=== Stage 3: Transcription & segment export (subprocess per file) ===")

    # IMPORTANT:
    # We DO NOT load Parakeet in this main process.
    # Instead, we invoke this script again for each file with --_internal_stage3_file.
    # That way each transcription runs in a fresh process and all MLX memory
    # is reclaimed when that process exits (workaround for MLX leaks).

    for file_path in tqdm(files, desc="Stage 3: files"):
        if is_file_already_transcribed(file_path, args.output):
            print(f"Stage3: Skipping (already transcribed): {file_path}")
            continue

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--output", args.output,
            "--_internal_stage3_file", file_path,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Stage3: Error in subprocess while processing '{file_path}': {e}")

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
