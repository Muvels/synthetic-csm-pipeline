import re
import json
import shutil
from pathlib import Path

# Base directories
BASE_DIR = Path("synthetic-dataset")
DIAR_DIR = BASE_DIR / "_meta" / "diarization"
EMB_DIR = BASE_DIR / "_meta" / "embeddings"

# Valid format: 001_speaker1.wav or 001_speaker2.wav
WAV_PATTERN = re.compile(r"^(\d{3})_speaker([12])\.wav$")


def print_title(title: str):
    line = "=" * len(title)
    print(line)
    print(title)
    print(line)


def print_subtitle(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def check_wav_filenames():
    """
    Validate .wav filenames and return:
    - stats: dict with numeric info
    - invalid_files: list[Path] (actual files with bad names)
    - bad_directories: set[Path] (sample-level dirs, NOT 'segments')
    """
    invalid_files = []
    valid_files = 0
    bad_directories = set()

    for file_path in BASE_DIR.rglob("*.wav"):
        name = file_path.name
        segments_dir = file_path.parent

        # Determine the sample directory:
        # if path is .../<sample>/segments/001_speakerX.wav,
        # we consider <sample> the "faulty directory", not the 'segments' dir.
        if segments_dir.name == "segments":
            sample_dir = segments_dir.parent
        else:
            sample_dir = segments_dir

        if WAV_PATTERN.match(name):
            valid_files += 1
        else:
            invalid_files.append(file_path)
            if sample_dir != BASE_DIR:
                bad_directories.add(sample_dir)

    total_files = valid_files + len(invalid_files)

    stats = {
        "total_wav": total_files,
        "valid_wav": valid_files,
        "invalid_wav": len(invalid_files),
        "dirs_with_invalid_wav": len(bad_directories),
    }

    print_title("WAV FILENAME CHECK")

    print(f"Total WAV files      : {stats['total_wav']}")
    print(f"Valid WAV files      : {stats['valid_wav']}")
    print(f"Invalid WAV files    : {stats['invalid_wav']}")
    print(f"Sample dirs affected : {stats['dirs_with_invalid_wav']}")

    print_subtitle("Invalid WAV files")
    if invalid_files:
        for path in invalid_files:
            print(f"  - {path}")
    else:
        print("  None (all filenames are valid).")

    print_subtitle("Sample directories with invalid WAV files")
    if bad_directories:
        for d in sorted(bad_directories):
            print(f"  - {d}")
    else:
        print("  None.")

    print()  # empty line
    return stats, invalid_files, bad_directories


def load_diarization_speakers(json_path: Path):
    """
    Load diarization JSON and return (num_speakers, speakers_set, broken_flag).
    broken_flag is True if JSON has >2 speakers or cannot be parsed.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Parsing error -> treat as broken
        return 0, set(), True

    segments = data.get("segments", [])
    speakers = set()

    for seg in segments:
        speaker = seg.get("speaker")
        if speaker is not None:
            speakers.add(speaker)

    num_speakers = len(speakers)
    broken = num_speakers > 2  # rule: more than 2 speakers -> broken

    return num_speakers, speakers, broken


def check_diarization():
    """
    Check diarization JSON files and return:
    - stats: dict with numeric info
    - diarization_info: dict[sample_dir_path] = {
          "json_path": Path,
          "num_speakers": int,
          "speakers": set[str],
          "broken": bool
      }
    """
    diarization_info = {}

    print_title("DIARIZATION JSON CHECK")

    if not DIAR_DIR.exists():
        print(f"Diarization directory does not exist: {DIAR_DIR}")
        print()
        return {
            "total_json": 0,
            "valid_json": 0,
            "broken_json": 0,
        }, diarization_info

    json_files = sorted(DIAR_DIR.glob("*.json"))
    if not json_files:
        print(f"No diarization JSON files found in: {DIAR_DIR}")
        print()
        return {
            "total_json": 0,
            "valid_json": 0,
            "broken_json": 0,
        }, diarization_info

    for json_path in json_files:
        num_speakers, speakers, broken = load_diarization_speakers(json_path)

        # ASSUMPTION:
        # Folder for this diarization is synthetic-dataset/<stem_of_json>/
        # e.g. diarization/foo.json -> synthetic-dataset/foo/
        sample_dir = BASE_DIR / json_path.stem

        diarization_info[sample_dir] = {
            "json_path": json_path,
            "num_speakers": num_speakers,
            "speakers": speakers,
            "broken": broken,
        }

    broken_entries = {d: info for d, info in diarization_info.items() if info["broken"]}
    total_json = len(json_files)
    broken_json = len(broken_entries)
    valid_json = total_json - broken_json

    stats = {
        "total_json": total_json,
        "valid_json": valid_json,
        "broken_json": broken_json,
    }

    print(f"Total diarization JSON files  : {stats['total_json']}")
    print(f"Valid diarization JSON files  : {stats['valid_json']}")
    print(f"Broken diarization JSON files : {stats['broken_json']}")

    print_subtitle("Broken diarization files ( >2 speakers or parse error )")
    if broken_entries:
        for d, info in sorted(broken_entries.items()):
            speakers_sorted = ", ".join(sorted(info["speakers"])) if info["speakers"] else "None"
            print(f"  - {info['json_path']} -> {info['num_speakers']} speakers ({speakers_sorted})")
    else:
        print("  None (all have 2 or fewer speakers).")

    print()
    return stats, diarization_info


def prompt_delete_faulty_dirs(faulty_dirs):
    """
    Ask user if faulty sample directories should be deleted.
    These dirs are expected to be like synthetic-dataset/<sample_name>,
    NOT 'segments' and NOT metadata dirs.
    """
    # Filter out anything we must never delete
    safe_faulty_dirs = set()
    for d in faulty_dirs:
        # Must exist and be a dir
        if not d.exists() or not d.is_dir():
            continue
        # Never delete the root dataset
        if d == BASE_DIR:
            continue
        # Never delete _meta or any of its subdirs
        if d == BASE_DIR / "_meta" or (BASE_DIR / "_meta") in d.parents:
            continue
        # Just in case: never directly delete a 'segments' dir;
        # if we somehow got one, skip it.
        if d.name == "segments":
            continue

        safe_faulty_dirs.add(d)

    if not safe_faulty_dirs:
        print("\nNo faulty sample directories to delete.")
        return

    print_title("FAULTY SAMPLE DIRECTORIES (CANDIDATES FOR DELETION)")
    for d in sorted(safe_faulty_dirs):
        print(f"  - {d}")

    print(
        "\nWARNING: The above sample directories will be deleted recursively (rm -rf style),"
        " and their corresponding diarization + embeddings files (if present) will also be removed."
    )
    answer = input("Do you want to delete ALL of these faulty sample directories? [yes/no]: ").strip().lower()

    if answer in ("yes", "y"):
        print("\nDeleting faulty sample directories...")
        for d in sorted(safe_faulty_dirs):
            try:
                shutil.rmtree(d)
                print(f"  Deleted sample dir: {d}")

                # Also delete matching diarization + embeddings files, if they exist.
                # Convention (see dataset/process_podcast.py):
                #   BASE_DIR/_meta/diarization/<sample_name>_diarization.json
                #   BASE_DIR/_meta/embeddings/<sample_name>_embeddings.npz
                sample_name = d.name
                diar_path = DIAR_DIR / f"{sample_name}_diarization.json"
                emb_path = EMB_DIR / f"{sample_name}_embeddings.npz"

                for meta_path, label in ((diar_path, "diarization"), (emb_path, "embeddings")):
                    if meta_path.exists():
                        try:
                            meta_path.unlink()
                            print(f"  Deleted {label} file: {meta_path}")
                        except Exception as e:
                            print(f"  FAILED to delete {label} file {meta_path}: {e}")
            except Exception as e:
                print(f"  FAILED to delete {d}: {e}")
        print("Deletion step finished.")
    else:
        print("\nNo directories were deleted.")


def main():
    if not BASE_DIR.exists():
        print(f"Base directory does not exist: {BASE_DIR}")
        return

    # ---------- 1) WAV checks ----------
    wav_stats, _, bad_directories = check_wav_filenames()

    # ---------- 2) Diarization checks ----------
    diar_stats, diarization_info = check_diarization()

    # ---------- 3) Combine results ----------
    dirs_with_broken_diar = {d for d, info in diarization_info.items() if info["broken"]}

    total_recalc_dirs = bad_directories & dirs_with_broken_diar
    wav_only_recalc = bad_directories - total_recalc_dirs
    diar_only_recalc = dirs_with_broken_diar - total_recalc_dirs

    print_title("RE-CALCULATION SUMMARY")

    print(f"Sample dirs with WAV issues only         : {len(wav_only_recalc)}")
    print(f"Sample dirs with diarization issues only : {len(diar_only_recalc)}")
    print(f"Sample dirs needing TOTAL recalculation   : {len(total_recalc_dirs)}")

    print_subtitle("1) Sample dirs needing recalculation due to WAV filename / speaker ID issues")
    if wav_only_recalc:
        for d in sorted(wav_only_recalc):
            print(f"  - {d}")
    else:
        print("  None.")

    print_subtitle("2) Sample dirs needing recalculation due to diarization issues (>2 speakers)")
    if diar_only_recalc:
        for d in sorted(diar_only_recalc):
            info = diarization_info.get(d)
            if info:
                print(f"  - {d} -> {info['json_path']} ({info['num_speakers']} speakers)")
            else:
                print(f"  - {d}")
    else:
        print("  None.")

    print_subtitle("3) Sample dirs needing TOTAL recalculation (both WAV + diarization broken)")
    if total_recalc_dirs:
        for d in sorted(total_recalc_dirs):
            info = diarization_info.get(d)
            if info:
                print(f"  - {d} -> {info['json_path']} ({info['num_speakers']} speakers)")
            else:
                print(f"  - {d}")
    else:
        print("  None.")

    # ---------- 4) Global numeric overview ----------
    print_title("GLOBAL NUMERIC OVERVIEW")

    print("WAV FILES")
    print(f"  Total WAV files      : {wav_stats['total_wav']}")
    print(f"  Valid WAV files      : {wav_stats['valid_wav']}")
    print(f"  Invalid WAV files    : {wav_stats['invalid_wav']}")
    print(f"  Affected sample dirs : {wav_stats['dirs_with_invalid_wav']}")

    print("\nDIARIZATION JSON")
    print(f"  Total JSON files     : {diar_stats['total_json']}")
    print(f"  Valid JSON files     : {diar_stats['valid_json']}")
    print(f"  Broken JSON files    : {diar_stats['broken_json']}")

    print("\nSAMPLE DIRECTORIES (RECALCULATION)")
    print(f"  WAV issues only        : {len(wav_only_recalc)}")
    print(f"  Diarization issues only: {len(diar_only_recalc)}")
    print(f"  TOTAL recalculation    : {len(total_recalc_dirs)}")

    # ---------- 5) Prompt to delete faulty sample folders ----------
    # "Faulty" = any sample directory that needs some kind of recalculation
    faulty_dirs = wav_only_recalc | diar_only_recalc | total_recalc_dirs
    prompt_delete_faulty_dirs(faulty_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
