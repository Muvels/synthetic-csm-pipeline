from pathlib import Path

BASE_DIR = Path("synthetic-dataset")
FILENAME = "vibevoice-podcast-script.txt"
MAX_LINES = 999

def count_lines(file_path: Path) -> int:
    """Count the number of lines in a text file."""
    line_count = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            line_count += 1
    return line_count

def main():
    if not BASE_DIR.exists():
        print(f"Base directory '{BASE_DIR}' does not exist.")
        return

    too_long_files = []
    longest_file = None
    longest_line_count = -1

    # Recursively find all vibevoice-podcast-script.txt files
    for file_path in BASE_DIR.rglob(FILENAME):
        line_count = count_lines(file_path)

        # Track files over limit
        if line_count > MAX_LINES:
            too_long_files.append((file_path, line_count))

        # Track the longest file
        if line_count > longest_line_count:
            longest_line_count = line_count
            longest_file = file_path

    # Output files that exceed MAX_LINES
    if too_long_files:
        print(f"Files with more than {MAX_LINES} lines:\n")
        for path, lines in too_long_files:
            print(f"{path} -> {lines} lines")
    else:
        print(f"All '{FILENAME}' files under '{BASE_DIR}' have <= {MAX_LINES} lines.")

    # Output longest file
    if longest_file:
        print("\nLongest file:")
        print(f"{longest_file} -> {longest_line_count} lines")
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()
