"""
Moves 1/9 of the files in a source folder to a new destination folder
each time it is run, in order, without repeating files already moved.

- On the first run, it lists all files in SOURCE_DIR, sorts them for a
  stable order, and saves that full list + progress to a state file
  (STATE_FILE) inside SOURCE_DIR.
- Each run reads the state file, takes the next 1/9 chunk of files that
  hasn't been moved yet, creates a new folder for that batch, and moves
  the files there.
- Run the script 9 times total to move everything.

Usage:
    python move_batch.py
"""

import json
import math
import shutil
from pathlib import Path

# ---- CONFIG: edit these two paths ----
SOURCE_DIR = Path(r"C:\Users\qx398\MIMICWorkspace\MasterDataset_v1pickles0806")
DEST_ROOT = Path(r"C:\Users\qx398\MIMICWorkspace\MasterDataset_v1pickles0806_patch")  # batch folders created inside here
# ---------------------------------------

NUM_BATCHES = 9
STATE_FILE = SOURCE_DIR / ".move_batch_state.json"


def load_or_create_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        return state

    # First run: snapshot the current file list, sorted for a stable order.
    files = sorted(
        [p.name for p in SOURCE_DIR.iterdir() if p.is_file() and p != STATE_FILE]
    )
    state = {
        "files": files,
        "total": len(files),
        "batch_size": math.ceil(len(files) / NUM_BATCHES),
        "next_batch": 1,  # 1-indexed batch number to move next
    }
    save_state(state)
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source folder does not exist: {SOURCE_DIR}")

    state = load_or_create_state()

    batch_num = state["next_batch"]
    if batch_num > NUM_BATCHES:
        print("All batches have already been moved. Nothing to do.")
        return

    batch_size = state["batch_size"]
    files = state["files"]

    start = (batch_num - 1) * batch_size
    end = min(start + batch_size, len(files))
    chunk = files[start:end]

    if not chunk:
        print("No files left to move for this batch.")
        state["next_batch"] += 1
        save_state(state)
        return

    dest_dir = DEST_ROOT / f"batch_{batch_num}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    for name in chunk:
        src_path = SOURCE_DIR / name
        if not src_path.exists():
            # File was moved/deleted outside this script; skip but note it.
            missing += 1
            continue
        shutil.move(str(src_path), str(dest_dir / name))
        moved += 1

    state["next_batch"] += 1
    save_state(state)

    print(f"Batch {batch_num}/{NUM_BATCHES}: moved {moved} files to {dest_dir}")
    if missing:
        print(f"  Warning: {missing} files from this batch were already missing.")
    print(f"Next run will move batch {state['next_batch']} (if <= {NUM_BATCHES}).")


if __name__ == "__main__":
    main()
