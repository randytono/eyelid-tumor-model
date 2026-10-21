"""Check the repository for patient data and other things that must not be pushed.

Run before every push:
    python tools/check_repo_safety.py

Inside a git repository it checks exactly the files git would commit (tracked
plus untracked, minus .gitignore). Exits with status 1 if anything is found.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCKED_EXT = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".tif": "image", ".tiff": "image",
    ".bmp": "image", ".gif": "image", ".heic": "image", ".dcm": "DICOM image",
    ".csv": "tabular data", ".tsv": "tabular data", ".xlsx": "spreadsheet", ".xls": "spreadsheet",
    ".parquet": "tabular data", ".pth": "model weights", ".pt": "model weights",
    ".ckpt": "model weights", ".onnx": "model weights", ".h5": "model weights",
    ".ipynb": "notebook (outputs can embed patient images)", ".docx": "document", ".pdf": "document",
}
TEXT_EXT = {".py", ".md", ".txt", ".sh", ".cfg", ".toml", ".yaml", ".yml", ".cff", ".json", ""}
PATTERNS = [
    (re.compile(r"(?<![\d.\-/])\d{7,8}(?![\d.])"), "7-8 digit number (possible medical record number)"),
    (re.compile(r"_create_unverified_context"), "disabled SSL certificate check"),
    (re.compile(r"(/Users/|/home/[a-z]|C:\\\\Users\\\\|~/thesis)"), "absolute personal path"),
    (re.compile(r"\byavai\b", re.I), "internal platform call"),
]
MAX_BYTES = 2_000_000


def candidate_files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [f for f in out.splitlines() if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not a git repository yet: checking every file under the repo folder.\n")
        files = []
        for base, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            files += [os.path.relpath(os.path.join(base, n), ROOT) for n in names]
        return files


def main():
    problems = []
    for rel in candidate_files():
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path) or rel.startswith("tools/check_repo_safety"):
            continue
        ext = os.path.splitext(rel)[1].lower()
        if ext in BLOCKED_EXT:
            problems.append(f"{rel}: {BLOCKED_EXT[ext]} file")
            continue
        if os.path.getsize(path) > MAX_BYTES:
            problems.append(f"{rel}: larger than {MAX_BYTES // 1_000_000} MB")
        if ext in TEXT_EXT:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    for pattern, what in PATTERNS:
                        if pattern.search(line):
                            problems.append(f"{rel}:{lineno}: {what}")

    if problems:
        print("Do NOT push yet. Found:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("No images, data files, weights, notebooks, record-number patterns or personal paths found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
