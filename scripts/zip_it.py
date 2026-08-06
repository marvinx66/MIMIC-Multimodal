"""
Zip a file or an entire folder in Python.

Usage:
    python zip_it.py SOURCE OUTPUT_ZIP
    python zip_it.py /path/to/folder /path/to/output.zip
    python zip_it.py /path/to/file.txt /path/to/output.zip

Optional:
    -q, --quiet   suppress the "Created ..." message
"""

import argparse
import zipfile
from pathlib import Path


def zip_file(source_file: str, output_zip: str, quiet: bool = False):
    """Zip a single file."""
    source_file = Path(source_file)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, arcname=source_file.name)
    if not quiet:
        print(f"Created {output_zip}")


def zip_folder(source_folder: str, output_zip: str, quiet: bool = False):
    """Zip an entire folder, preserving its internal structure."""
    source_folder = Path(source_folder)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_folder.rglob("*"):
            if path.is_file():
                # arcname keeps paths relative to the folder itself
                arcname = path.relative_to(source_folder.parent)
                zf.write(path, arcname=arcname)
    if not quiet:
        print(f"Created {output_zip}")


def main():
    parser = argparse.ArgumentParser(description="Zip a file or folder.")
    parser.add_argument("source", help="Path to the file or folder to zip")
    parser.add_argument("output_zip", help="Path to the output .zip file")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress the success message"
    )
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source does not exist: {src}")

    if src.is_dir():
        zip_folder(src, args.output_zip, quiet=args.quiet)
    else:
        zip_file(src, args.output_zip, quiet=args.quiet)


if __name__ == "__main__":
    main()
