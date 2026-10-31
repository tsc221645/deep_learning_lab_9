"""Download the UCI electricity load dataset and extract its data file."""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)
DATA_FILENAME = "LD2011_2014.txt"


def download_dataset(output_dir: Path, force: bool = False) -> Path:
    """Download the UCI archive and safely extract only its data file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DATA_FILENAME
    if data_path.exists() and not force:
        return data_path

    archive_path = output_dir / "electricityloaddiagrams20112014.zip"
    temporary_archive = archive_path.with_suffix(".zip.part")
    temporary_data = data_path.with_suffix(".txt.part")
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "forecasting-lab/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_archive.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file, length=1024 * 1024)
        with zipfile.ZipFile(temporary_archive) as archive:
            members = [
                member
                for member in archive.infolist()
                if Path(member.filename).name == DATA_FILENAME
            ]
            if len(members) != 1:
                raise ValueError(
                    f"Expected one {DATA_FILENAME} in the UCI archive; "
                    f"found {len(members)}"
                )
            with archive.open(members[0]) as source, temporary_data.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary_data.stat().st_size != members[0].file_size:
            raise IOError("Extracted dataset size does not match the ZIP entry")
        temporary_data.replace(data_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_data.unlink(missing_ok=True)

    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = download_dataset(args.output_dir, force=args.force)
    print(f"Dataset ready: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
