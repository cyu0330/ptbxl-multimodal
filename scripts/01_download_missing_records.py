# scripts/01_download_missing_records.py
#
# Utility script for locating missing PTB-XL waveform files (.hea / .dat)
# and downloading them from PhysioNet if needed.

import argparse
import os
from typing import List, Tuple

import pandas as pd
import requests
from urllib.parse import urljoin
from time import sleep


def find_missing_records(base_dir: str) -> List[Tuple[str, str, str]]:
    """
    Check ptbxl_database.csv and identify entries whose .hea or .dat file
    is not present locally.

    Returns:
        A list of tuples (relative_record_path, local_hea_path, local_dat_path)
    """
    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")

    df = pd.read_csv(db_path)

    if "filename_hr" not in df.columns:
        raise KeyError("Column 'filename_hr' missing in ptbxl_database.csv")

    missing = []

    for _, row in df.iterrows():
        rel_path = row["filename_hr"]
        rec_path = os.path.join(base_dir, rel_path)

        hea_path = rec_path + ".hea"
        dat_path = rec_path + ".dat"

        if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
            missing.append((rel_path, hea_path, dat_path))

    return missing


def download_file(url: str, dst_path: str, session: requests.Session, retries: int = 3) -> bool:
    """
    Download a file from a given URL into dst_path.

    Returns:
        True if successful, False otherwise.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"  -> Downloading {url}")
            resp = session.get(url, stream=True, timeout=30)

            if resp.status_code != 200:
                print(f"     HTTP {resp.status_code} (attempt {attempt})")
                sleep(1.0)
                continue

            with open(dst_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True

        except Exception as e:
            print(f"     Error on attempt {attempt}: {e}")
            sleep(1.0)

    print(f"  !! Failed to download {url}")
    return False


def download_missing_records(
    base_dir: str,
    base_url: str = "https://physionet.org/files/ptb-xl/1.0.3/",
    max_missing: int = None,
) -> None:
    """
    Detect missing waveform files and download them from PhysioNet.

    Args:
        base_dir: Local PTB-XL directory.
        base_url: Remote root path of PTB-XL.
        max_missing: Limit number of records for testing.
    """
    print(f"Local PTB-XL directory: {base_dir}")

    missing = find_missing_records(base_dir)
    total = len(missing)
    print(f"Missing records: {total}")

    if total == 0:
        print("All waveform files are present.")
        return

    if max_missing is not None:
        missing = missing[:max_missing]
        print(f"Processing only first {len(missing)} records (max_missing={max_missing})")

    session = requests.Session()

    completed = 0
    for idx, (rel_path, hea_path, dat_path) in enumerate(missing, start=1):
        print(f"\n[{idx}/{len(missing)}] {rel_path}")

        hea_url = urljoin(base_url, rel_path + ".hea")
        dat_url = urljoin(base_url, rel_path + ".dat")

        ok_he = True
        ok_da = True

        if not os.path.exists(hea_path):
            ok_he = download_file(hea_url, hea_path, session)

        if not os.path.exists(dat_path):
            ok_da = download_file(dat_url, dat_path, session)

        if ok_he and ok_da:
            completed += 1
        else:
            print("  Incomplete after download attempts.")

    print(f"\nCompleted {completed} / {len(missing)} records.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Local PTB-XL directory containing ptbxl_database.csv.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://physionet.org/files/ptb-xl/1.0.3/",
        help="Remote base URL of PTB-XL dataset.",
    )
    parser.add_argument(
        "--max_missing",
        type=int,
        default=None,
        help="Limit number of records to download.",
    )

    args = parser.parse_args()
    download_missing_records(
        base_dir=args.base_dir,
        base_url=args.base_url,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
