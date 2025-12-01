# scripts/01_download_missing_records.py

import argparse
import os
from typing import List, Tuple

import pandas as pd
import requests
from urllib.parse import urljoin
from time import sleep


def find_missing_records(base_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scan ptbxl_database.csv and find records whose .hea or .dat file is missing.

    Returns:
        List of tuples: (rel_record_path, hea_path, dat_path)
        where rel_record_path is like 'records500/11000/11662_hr'
    """
    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")

    df = pd.read_csv(db_path)

    if "filename_hr" not in df.columns:
        raise KeyError("Column 'filename_hr' not found in ptbxl_database.csv")

    missing = []

    for _, row in df.iterrows():
        rel_path = row["filename_hr"]  # e.g. "records500/11000/11662_hr"
        rec_path = os.path.join(base_dir, rel_path)

        hea_path = rec_path + ".hea"
        dat_path = rec_path + ".dat"

        if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
            missing.append((rel_path, hea_path, dat_path))

    return missing


def download_file(url: str, dst_path: str, session: requests.Session, retries: int = 3) -> bool:
    """
    Download a single file from url to dst_path.

    Returns:
        True if success, False otherwise.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"  -> Downloading {url} -> {dst_path}")
            resp = session.get(url, stream=True, timeout=30)
            if resp.status_code != 200:
                print(f"     HTTP {resp.status_code} on attempt {attempt}")
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
    Find missing PTB-XL records (.hea/.dat) and download them from PhysioNet.

    Args:
        base_dir: local PTB-XL directory (contains ptbxl_database.csv, records500/, etc.)
        base_url: remote PTB-XL root url
        max_missing: if given, only process at most this many missing records
    """
    print(f"Base dir: {base_dir}")
    print(f"Remote base url: {base_url}")

    missing = find_missing_records(base_dir)
    total_missing = len(missing)
    print(f"Found {total_missing} records with missing .hea/.dat")

    if total_missing == 0:
        print("Nothing to download. All records seem complete.")
        return

    if max_missing is not None:
        missing = missing[:max_missing]
        print(f"Will download only the first {len(missing)} missing records (max_missing={max_missing})")

    session = requests.Session()

    completed = 0
    for idx, (rel_path, hea_path, dat_path) in enumerate(missing, start=1):
        print(f"\n[{idx}/{len(missing)}] Record: {rel_path}")

        # Construct remote URLs
        # rel_path example: 'records500/11000/11662_hr'
        hea_rel = rel_path + ".hea"
        dat_rel = rel_path + ".dat"

        hea_url = urljoin(base_url, hea_rel)
        dat_url = urljoin(base_url, dat_rel)

        ok_he = True
        ok_da = True

        if not os.path.exists(hea_path):
            ok_he = download_file(hea_url, hea_path, session)
        else:
            print(f"  hea exists: {hea_path}")

        if not os.path.exists(dat_path):
            ok_da = download_file(dat_url, dat_path, session)
        else:
            print(f"  dat exists: {dat_path}")

        if ok_he and ok_da:
            completed += 1
        else:
            print("  !! This record is still incomplete after download attempts.")

    print(f"\nDone. Successfully completed {completed} / {len(missing)} missing records.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Local PTB-XL directory (contains ptbxl_database.csv, records500/, etc.)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://physionet.org/files/ptb-xl/1.0.3/",
        help="Remote PTB-XL base URL",
    )
    parser.add_argument(
        "--max_missing",
        type=int,
        default=None,
        help="Maximum number of missing records to download (for testing).",
    )
    args = parser.parse_args()

    download_missing_records(
        base_dir=args.base_dir,
        base_url=args.base_url,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
