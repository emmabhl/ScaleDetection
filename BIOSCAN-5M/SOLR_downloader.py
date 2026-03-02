import requests
import csv
import time
import os
from typing import Optional

SOLR_URL = "https://spathi.cmpt.sfu.ca/solr/bioscan5m/select"
ROWS_PER_BATCH = 50_000
OUTPUT_CSV = "ScaleDetection/BIOSCAN-5M/data/BIOSCAN_metadata_spathi.csv"
CURSOR_FILE = "cursor.txt"
SLEEP_BETWEEN_REQUESTS = 0.2  # be polite to the server


def load_cursor() -> str:
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE, "r") as f:
            return f.read().strip()
    return "*"


def save_cursor(cursor: str):
    with open(CURSOR_FILE, "w") as f:
        f.write(cursor)


def write_rows(csv_writer, docs, header_written: bool) -> bool:
    if not docs:
        return header_written

    if not header_written:
        csv_writer.writeheader()
        header_written = True

    for doc in docs:
        csv_writer.writerow(doc)

    return header_written


def download_all():
    cursor_mark = load_cursor()
    header_written = os.path.exists(OUTPUT_CSV)

    total_downloaded = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer: Optional[csv.DictWriter] = None

        while True:
            params = {
                "q": "*:*",
                "sort": "id asc",
                "rows": ROWS_PER_BATCH,
                "cursorMark": cursor_mark,
                "wt": "json"
            }

            response = requests.get(SOLR_URL, params=params, timeout=120)
            response.raise_for_status()

            data = response.json()
            docs = data["response"]["docs"]
            next_cursor = data.get("nextCursorMark")

            if csv_writer is None and docs:
                fieldnames = sorted({k for doc in docs for k in doc.keys()})
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            header_written = write_rows(csv_writer, docs, header_written)

            batch_size = len(docs)
            total_downloaded += batch_size

            print(
                f"Downloaded {batch_size} docs | "
                f"Total: {total_downloaded} | "
                f"Cursor: {cursor_mark[:20]}..."
            )

            if next_cursor == cursor_mark:
                print("Reached end of collection.")
                break

            cursor_mark = next_cursor
            save_cursor(cursor_mark)

            time.sleep(SLEEP_BETWEEN_REQUESTS)


if __name__ == "__main__":
    download_all()
