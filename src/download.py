"""Download the two NYC Open Data tables used in this project.

Capital Projects Dashboard - Citywide Budget and Schedule (fb86-vt7u):
    one row per project (FMS ID) per reporting period, every four months.
Capital Projects Dashboard - Citywide Budget Spend History and Variance (qj5n-h5qp):
    budget snapshots per project going back as far as 2006.

Usage: python src/download.py
"""
from pathlib import Path
import urllib.request

DATASETS = {
    "budget_schedule": "fb86-vt7u",
    "budget_history": "qj5n-h5qp",
}
URL = "https://data.cityofnewyork.us/api/views/{id}/rows.csv?accessType=DOWNLOAD"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, ds_id in DATASETS.items():
        out = RAW / f"{ds_id}.csv"
        if out.exists():
            print(f"{name}: already present ({out.name})")
            continue
        print(f"{name}: downloading {ds_id} ...")
        urllib.request.urlretrieve(URL.format(id=ds_id), out)
        print(f"  saved {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
