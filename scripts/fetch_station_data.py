"""Rebuild railsentinel/data/stations.csv from the datameet Indian Railways dataset.

Source: https://github.com/datameet/railways (stations.json, released under CC0).

The datameet file predates three station-code changes, so coordinates are read
from the old code and written under the current one:
  CSTM -> CSMT (Mumbai CSMT), ALD -> PRYJ (Prayagraj Jn), MGS -> DDU (Pt. DD Upadhyaya Jn).

State and region columns are filled from the mapping below because the source
leaves `state` empty for most major stations. `region` is the twin's own
grouping used for weather and fairness segments; it is not an official zone.

Usage:
    python scripts/fetch_station_data.py            # download and rebuild
    python scripts/fetch_station_data.py --check    # verify the committed CSV against the source
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "railsentinel" / "data" / "stations.csv"

# current_code: (source_code, display_name, city, state, region)
CURATED = {
    "NDLS": ("NDLS", "New Delhi", "Delhi", "Delhi", "North"),
    "NZM": ("NZM", "Hazrat Nizamuddin", "Delhi", "Delhi", "North"),
    "DEE": ("DEE", "Delhi Sarai Rohilla", "Delhi", "Delhi", "North"),
    "CDG": ("CDG", "Chandigarh", "Chandigarh", "Chandigarh", "North"),
    "UMB": ("UMB", "Ambala Cantt Jn", "Ambala", "Haryana", "North"),
    "LDH": ("LDH", "Ludhiana Jn", "Ludhiana", "Punjab", "North"),
    "JUC": ("JUC", "Jalandhar City", "Jalandhar", "Punjab", "North"),
    "ASR": ("ASR", "Amritsar Jn", "Amritsar", "Punjab", "North"),
    "MTJ": ("MTJ", "Mathura Jn", "Mathura", "Uttar Pradesh", "North"),
    "AGC": ("AGC", "Agra Cantt", "Agra", "Uttar Pradesh", "North"),
    "CNB": ("CNB", "Kanpur Central", "Kanpur", "Uttar Pradesh", "North"),
    "LKO": ("LKO", "Lucknow", "Lucknow", "Uttar Pradesh", "North"),
    "PRYJ": ("ALD", "Prayagraj Jn", "Prayagraj", "Uttar Pradesh", "North"),
    "BSB": ("BSB", "Varanasi Jn", "Varanasi", "Uttar Pradesh", "North"),
    "DDU": ("MGS", "Pt. Deen Dayal Upadhyaya Jn", "Mughalsarai", "Uttar Pradesh", "North"),
    "JP": ("JP", "Jaipur Jn", "Jaipur", "Rajasthan", "North"),
    "AII": ("AII", "Ajmer Jn", "Ajmer", "Rajasthan", "North"),
    "KOTA": ("KOTA", "Kota Jn", "Kota", "Rajasthan", "North"),
    "GWL": ("GWL", "Gwalior Jn", "Gwalior", "Madhya Pradesh", "Central"),
    "BPL": ("BPL", "Bhopal Jn", "Bhopal", "Madhya Pradesh", "Central"),
    "ET": ("ET", "Itarsi Jn", "Itarsi", "Madhya Pradesh", "Central"),
    "JBP": ("JBP", "Jabalpur", "Jabalpur", "Madhya Pradesh", "Central"),
    "RTM": ("RTM", "Ratlam Jn", "Ratlam", "Madhya Pradesh", "Central"),
    "NGP": ("NGP", "Nagpur", "Nagpur", "Maharashtra", "Central"),
    "R": ("R", "Raipur Jn", "Raipur", "Chhattisgarh", "Central"),
    "BSP": ("BSP", "Bilaspur Jn", "Bilaspur", "Chhattisgarh", "Central"),
    "CSMT": ("CSTM", "Mumbai CSMT", "Mumbai", "Maharashtra", "West"),
    "LTT": ("LTT", "Lokmanya Tilak Terminus", "Mumbai", "Maharashtra", "West"),
    "BDTS": ("BDTS", "Mumbai Bandra Terminus", "Mumbai", "Maharashtra", "West"),
    "KYN": ("KYN", "Kalyan Jn", "Kalyan", "Maharashtra", "West"),
    "PNVL": ("PNVL", "Panvel", "Panvel", "Maharashtra", "West"),
    "PUNE": ("PUNE", "Pune Jn", "Pune", "Maharashtra", "West"),
    "DD": ("DD", "Daund Jn", "Daund", "Maharashtra", "West"),
    "NK": ("NK", "Nasik Road", "Nashik", "Maharashtra", "West"),
    "MMR": ("MMR", "Manmad Jn", "Manmad", "Maharashtra", "West"),
    "BSL": ("BSL", "Bhusaval Jn", "Bhusaval", "Maharashtra", "West"),
    "AK": ("AK", "Akola Jn", "Akola", "Maharashtra", "West"),
    "BD": ("BD", "Badnera Jn", "Badnera", "Maharashtra", "West"),
    "SUR": ("SUR", "Solapur Jn", "Solapur", "Maharashtra", "West"),
    "MRJ": ("MRJ", "Miraj Jn", "Miraj", "Maharashtra", "West"),
    "RN": ("RN", "Ratnagiri", "Ratnagiri", "Maharashtra", "West"),
    "MAO": ("MAO", "Madgaon", "Madgaon", "Goa", "West"),
    "ST": ("ST", "Surat", "Surat", "Gujarat", "West"),
    "BRC": ("BRC", "Vadodara Jn", "Vadodara", "Gujarat", "West"),
    "ADI": ("ADI", "Ahmedabad Jn", "Ahmedabad", "Gujarat", "West"),
    "HWH": ("HWH", "Howrah Jn", "Kolkata", "West Bengal", "East"),
    "SDAH": ("SDAH", "Sealdah", "Kolkata", "West Bengal", "East"),
    "KGP": ("KGP", "Kharagpur Jn", "Kharagpur", "West Bengal", "East"),
    "NJP": ("NJP", "New Jalpaiguri", "Siliguri", "West Bengal", "East"),
    "TATA": ("TATA", "Tatanagar Jn", "Jamshedpur", "Jharkhand", "East"),
    "DHN": ("DHN", "Dhanbad Jn", "Dhanbad", "Jharkhand", "East"),
    "GAYA": ("GAYA", "Gaya Jn", "Gaya", "Bihar", "East"),
    "PNBE": ("PNBE", "Patna Jn", "Patna", "Bihar", "East"),
    "BBS": ("BBS", "Bhubaneswar", "Bhubaneswar", "Odisha", "East"),
    "GHY": ("GHY", "Guwahati", "Guwahati", "Assam", "Northeast"),
    "MAS": ("MAS", "Chennai Central", "Chennai", "Tamil Nadu", "South"),
    "KPD": ("KPD", "Katpadi Jn", "Vellore", "Tamil Nadu", "South"),
    "JTJ": ("JTJ", "Jolarpettai Jn", "Jolarpettai", "Tamil Nadu", "South"),
    "CBE": ("CBE", "Coimbatore Jn", "Coimbatore", "Tamil Nadu", "South"),
    "SBC": ("SBC", "KSR Bengaluru City", "Bengaluru", "Karnataka", "South"),
    "UBL": ("UBL", "Hubballi Jn", "Hubballi", "Karnataka", "South"),
    "WADI": ("WADI", "Wadi Jn", "Wadi", "Karnataka", "South"),
    "SC": ("SC", "Secunderabad Jn", "Hyderabad", "Telangana", "South"),
    "GTL": ("GTL", "Guntakal Jn", "Guntakal", "Andhra Pradesh", "South"),
    "RU": ("RU", "Renigunta Jn", "Renigunta", "Andhra Pradesh", "South"),
    "BZA": ("BZA", "Vijayawada Jn", "Vijayawada", "Andhra Pradesh", "South"),
    "VSKP": ("VSKP", "Visakhapatnam", "Visakhapatnam", "Andhra Pradesh", "South"),
    "ERS": ("ERS", "Ernakulam Jn", "Kochi", "Kerala", "South"),
    "TVC": ("TVC", "Thiruvananthapuram Central", "Thiruvananthapuram", "Kerala", "South"),
    "PGT": ("PGT", "Palakkad Jn", "Palakkad", "Kerala", "South"),
}

FIELDS = ["code", "name", "city", "state", "region", "lat", "lon", "source_code"]


def _download() -> dict:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        return json.load(resp)


def build_rows(geojson: dict) -> list[dict]:
    index = {f["properties"]["code"]: f for f in geojson["features"]}
    rows = []
    for code, (src, name, city, state, region) in CURATED.items():
        feat = index.get(src)
        if feat is None or not feat.get("geometry"):
            raise SystemExit(f"Station {src} not found in source data")
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({"code": code, "name": name, "city": city, "state": state, "region": region,
                     "lat": round(float(lat), 5), "lon": round(float(lon), 5), "source_code": src})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare committed CSV with the source instead of writing")
    args = parser.parse_args()
    rows = build_rows(_download())
    if args.check:
        with OUT_PATH.open(newline="", encoding="utf-8") as fh:
            committed = {r["code"]: r for r in csv.DictReader(fh)}
        mismatches = [r["code"] for r in rows
                      if r["code"] not in committed
                      or abs(float(committed[r["code"]]["lat"]) - r["lat"]) > 1e-4
                      or abs(float(committed[r["code"]]["lon"]) - r["lon"]) > 1e-4]
        print("OK: committed station table matches source" if not mismatches else f"Mismatch: {mismatches}")
        return 1 if mismatches else 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} stations to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
