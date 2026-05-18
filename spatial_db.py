import geopandas as gpd
from shapely.geometry import box
import csv
import os
from datetime import datetime
import json
from typing import List, Dict

DB_PATH = "heart_detections.gpkg"
CSV_PATH = "detections.csv"
STORE_JSON = "spatial_data.json"

def insert_spatial_data(label: str, bbox: List[int], image_name: str = None) -> None:
    """
    Existing function in your project may already do more (CSV / geopkg).
    Keep that logic, but also append a minimal JSON record so the app can read it back.
    """
    # Save to GeoPackage (optional but nice to keep)
    x1, y1, x2, y2 = bbox
    geom = box(x1, y1, x2, y2)
    gdf = gpd.GeoDataFrame(
        [{"region_label": label, "geometry": geom, "image_name": image_name, "created_at": datetime.utcnow().isoformat()}],
        crs="EPSG:4326"
    )
    try:
        gdf.to_file(DB_PATH, layer="detections", driver="GPKG", mode="a")
    except Exception:
        gdf.to_file(DB_PATH, layer="detections", driver="GPKG")

    # Also append to CSV (flat, easy to read)
    append_region_to_csv(label, bbox, image_name)

    # after existing persistence logic, also append to JSON index for quick retrieval:
    try:
        rec = {"label": label, "bbox": bbox, "image": image_name}
        data = []
        if os.path.exists(STORE_JSON):
            with open(STORE_JSON, "r", encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                except Exception:
                    data = []
        data.append(rec)
        with open(STORE_JSON, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass

def append_region_to_csv(label, bbox, image_name="temp.jpg"):
    x1, y1, x2, y2 = bbox
    row = {
        "image_name": image_name,
        "label": label,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "created_at": datetime.utcnow().isoformat()
    }

    # Create CSV with header if not exists
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def get_all_detections():
    # If you want to read GeoPackage layer
    return gpd.read_file(DB_PATH, layer="detections")

def get_spatial_data() -> List[Dict]:
    """
    Return list of stored spatial entries. Tries JSON first, falls back to CSV if present.
    """
    try:
        if os.path.exists(STORE_JSON):
            with open(STORE_JSON, "r", encoding="utf-8") as fh:
                return json.load(fh)
        # fallback: try CSV (simple format: label,x1,y1,x2,y2,image)
        csv_path = "spatial_data.csv"
        if os.path.exists(csv_path):
            out = []
            with open(csv_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split(",")
                    if len(parts) >= 5:
                        label = parts[0]
                        bbox = list(map(int, parts[1:5]))
                        img = parts[5] if len(parts) > 5 else None
                        out.append({"label": label, "bbox": bbox, "image": img})
            return out
    except Exception:
        pass
    return []

import os
import json

STORE_JSON = "spatial_data.json"
CSV_PATH = "spatial_data.csv"
GPKG_PATH = "spatial_data.gpkg"

def clear_spatial_data(remove_json: bool = True, remove_csv: bool = True, remove_geopkg: bool = True) -> bool:
    """
    Remove local spatial store files (JSON/CSV/GeoPackage). Returns True if any file removed or truncated.
    This is safe for local test data; adapt if you use a real DB (PostGIS/Neo4j) — then implement SQL/driver deletion.
    """
    removed_any = False

    try:
        if remove_json and os.path.exists(STORE_JSON):
            try:
                os.remove(STORE_JSON)
            except Exception:
                # fallback: truncate file
                with open(STORE_JSON, "w", encoding="utf-8") as fh:
                    fh.write("[]")
            removed_any = True

        if remove_csv and os.path.exists(CSV_PATH):
            try:
                os.remove(CSV_PATH)
            except Exception:
                # truncate fallback
                with open(CSV_PATH, "w", encoding="utf-8") as fh:
                    fh.write("")
            removed_any = True

        if remove_geopkg and os.path.exists(GPKG_PATH):
            try:
                os.remove(GPKG_PATH)
            except Exception:
                # ignore if cannot remove
                pass
            removed_any = True if os.path.exists(GPKG_PATH) is False else removed_any

    except Exception:
        # keep function resilient
        pass

    return removed_any
