#!/usr/bin/env python3
"""
Analyse Regions from semantic segmentation masks.

Iterates image/mask pairs in a dataset directory, loads them at a
specified pyramid level, converts RGB → CIELAB, labels connected
components (8-connectivity) per class as instances, and computes:

  • Area (px)
  • Mean LAB colour
  • 3×3 covariance matrix of LAB values
  • Number of pixels within a Euclidean distance threshold from a
    reference LAB colour

Results are saved to a descriptively-named CSV file.

Usage:
    conda run -n wsi python analyse_regions.py /path/to/dataset \\
        --level 3 \\
        --ref-lab 50 0 0 \\
        --threshold-dist 15 \\
        --output-dir .
"""

import argparse
import csv
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as connected_components

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from qupath_handler import PyramidTiff, CLASS_NAMES


# ---------------------------------------------------------------------------
# BIDS / participants.csv helpers
# ---------------------------------------------------------------------------

def load_participants_csv(dataset_root: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata from participants.csv; return {IMAGE_NAME: {cols}}."""
    participants_data: Dict[str, Dict[str, str]] = {}
    csv_path = dataset_root / "participants.csv"
    if not csv_path.exists():
        csv_path = dataset_root / "participants.tsv"
    if not csv_path.exists():
        csv_path = dataset_root / "data" / "participants.csv"
    if not csv_path.exists():
        print(f"[WARNING] participants.csv not found in {dataset_root}")
        return participants_data

    print(f"[INFO] Loading participants from {csv_path}")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('IMAGE_NAME', '').strip()
                if name:
                    participants_data[name] = {
                        'subject_id': row.get('SUBJECT_ID', ''),
                        'session_id': row.get('SESSION_ID', ''),
                        'age': row.get('AGE', ''),
                        'psa': row.get('PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL', ''),
                        'diagnosis': row.get('SLIDE_DIAGNOSIS', ''),
                        'isup': row.get('ISUP_Grade_Group_', ''),
                        'gleason': row.get('Gleason_score', ''),
                        'scanner': row.get('Scanner', ''),
                    }
    except Exception as e:
        print(f"[WARNING] Failed to read participants.csv: {e}")
    return participants_data


def image_name_to_subdirs(image_name: str) -> Tuple[str, str]:
    """'sub-01_ses-01' → ('sub-01', 'ses-01')."""
    parts = image_name.split('_ses-', 1)
    participant_dir = parts[0]
    session_str = f"ses-{parts[1]}" if len(parts) > 1 else ""
    return participant_dir, session_str


# ---------------------------------------------------------------------------
# Colour conversion  (RGB → CIELAB, no external dependencies)
# ---------------------------------------------------------------------------

def _linearize_srgb(rgb: np.ndarray) -> np.ndarray:
    """Inverse sRGB gamma (ITU-R BT.709 / IEC 61966-2-1)."""
    mask = rgb > 0.04045
    out = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    return out


_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# D65 white point
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _lab_f(t: np.ndarray) -> np.ndarray:
    """CIE Lab non-linear compression."""
    delta = 6 / 29
    return np.where(t > delta ** 3, t ** (1 / 3), t / (3 * delta ** 2) + 4 / 29)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB (uint8, 0–255) to CIELAB (float64).

    Args:
        rgb: (H, W, 3) array in sRGB colour space.

    Returns:
        (H, W, 3) array with L* (0–100), a*, b* (roughly -128–127).
    """
    rgb_f = rgb.astype(np.float64) / 255.0
    rgb_lin = _linearize_srgb(rgb_f)
    xyz = rgb_lin @ _M_RGB2XYZ.T

    fx = _lab_f(xyz[..., 0] / _XN)
    fy = _lab_f(xyz[..., 1] / _YN)
    fz = _lab_f(xyz[..., 2] / _ZN)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


# ---------------------------------------------------------------------------
# File helpers (BIDS-aware)
# ---------------------------------------------------------------------------

def build_image_path(data_dir: Path, image_name: str, stain: str) -> Path:
    """Return path to the WSI image for a given stain.

    stain = 'h-e'       → wsi_h-e/{image_name}_h-e.ome.tif
    stain = 'hmwck-amacr' → wsi_hmwck-amacr/{image_name}_hmwck-amacr.ome.tif
    """
    pdir, sdir = image_name_to_subdirs(image_name)
    stain_dir = f"wsi_{stain}"
    return data_dir / pdir / sdir / stain_dir / f"{image_name}_{stain}.ome.tif"


def build_mask_path(data_dir: Path, image_name: str, stain: str) -> Path:
    """Return path to the semantic mask for a given stain."""
    pdir, sdir = image_name_to_subdirs(image_name)
    stain_dir = f"wsi_{stain}"
    return data_dir / pdir / sdir / stain_dir / f"{image_name}_{stain}_mask.ome.tif"


def find_pairs(data_dir: Path, participants: Dict[str, Dict[str, str]],
               stain: str) -> List[Tuple[str, Path, Path]]:
    """Return list of (image_name, image_path, mask_path) from participants."""
    pairs: List[Tuple[str, Path, Path]] = []
    for name in sorted(participants.keys()):
        img_path = build_image_path(data_dir, name, stain)
        mask_path = build_mask_path(data_dir, name, stain)
        if img_path.exists() and mask_path.exists():
            pairs.append((name, img_path, mask_path))
    return pairs


def load_classes(path: Path) -> Dict[int, str]:
    """Load {class_id: class_name} from *classes.txt*."""
    classes: Dict[int, str] = {}
    if not path.exists():
        return classes
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            try:
                cid = int(parts[0].strip())
                cname = parts[1].strip()
                if cid > 0:
                    classes[cid] = cname
            except ValueError:
                pass
    return classes


# ---------------------------------------------------------------------------
# CSV filename
# ---------------------------------------------------------------------------

def make_csv_path(output_dir: Path, ref_lab: Tuple[float, float, float],
                  threshold: float) -> Path:
    """Generate a descriptive CSV filename."""
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    ref_str = f"L{ref_lab[0]}A{ref_lab[1]}B{ref_lab[2]}"
    thr_str = f"{threshold}"
    fname = f"rosa_{date_str}_{time_str}_pnr-ref-{ref_str}_thr-{thr_str}.csv"
    return output_dir / fname


def make_csv_path_stain(output_dir: Path, stain: str,
                         ref_lab: Tuple[float, float, float],
                         threshold: float) -> Path:
    """Descriptive CSV name with stain tag."""
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    ref_str = f"L{ref_lab[0]}A{ref_lab[1]}B{ref_lab[2]}"
    thr_str = f"{threshold}"
    fname = f"{stain}_rosa_{date_str}_{time_str}_pnr-ref-{ref_str}_thr-{thr_str}.csv"
    return output_dir / fname


# ---------------------------------------------------------------------------
# Per-instance analysis
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "image_name", "class_id", "class_name", "instance_id", "area_px",
    "L_mean", "A_mean", "B_mean",
    "cov_LL", "cov_LA", "cov_LB",
    "cov_AL", "cov_AA", "cov_AB",
    "cov_BL", "cov_BA", "cov_BB",
    "pixels_near_ref",
    "ref_L", "ref_A", "ref_B", "threshold_distance",
]


def _compute_cov(pixels: np.ndarray) -> np.ndarray:
    """3×3 covariance matrix (returns zeros for < 2 pixels)."""
    if pixels.shape[0] < 2:
        return np.zeros((3, 3), dtype=np.float64)
    return np.cov(pixels, rowvar=False)


def analyse_image(image_name: str,
                  image_tiff: PyramidTiff,
                  mask_tiff: PyramidTiff,
                  level: int,
                  ref_lab: Tuple[float, float, float],
                  threshold_dist: float,
                  classes: Dict[int, str]) -> List[Dict]:
    """Analyse a single image/mask pair; return list of row-dicts."""
    # ---- load data --------------------------------------------------------
    img = image_tiff.read_level(level)
    msk = mask_tiff.read_level(level)
    if msk.ndim == 3:
        msk = msk[:, :, 0]

    if img.shape[:2] != msk.shape[:2]:
        dh = abs(img.shape[0] - msk.shape[0])
        dw = abs(img.shape[1] - msk.shape[1])
        if dh > 2 or dw > 2:
            print(f"  [WARNING] Size mismatch img={img.shape[:2]} vs mask={msk.shape[:2]} "
                  f"(Δh={dh}, Δw={dw}) — likely different base resolutions, skipping")
            return []
        h = min(img.shape[0], msk.shape[0])
        w = min(img.shape[1], msk.shape[1])
        print(f"  [INFO] Size mismatch img={img.shape[:2]} vs mask={msk.shape[:2]} "
              f"(Δh={dh}, Δw={dw}) — cropping to ({h}, {w})")
        img = img[:h, :w]
        msk = msk[:h, :w]

    # ---- RGB → LAB --------------------------------------------------------
    lab = rgb_to_lab(img)

    # free image data early
    del img
    gc.collect()

    ref = np.array(ref_lab, dtype=np.float64)

    rows: List[Dict] = []
    structure = np.ones((3, 3), dtype=bool)          # 8-connectivity

    unique_ids = [c for c in np.unique(msk) if c != 0]

    for cls_id in unique_ids:
        cls_name = classes.get(int(cls_id)) or CLASS_NAMES.get(int(cls_id), f"Class {cls_id}")

        binary = (msk == cls_id)
        labeled, n_inst = connected_components(binary, structure=structure)

        for inst_id in range(1, n_inst + 1):
            inst_mask = (labeled == inst_id)
            area = int(np.sum(inst_mask))
            if area == 0:
                continue

            inst_pixels = lab[inst_mask]  # (N, 3)

            mean_lab = np.mean(inst_pixels, axis=0)
            cov = _compute_cov(inst_pixels)

            # count pixels within Euclidean distance of ref LAB
            dist = np.sqrt(np.sum((inst_pixels - ref) ** 2, axis=1))
            near_count = int(np.sum(dist <= threshold_dist))

            rows.append({
                "image_name": image_name,
                "class_id": int(cls_id),
                "class_name": cls_name,
                "instance_id": inst_id,
                "area_px": area,
                "L_mean": mean_lab[0],
                "A_mean": mean_lab[1],
                "B_mean": mean_lab[2],
                "cov_LL": cov[0, 0],
                "cov_LA": cov[0, 1],
                "cov_LB": cov[0, 2],
                "cov_AL": cov[1, 0],
                "cov_AA": cov[1, 1],
                "cov_AB": cov[1, 2],
                "cov_BL": cov[2, 0],
                "cov_BA": cov[2, 1],
                "cov_BB": cov[2, 2],
                "pixels_near_ref": near_count,
                "ref_L": ref_lab[0],
                "ref_A": ref_lab[1],
                "ref_B": ref_lab[2],
                "threshold_distance": threshold_dist,
            })

        # free per-class mask
        del binary, labeled
        gc.collect()

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse connected regions in semantic segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  conda run -n wsi python analyse_regions.py "/media/abel/TOSHIBA EXT/prostate_HnE-IHC_dataset" --level 3
  conda run -n wsi python analyse_regions.py /path/to/dataset --stain hmwck-amacr --level 3

Expected dataset structure (BIDS):
  dataset_root/
    participants.csv
    data/
      sub-XX/
        ses-YY/
          wsi_h-e/
            sub-XX_ses-YY_h-e.ome.tif
            sub-XX_ses-YY_h-e_mask.ome.tif
          wsi_hmwck-amacr/
            sub-XX_ses-YY_hmwck-amacr.ome.tif
            sub-XX_ses-YY_hmwck-amacr_mask.ome.tif

classes.txt is read from the script directory.
        """,
    )
    parser.add_argument("dataset_root",
                        help="Dataset root (contains participants.csv and data/)")
    parser.add_argument("--stain", "-s", default="hmwck-amacr",
                        choices=["h-e", "hmwck-amacr"],
                        help="Stain to process: h-e or hmwck-amacr (default: hmwck-amacr)")
    parser.add_argument("--data-subdir", default="data",
                        help="Subdirectory with BIDS structure (default: data)")
    parser.add_argument("--level", "-l", type=int, default=3,
                        help="Pyramid level to load (default: 3)")
    parser.add_argument("--ref-lab", nargs=3, type=float,
                        default=[50.0, 0.0, 0.0],
                        metavar=("L", "A", "B"),
                        help="Reference LAB colour (default: 50 0 0)")
    parser.add_argument("--threshold-dist", type=float, default=15.0,
                        help="Euclidean distance threshold in LAB space "
                             "(default: 15.0)")
    parser.add_argument("--output-dir", "-o",
                        default=".",
                        help="Output directory for CSV (default: current dir)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    dataset_root = Path(args.dataset_root)
    data_dir = dataset_root / args.data_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_lab = (args.ref_lab[0], args.ref_lab[1], args.ref_lab[2])
    threshold_dist = args.threshold_dist
    stain = args.stain

    # ---- participants -----------------------------------------------------
    participants = load_participants_csv(dataset_root)
    if not participants:
        print("[ERROR] No participants found in participants.csv")
        sys.exit(1)
    print(f"[INFO] Loaded {len(participants)} participant(s) from participants.csv")

    # ---- classes ----------------------------------------------------------
    script_dir = Path(__file__).parent.resolve()
    classes = load_classes(script_dir / "classes.txt")
    if not classes:
        print("[INFO] classes.txt not found – falling back to CLASS_NAMES")
    n_classes = max(len(classes), len(CLASS_NAMES))
    print(f"[INFO] {n_classes} classes available")

    # ---- find pairs -------------------------------------------------------
    pairs = find_pairs(data_dir, participants, stain)
    if not pairs:
        print(f"[ERROR] No image-mask pairs found for stain '{stain}' in {data_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(pairs)} image-mask pair(s) for stain '{stain}'")

    # ---- output CSV path --------------------------------------------------
    csv_path = make_csv_path_stain(output_dir, stain, ref_lab, threshold_dist)
    print(f"[INFO] Output CSV: {csv_path}")

    # ---- process ----------------------------------------------------------
    total_regions = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for idx, (name, img_path, mask_path) in enumerate(pairs, 1):
            print(f"\n[{idx}/{len(pairs)}] {name}  [{stain}]")
            print(f"  Image: {img_path}")
            print(f"  Mask:  {mask_path}")

            try:
                with (PyramidTiff(str(img_path)) as img_tif,
                      PyramidTiff(str(mask_path)) as mask_tif):

                    level = min(args.level,
                                img_tif.n_levels - 1,
                                mask_tif.n_levels - 1)
                    print(f"  Level: {level} "
                          f"(requested {args.level}, "
                          f"available {img_tif.n_levels})")

                    rows = analyse_image(
                        name,
                        img_tif,
                        mask_tif,
                        level,
                        ref_lab,
                        threshold_dist,
                        classes,
                    )

                for row in rows:
                    writer.writerow(row)

                n_regions = len(rows)
                total_regions += n_regions
                print(f"  Regions found: {n_regions}")

            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()
            finally:
                gc.collect()

    print(f"\n[DONE] {total_regions} regions written to {csv_path}")


if __name__ == "__main__":
    main()
