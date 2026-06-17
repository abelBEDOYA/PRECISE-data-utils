#!/usr/bin/env python3
"""
QuPath Export Handler v2

Adapted for BIDS-like dataset structure with participants.csv.

New directory structure (input):
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

Clinical/metadata is read from participants.csv (IMAGE_NAME row key).
"""

import os
import sys
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

if "--batch-save" in sys.argv or "--save-all" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import tifffile


def load_participants_csv(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load metadata from participants.csv.

    Returns:
        Dict mapping IMAGE_NAME -> {age, psa, dre, mri, diagnosis, isup, gleason, scanner, ...}
    """
    participants_data = {}

    possible_paths = [
        data_dir / "participants.csv",
        data_dir / "participants.tsv",
        data_dir / "data" / "participants.csv",
        data_dir.parent / "participants.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        print(f"[Warning] participants.csv not found in {data_dir}")
        return participants_data

    print(f"[Participants] Loading from: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get('IMAGE_NAME', '').strip()
                if image_name:
                    participants_data[image_name] = {
                        'subject_id': row.get('SUBJECT_ID', ''),
                        'session_id': row.get('SESSION_ID', ''),
                        'age': row.get('AGE', ''),
                        'psa': row.get('PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL', ''),
                        'digital_rectal_exam': row.get('DIGITAL_RECTAL_EXAM', ''),
                        'mri_findings': row.get('FINDINGS_IN_PELVIC_MRI', ''),
                        'diagnosis': row.get('SLIDE_DIAGNOSIS', ''),
                        'isup': row.get('ISUP_Grade_Group_', ''),
                        'gleason': row.get('Gleason_score', ''),
                        'scanner': row.get('Scanner', ''),
                    }
    except Exception as e:
        print(f"[Warning] Failed to read participants.csv: {e}")

    return participants_data


def image_name_to_subdirs(image_name: str) -> Tuple[str, str]:
    """
    Convert IMAGE_NAME (e.g. 'sub-01_ses-01') to participant and session dirs.
    Returns (participant_dir, session_dir) = ('sub-01', 'ses-01').
    """
    parts = image_name.split('_ses-', 1)
    participant_dir = parts[0]
    session_str = f"ses-{parts[1]}" if len(parts) > 1 else ""
    return participant_dir, session_str


CLASS_NAMES = {
    1: "Tumor",
    2: "Bening gland",
    3: "Artifact",
    4: "High grade prostatic intraepithelial neoplasia (HGPIN)",
    5: "Intraductal carcinoma",
    6: "Atypical intraductal proliferation",
    7: "Stroma",
}

CLASS_COLORS_HEX = [
    "#000000",  # 0: Background
    "#B83B5E",  # 1: Tumor
    "#F38181",  # 2: Bening gland
    "#AA96DA",  # 3: Artifact
    "#FCBAD3",  # 4: HGPIN
    "#FF6B6B",  # 5: Intraductal carcinoma
    "#9B59B6",  # 6: Atypical intraductal proliferation
    "#FAE3D9",  # 7: Stroma
]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = [hex_to_rgb(c) for c in CLASS_COLORS_HEX]


class PyramidTiff:
    """
    Efficient wrapper for pyramidal TIFF (OME-TIFF).
    (Identical to original version.)
    """

    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.tif = tifffile.TiffFile(path)
        self._detect_pyramid_structure()
        self._cache_level_info()
        if self.verbose:
            self._print_info()

    def _detect_pyramid_structure(self) -> None:
        self._pyramid_type = None
        self._levels_source = []

        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._pyramid_type = 'series_levels'
                self._levels_source = first_series.levels
                self.n_levels = len(self._levels_source)
                return

        if len(self.tif.series) > 1:
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'multiple_series'
                self._levels_source = self.tif.series
                self.n_levels = len(self._levels_source)
                return

        if len(self.tif.pages) > 1:
            shapes = [p.shape for p in self.tif.pages]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'pages'
                self._levels_source = list(self.tif.pages)
                self.n_levels = len(self._levels_source)
                return

        self._pyramid_type = 'single'
        if len(self.tif.series) > 0:
            self._levels_source = [self.tif.series[0]]
        else:
            self._levels_source = [self.tif.pages[0]]
        self.n_levels = 1

    def _shapes_are_pyramid(self, shapes: List[Tuple]) -> bool:
        if len(shapes) < 2:
            return False
        sizes = []
        for shape in shapes:
            dims = sorted(shape, reverse=True)[:2]
            sizes.append(max(dims))
        for i in range(1, len(sizes)):
            if sizes[i] >= sizes[i-1]:
                return False
        return True

    def _cache_level_info(self) -> None:
        self.level_info: List[Dict[str, Any]] = []
        base_w, base_h = 0, 0
        for i, level_src in enumerate(self._levels_source):
            shape = level_src.shape
            h, w, c = self._parse_shape(shape)
            if i == 0:
                ds = 1.0
                base_w, base_h = w, h
            else:
                ds = base_w / w if w > 0 else 1.0
            self.level_info.append({
                'index': i,
                'shape': shape,
                'width': w,
                'height': h,
                'channels': c,
                'downsample': ds,
            })

    def _parse_shape(self, shape: Tuple) -> Tuple[int, int, int]:
        if len(shape) == 2:
            return shape[0], shape[1], 1
        elif len(shape) == 3:
            if shape[0] <= 4:
                return shape[1], shape[2], shape[0]
            else:
                return shape[0], shape[1], shape[2]
        elif len(shape) >= 4:
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1

    def _print_info(self) -> None:
        print(f"  [PyramidTiff] Type: {self._pyramid_type}")
        print(f"  [PyramidTiff] Levels: {self.n_levels}")
        for info in self.level_info:
            print(f"    Level {info['index']}: {info['width']}x{info['height']} "
                  f"(ds: {info['downsample']:.0f}x, shape: {info['shape']})")

    def get_level_for_display(self, max_pixels: int = 4_000_000) -> int:
        for info in self.level_info:
            pixels = info['width'] * info['height']
            if pixels <= max_pixels:
                return info['index']
        return self.n_levels - 1

    def read_level(self, level: int = 0) -> np.ndarray:
        level = min(level, self.n_levels - 1)
        if self.verbose:
            info = self.level_info[level]
            print(f"  [PyramidTiff] Reading level {level}: {info['width']}x{info['height']}")
        level_src = self._levels_source[level]
        data = level_src.asarray()
        if self.verbose:
            print(f"  [PyramidTiff] Loaded: shape={data.shape}, RAM={data.nbytes/1024/1024:.1f}MB")
        return self._normalize_shape(data)

    def _normalize_shape(self, data: np.ndarray) -> np.ndarray:
        data = np.squeeze(data)
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)
            return data
        while data.ndim > 3:
            data = data[0]
        return self._normalize_shape(data)

    @property
    def base_shape(self) -> Tuple[int, int]:
        return (self.level_info[0]['width'], self.level_info[0]['height'])

    def close(self) -> None:
        self.tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class QuPathHandler:
    """
    Handler for H&E / Immuno stain image pairs and their masks.
    Adapted for BIDS-like structure with participants.csv.
    """

    def __init__(
        self,
        dataset_root: str,
        data_subdir: str = "data",
        save_resolution: int = 3840,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_root: Root directory containing participants.csv and data/
            data_subdir: Subdirectory with the BIDS structure (default: data)
            save_resolution: Width in pixels for saving images
            output_dir: Output directory for saved images
        """
        self.dataset_root = Path(dataset_root)
        self.data_dir = self.dataset_root / data_subdir

        self.save_resolution = save_resolution
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).parent.resolve()
            self.output_dir = script_dir / "preview"

        self.current_name: str | None = None
        self.image_tiff: PyramidTiff | None = None
        self.mask_tiff: PyramidTiff | None = None
        self.hne_mask_tiff: PyramidTiff | None = None
        self.immuno_mask_tiff: PyramidTiff | None = None
        self.current_level: int = 0

        self.image_data: np.ndarray | None = None
        self.mask_data: np.ndarray | None = None
        self.hne_mask_data: np.ndarray | None = None
        self.immuno_mask_data: np.ndarray | None = None

        # Load metadata from participants.csv
        self.participants_data = load_participants_csv(self.dataset_root)
        if self.participants_data:
            print(f"Loaded metadata for {len(self.participants_data)} images from participants.csv")
        else:
            print("participants.csv not found or empty")

        self.mask_cmap = ListedColormap(CLASS_COLORS_HEX)
        self.mask_norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS_HEX) + 0.5, 1), self.mask_cmap.N)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _image_name_to_paths(self, image_name: str) -> Dict[str, Path]:
        """
        Build all four file paths from an IMAGE_NAME.
        Returns dict with keys: hne_img, hne_mask, immuno_img, immuno_mask
        (Paths that don't exist will still be included; caller checks existence.)
        """
        participant_dir, session_dir = image_name_to_subdirs(image_name)

        base_dir = self.data_dir / participant_dir / session_dir
        hne_dir = base_dir / "wsi_h-e"
        immuno_dir = base_dir / "wsi_hmwck-amacr"

        return {
            'hne_img': hne_dir / f"{image_name}_h-e.ome.tif",
            'hne_mask': hne_dir / f"{image_name}_h-e_mask.ome.tif",
            'immuno_img': immuno_dir / f"{image_name}_hmwck-amacr.ome.tif",
            'immuno_mask': immuno_dir / f"{image_name}_hmwck-amacr_mask.ome.tif",
        }

    def list_images(self) -> List[str]:
        """
        List all available images from participants.csv that have all 4 files.
        """
        valid = []
        for image_name in sorted(self.participants_data.keys()):
            paths = self._image_name_to_paths(image_name)
            if all(p.exists() for p in paths.values()):
                valid.append(image_name)
            # else silently skip — some participants may not have all files yet
        return valid

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_pair(self, name: str, level: int | None = None) -> None:
        self.close()
        self.current_name = name
        paths = self._image_name_to_paths(name)

        # H&E image
        hne_path = paths['hne_img']
        if hne_path.exists():
            print(f"Loading H&E: {hne_path.name}")
            self.image_tiff = PyramidTiff(str(hne_path))
        else:
            print(f"H&E image not found: {hne_path}")
            return

        # H&E mask
        hne_mask_path = paths['hne_mask']
        if hne_mask_path.exists():
            print(f"Loading H&E mask: {hne_mask_path.name}")
            self.hne_mask_tiff = PyramidTiff(str(hne_mask_path))
        else:
            print(f"H&E mask not found: {hne_mask_path}")
            self.hne_mask_tiff = None

        # Immuno image
        immuno_path = paths['immuno_img']
        if immuno_path.exists():
            print(f"Loading Immuno: {immuno_path.name}")
            self.mask_tiff = PyramidTiff(str(immuno_path))
        else:
            print(f"Immuno image not found: {immuno_path}")
            self.mask_tiff = None

        # Immuno mask
        immuno_mask_path = paths['immuno_mask']
        if immuno_mask_path.exists():
            print(f"Loading Immuno mask: {immuno_mask_path.name}")
            self.immuno_mask_tiff = PyramidTiff(str(immuno_mask_path))
        else:
            print(f"Immuno mask not found: {immuno_mask_path}")
            self.immuno_mask_tiff = None

        if level is None:
            level = self.image_tiff.get_level_for_display(max_pixels=4_000_000)
        self._load_level(level)

    def _load_level(self, level: int) -> None:
        import gc

        if self.image_tiff is None:
            return

        level = min(level, self.image_tiff.n_levels - 1)
        self.current_level = level

        print(f"\nLoading level {level}...")

        self.image_data = None
        self.mask_data = None
        self.hne_mask_data = None
        self.immuno_mask_data = None
        gc.collect()

        self.image_data = self.image_tiff.read_level(level)
        info = self.image_tiff.level_info[level]
        print(f"  H&E: {info['width']} x {info['height']} (ds: {info['downsample']:.1f}x)")
        print(f"  H&E RAM: {self.image_data.nbytes / 1024 / 1024:.1f} MB")

        if self.mask_tiff is not None:
            mlevel = min(level, self.mask_tiff.n_levels - 1)
            self.mask_data = self.mask_tiff.read_level(mlevel)
            print(f"  Immuno: {self.mask_data.shape}")
            print(f"  Immuno RAM: {self.mask_data.nbytes / 1024 / 1024:.1f} MB")

        if self.hne_mask_tiff is not None:
            mlevel = min(level, self.hne_mask_tiff.n_levels - 1)
            self.hne_mask_data = self.hne_mask_tiff.read_level(mlevel)
            if self.hne_mask_data.ndim == 3:
                self.hne_mask_data = self.hne_mask_data[:, :, 0]
            print(f"  H&E mask: {self.hne_mask_data.shape}")
            print(f"  H&E mask RAM: {self.hne_mask_data.nbytes / 1024 / 1024:.1f} MB")

        if self.immuno_mask_tiff is not None:
            mlevel = min(level, self.immuno_mask_tiff.n_levels - 1)
            self.immuno_mask_data = self.immuno_mask_tiff.read_level(mlevel)
            if self.immuno_mask_data.ndim == 3:
                self.immuno_mask_data = self.immuno_mask_data[:, :, 0]
            print(f"  Immuno mask: {self.immuno_mask_data.shape}")
            print(f"  Immuno mask RAM: {self.immuno_mask_data.nbytes / 1024 / 1024:.1f} MB")

        gc.collect()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        if self.image_tiff is None:
            return {}
        return {
            'name': self.current_name,
            'current_level': self.current_level,
            'image_levels': [
                {'level': i, 'size': f"{info['width']} x {info['height']}", 'downsample': info['downsample']}
                for i, info in enumerate(self.image_tiff.level_info)
            ],
            'mask_levels': [
                {'level': i, 'size': f"{info['width']} x {info['height']}", 'downsample': info['downsample']}
                for i, info in enumerate(self.mask_tiff.level_info)
            ] if self.mask_tiff else [],
            'hne_mask_levels': [
                {'level': i, 'size': f"{info['width']} x {info['height']}", 'downsample': info['downsample']}
                for i, info in enumerate(self.hne_mask_tiff.level_info)
            ] if self.hne_mask_tiff else [],
            'immuno_mask_levels': [
                {'level': i, 'size': f"{info['width']} x {info['height']}", 'downsample': info['downsample']}
                for i, info in enumerate(self.immuno_mask_tiff.level_info)
            ] if self.immuno_mask_tiff else [],
            'base_size': self.image_tiff.base_shape,
        }

    def get_data(self):
        return self.image_data, self.mask_data, self.hne_mask_data, self.immuno_mask_data

    def get_clinical_info(self, name: Optional[str] = None) -> Optional[Dict[str, str]]:
        if name is None:
            name = self.current_name
        if name is None:
            return None

        result = self.participants_data.get(name)
        if result is not None:
            return result

        for key in self.participants_data:
            if name in key or key in name:
                return self.participants_data[key]
        return None

    def format_clinical_title(self, name: Optional[str] = None) -> str:
        if name is None:
            name = self.current_name
        info = self.get_clinical_info(name)
        if info is None:
            return ""

        line1_parts = []
        if info.get('diagnosis'):
            line1_parts.append(f"{info['diagnosis']}")
        if info.get('age'):
            line1_parts.append(f"Age: {info['age']}")
        if info.get('psa'):
            line1_parts.append(f"PSA: {info['psa']}")
        if info.get('mri_findings'):
            line1_parts.append(f"MRI: {info['mri_findings']}")

        line2_parts = []
        if info.get('isup') and info['isup'] != '0':
            line2_parts.append(f"ISUP: {info['isup']}")
        if info.get('gleason') and info['gleason'] != '0':
            line2_parts.append(f"Gleason: {info['gleason']}")
        if info.get('scanner'):
            line2_parts.append(f"Scanner: {info['scanner']}")

        lines = []
        if line1_parts:
            lines.append(" | ".join(line1_parts))
        if line2_parts:
            lines.append(" | ".join(line2_parts))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save / Visualize
    # ------------------------------------------------------------------
    def _save_figure(self, fig: plt.Figure, show_legend: bool = True) -> None:
        if self.current_name is None:
            print("  No image loaded to save.")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.current_name}.png"
        filepath = self.output_dir / filename
        fig_width_inches = fig.get_figwidth()
        target_dpi = self.save_resolution / fig_width_inches
        print(f"  Saving: {filepath}")
        print(f"  Resolution: {self.save_resolution}px (DPI: {target_dpi:.0f})")
        fig.savefig(
            filepath, dpi=target_dpi, bbox_inches='tight', pad_inches=0.2,
            facecolor='white', edgecolor='none', format='png',
        )
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                w, h = img.size
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  Saved: {w}x{h}px ({file_size_mb:.1f} MB)")
        except ImportError:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Saved: {file_size_mb:.1f} MB")

    def visualize(self, show_legend: bool = True, save_only: bool = False) -> None:
        if self.image_data is None:
            print("No data loaded. Use load_pair() first.")
            return

        has_hne_mask = self.hne_mask_data is not None
        has_immuno = self.mask_data is not None
        has_immuno_mask = self.immuno_mask_data is not None

        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        ax_hne = axes[0, 0]
        ax_hne_mask = axes[1, 0]
        ax_immuno = axes[0, 1]
        ax_immuno_mask = axes[1, 1]

        level_info = self.image_tiff.level_info[self.current_level]
        title = (
            f"{self.current_name} | Level {self.current_level} | "
            f"{level_info['width']}x{level_info['height']} (ds: {level_info['downsample']:.0f}x)"
        )
        clinical_str = self.format_clinical_title()
        if clinical_str:
            title = f"{title}\n{clinical_str}"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        ax_hne.imshow(self.image_data)
        ax_hne.set_title("H&E", fontsize=13, fontweight='bold')
        ax_hne.axis('off')

        if has_hne_mask:
            ax_hne_mask.imshow(
                self.hne_mask_data, cmap=self.mask_cmap,
                norm=self.mask_norm, interpolation='nearest',
            )
        ax_hne_mask.set_title("H&E mask", fontsize=13, fontweight='bold')
        ax_hne_mask.axis('off')

        if has_immuno:
            ax_immuno.imshow(self.mask_data)
        ax_immuno.set_title("HMWCK-AMACR", fontsize=13, fontweight='bold')
        ax_immuno.axis('off')

        if has_immuno_mask:
            ax_immuno_mask.imshow(
                self.immuno_mask_data, cmap=self.mask_cmap,
                norm=self.mask_norm, interpolation='nearest',
            )
        ax_immuno_mask.set_title("HMWCK-AMACR mask", fontsize=13, fontweight='bold')
        ax_immuno_mask.axis('off')

        if show_legend and (has_hne_mask or has_immuno_mask):
            mask_data_for_legend = self.hne_mask_data if has_hne_mask else self.immuno_mask_data
            unique_classes = np.unique(mask_data_for_legend)
            legend_patches = [
                Patch(
                    facecolor=CLASS_COLORS_HEX[c], edgecolor='black',
                    label=f"{c}: {CLASS_NAMES.get(c, '?')}",
                )
                for c in sorted(unique_classes) if c in CLASS_NAMES
            ]
            ax_immuno_mask.legend(
                handles=legend_patches, loc='center left',
                bbox_to_anchor=(1.02, 0.5), fontsize=11,
            )

        all_axes = [ax_hne, ax_hne_mask, ax_immuno, ax_immuno_mask]
        self._syncing = False

        def make_sync(src_ax):
            def sync_fn(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    for ax in all_axes:
                        if ax is not src_ax:
                            ax.set_xlim(src_ax.get_xlim())
                            ax.set_ylim(src_ax.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            return sync_fn

        for ax in all_axes:
            ax.callbacks.connect('xlim_changed', make_sync(ax))
            ax.callbacks.connect('ylim_changed', make_sync(ax))

        if not save_only:
            def on_key(event):
                if event.key == 's':
                    self._save_figure(fig, show_legend=show_legend)
            fig.canvas.mpl_connect('key_press_event', on_key)
            print("  [Controls] S: save PNG | Q: close")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        if save_only:
            self._save_figure(fig, show_legend=show_legend)
            plt.close(fig)
        else:
            plt.show()

    def visualize_interactive(self) -> None:
        if self.image_tiff is None:
            print("No data loaded. Use load_pair() first.")
            return

        has_hne_mask = self.hne_mask_data is not None
        has_immuno = self.mask_data is not None
        has_immuno_mask = self.immuno_mask_data is not None

        fig = plt.figure(figsize=(18, 17))
        ax_hne = fig.add_axes([0.05, 0.55, 0.4, 0.38])
        ax_immuno = fig.add_axes([0.55, 0.55, 0.4, 0.38])
        ax_hne_mask = fig.add_axes([0.05, 0.08, 0.4, 0.38])
        ax_immuno_mask = fig.add_axes([0.55, 0.08, 0.4, 0.38])
        ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.025])

        n_levels = self.image_tiff.n_levels
        slider = Slider(ax_slider, 'Level', 0, n_levels - 1, valinit=self.current_level, valstep=1)

        hne_display = ax_hne.imshow(self.image_data)
        ax_hne.set_title("H&E", fontsize=13, fontweight='bold')
        ax_hne.axis('off')

        immuno_display = None
        if has_immuno:
            immuno_display = ax_immuno.imshow(self.mask_data)
        ax_immuno.set_title("HMWCK-AMACR", fontsize=13, fontweight='bold')
        ax_immuno.axis('off')

        hne_mask_display = None
        if has_hne_mask:
            hne_mask_display = ax_hne_mask.imshow(
                self.hne_mask_data, cmap=self.mask_cmap,
                norm=self.mask_norm, interpolation='nearest',
            )
        ax_hne_mask.set_title("H&E mask", fontsize=13, fontweight='bold')
        ax_hne_mask.axis('off')

        immuno_mask_display = None
        if has_immuno_mask:
            immuno_mask_display = ax_immuno_mask.imshow(
                self.immuno_mask_data, cmap=self.mask_cmap,
                norm=self.mask_norm, interpolation='nearest',
            )
        ax_immuno_mask.set_title("HMWCK-AMACR mask", fontsize=13, fontweight='bold')
        ax_immuno_mask.axis('off')

        def update_title():
            info = self.image_tiff.level_info[self.current_level]
            title = (
                f"{self.current_name} | Level {self.current_level} | "
                f"{info['width']}x{info['height']} (ds: {info['downsample']:.0f}x)"
            )
            clinical_str = self.format_clinical_title()
            if clinical_str:
                title = f"{title}\n{clinical_str}"
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)

        update_title()

        def on_slider_change(val):
            level = int(val)
            if level != self.current_level:
                self._load_level(level)
                hne_display.set_data(self.image_data)
                hne_display.set_extent([0, self.image_data.shape[1], self.image_data.shape[0], 0])
                ax_hne.set_xlim(0, self.image_data.shape[1])
                ax_hne.set_ylim(self.image_data.shape[0], 0)

                if immuno_display is not None and self.mask_data is not None:
                    immuno_display.set_data(self.mask_data)
                    immuno_display.set_extent([0, self.mask_data.shape[1], self.mask_data.shape[0], 0])
                    ax_immuno.set_xlim(0, self.mask_data.shape[1])
                    ax_immuno.set_ylim(self.mask_data.shape[0], 0)

                if hne_mask_display is not None and self.hne_mask_data is not None:
                    hne_mask_display.set_data(self.hne_mask_data)
                    hne_mask_display.set_extent([0, self.hne_mask_data.shape[1], self.hne_mask_data.shape[0], 0])
                    ax_hne_mask.set_xlim(0, self.hne_mask_data.shape[1])
                    ax_hne_mask.set_ylim(self.hne_mask_data.shape[0], 0)

                if immuno_mask_display is not None and self.immuno_mask_data is not None:
                    immuno_mask_display.set_data(self.immuno_mask_data)
                    immuno_mask_display.set_extent([0, self.immuno_mask_data.shape[1], self.immuno_mask_data.shape[0], 0])
                    ax_immuno_mask.set_xlim(0, self.immuno_mask_data.shape[1])
                    ax_immuno_mask.set_ylim(self.immuno_mask_data.shape[0], 0)

                update_title()
                fig.canvas.draw_idle()

        slider.on_changed(on_slider_change)

        all_axes = [ax_hne, ax_hne_mask, ax_immuno, ax_immuno_mask]
        self._syncing = False

        def make_sync(src_ax):
            def sync_fn(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    for ax in all_axes:
                        if ax is not src_ax:
                            ax.set_xlim(src_ax.get_xlim())
                            ax.set_ylim(src_ax.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            return sync_fn

        for ax in all_axes:
            ax.callbacks.connect('xlim_changed', make_sync(ax))
            ax.callbacks.connect('ylim_changed', make_sync(ax))

        def on_key(event):
            if event.key == 's':
                self._save_figure(fig, show_legend=True)

        fig.canvas.mpl_connect('key_press_event', on_key)
        print("  [Controls] S: save PNG | Q: close | Slider: change level")
        plt.show()

    def change_level(self, level: int) -> None:
        if self.image_tiff is None:
            print("No data loaded.")
            return
        self._load_level(level)

    def close(self) -> None:
        if self.image_tiff is not None:
            self.image_tiff.close()
            self.image_tiff = None
        if self.mask_tiff is not None:
            self.mask_tiff.close()
            self.mask_tiff = None
        if self.hne_mask_tiff is not None:
            self.hne_mask_tiff.close()
            self.hne_mask_tiff = None
        if self.immuno_mask_tiff is not None:
            self.immuno_mask_tiff.close()
            self.immuno_mask_tiff = None
        self.image_data = None
        self.mask_data = None
        self.hne_mask_data = None
        self.immuno_mask_data = None
        self.current_name = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="H&E / Immuno stain pair visualizer (v2 — BIDS + participants.csv)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qupath_handler_hne_inmuno_v2.py "/media/abel/TOSHIBA EXT/prostate_HnE-IHC_dataset"

  # Specify resolution level
  python qupath_handler_hne_inmuno_v2.py /path/to/dataset --level 2

  # Batch save all images
  python qupath_handler_hne_inmuno_v2.py /path/to/dataset --batch-save

Expected directory structure:
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
        """,
    )

    parser.add_argument("dataset_root", help="Dataset root (contains participants.csv and data/)")
    parser.add_argument("--level", "-l", type=int, default=None, help="Pyramid level (default: auto)")
    parser.add_argument("--save-resolution", "-r", type=int, default=3840,
                        help="Width in pixels for saving images (default: 3840 = 4K)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for screenshots (default: script_dir/preview)")
    parser.add_argument("--batch-save", "--save-all", dest="batch_save", action="store_true",
                        help="Iterate all images and save PNG without opening windows")
    parser.add_argument("--data-subdir", default="data",
                        help="Subdirectory with BIDS structure (default: data)")

    args = parser.parse_args()

    handler = QuPathHandler(
        args.dataset_root,
        data_subdir=args.data_subdir,
        save_resolution=args.save_resolution,
        output_dir=args.output_dir,
    )

    images = handler.list_images()
    if not images:
        print("No images found. Check dataset_root and participants.csv.")
        return

    print(f"\nFound {len(images)} images")
    if args.batch_save:
        print("Batch mode: saving PNG without opening windows.\n")
    else:
        print("Close each window to proceed to the next.\n")

    for i, name in enumerate(images):
        print(f"[{i+1}/{len(images)}] {name}")
        try:
            handler.load_pair(name, level=args.level)

            meta = handler.get_metadata()
            print(f"  Base size: {meta.get('base_size')}")
            print(f"  Loaded level: {meta.get('current_level')}")

            clinical = handler.get_clinical_info(name)
            if clinical:
                print(f"  Diagnosis: {clinical.get('diagnosis', 'N/A')}")
                age = clinical.get('age', '')
                psa = clinical.get('psa', '')
                if age or psa:
                    info_parts = []
                    if age:
                        info_parts.append(f"Age: {age}")
                    if psa:
                        info_parts.append(f"PSA: {psa}")
                    print(f"  {' | '.join(info_parts)}")
                mri = clinical.get('mri_findings', '')
                if mri:
                    print(f"  MRI: {mri}")
                isup = clinical.get('isup', '')
                gleason = clinical.get('gleason', '')
                if isup and isup != '0':
                    print(f"  ISUP Grade: {isup} | Gleason: {gleason}")
                print(f"  Scanner: {clinical.get('scanner', 'N/A')}")

            if args.batch_save:
                handler.visualize(save_only=True)
            else:
                handler.visualize()

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            handler.close()

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
