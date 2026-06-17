#!/usr/bin/env python3
"""
Add Stroma to Masks

Script to detect tissue in histological images via flexible thresholding
and label unannotated tissue regions as Stroma (class 28).

Generates pyramidal TIFFs consistent with the original structure.

Usage:
    python add_stroma.py /path/to/dataset
    python add_stroma.py /path/to/dataset --threshold 235 --dilate 10 --erode 5
    python add_stroma.py /path/to/dataset --preview
"""

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter, binary_dilation, binary_erosion, binary_fill_holes


STROMA_CLASS_ID = 7
BACKGROUND_CLASS_ID = 0
CLASS_NAMES = {
    1: "Tumor",
    2: "Bening gland",
    3: "Artifact",
    4: "High grade prostatic intraepithelial neoplasia (HGPIN)",
    5: "Intraductal carcinoma",
    6: "Atypical intraductal proliferation",
    7: "Stroma"

}

CLASS_COLORS_HEX = [
    "#000000",  # 0: Background (black)
    "#B83B5E",  # 1: Tumor
    "#F38181",  # 2: Bening gland
    "#AA96DA",  # 3: Artifact
    "#FCBAD3",  # 4: High grade prostatic intraepithelial neoplasia (HGPIN)
    "#FF6B6B",  # 5: Intraductal carcinoma
    "#9B59B6",  # 6: Atypical intraductal proliferation
    "#FAE3D9",  # 7: Stroma
]

class TissueDetector:
    """
    Detects tissue regions in tiles of histological images.
    
    Uses thresholding with optional morphological operations.
    """
    
    def __init__(
        self,
        threshold: int = 240,
        blur: int = 0,
        dilate: int = 0,
        erode: int = 0,
        min_area: int = 0,
        erode_exterior_only: bool = False
    ):
        """
        Args:
            threshold: Whiteness threshold (0-255, lower = more sensitive)
            blur: Gaussian blur before threshold (0 = disabled)
            dilate: Dilation of tissue mask (0 = disabled)
            erode: Erosion of tissue mask (0 = disabled)
            min_area: Minimum region area to consider (0 = no filtering)
            erode_exterior_only: Only erode exterior boundaries, preserve interior holes
        """
        self.threshold = threshold
        self.blur = blur
        self.dilate = dilate
        self.erode = erode
        self.min_area = min_area
        self.erode_exterior_only = erode_exterior_only
    
    def detect(self, rgb_tile: np.ndarray) -> np.ndarray:
        """
        Detects tissue in an RGB tile.
        
        Args:
            rgb_tile: Array (H, W, 3) with values 0-255
            
        Returns:
            Boolean mask (H, W) where True = detected tissue
        """
        if rgb_tile.ndim == 2:
            gray = rgb_tile.astype(np.float32)
        else:
            gray = np.mean(rgb_tile.astype(np.float32), axis=2)
        
        # Apply blur if configured (uniform_filter is faster than gaussian)
        if self.blur > 0:
            gray = uniform_filter(gray, size=self.blur * 2 + 1)
        
        # Threshold: tissue = non-white
        tissue_mask = gray < self.threshold
        
        # Apply dilation if configured
        if self.dilate > 0:
            tissue_mask = binary_dilation(tissue_mask, iterations=self.dilate)
        
        # Apply erosion if configured
        if self.erode > 0:
            if self.erode_exterior_only:
                filled = binary_fill_holes(tissue_mask)
                interior_holes = filled & ~tissue_mask
                eroded_exterior = binary_erosion(filled, iterations=self.erode)
                tissue_mask = eroded_exterior | interior_holes
            else:
                tissue_mask = binary_erosion(tissue_mask, iterations=self.erode)
        
        # Remove small regions if configured
        if self.min_area > 0:
            tissue_mask = self._remove_small_regions(tissue_mask, self.min_area)
        
        return tissue_mask
    
    def _remove_small_regions(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Removes small regions from the mask."""
        from scipy.ndimage import label
        
        labeled, num_features = label(mask)
        if num_features == 0:
            return mask
        
        component_sizes = np.bincount(labeled.ravel())
        too_small = component_sizes < min_size
        too_small[0] = False
        
        mask_cleaned = mask.copy()
        mask_cleaned[too_small[labeled]] = False
        
        return mask_cleaned


class PyramidReader:
    """Reads pyramidal TIFFs by tiles."""
    
    def __init__(self, path: str):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self._detect_structure()
    
    def _detect_structure(self) -> None:
        """Detects the pyramid structure."""
        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._levels = first_series.levels
                self.n_levels = len(self._levels)
                return
        
        if len(self.tif.series) > 1:
            self._levels = self.tif.series
            self.n_levels = len(self._levels)
            return
        
        self._levels = [self.tif.series[0]] if self.tif.series else [self.tif.pages[0]]
        self.n_levels = 1
    
    def get_level_shape(self, level: int = 0) -> Tuple[int, int]:
        """Returns (height, width) of the specified level."""
        shape = self._levels[level].shape
        if len(shape) == 2:
            return shape[0], shape[1]
        elif len(shape) == 3:
            if shape[0] <= 4:
                return shape[1], shape[2]
            return shape[0], shape[1]
        return shape[-2], shape[-1]
    
    def get_downsamples(self) -> List[float]:
        """Computes the downsample factor for each level."""
        base_h, base_w = self.get_level_shape(0)
        downsamples = []
        for i in range(self.n_levels):
            h, w = self.get_level_shape(i)
            ds = base_w / w if w > 0 else 1.0
            downsamples.append(ds)
        return downsamples
    
    def read_level(self, level: int = 0) -> np.ndarray:
        """Reads a full level."""
        data = self._levels[level].asarray()
        return self._normalize(data)
    
    def read_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """
        Reads a region from a level.
        
        Tries multiple methods in order of efficiency:
        1. Zarr (if available)
        2. Direct tile reading (for tiled TIFFs)
        3. Fallback with cache (for small levels)
        """
        level_h, level_w = self.get_level_shape(level)
        
        x = max(0, min(x, level_w))
        y = max(0, min(y, level_h))
        width = min(width, level_w - x)
        height = min(height, level_h - y)
        
        if width <= 0 or height <= 0:
            return np.zeros((height, width), dtype=np.uint8)
        
        # Method 1: Try zarr
        try:
            store = self._levels[level].aszarr()
            import zarr
            z = zarr.open(store, mode='r')
            
            if z.ndim == 2:
                data = z[y:y+height, x:x+width]
            elif z.ndim == 3:
                if z.shape[0] <= 4:
                    data = z[:, y:y+height, x:x+width]
                    data = np.moveaxis(data, 0, -1)
                else:
                    data = z[y:y+height, x:x+width, :]
            else:
                data = z[..., y:y+height, x:x+width]
                data = np.squeeze(data)
            
            return np.asarray(data)
        except Exception:
            pass
        
        # Method 2: Direct tile reading (memory efficient)
        try:
            data = self._read_tiles_direct(level, x, y, width, height)
            if data is not None:
                return data
        except Exception:
            pass
        
        # Method 3: Fallback with cache (for small levels only)
        return self._read_region_with_cache(level, x, y, width, height)
    
    def _get_page(self, level: int):
        """Gets the TiffPage for a level."""
        level_src = self._levels[level]
        # For series, get the first page
        if hasattr(level_src, 'pages') and len(level_src.pages) > 0:
            return level_src.pages[0]
        # For levels that are pages directly
        if hasattr(level_src, 'keyframe'):
            return level_src.keyframe
        # If it is a page directly
        if hasattr(level_src, 'is_tiled'):
            return level_src
        return None
    
    def _read_tiles_direct(self, level: int, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Reads a region by loading only the necessary tiles directly.
        
        Memory efficient: only loads tiles that intersect with the region.
        """
        page = self._get_page(level)
        if page is None or not page.is_tiled:
            return None
        
        tile_w = page.tilewidth
        tile_h = page.tilelength
        img_w = page.imagewidth
        img_h = page.imagelength
        
        # Determine output shape
        shape = page.shape
        if len(shape) == 2:
            n_channels = 1
            output = np.zeros((height, width), dtype=page.dtype)
        elif len(shape) == 3:
            if shape[0] <= 4:  # (C, H, W)
                n_channels = shape[0]
                output = np.zeros((height, width, n_channels), dtype=page.dtype)
            else:  # (H, W, C)
                n_channels = shape[2]
                output = np.zeros((height, width, n_channels), dtype=page.dtype)
        else:
            return None
        
        # Calculate which tiles we need
        tile_col_start = x // tile_w
        tile_row_start = y // tile_h
        tile_col_end = (x + width - 1) // tile_w + 1
        tile_row_end = (y + height - 1) // tile_h + 1
        
        tiles_per_row = (img_w + tile_w - 1) // tile_w
        
        # Read each required tile
        fh = page.parent.filehandle
        
        for tile_row in range(tile_row_start, tile_row_end):
            for tile_col in range(tile_col_start, tile_col_end):
                tile_idx = tile_row * tiles_per_row + tile_col
                
                if tile_idx >= len(page.dataoffsets):
                    continue
                
                offset = page.dataoffsets[tile_idx]
                bytecount = page.databytecounts[tile_idx]
                
                if bytecount == 0:
                    continue
                
                # Read and decode the tile
                fh.seek(offset)
                tile_data = fh.read(bytecount)
                
                try:
                    tile = page.decode(tile_data, tile_idx)[0]
                except Exception:
                    continue
                
                # Normalize tile to (H, W) or (H, W, C)
                tile = np.squeeze(tile)
                if tile.ndim == 3 and tile.shape[0] <= 4 and tile.shape[0] < tile.shape[1]:
                    tile = np.moveaxis(tile, 0, -1)
                
                # Tile coordinates in the full image
                tile_x = tile_col * tile_w
                tile_y = tile_row * tile_h
                
                # Calculate intersection with requested region
                src_x1 = max(0, x - tile_x)
                src_y1 = max(0, y - tile_y)
                src_x2 = min(tile_w, x + width - tile_x)
                src_y2 = min(tile_h, y + height - tile_y)
                
                dst_x1 = max(0, tile_x - x)
                dst_y1 = max(0, tile_y - y)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                # Copy data
                if tile.ndim == 2 and output.ndim == 2:
                    output[dst_y1:dst_y2, dst_x1:dst_x2] = tile[src_y1:src_y2, src_x1:src_x2]
                elif tile.ndim == 2 and output.ndim == 3:
                    output[dst_y1:dst_y2, dst_x1:dst_x2, 0] = tile[src_y1:src_y2, src_x1:src_x2]
                else:
                    output[dst_y1:dst_y2, dst_x1:dst_x2] = tile[src_y1:src_y2, src_x1:src_x2]
        
        return output
    
    def _read_region_with_cache(self, level: int, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Fallback: reads region using full-level cache.
        For small levels only when other methods fail.
        """
        level_h, level_w = self.get_level_shape(level)
        level_pixels = level_w * level_h
        
        # Strict limit for fallback
        max_pixels = 100_000_000  # 100 MP
        
        if level_pixels > max_pixels:
            raise RuntimeError(
                f"Level {level} too large ({level_pixels:,} px). "
                f"Could not read by tiles or zarr. "
                f"Verify that the TIFF is properly tiled."
            )
        
        cache_key = f"_level_cache_{level}"
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self.read_level(level))
        
        cached_data = getattr(self, cache_key)
        return cached_data[y:y+height, x:x+width]
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalizes data to (H, W) or (H, W, C)."""
        data = np.squeeze(data)
        if data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
            data = np.moveaxis(data, 0, -1)
        return data
    
    def clear_cache(self) -> None:
        """Clears the cache of loaded levels."""
        for attr in list(vars(self).keys()):
            if attr.startswith('_level_cache_'):
                delattr(self, attr)
        gc.collect()
    
    def close(self) -> None:
        self.clear_cache()
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class StromaAdder:
    """
    Adds Stroma class to masks in unannotated tissue regions.
    
    Processes large images by tiles with overlap to avoid edge artifacts
    and generates coherent pyramidal TIFFs.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str | None = None,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        tile_size: int = 2048,
        overlap: int = 100,
        detector_params: Dict[str, Any] | None = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / images_subdir
        self.masks_dir = self.dataset_dir / masks_subdir
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.dataset_dir / "masks_with_stroma"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_size = tile_size
        self.overlap = overlap
        
        params = detector_params or {}
        self.detector = TissueDetector(**params)
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Finds image-mask pairs."""
        pairs = []
        
        for img_path in sorted(self.images_dir.glob("*.ome.tif")):
            base_name = img_path.stem
            if base_name.endswith(".ome"):
                base_name = base_name[:-4]
            
            mask_patterns = [
                f"{base_name}__mask_multiclass.ome.tif",
                f"{base_name}_mask.ome.tif",
                f"{base_name}__mask.ome.tif",
            ]
            
            mask_path = None
            for pattern in mask_patterns:
                candidate = self.masks_dir / pattern
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                pairs.append((img_path, mask_path))
            else:
                print(f"  Warning: No mask found for {img_path.name}")
        
        return pairs
    
    def process_all(self, specific_name: str | None = None, skip_first: int = 0) -> None:
        """
        Processes all image-mask pairs.
        
        Args:
            specific_name: If specified, only processes images containing this name
            skip_first: Number of images to skip at the start (default: 0)
        """
        pairs = self.find_pairs()
        
        if specific_name:
            pairs = [(i, m) for i, m in pairs if specific_name in i.stem]
        
        total_pairs = len(pairs)
        
        # Skip the first N images if specified
        if skip_first > 0:
            if skip_first >= total_pairs:
                print(f"Error: --skip {skip_first} is greater than or equal to total images ({total_pairs})")
                return
            pairs = pairs[skip_first:]
            print(f"Found {total_pairs} pairs, skipping the first {skip_first}")
            print(f"Processing from image {skip_first + 1} to {total_pairs}")
        else:
            print(f"Found {total_pairs} image-mask pairs")
        
        print(f"Output: {self.output_dir}")
        print(f"Tissue detection parameters:")
        print(f"  - threshold: {self.detector.threshold}")
        print(f"  - blur: {self.detector.blur} px")
        print(f"  - dilate: {self.detector.dilate} px")
        print(f"  - erode: {self.detector.erode} px")
        print(f"  - erode_exterior_only: {self.detector.erode_exterior_only}")
        print(f"  - min_area: {self.detector.min_area} px")
        print(f"Processing parameters:")
        print(f"  - tile_size: {self.tile_size} px")
        print(f"  - overlap: {self.overlap} px")
        print()
        
        for i, (img_path, mask_path) in enumerate(pairs):
            # Show actual index (considering skipped ones)
            real_index = i + skip_first + 1
            print(f"[{real_index}/{total_pairs}] Processing: {img_path.name}")
            try:
                self.process_pair(img_path, mask_path)
                gc.collect()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    def process_pair(self, image_path: Path, mask_path: Path) -> None:
        """
        Processes an image-mask pair.
        
        Optimized for low RAM usage:
        - Uses memmap for level 0 (stored on temporary disk)
        - Generates and writes each pyramid level one by one
        """
        timings = {}
        total_start = time.time()
        
        # Create temporary file for level 0 memmap
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp:
            memmap_path = tmp.name
        
        try:
            with PyramidReader(str(image_path)) as img_reader, \
                 PyramidReader(str(mask_path)) as mask_reader:
                
                img_h, img_w = img_reader.get_level_shape(0)
                mask_h, mask_w = mask_reader.get_level_shape(0)
                
                print(f"  Image: {img_w} x {img_h}")
                print(f"  Mask: {mask_w} x {mask_h}")
                
                downsamples = mask_reader.get_downsamples()
                print(f"  Levels: {len(downsamples)} ({', '.join(f'{d:.0f}x' for d in downsamples)})")
                
                # Process level 0 using memmap (on disk, not RAM)
                print(f"  Processing level 0 by tiles ({self.tile_size}x{self.tile_size}, overlap={self.overlap})...")
                t0 = time.time()
                level0_mask, tile_timings = self._process_level0(img_reader, mask_reader, memmap_path)
                timings['tiles_total'] = time.time() - t0
                timings.update(tile_timings)
                
                output_name = mask_path.name
                output_path = self.output_dir / output_name
                
                # Generate pyramid and write TIFF level by level (without accumulating in RAM)
                print(f"  Generating pyramid and writing {output_path.name}...")
                t0 = time.time()
                self._write_pyramid_tiff_streaming(output_path, level0_mask, downsamples)
                timings['pyramid_and_write'] = time.time() - t0
                
                # Close and release the memmap
                del level0_mask
                gc.collect()
                
                file_size = output_path.stat().st_size
                timings['total'] = time.time() - total_start
                
                # Show timing summary
                print(f"  Completed: {file_size / 1024 / 1024:.1f} MB")
                print(f"  ─── Timings ───")
                print(f"    Tile reading:        {timings.get('read', 0):6.1f}s ({100*timings.get('read', 0)/timings['total']:5.1f}%)")
                print(f"    Tissue detection:    {timings.get('detect', 0):6.1f}s ({100*timings.get('detect', 0)/timings['total']:5.1f}%)")
                print(f"    Resize:              {timings.get('resize', 0):6.1f}s ({100*timings.get('resize', 0)/timings['total']:5.1f}%)")
                print(f"    Pyramid + write:    {timings.get('pyramid_and_write', 0):6.1f}s ({100*timings.get('pyramid_and_write', 0)/timings['total']:5.1f}%)")
                print(f"    ─────────────────────────────")
                print(f"    TOTAL:               {timings['total']:6.1f}s")
        
        finally:
            # Clean up temporary memmap file
            try:
                Path(memmap_path).unlink()
            except Exception:
                pass
    
    def _process_level0(
        self,
        img_reader: PyramidReader,
        mask_reader: PyramidReader,
        memmap_path: str
    ) -> Tuple[np.memmap, Dict[str, float]]:
        """
        Processes level 0 by tiles with overlap and returns the modified mask and timings.
        
        Uses memmap to store the mask on disk instead of RAM.
        """
        mask_h, mask_w = mask_reader.get_level_shape(0)
        img_h, img_w = img_reader.get_level_shape(0)
        
        scale_x = img_w / mask_w
        scale_y = img_h / mask_h
        
        # Use memmap to avoid saturating RAM with full level 0
        output_mask = np.memmap(
            memmap_path, 
            dtype=np.uint8, 
            mode='w+', 
            shape=(mask_h, mask_w)
        )
        ram_saved_gb = (mask_h * mask_w) / (1024**3)
        print(f"    Using memmap on disk ({ram_saved_gb:.1f} GB saved in RAM)")
        
        stroma_pixels_added = 0
        
        # Accumulated timings
        time_read = 0.0
        time_detect = 0.0
        time_resize = 0.0
        
        # Calculate number of tiles
        step = self.tile_size
        n_tiles_y = (mask_h + step - 1) // step
        n_tiles_x = (mask_w + step - 1) // step
        total_tiles = n_tiles_y * n_tiles_x
        tile_count = 0
        
        for tile_y in range(n_tiles_y):
            for tile_x in range(n_tiles_x):
                tile_count += 1
                
                # Output tile coordinates (without overlap)
                out_x = tile_x * step
                out_y = tile_y * step
                out_w = min(step, mask_w - out_x)
                out_h = min(step, mask_h - out_y)
                
                # Input tile coordinates (with overlap)
                in_x = max(0, out_x - self.overlap)
                in_y = max(0, out_y - self.overlap)
                in_x_end = min(mask_w, out_x + out_w + self.overlap)
                in_y_end = min(mask_h, out_y + out_h + self.overlap)
                in_w = in_x_end - in_x
                in_h = in_y_end - in_y
                
                # Offset within processed tile where output region is
                offset_x = out_x - in_x
                offset_y = out_y - in_y
                
                # Read mask tile with overlap
                t0 = time.time()
                mask_tile = mask_reader.read_region(in_x, in_y, in_w, in_h, level=0)
                if mask_tile.ndim == 3:
                    mask_tile = mask_tile[:, :, 0]
                
                # Read image tile with overlap (scaled)
                img_in_x = int(in_x * scale_x)
                img_in_y = int(in_y * scale_y)
                img_in_w = int(in_w * scale_x)
                img_in_h = int(in_h * scale_y)
                
                img_tile = img_reader.read_region(img_in_x, img_in_y, img_in_w, img_in_h, level=0)
                time_read += time.time() - t0
                
                # Resize image if necessary
                t0 = time.time()
                if img_tile.shape[:2] != mask_tile.shape[:2]:
                    from scipy.ndimage import zoom
                    if img_tile.ndim == 3:
                        factors = (in_h / img_tile.shape[0], in_w / img_tile.shape[1], 1)
                    else:
                        factors = (in_h / img_tile.shape[0], in_w / img_tile.shape[1])
                    img_tile = zoom(img_tile, factors, order=1)
                time_resize += time.time() - t0
                
                # Detect tissue (with blur, dilate, erode applied internally)
                t0 = time.time()
                tissue_mask = self.detector.detect(img_tile)
                
                # Stroma candidates: tissue where there is no label
                stroma_candidates = tissue_mask & (mask_tile == BACKGROUND_CLASS_ID)
                
                # Apply stroma
                mask_tile[stroma_candidates] = STROMA_CLASS_ID
                time_detect += time.time() - t0
                
                # Extract only the output region (without overlap)
                out_tile = mask_tile[offset_y:offset_y+out_h, offset_x:offset_x+out_w]
                stroma_in_out = stroma_candidates[offset_y:offset_y+out_h, offset_x:offset_x+out_w]
                stroma_pixels_added += np.sum(stroma_in_out)
                
                # Save to output mask
                output_mask[out_y:out_y+out_h, out_x:out_x+out_w] = out_tile
                
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"    Tiles: {tile_count}/{total_tiles} "
                          f"({100*tile_count/total_tiles:.0f}%)", end='\r')
        
        print(f"    Tiles: {total_tiles}/{total_tiles} (100%) - "
              f"Stroma pixels added: {stroma_pixels_added:,}")
        
        timings = {
            'read': time_read,
            'detect': time_detect,
            'resize': time_resize,
        }
        
        return output_mask, timings
    
    def _write_pyramid_tiff_streaming(
        self,
        output_path: Path,
        level0: np.ndarray,
        downsamples: List[float],
        tile_size: int = 512
    ) -> None:
        """
        Generates and writes the pyramid level by level without accumulating in memory.
        
        Each level is generated, written and released before moving to the next,
        minimizing RAM usage.
        """
        from scipy.ndimage import zoom
        
        h, w = level0.shape
        n_levels = len(downsamples)
        
        with tifffile.TiffWriter(str(output_path), ome=True, bigtiff=True) as tif:
            options = {
                'tile': (tile_size, tile_size),
                'compression': 'lzw',
                'photometric': 'minisblack',
            }
            
            # Write level 0 (already in memmap, does not add RAM)
            print(f"    Writing level 0: {w} x {h}")
            tif.write(
                level0,
                subifds=n_levels - 1,
                **options
            )
            
            # Generate and write each subsequent level one by one
            for i, ds in enumerate(downsamples[1:], start=1):
                new_h = int(h / ds)
                new_w = int(w / ds)
                
                print(f"    Generating level {i} (ds={ds:.0f}x): {new_w} x {new_h}...", end=' ')
                
                factor = 1.0 / ds
                level_n = zoom(level0, factor, order=0, mode='nearest')
                
                if level_n.shape != (new_h, new_w):
                    level_n = level_n[:new_h, :new_w]
                
                level_n = level_n.astype(np.uint8)
                
                # Write immediately
                tif.write(
                    level_n,
                    subfiletype=1,
                    **options
                )
                
                # Release level memory before generating the next
                level_size_mb = level_n.nbytes / (1024 * 1024)
                del level_n
                gc.collect()
                
                print(f"written and released ({level_size_mb:.1f} MB)")


def preview_thresholding(
    image_path: str,
    mask_path: str,
    detector: TissueDetector,
    level: int = 2
) -> None:
    """Shows a preview processing the full image at L0 (exact same pipeline as real)."""
    import matplotlib.pyplot as plt
    import tempfile
    
    with PyramidReader(image_path) as img_reader, \
         PyramidReader(mask_path) as mask_reader:
        
        mask_h, mask_w = mask_reader.get_level_shape(0)
        img_h, img_w = img_reader.get_level_shape(0)
        scale_x = img_w / mask_w
        scale_y = img_h / mask_h
        
        downsamples = img_reader.get_downsamples()
        preview_level = min(level, img_reader.n_levels - 1, mask_reader.n_levels - 1)
        ds = downsamples[preview_level]
        
        print(f"  Image L0: {img_w} x {img_h}")
        print(f"  Mask L0:  {mask_w} x {mask_h}")
        print(f"  Scale:    {scale_x:.2f}x, {scale_y:.2f}x")
        print()
        print(f"  Parameters (L0, sin escalado):")
        print(f"    threshold: {detector.threshold}")
        print(f"    blur:      {detector.blur} px")
        print(f"    dilate:    {detector.dilate} px")
        print(f"    erode:     {detector.erode} px")
        print(f"    erode_exterior_only: {detector.erode_exterior_only}")
        print(f"    min_area:  {detector.min_area} px²")
        print()
        
        # Process full mask at L0 using exact same pipeline as _process_level0
        tile_size = 2048
        overlap = 100
        step = tile_size
        
        n_tiles_y = (mask_h + step - 1) // step
        n_tiles_x = (mask_w + step - 1) // step
        total_tiles = n_tiles_y * n_tiles_x
        
        print(f"  Processing full image at L0 by tiles ({tile_size}x{tile_size}, "
              f"{total_tiles} tiles)...")
        
        # Temp memmap for the result
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp:
            memmap_path = tmp.name
        
        try:
            output_mask = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=(mask_h, mask_w))
            tile_count = 0
            total_stroma = 0
            
            total_stroma = 0
            
            for tile_y in range(n_tiles_y):
                for tile_x in range(n_tiles_x):
                    tile_count += 1
                    
                    out_x = tile_x * step
                    out_y = tile_y * step
                    out_w = min(step, mask_w - out_x)
                    out_h = min(step, mask_h - out_y)
                    
                    in_x = max(0, out_x - overlap)
                    in_y = max(0, out_y - overlap)
                    in_w = min(mask_w, out_x + out_w + overlap) - in_x
                    in_h = min(mask_h, out_y + out_h + overlap) - in_y
                    offset_x = out_x - in_x
                    offset_y = out_y - in_y
                    
                    # Read mask tile at L0
                    mask_tile = mask_reader.read_region(in_x, in_y, in_w, in_h, level=0)
                    if mask_tile.ndim == 3:
                        mask_tile = mask_tile[:, :, 0]
                    
                    # Read image tile at L0 (scaled)
                    img_in_x = int(in_x * scale_x)
                    img_in_y = int(in_y * scale_y)
                    img_in_w = int(in_w * scale_x)
                    img_in_h = int(in_h * scale_y)
                    img_tile = img_reader.read_region(img_in_x, img_in_y, img_in_w, img_in_h, level=0)
                    
                    # Resize image to match mask tile dimensions (zoom bilinear)
                    if img_tile.shape[:2] != mask_tile.shape[:2]:
                        from scipy.ndimage import zoom
                        if img_tile.ndim == 3:
                            factors = (in_h / img_tile.shape[0], in_w / img_tile.shape[1], 1)
                        else:
                            factors = (in_h / img_tile.shape[0], in_w / img_tile.shape[1])
                        img_tile = zoom(img_tile, factors, order=1)
                    
                    # Detect tissue
                    tissue_mask = detector.detect(img_tile)
                    
                    # Stroma candidates: unlabeled tissue
                    stroma_candidates = tissue_mask & (mask_tile == BACKGROUND_CLASS_ID)
                    
                    # Apply stroma
                    mask_tile[stroma_candidates] = STROMA_CLASS_ID
                    
                    # Extract output region (without overlap)
                    out_tile = mask_tile[offset_y:offset_y + out_h, offset_x:offset_x + out_w]
                    output_mask[out_y:out_y + out_h, out_x:out_x + out_w] = out_tile
                    
                    stroma_in_out = stroma_candidates[offset_y:offset_y + out_h, offset_x:offset_x + out_w]
                    total_stroma += np.sum(stroma_in_out)
                    
                    if tile_count % 10 == 0 or tile_count == total_tiles:
                        print(f"    Tiles: {tile_count}/{total_tiles} "
                              f"({100 * tile_count / total_tiles:.0f}%)", end='\r')
            
            print(f"    Tiles: {total_tiles}/{total_tiles} (100%)")
            print(f"    Stroma pixels added: {total_stroma:,}")
            print()
            
            # Flush memmap changes
            output_mask.flush()
            
            # Downsample result for visualization at preview level
            from scipy.ndimage import zoom
            vis_h = mask_h / ds
            vis_w = mask_w / ds
            
            # Read image at preview level for context
            img_viz = img_reader.read_level(preview_level)
            if img_viz.ndim == 3:
                img_viz = img_viz[:, :, :3]  # ensure no alpha
            
            # Read original mask at preview level
            orig_mask_viz = mask_reader.read_level(preview_level)
            if orig_mask_viz.ndim == 3:
                orig_mask_viz = orig_mask_viz[:, :, 0]
            
            # Downsample result to match preview level dimensions
            result_down = zoom(output_mask.astype(np.float32), 1.0 / ds, order=0).astype(np.uint8)
            
            # Ensure same dimensions
            min_h = min(result_down.shape[0], orig_mask_viz.shape[0])
            min_w = min(result_down.shape[1], orig_mask_viz.shape[1])
            result_viz = result_down[:min_h, :min_w]
            orig_viz = orig_mask_viz[:min_h, :min_w]
            img_viz = img_viz[:min_h, :min_w]
            
            # Compute stroma regions for display
            stroma_viz = (result_viz == STROMA_CLASS_ID) & (orig_viz != STROMA_CLASS_ID)
            tissue_viz = (result_viz != BACKGROUND_CLASS_ID) | (orig_viz != BACKGROUND_CLASS_ID)
            
            # RGB overlay
            zones_rgb = np.zeros((*tissue_viz.shape, 3), dtype=np.uint8)
            zones_rgb[tissue_viz] = [128, 128, 128]
            zones_rgb[stroma_viz] = [100, 255, 100]
            
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            
            from matplotlib.colors import ListedColormap, BoundaryNorm
            mask_cmap = ListedColormap(CLASS_COLORS_HEX)
            mask_norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS_HEX) + 0.5, 1), mask_cmap.N)
            
            axes[0, 0].imshow(img_viz)
            axes[0, 0].set_title(f"Original image (level {preview_level})")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(zones_rgb, interpolation='nearest')
            axes[0, 1].set_title(f"Tissue detection (L0 pipeline, shown at {ds:.0f}x)\n"
                                f"Gray=tissue, Green=stroma added\n"
                                f"blur={detector.blur}px dilate={detector.dilate}px "
                                f"erode={detector.erode}px")
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(orig_viz, cmap=mask_cmap, norm=mask_norm, interpolation='nearest')
            axes[1, 0].set_title("Original mask")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(result_viz, cmap=mask_cmap, norm=mask_norm, interpolation='nearest')
            axes[1, 1].set_title(f"Mask with stroma added\n"
                                f"(new pixels at L0: {total_stroma:,})")
            axes[1, 1].axis('off')
            
            all_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
            syncing = [False]
            
            def sync_axes(source_ax):
                if syncing[0]:
                    return
                syncing[0] = True
                try:
                    xlim = source_ax.get_xlim()
                    ylim = source_ax.get_ylim()
                    for ax in all_axes:
                        if ax is not source_ax:
                            ax.set_xlim(xlim)
                            ax.set_ylim(ylim)
                    fig.canvas.draw_idle()
                finally:
                    syncing[0] = False
            
            for ax in all_axes:
                ax.callbacks.connect('xlim_changed', lambda event_ax, src=ax: sync_axes(src))
                ax.callbacks.connect('ylim_changed', lambda event_ax, src=ax: sync_axes(src))
            
            plt.suptitle(f"Preview (L0 pipeline, full image) - {Path(image_path).name}", fontsize=12)
            plt.tight_layout()
            plt.show()
        
        finally:
            Path(memmap_path).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Adds Stroma class to masks in unannotated tissue regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire dataset
  python add_stroma.py /path/to/dataset
  
  # With custom parameters
  python add_stroma.py /path/to/dataset --threshold 235 --blur 5 --dilate 10 --erode 3
  
  # Preview to tune parameters
  python add_stroma.py /path/to/dataset --preview
  
  # Process only one image
  python add_stroma.py /path/to/dataset --name "image_001"
  
  # Skip the first 14 images (start from the 15th)
  python add_stroma.py /path/to/dataset --skip 14

Expected dataset structure:
  dataset/
    images/
      image_001.ome.tif
      ...
    masks/
      image_001__mask_multiclass.ome.tif
      ...
        """
    )
    
    parser.add_argument(
        "dataset_dir",
        help="Dataset directory with images/ and masks/"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: dataset/masks_with_stroma/)"
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Images subdirectory (default: images)"
    )
    parser.add_argument(
        "--masks-dir",
        default="masks",
        help="Masks subdirectory (default: masks)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=240,
        help="Whiteness threshold 0-255 (default: 240, lower = more sensitive)"
    )
    parser.add_argument(
        "--blur", "-b",
        type=int,
        default=0,
        help="Gaussian blur before threshold in px (default: 0 = disabled)"
    )
    parser.add_argument(
        "--dilate", "-d",
        type=int,
        default=0,
        help="Tissue mask dilation in px (default: 0 = disabled)"
    )
    parser.add_argument(
        "--erode", "-e",
        type=int,
        default=0,
        help="Tissue mask erosion in px (default: 0 = disabled)"
    )
    parser.add_argument(
        "--erode-exterior-only",
        action="store_true",
        default=True,
        help="Only erode exterior boundaries, preserve interior holes (default: True)"
    )
    parser.add_argument(
        "--min-area", "-a",
        type=int,
        default=0,
        help="Minimum region area in pixels (default: 0 = no filtering)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile size for processing (default: 2048)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap between tiles to avoid edge artifacts (default: 100)"
    )
    parser.add_argument(
        "--name", "-n",
        help="Process only the image containing this name"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Show thresholding preview without saving"
    )
    parser.add_argument(
        "--preview-level",
        type=int,
        default=2,
        help="Pyramid level for preview (default: 2)"
    )
    parser.add_argument(
        "--skip", "-s",
        type=int,
        default=0,
        help="Skip the first N images (default: 0, start from the first)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    detector_params = {
        'threshold': args.threshold,
        'blur': args.blur,
        'dilate': args.dilate,
        'erode': args.erode,
        'min_area': args.min_area,
        'erode_exterior_only': args.erode_exterior_only,
    }
    
    if args.preview:
        import random
        
        adder = StromaAdder(
            args.dataset_dir,
            output_dir=args.output,
            images_subdir=args.images_dir,
            masks_subdir=args.masks_dir,
            detector_params=detector_params
        )
        
        pairs = adder.find_pairs()
        if args.name:
            pairs = [(i, m) for i, m in pairs if args.name in i.stem]
        
        if not pairs:
            print("No image-mask pairs found")
            sys.exit(1)
        
        img_path, mask_path = random.choice(pairs)
        print(f"Preview of: {img_path.name} (randomly selected from {len(pairs)})")
        
        preview_thresholding(
            str(img_path),
            str(mask_path),
            adder.detector,
            level=args.preview_level
        )
    else:
        print("=" * 60)
        print("Add Stroma to Masks")
        print("=" * 60)
        
        adder = StromaAdder(
            args.dataset_dir,
            output_dir=args.output,
            images_subdir=args.images_dir,
            masks_subdir=args.masks_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            detector_params=detector_params
        )
        
        adder.process_all(specific_name=args.name, skip_first=args.skip)
        
        print()
        print("=" * 60)
        print("Process completed")
        print("=" * 60)


if __name__ == "__main__":
    main()
