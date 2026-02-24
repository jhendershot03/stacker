"""
Astrophotography Image Stacker (v4 - Large Image Edition)
===========================================================
Key fix: alignment is computed on a downsampled copy (max 1500px wide),
then the transform is scaled back up and applied to the full-resolution image.
This makes alignment fast even on 6000×4000 images.

Requirements:
    pip install numpy opencv-python-headless Pillow tqdm astropy

Usage:
    python stack_images.py --input ./frames --output stacked.tif
    python stack_images.py --input ./frames --output stacked.tif --method median
    python stack_images.py --input ./frames --output stacked.tif --no-align
    python stack_images.py --input ./frames --output stacked.tif --debug

Supported formats: .jpg, .jpeg, .png, .tif, .tiff, .fit, .fits
"""

import argparse
import sys
import os
import glob
import signal
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found.  Run: pip install opencv-python-headless")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not found.  Run: pip install Pillow")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc  = kwargs.get("desc", "")
        total = kwargs.get("total", "?")
        for i, item in enumerate(iterable, 1):
            print(f"\r{desc}: {i}/{total}", end="", flush=True)
            yield item
        print()

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

# Resolution used for alignment computation (keeps things fast)
ALIGN_MAX_WIDTH = 1500


# ── Timeout helper (Unix/macOS only) ──────────────────────────────────────────
class AlignTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise AlignTimeoutError()

def with_timeout(seconds, func, *args, **kwargs):
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        return result
    return func(*args, **kwargs)  # Windows: no timeout


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext in (".fit", ".fits"):
        if not HAS_FITS:
            raise RuntimeError("astropy required for FITS: pip install astropy")
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 3:
                data = np.moveaxis(data, 0, -1)
            return data
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def collect_paths(input_dir: str) -> list:
    exts = ["jpg", "jpeg", "png", "tif", "tiff", "fit", "fits"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(input_dir, f"*.{ext}"))
        paths += glob.glob(os.path.join(input_dir, f"*.{ext.upper()}"))
    return sorted(set(paths))


def save_image(array: np.ndarray, path: str, debug: bool = False):
    ext   = Path(path).suffix.lower()
    array = np.clip(array, 0, None)
    mn, mx = float(array.min()), float(array.max())
    if debug:
        print(f"  [debug] output  min={mn:.2f}  max={mx:.2f}  shape={array.shape}")
    if mx == 0:
        print("  ⚠️  Result is all zeros — try --no-align to verify base stacking works.")
        mx = 1.0
    norm = (array - mn) / (mx - mn)
    if ext in (".tif", ".tiff"):
        out = (norm * 65535).astype(np.uint16)
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 else out)
    else:
        out = (norm * 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 else out)
    print(f"✅  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Alignment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_thumb(img: np.ndarray, max_width: int = ALIGN_MAX_WIDTH):
    """Return a downsampled uint8 grayscale thumbnail and the scale factor."""
    h, w = img.shape[:2]
    scale = min(1.0, max_width / w)
    tw, th = int(w * scale), int(h * scale)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    gray = np.clip(gray, 0, None)
    mx = gray.max()
    gray = (gray / mx * 255).astype(np.uint8) if mx > 0 else gray.astype(np.uint8)
    thumb = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
    return thumb, scale


def scale_affine(M: np.ndarray, scale: float) -> np.ndarray:
    """Scale a 2×3 affine matrix computed on a thumbnail back to full resolution."""
    M_full = M.copy()
    M_full[0, 2] /= scale   # tx
    M_full[1, 2] /= scale   # ty
    return M_full


def detect_stars(gray: np.ndarray, max_stars: int = 200) -> np.ndarray:
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    h, w = gray.shape
    min_area, max_area = 2, int(h * w * 0.002)
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            regions.append((float(gray[labels == i].mean()), centroids[i]))
    regions.sort(key=lambda x: -x[0])
    pts = [r[1] for r in regions[:max_stars]]
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def match_stars(ref_pts, img_pts, max_dist: float = 50.0):
    if len(ref_pts) == 0 or len(img_pts) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    ref_m, img_m, used = [], [], set()
    for rp in ref_pts:
        dists = np.linalg.norm(img_pts - rp, axis=1)
        idx   = int(np.argmin(dists))
        if dists[idx] < max_dist and idx not in used:
            ref_m.append(rp); img_m.append(img_pts[idx]); used.add(idx)
    return np.array(ref_m, dtype=np.float32), np.array(img_m, dtype=np.float32)


def _star_align_thumb(ref_thumb, img_thumb):
    """Run star centroid alignment on thumbnails. Returns 2×3 affine M or None."""
    ref_stars = detect_stars(ref_thumb)
    img_stars = detect_stars(img_thumb)
    if len(ref_stars) < 3 or len(img_stars) < 3:
        return None, len(ref_stars), len(img_stars)
    ref_m, img_m = match_stars(ref_stars, img_stars)
    if len(ref_m) < 3:
        return None, len(ref_stars), len(img_stars)
    M, _ = cv2.estimateAffinePartial2D(
        img_m, ref_m, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M, len(ref_stars), len(img_stars)


def _orb_align_thumb(ref_thumb, img_thumb):
    """Run ORB alignment on thumbnails. Returns 2×3 affine M or None."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_thumb, None)
    kp2, des2 = orb.detectAndCompute(img_thumb, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    good = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
    if len(good) < 4:
        return None
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # Convert homography to affine via estimateAffinePartial2D on matched points
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                        ransacReprojThreshold=5.0)
    return M


def align_frame(ref: np.ndarray, img: np.ndarray,
                timeout: int = 15, debug: bool = False) -> np.ndarray:
    """
    Align img to ref.
    1. Build small thumbnails for fast feature detection
    2. Compute affine transform on thumbnails
    3. Scale transform back to full resolution and warp full image
    """
    h, w = ref.shape[:2]

    ref_thumb, ref_scale = make_thumb(ref)
    img_thumb, img_scale = make_thumb(img)
    # Use the same scale for both (they should be identical resolution)
    scale = ref_scale

    # ── 1. Star centroid on thumbnail ─────────────────────────────────────────
    try:
        M, n_ref, n_img = with_timeout(timeout, _star_align_thumb, ref_thumb, img_thumb)
        if debug:
            print(f"    [debug] thumb size: {ref_thumb.shape[1]}×{ref_thumb.shape[0]}  "
                  f"scale: {scale:.3f}  stars: ref={n_ref} frame={n_img}")
        if M is not None:
            M_full   = scale_affine(M, scale)
            aligned  = cv2.warpAffine(img, M_full, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if aligned.max() > 0:
                return aligned
            elif debug:
                print("    [debug] star warp produced black frame — trying ORB")
    except AlignTimeoutError:
        print(f"    ⚠️  Star alignment timed out (>{timeout}s) — trying ORB")
    except Exception as e:
        if debug: print(f"    [debug] star align error: {e}")

    # ── 2. ORB on thumbnail ───────────────────────────────────────────────────
    try:
        M = with_timeout(timeout, _orb_align_thumb, ref_thumb, img_thumb)
        if M is not None:
            M_full  = scale_affine(M, scale)
            aligned = cv2.warpAffine(img, M_full, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if aligned.max() > 0:
                return aligned
    except AlignTimeoutError:
        print(f"    ⚠️  ORB timed out (>{timeout}s) — using unaligned")
    except Exception as e:
        if debug: print(f"    [debug] ORB error: {e}")

    print("    ⚠️  Alignment failed — using unaligned frame")
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  Stacking
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_clip_stack(frames: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    print(f"  Sigma-clipping (σ={sigma}) + mean combine…")
    mean  = np.mean(frames, axis=0)
    std   = np.std(frames,  axis=0)
    mask  = (frames >= mean - sigma * std) & (frames <= mean + sigma * std)
    count = np.maximum(mask.sum(axis=0).astype(np.float32), 1)
    return ((frames * mask).sum(axis=0) / count).astype(np.float32)

def median_stack(frames: np.ndarray) -> np.ndarray:
    print("  Median combining…")
    return np.median(frames, axis=0).astype(np.float32)

def mean_stack(frames: np.ndarray) -> np.ndarray:
    print("  Mean combining…")
    return np.mean(frames, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global ALIGN_MAX_WIDTH
    parser = argparse.ArgumentParser(
        description="Stack astrophotography frames (v4 — thumbnail alignment, no hangs)."
    )
    parser.add_argument("--input",     "-i", required=True)
    parser.add_argument("--output",    "-o", default="stacked.tif")
    parser.add_argument("--method",    "-m", choices=["sigma","median","mean"], default="sigma")
    parser.add_argument("--sigma",     "-s", type=float, default=2.5)
    parser.add_argument("--no-align",  action="store_true")
    parser.add_argument("--reference", "-r", type=int, default=0)
    parser.add_argument("--timeout",   "-t", type=int, default=15,
                        help="Max seconds per frame alignment (default: 15)")
    parser.add_argument("--align-width", type=int, default=ALIGN_MAX_WIDTH,
                        help=f"Width to downsample to for alignment (default: {ALIGN_MAX_WIDTH})")
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    ALIGN_MAX_WIDTH = args.align_width

    paths = collect_paths(args.input)
    if not paths:
        sys.exit(f"No supported image files found in: {args.input}")
    print(f"Found {len(paths)} frames in '{args.input}'")

    ref_idx   = min(args.reference, len(paths) - 1)
    reference = load_image(paths[ref_idx])
    print(f"Reference [{ref_idx}]: {Path(paths[ref_idx]).name}  "
          f"{reference.shape[1]}×{reference.shape[0]}  max={reference.max():.1f}")
    if not args.no_align:
        _, sc = make_thumb(reference)
        tw = int(reference.shape[1] * sc)
        th = int(reference.shape[0] * sc)
        print(f"Alignment thumbnail size: {tw}×{th}  (scale factor: {sc:.3f})")

    frames, failed = [], 0
    for i, p in enumerate(tqdm(paths, desc="Loading & aligning", total=len(paths))):
        img = load_image(p)
        if img.shape[:2] != reference.shape[:2]:
            img = cv2.resize(img, (reference.shape[1], reference.shape[0]))
        if not args.no_align and i != ref_idx:
            aligned = align_frame(reference, img,
                                  timeout=args.timeout, debug=args.debug)
            if aligned.max() == 0:
                failed += 1
            frames.append(aligned)
        else:
            frames.append(img)

    print()
    if failed:
        print(f"  ⚠️  {failed} frame(s) fell back to unaligned.")

    frames_arr = np.array(frames, dtype=np.float32)
    print(f"Stack shape: {frames_arr.shape}  "
          f"min={frames_arr.min():.1f}  max={frames_arr.max():.1f}")

    print(f"\nStacking ({args.method})…")
    if   args.method == "sigma":  result = sigma_clip_stack(frames_arr, args.sigma)
    elif args.method == "median": result = median_stack(frames_arr)
    else:                         result = mean_stack(frames_arr)

    print(f"Result  min={result.min():.1f}  max={result.max():.1f}")
    save_image(result, args.output, debug=args.debug)
    print("\nDone! 🔭")

if __name__ == "__main__":
    main()