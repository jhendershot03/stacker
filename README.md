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
