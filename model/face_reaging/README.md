# FRAN Face Re-Aging Model (Disney Research FRAN Architecture)

## Overview
This directory houses the neural network weights and visual blending masks for the **Face Re-Aging** module.

## Files
- `face_reaging.onnx` (~118.48 MB) — Face Re-Aging Network (FRAN) exported to ONNX format for CPU inference. *(Excluded from Git due to GitHub 100MB file limit and license considerations).*
- `mask1024.jpg` (200 KB) — High-resolution anatomical facial skin mask for continuous boundary blending.
- `mask512.jpg` (10 KB) — Standard resolution facial skin mask.

## Provenance & Attribution
- **Architecture:** Disney Research FRAN (Production-Ready Face Re-Aging for Visual Effects).
- **ONNX Export Source:** [Glat0s/face_reaging-onnx](https://github.com/Glat0s/face_reaging-onnx).
- **License:** Non-commercial research / experimental use.

## How to Set Up the Weight
Because `face_reaging.onnx` is 118.48 MB, it cannot be hosted directly in standard public Git repositories without Git LFS.

To enable deep learned re-aging locally:
1. Download `face_reaging.onnx` from the model source repository:
   - Source URL: `https://github.com/Glat0s/face_reaging-onnx`
2. Place `face_reaging.onnx` in this directory:
   ```
   model/face_reaging/face_reaging.onnx
   ```
3. If this file is absent, ThirdEye automatically falls back to procedural landmark-guided skin tension and contrast adjustments.
