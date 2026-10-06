# Sketch-to-Face Reconstruction Models

## Overview
This directory contains generative neural network models for converting forensic composite sketches into realistic photorealistic face reconstructions.

## Files
- `cyclegan_model.pt` (~30.04 MB) — Lightweight TorchScript-compiled CycleGAN generator trained for sketch-to-photo domain translation.

## Provenance & Details
- **Framework:** PyTorch (TorchScript `torch.jit`)
- **Input:** $256 \times 256$ standardized grayscale/RGB composite sketch
- **Output:** $256 \times 256$ RGB reconstructed photographic face representation
- **Inference Hardware:** CPU execution (optimized with native PyTorch operators)
- **License:** Open Source Research Model

## Setup
The primary model `cyclegan_model.pt` (30.04 MB) is within GitHub's individual file limit (< 50 MB) and can be included directly with the codebase.
