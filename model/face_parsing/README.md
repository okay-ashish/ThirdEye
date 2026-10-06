# BiSeNet Face Parsing Model

## Overview
This directory contains deep neural networks for semantic face parsing and facial component segmentation.

## Files
- `efficientnet_b0.onnx` (~26.28 MB) — BiSeNet model with EfficientNet-B0 backbone exported to ONNX format.

## Semantic Categories
The model segments facial images into 19 distinct semantic classes (CelebAMask-HQ standard):
0: Background, 1: Skin, 2: Left Eyebrow, 3: Right Eyebrow, 4: Left Eye, 5: Right Eye, 6: Eyeglasses, 7: Left Ear, 8: Right Ear, 9: Earring, 10: Nose, 11: Mouth, 12: Upper Lip, 13: Lower Lip, 14: Neck, 15: Necklace, 16: Cloth, 17: Hair, 18: Hat.

## Provenance & Attribution
- **Source:** [Mrkomiljon/face-parsing](https://github.com/Mrkomiljon/face-parsing)
- **Dataset:** CelebAMask-HQ (30,000 high-resolution images)
- **License:** MIT License
- **Inference Hardware:** CPU execution via ONNX Runtime (`CPUExecutionProvider`)
