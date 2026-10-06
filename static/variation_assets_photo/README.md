# Photographic Variation Assets

This directory is reserved for high-resolution photographic appearance assets used in the **Face Variations (Phase 4.6)** appearance module.

## Supported Asset Files
When present, ThirdEye loads the following 1024x1024 RGBA transparent PNG assets:
- `short_hair.png` — Skull-conforming short cropped hair
- `long_hair.png` — Natural shoulder-length hair tresses
- `straight_hair.png` — Sleek straight strands
- `wavy_hair.png` — Voluminous wavy locks
- `curly_hair.png` — High-definition coiled curls
- `cap.png` — Standard athletic baseball cap with isolated visor
- `turban.png` — Formal headwear with frontal V-arch contour
- `moustache.png` — Upper lip facial hair
- `full_beard.png` — Full jawline beard with natural oral cavity
- `bald.png` — Cranial dome shading mask

## Fallback Behavior
These photographic PNG assets are excluded from the public repository due to third-party rights and provenance considerations.

ThirdEye includes a built-in graceful fallback:
If photographic PNG assets are absent from `static/variation_assets_photo/`, the variation engine automatically falls back to the clean vector sketch assets in [`static/variation_assets/`](../variation_assets/), ensuring full functionality without external downloads.

## Local Asset Acquisition
To utilize photographic overlays locally:
1. Obtain appropriately licensed or synthetically generated 1024x1024 RGBA PNGs.
2. Ensure images have a transparent alpha channel and are centered on standard anthropometric landmarks.
3. Place them in this folder with the filenames listed above.
