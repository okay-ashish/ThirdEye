"""
ThirdEye System — Phase 3.1
Forensic Facial Reconstruction Pipeline
Synthesizes a computer-generated facial appearance hypothesis from an input sketch
using a learned Sketch-to-Face neural generator (CycleGAN / ResNet-6) with
structural guidance, robust 68-point facial landmark geometry, calibrated Fitzpatrick
skin tone color grading, and an anatomical fallback engine.
"""

import os
import time
import uuid
import base64
import numpy as np
import cv2
from PIL import Image

# --- CALIBRATED FORENSIC SKIN TONE PALETTE (Fitzpatrick Scale Reference) ---
SKIN_TONE_PALETTE = {
    'fair': {
        'id': 'fair',
        'name': 'Fair / Porcelain',
        'hex': '#FCE2D6',
        'base_bgr': np.array([205.0, 222.0, 246.0], dtype=np.float32),
        'lip_bgr': np.array([135.0, 140.0, 210.0], dtype=np.float32),
        'warmth_bgr': np.array([25.0, 12.0, 28.0], dtype=np.float32),
        'hair_bgr': np.array([40.0, 50.0, 65.0], dtype=np.float32)
    },
    'light': {
        'id': 'light',
        'name': 'Light / Peach',
        'hex': '#F3D0B7',
        'base_bgr': np.array([182.0, 208.0, 240.0], dtype=np.float32),
        'lip_bgr': np.array([125.0, 130.0, 195.0], dtype=np.float32),
        'warmth_bgr': np.array([22.0, 10.0, 26.0], dtype=np.float32),
        'hair_bgr': np.array([35.0, 45.0, 58.0], dtype=np.float32)
    },
    'medium': {
        'id': 'medium',
        'name': 'Warm Beige / Natural',
        'hex': '#E0B18B',
        'base_bgr': np.array([142.0, 178.0, 222.0], dtype=np.float32),
        'lip_bgr': np.array([110.0, 115.0, 175.0], dtype=np.float32),
        'warmth_bgr': np.array([20.0, 8.0, 24.0], dtype=np.float32),
        'hair_bgr': np.array([30.0, 38.0, 50.0], dtype=np.float32)
    },
    'olive': {
        'id': 'olive',
        'name': 'Tan / Olive',
        'hex': '#C48D5E',
        'base_bgr': np.array([108.0, 148.0, 198.0], dtype=np.float32),
        'lip_bgr': np.array([95.0, 98.0, 155.0], dtype=np.float32),
        'warmth_bgr': np.array([18.0, 7.0, 20.0], dtype=np.float32),
        'hair_bgr': np.array([25.0, 32.0, 42.0], dtype=np.float32)
    },
    'brown': {
        'id': 'brown',
        'name': 'Deep Tan / Bronze',
        'hex': '#9E6B47',
        'base_bgr': np.array([78.0, 112.0, 162.0], dtype=np.float32),
        'lip_bgr': np.array([72.0, 75.0, 128.0], dtype=np.float32),
        'warmth_bgr': np.array([15.0, 6.0, 18.0], dtype=np.float32),
        'hair_bgr': np.array([20.0, 25.0, 35.0], dtype=np.float32)
    },
    'dark': {
        'id': 'dark',
        'name': 'Rich Dark / Espresso',
        'hex': '#5C3A24',
        'base_bgr': np.array([48.0, 68.0, 104.0], dtype=np.float32),
        'lip_bgr': np.array([50.0, 52.0, 88.0], dtype=np.float32),
        'warmth_bgr': np.array([12.0, 5.0, 14.0], dtype=np.float32),
        'hair_bgr': np.array([15.0, 18.0, 25.0], dtype=np.float32)
    }
}

DEFAULT_RECONSTRUCTION_CONFIG = {
    'max_dimension': 800,
    'model_path': os.path.join(os.path.dirname(__file__), 'model', 'sketch_to_face', 'cyclegan_model.pt'),
    'force_fallback': False
}

# Global singleton model cache
_CACHED_MODEL = None
_MODEL_LOAD_ATTEMPTED = False
_MODEL_LOAD_ERROR = None


def get_sketch_to_face_model(model_path=None):
    """
    Returns the cached learned sketch-to-face PyTorch TorchScript model.
    Loads lazily on first invocation.
    """
    global _CACHED_MODEL, _MODEL_LOAD_ATTEMPTED, _MODEL_LOAD_ERROR
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if model_path is None:
        model_path = DEFAULT_RECONSTRUCTION_CONFIG['model_path']

    if not os.path.exists(model_path):
        _MODEL_LOAD_ERROR = f"Model file not found at {model_path}"
        return None

    try:
        import torch
        device = torch.device('cpu')
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        _CACHED_MODEL = model
        _MODEL_LOAD_ATTEMPTED = True
        return _CACHED_MODEL
    except Exception as e:
        _MODEL_LOAD_ERROR = str(e)
        _MODEL_LOAD_ATTEMPTED = True
        return None


def decode_sketch_input(sketch_input):
    """
    Decodes input sketch into an OpenCV BGR uint8 numpy array.
    Supports Base64 data URL, raw bytes, or filesystem path.
    Composites 4-channel RGBA transparent canvases onto clean white backdrop.
    """
    if isinstance(sketch_input, str):
        if ',' in sketch_input and 'data:image' in sketch_input:
            header, encoded = sketch_input.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif os.path.exists(sketch_input):
            img = cv2.imread(sketch_input, cv2.IMREAD_UNCHANGED)
        else:
            try:
                img_bytes = base64.b64decode(sketch_input)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            except Exception:
                img = None
    elif isinstance(sketch_input, (bytes, bytearray)):
        nparr = np.frombuffer(sketch_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    else:
        img = None

    if img is None:
        raise ValueError("Could not decode sketch image input. Unsupported format or corrupted data.")

    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(bgr, dtype=np.float32) * 255.0
        blended = (bgr * alpha[:, :, np.newaxis] + white_bg * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)
        return blended
    elif len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        return img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def detect_landmarks_robust(rgb_img):
    """
    Robust 68-point facial landmark detector with 4-tier fallbacks:
      Tier 1: Direct Dlib HOG face landmarks
      Tier 2: Up-sampled Dlib face landmarks
      Tier 3: Stroke-contour bounding-box guided Dlib shape prediction
      Tier 4: Geometric anatomical cranial estimation
    """
    import face_recognition
    h, w = rgb_img.shape[:2]

    # Tier 1: Direct Dlib search
    try:
        lms = face_recognition.face_landmarks(rgb_img)
        if lms and len(lms) > 0:
            return lms[0], 'dlib_direct'
    except Exception:
        pass

    # Tier 2: Up-sampled Dlib search
    try:
        locs = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=1)
        if locs:
            lms = face_recognition.face_landmarks(rgb_img, face_locations=locs)
            if lms and len(lms) > 0:
                return lms[0], 'dlib_upsampled'
    except Exception:
        pass

    # Tier 3: Stroke-bounding-box guided Dlib prediction
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    pts = np.argwhere(inv > 25)
    if len(pts) > 100:
        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
        bw = max_x - min_x
        bh = max_y - min_y

        top = max(0, int(min_y + bh * 0.18))
        bottom = min(h - 1, int(max_y + bh * 0.02))
        left = max(0, int(min_x + bw * 0.06))
        right = min(w - 1, int(max_x - bw * 0.06))

        bbox = (top, right, bottom, left)
        try:
            lms = face_recognition.face_landmarks(rgb_img, face_locations=[bbox])
            if lms and len(lms) > 0:
                return lms[0], 'dlib_stroke_guided'
        except Exception:
            pass

    # Tier 4: Synthetic Anatomical Landmarks Fallback
    if len(pts) > 100:
        cx = int((min_x + max_x) / 2)
        cy = int((min_y + max_y) / 2)
        bw = max(120, bw)
        bh = max(150, bh)
    else:
        cx, cy = w // 2, h // 2
        bw, bh = int(w * 0.45), int(h * 0.6)

    chin = []
    for i in range(17):
        ang = np.pi * 0.85 * (i / 16.0) + np.pi * 0.075
        px = int(cx + (bw * 0.48) * np.cos(np.pi - ang))
        py = int(cy + (bh * 0.44) * np.sin(ang))
        chin.append((px, py))

    synthetic = {
        'chin': chin,
        'left_eyebrow': [(int(cx - bw * 0.35 + i * bw * 0.06), int(cy - bh * 0.18 - (2 - abs(i - 2)) * 4)) for i in range(5)],
        'right_eyebrow': [(int(cx + bw * 0.11 + i * bw * 0.06), int(cy - bh * 0.18 - (2 - abs(i - 2)) * 4)) for i in range(5)],
        'nose_bridge': [(cx, int(cy - bh * 0.14 + i * bh * 0.07)) for i in range(4)],
        'nose_tip': [(int(cx - bw * 0.09 + i * bw * 0.045), int(cy + bh * 0.1 + (1 if i in [0, 4] else 4))) for i in range(5)],
        'left_eye': [(int(cx - bw * 0.3 + i * bw * 0.04), int(cy - bh * 0.09)) for i in range(6)],
        'right_eye': [(int(cx + bw * 0.12 + i * bw * 0.04), int(cy - bh * 0.09)) for i in range(6)],
        'top_lip': [(int(cx - bw * 0.14 + i * bw * 0.025), int(cy + bh * 0.22)) for i in range(12)],
        'bottom_lip': [(int(cx - bw * 0.14 + i * bw * 0.025), int(cy + bh * 0.25)) for i in range(12)]
    }
    return synthetic, 'synthetic_geometry'


# ----------------------------------------------------------------------
# PRIMARY ENGINE: LEARNED SKETCH-TO-FACE SYNTHESIS (CYCLEGAN / RESNET)
# ----------------------------------------------------------------------
def reconstruct_face_learned(bgr_img, lm, skin_tone='medium', model=None):
    """
    Executes deep learned sketch-to-face synthesis.
    Transforms input sketch into a photorealistic facial appearance hypothesis using:
      1. Tightly aligned bounding box and scale normalization.
      2. Neural generator inference on CPU (TorchScript ResNet-6 Generator).
      3. High-fidelity super-sampling and selective Fitzpatrick skin tone grading.
      4. Anatomically natural silhouette termination (zero rectangular neck extension).
      5. High-frequency structural sketch guidance preservation.
    """
    import torch
    h, w = bgr_img.shape[:2]
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 1. Determine bounding box for subject alignment
    pts = np.argwhere(gray < 245)
    if len(pts) > 50:
        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
    else:
        min_y, max_y = int(h * 0.1), int(h * 0.9)
        min_x, max_x = int(w * 0.2), int(w * 0.8)

    margin = 30
    min_y = max(0, min_y - margin)
    max_y = min(h, max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(w, max_x + margin)

    bh = max_y - min_y
    bw = max_x - min_x
    size = max(bh, bw)
    cy = (min_y + max_y) // 2
    cx = (min_x + max_x) // 2

    y1 = max(0, cy - size // 2)
    y2 = min(h, cy + size // 2)
    x1 = max(0, cx - size // 2)
    x2 = min(w, cx + size // 2)

    crop_bgr = bgr_img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    ch, cw = crop_rgb.shape[:2]

    # 2. Rescale to 128x128 and normalize to [-1.0, 1.0]
    crop_128 = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    tensor_in = ((crop_128.astype(np.float32) / 127.5) - 1.0)
    tensor_torch = torch.from_numpy(tensor_in).permute(2, 0, 1).unsqueeze(0)

    # 3. Neural Generator Inference
    with torch.no_grad():
        tensor_out = model(tensor_torch)

    out_np = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_rgb = np.clip((out_np + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(cv2.resize(out_rgb, (cw, ch), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)

    # 4. Photometric Fitzpatrick Skin Tone Transfer
    tone_key = skin_tone.lower() if skin_tone and skin_tone.lower() in SKIN_TONE_PALETTE else 'medium'
    skin_cfg = SKIN_TONE_PALETTE[tone_key]
    base_bgr = skin_cfg['base_bgr']

    synth_gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    l_norm = np.clip(synth_gray / 0.68, 0.0, 1.25)
    skin_bgr = np.zeros_like(out_bgr, dtype=np.float32)
    for c in range(3):
        skin_bgr[:, :, c] = np.clip(base_bgr[c] * l_norm, 0.0, 255.0)

    # Preserve dark features (hair, iris/pupil ocular pigment, shadows)
    dark_factor = np.clip((85.0 - synth_gray * 255.0) / 45.0, 0.0, 1.0)[:, :, np.newaxis]
    graded_patch = skin_bgr * (1.0 - dark_factor) + out_bgr.astype(np.float32) * dark_factor

    # 5. Integrate High-Frequency Structural Sketch Linework
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sketch_lines = np.clip((240.0 - crop_gray) / 255.0 * 0.28, 0.0, 0.28)[:, :, np.newaxis]
    graded_patch = graded_patch * (1.0 - sketch_lines)

    # 6. Anatomically Natural Subject Silhouette (Zero bottom-of-canvas neck defect)
    bg_color = np.array([248.0, 245.0, 242.0], dtype=np.float32)  # Clean presentation backdrop #F2F5F8
    bg_canvas = np.full((h, w, 3), bg_color, dtype=np.float32)

    full_patch = bg_canvas.copy()
    full_patch[y1:y2, x1:x2] = graded_patch

    # Extract silhouette contour using morphological closure on sketch strokes
    binary = (gray < 242).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.float32)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 1.0, -1)
    else:
        # Elliptical fallback
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 1.0, -1)

    mask_soft = cv2.GaussianBlur(mask, (13, 13), 3.5)[:, :, np.newaxis]

    # 7. Composite onto clean studio backdrop
    final = bg_canvas * (1.0 - mask_soft) + full_patch * mask_soft

    # 8. Subtle epidermal microtexture grain
    np.random.seed(42)
    noise = np.random.normal(0, 1.2, (h, w, 3)).astype(np.float32)
    final = np.clip(final + noise * mask_soft, 0, 255).astype(np.uint8)

    return final, 'learned_cyclegan'


# ----------------------------------------------------------------------
# FALLBACK ENGINE: COMPUTATIONAL ANATOMICAL RECONSTRUCTION
# ----------------------------------------------------------------------
def estimate_face_geometry_fallback(lm, h, w):
    """
    Derives 3D facial planes and natural anatomical silhouette for the fallback engine.
    Terminates submental neck contour naturally just below chin (ZERO h-1 rectangle).
    """
    chin = np.array(lm['chin'], dtype=np.int32)
    left_eyebrow = np.array(lm['left_eyebrow'], dtype=np.int32)
    right_eyebrow = np.array(lm['right_eyebrow'], dtype=np.int32)
    nose_bridge = np.array(lm['nose_bridge'], dtype=np.int32)
    nose_tip = np.array(lm['nose_tip'], dtype=np.int32)
    left_eye = np.array(lm['left_eye'], dtype=np.int32)
    right_eye = np.array(lm['right_eye'], dtype=np.int32)
    top_lip = np.array(lm['top_lip'], dtype=np.int32)
    bottom_lip = np.array(lm['bottom_lip'], dtype=np.int32)

    chin_top_left = chin[0]
    chin_top_right = chin[-1]
    brow_top_left = np.min(left_eyebrow, axis=0)
    brow_top_right = np.min(right_eyebrow, axis=0)
    mid_brow_x = int((brow_top_left[0] + brow_top_right[0]) / 2)

    face_height = chin[8][1] - min(brow_top_left[1], brow_top_right[1])
    forehead_y = max(8, int(min(brow_top_left[1], brow_top_right[1]) - face_height * 0.40))

    forehead_pts = [
        chin_top_right,
        [int(chin_top_right[0] * 0.88 + brow_top_right[0] * 0.12), int(brow_top_right[1] - 22)],
        [brow_top_right[0] + 5, int(forehead_y + 22)],
        [int(mid_brow_x * 0.5 + brow_top_right[0] * 0.5), int(forehead_y + 6)],
        [mid_brow_x, forehead_y],
        [int(mid_brow_x * 0.5 + brow_top_left[0] * 0.5), int(forehead_y + 6)],
        [brow_top_left[0] - 5, int(forehead_y + 22)],
        [int(chin_top_left[0] * 0.88 + brow_top_left[0] * 0.12), int(brow_top_left[1] - 22)],
        chin_top_left
    ]
    full_face_polygon = np.vstack([chin, forehead_pts])

    # NATURAL SUBMENTAL CURVE: extends at most ~12% face height below chin (zero h-1 bottom artifact)
    neck_drop = min(30, int(face_height * 0.12))
    neck_polygon = np.array([
        chin[3], chin[5],
        [chin[8][0], chin[8][1] + neck_drop],
        chin[11], chin[13]
    ], dtype=np.int32)

    return {
        'chin': chin,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
        'nose_bridge': nose_bridge,
        'nose_tip': nose_tip,
        'left_eye': left_eye,
        'right_eye': right_eye,
        'top_lip': top_lip,
        'bottom_lip': bottom_lip,
        'mid_brow_x': mid_brow_x,
        'forehead_y': forehead_y,
        'full_face_polygon': full_face_polygon,
        'neck_polygon': neck_polygon,
        'face_height': face_height
    }


def reconstruct_face_procedural(bgr_img, lm, skin_tone='medium'):
    """
    Fallback procedural reconstruction engine with corrected natural silhouette (zero neck block).
    """
    h, w = bgr_img.shape[:2]
    geom = estimate_face_geometry_fallback(lm, h, w)

    # Face mask without bottom rectangular chimney
    face_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(face_mask, [geom['full_face_polygon']], 1.0)
    cv2.fillPoly(face_mask, [geom['neck_polygon']], 0.85)
    face_mask_soft = cv2.GaussianBlur(face_mask, (25, 25), 8)[:, :, np.newaxis]

    tone_key = skin_tone.lower() if skin_tone and skin_tone.lower() in SKIN_TONE_PALETTE else 'medium'
    skin_cfg = SKIN_TONE_PALETTE[tone_key]
    base_bgr = skin_cfg['base_bgr']

    bg_color = np.array([248.0, 245.0, 242.0], dtype=np.float32)
    bg_canvas = np.full((h, w, 3), bg_color, dtype=np.float32)

    # Shaded skin tone base
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    skin_canvas = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        skin_canvas[:, :, c] = base_bgr[c] * (gray * 0.5 + 0.5)

    # Soft sketch integration
    sketch_inv = (255.0 - cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)) / 255.0
    shaded = skin_canvas * (1.0 - sketch_inv[:, :, np.newaxis] * 0.45)

    final = bg_canvas * (1.0 - face_mask_soft) + shaded * face_mask_soft
    return np.clip(final, 0, 255).astype(np.uint8), 'procedural_fallback'


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------
def reconstruct_face(sketch_input, output_dir='static/generated_faces', skin_tone='medium', config=None):
    """
    Main entry point for computer-generated facial reconstruction.
    Executes the primary learned sketch-to-face generator with automatic procedural fallback.
    """
    start_time = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    cfg = DEFAULT_RECONSTRUCTION_CONFIG.copy()
    if config and isinstance(config, dict):
        cfg.update(config)

    # 1. Input decoding and validation
    bgr_img = decode_sketch_input(sketch_input)
    h, w = bgr_img.shape[:2]

    # Standardize dimensions for consistent quality & sub-second CPU performance
    max_dim = cfg.get('max_dimension', 800)
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr_img.shape[:2]

    # Unique filenames
    uid = uuid.uuid4().hex[:10]
    sketch_filename = f"sketch_{uid}.png"
    reconstruction_filename = f"reconstruction_{uid}.png"
    sketch_path = os.path.join(output_dir, sketch_filename)
    reconstruction_path = os.path.join(output_dir, reconstruction_filename)

    # Save pristine sketch
    cv2.imwrite(sketch_path, bgr_img)

    # 2. Robust landmark detection
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    lm, tier = detect_landmarks_robust(rgb_img)

    # 3. Model acquisition & inference execution
    model = None
    force_fallback = cfg.get('force_fallback', False)
    if not force_fallback:
        model = get_sketch_to_face_model(cfg.get('model_path'))

    engine_method = "AI-Assisted Neural Synthesis (CycleGAN / ResNet-6)"
    ai_status = "AI Reconstruction Model Active"

    if model is not None:
        try:
            final_output, used_engine = reconstruct_face_learned(bgr_img, lm, skin_tone=skin_tone, model=model)
        except Exception as e:
            # Fallback to procedural reconstruction on inference error
            final_output, used_engine = reconstruct_face_procedural(bgr_img, lm, skin_tone=skin_tone)
            engine_method = "Computational Fallback Engine (Procedural)"
            ai_status = f"AI model inference failed ({str(e)}). Using computational fallback."
    else:
        # Fallback to procedural reconstruction when model is unavailable
        final_output, used_engine = reconstruct_face_procedural(bgr_img, lm, skin_tone=skin_tone)
        engine_method = "Computational Fallback Engine (Procedural)"
        ai_status = f"AI reconstruction model unavailable ({_MODEL_LOAD_ERROR or 'forced'}). Using computational fallback."

    # 4. Save final reconstruction image
    cv2.imwrite(reconstruction_path, final_output)

    elapsed_time = round(time.perf_counter() - start_time, 3)
    tone_key = skin_tone.lower() if skin_tone and skin_tone.lower() in SKIN_TONE_PALETTE else 'medium'
    skin_cfg = SKIN_TONE_PALETTE[tone_key]

    return {
        "success": True,
        "sketch_filename": sketch_filename,
        "reconstruction_filename": reconstruction_filename,
        "processing_time": elapsed_time,
        "dimensions": [w, h],
        "landmarks_detected": tier != 'synthetic_geometry',
        "detection_tier": tier,
        "skin_tone": tone_key,
        "skin_tone_name": skin_cfg['name'],
        "skin_tone_hex": skin_cfg['hex'],
        "method": engine_method,
        "ai_model_status": ai_status,
        "engine_used": used_engine
    }
