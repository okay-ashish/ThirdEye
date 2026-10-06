"""
ThirdEye Forensic Biometric System — Phase 4.2
Face Variations: Geometry-Aware Realism & Seamless Appearance Replacement

Architecture:
  1. Standardized Image Loading & Multi-tier Robust Landmark Extraction
  2. Anatomical Cranial Geometry Analysis:
     - Head roll/tilt angle theta, eye axis unit vectors (u_par, u_perp)
     - Trichion (forehead hairline center), Vertex (cranial apex), Parietal crests
     - Temple crests, Gonion angles, Mandibular contour, Philtrum, Vermilion border
     - Local skin sampling (forehead/cheek in LAB space) & Ambient background sampling
  3. Original Feature Removal & Suppression:
     - Outer hair flare suppression smoothly blended into background
     - Clean face polygon strictly preserving eyes, nose, lips, and facial proportions
  4. Person-Specific Appearance Synthesis:
     - Piecewise Affine / Delaunay Mesh Warping adapting hairstyle to skull contour and head tilt
     - Controlled Hair Lengths (Short hugs ears, Curly frames jaw, Wavy/Long flow to collar)
     - Anatomical 3D Shaded Cranial Dome for Bald reconstruction with pore texture & forehead fade
     - Clean Shave with full lower-face & moustache inpainting, LAB color matching, and skin grain
     - Jaw-conforming Full Beard with anti-halo fringe tinting & philtrum-aligned Moustache
     - Cap seated strictly on cranial vault above brows (visor shadow + hair-under-cap layering)
     - Royal Turban wrapping naturally around parietal contour
  5. Independent Age Adjustment (-20 to +20)
  6. Single Output Workflow (One Primary Appearance Variation only, session-scoped lifecycle)
"""

import os
import time
import uuid
import logging
import cv2
import numpy as np
import scipy.spatial
from PIL import Image
from face_reconstruction import detect_landmarks_robust

logger = logging.getLogger(__name__)

# Model paths for Phase 4.4
PARSER_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'face_parsing', 'efficientnet_b0.onnx')
FRAN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'face_reaging', 'face_reaging.onnx')

_GLOBAL_PARSER_SESSION = None
_GLOBAL_FRAN_SESSION = None

def get_face_parser_session():
    """Returns cached singleton ONNX session for BiSeNet face parsing (CPU)."""
    global _GLOBAL_PARSER_SESSION
    if _GLOBAL_PARSER_SESSION is None and os.path.exists(PARSER_MODEL_PATH):
        try:
            import onnxruntime as ort
            _GLOBAL_PARSER_SESSION = ort.InferenceSession(PARSER_MODEL_PATH, providers=['CPUExecutionProvider'])
            logger.info("BiSeNet EfficientNet-B0 face parser ONNX loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load BiSeNet parser: {e}")
    return _GLOBAL_PARSER_SESSION

def get_fran_session():
    """Returns cached singleton ONNX session for FRAN face re-aging (CPU)."""
    global _GLOBAL_FRAN_SESSION
    if _GLOBAL_FRAN_SESSION is None and os.path.exists(FRAN_MODEL_PATH):
        try:
            import onnxruntime as ort
            _GLOBAL_FRAN_SESSION = ort.InferenceSession(FRAN_MODEL_PATH, providers=['CPUExecutionProvider'])
            logger.info("FRAN face re-aging ONNX loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load FRAN model: {e}")
    return _GLOBAL_FRAN_SESSION

def run_face_parser_segmentation(bgr_img):
    """
    Runs BiSeNet ONNX face parser (EfficientNet-B0) on standardized face image.
    Returns 2D uint8 segmentation mask (H, W) with CelebAMask-HQ 19 classes:
      0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
      6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
      12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    Returns None if model is unavailable or on failure.
    """
    sess = get_face_parser_session()
    if sess is None:
        return None
    try:
        orig_h, orig_w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)

        inp_name = sess.get_inputs()[0].name
        outs = sess.run(None, {inp_name: arr})
        logits = outs[0][0] # (19, 256, 256)
        seg_256 = np.argmax(logits, axis=0).astype(np.uint8)
        seg = cv2.resize(seg_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return seg
    except Exception as e:
        logger.warning(f"Error during face parsing inference: {e}")
        return None

# -------------------------------------------------------------------------
# CONFIGURATION DEFINITIONS
# -------------------------------------------------------------------------

SUPPORTED_HAIR_STYLES = [
    'Keep Original',
    'Short',
    'Long',
    'Straight',
    'Wavy',
    'Curly',
    'Bald'
]

SUPPORTED_FACIAL_HAIR = [
    'Keep Original',
    'Clean Shaven',
    'Light Stubble',
    'Heavy Stubble',
    'Moustache',
    'Full Beard',
    'Beard + Moustache'
]

SUPPORTED_HEADWEAR = [
    'None',
    'Cap',
    'Turban'
]

VALID_AGE_DELTAS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

DEFAULT_APPEARANCE_CONFIG = {
    'hair_style': 'Keep Original',
    'facial_hair': 'Keep Original',
    'current_age': 25,
    'age_delta': 0,
    'target_age': 25,
    'headwear': 'None'
}

ASSET_PHOTO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'variation_assets_photo')
ASSET_DIR = os.path.join(os.path.dirname(__file__), 'static', 'variation_assets')

def get_asset_path(filename):
    """
    Phase 4.6: Prefers high-resolution photographic appearance assets in
    static/variation_assets_photo/, falling back gracefully to static/variation_assets/.
    """
    photo_path = os.path.join(ASSET_PHOTO_DIR, filename)
    if os.path.exists(photo_path):
        return photo_path
    fallback_path = os.path.join(ASSET_DIR, filename)
    if os.path.exists(fallback_path):
        return fallback_path
    return photo_path



# -------------------------------------------------------------------------
# STAGE 1: IMAGE LOADING & STANDARDIZATION
# -------------------------------------------------------------------------

def load_and_standardize_image(image_input, max_dimension=750):
    """
    Decodes image input from file path, numpy array, or PIL Image.
    Scales to max_dimension for optimal CPU performance while preserving aspect ratio.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image not found: {image_input}")
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not decode image file: {image_input}")
    elif isinstance(image_input, np.ndarray):
        bgr = image_input.copy()
        if len(bgr.shape) == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif len(bgr.shape) == 3 and bgr.shape[2] == 4:
            alpha = bgr[:, :, 3].astype(np.float32) / 255.0
            white = np.ones_like(bgr[:, :, :3], dtype=np.float32) * 255.0
            bgr = (bgr[:, :, :3] * alpha[:, :, np.newaxis] + white * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)
    elif isinstance(image_input, Image.Image):
        bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    h, w = bgr.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return bgr


# -------------------------------------------------------------------------
# STAGE 2: GEOMETRY & LANDMARK EXTRACTION
# -------------------------------------------------------------------------

def get_landmarks(bgr_img):
    """
    Robust landmark detector with multi-tier fallback.
    Returns 68-point dictionary or None if no face detected.
    """
    if bgr_img is None or bgr_img.size == 0:
        return None

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 8.0:
        return None

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    try:
        lm, tier = detect_landmarks_robust(rgb)
        if tier == 'synthetic_geometry':
            import face_recognition
            locs = face_recognition.face_locations(rgb)
            if not locs:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    return None
        return lm
    except Exception:
        return None


def extract_face_geometry(bgr):
    """
    Comprehensive anatomical cranial and facial geometry analysis for Phase 4.2.
    Computes head roll/tilt angle theta, eye axis vectors (u_par, u_perp),
    Trichion hairline center, Vertex cranial apex, Parietal crests,
    Mandibular angles, Philtrum, and sampled skin and background palettes.
    """
    h, w = bgr.shape[:2]
    lm = get_landmarks(bgr)
    if lm is None:
        return None

    chin = np.array(lm['chin'], dtype=np.float32)
    l_brow = np.array(lm['left_eyebrow'], dtype=np.float32)
    r_brow = np.array(lm['right_eyebrow'], dtype=np.float32)
    brows = np.vstack([l_brow, r_brow])

    l_eye = np.array(lm['left_eye'], dtype=np.float32)
    r_eye = np.array(lm['right_eye'], dtype=np.float32)
    eyes = np.vstack([l_eye, r_eye])

    nose_tip = np.array(lm['nose_tip'], dtype=np.float32)
    top_lip = np.array(lm['top_lip'], dtype=np.float32)
    bottom_lip = np.array(lm['bottom_lip'], dtype=np.float32)
    lips = np.vstack([top_lip, bottom_lip])

    pupil_L = np.mean(l_eye, axis=0)
    pupil_R = np.mean(r_eye, axis=0)

    d_eye = pupil_R - pupil_L
    angle = float(np.arctan2(d_eye[1], d_eye[0]))
    u_par = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    u_perp = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)

    temple_L = chin[0]
    temple_R = chin[16]
    ear_L = chin[1]
    ear_R = chin[15]
    gonion_L = chin[4]
    gonion_R = chin[12]
    chin_tip = chin[8]

    nasion = (np.mean(l_brow, axis=0) + np.mean(r_brow, axis=0)) / 2.0
    mid_brow = np.mean(brows, axis=0)
    subnasale = nose_tip[2] if len(nose_tip) >= 3 else nose_tip[-1]
    philtrum = (subnasale + top_lip[3]) / 2.0
    mouth_L = top_lip[0]
    mouth_R = top_lip[6]

    fh = float(np.dot(chin_tip - mid_brow, u_perp))
    if fh < 20.0:
        fh = float(np.linalg.norm(chin_tip - mid_brow))
    fw = float(np.linalg.norm(temple_R - temple_L))

    hairline_C = nasion - (0.32 * fh) * u_perp
    apex = nasion - (0.68 * fh) * u_perp

    brow_y = int(np.mean(brows[:, 1]))
    chin_y = int(np.max(chin[:, 1]))
    left_x = int(np.min(chin[:, 0]))
    right_x = int(np.max(chin[:, 0]))
    mid_x = (left_x + right_x) // 2
    hairline_y = max(0, int(hairline_C[1]))
    apex_y = max(0, int(apex[1]))
    nose_base_y = int(np.max(nose_tip[:, 1]))

    # Forehead skin sampling
    sample_y = int(np.clip(nasion[1] - 0.15 * fh * u_perp[1], 0, h - 1))
    sample_x = int(np.clip(nasion[0] - 0.15 * fh * u_perp[0], 0, w - 1))
    patch = bgr[max(0, sample_y - 12):min(h, sample_y + 12), max(0, sample_x - 16):min(w, sample_x + 16)]
    if patch.size > 0:
        skin_bgr = np.median(patch, axis=(0, 1)).astype(np.float32)
    else:
        skin_bgr = np.array([160.0, 190.0, 220.0], dtype=np.float32)

    # Ambient background sampling
    c1 = bgr[:max(10, int(sample_y // 2)), :max(10, int(w * 0.15))]
    c2 = bgr[:max(10, int(sample_y // 2)), min(w - 10, int(w * 0.85)):]
    corners = []
    if c1.size > 0: corners.append(c1.reshape(-1, 3))
    if c2.size > 0: corners.append(c2.reshape(-1, 3))
    if corners:
        bg_bgr = np.median(np.vstack(corners), axis=0).astype(np.float32)
    else:
        bg_bgr = np.array([240.0, 240.0, 240.0], dtype=np.float32)

    return {
        'h': h, 'w': w,
        'chin': chin,
        'brows': brows, 'l_brow': l_brow, 'r_brow': r_brow,
        'eyes': eyes, 'l_eye': l_eye, 'r_eye': r_eye,
        'nose_tip': nose_tip, 'subnasale': subnasale, 'philtrum': philtrum,
        'lips': lips, 'top_lip': top_lip, 'bottom_lip': bottom_lip,
        'mouth_L': mouth_L, 'mouth_R': mouth_R,
        'pupil_L': pupil_L, 'pupil_R': pupil_R,
        'temple_L': temple_L, 'temple_R': temple_R,
        'ear_L': ear_L, 'ear_R': ear_R,
        'gonion_L': gonion_L, 'gonion_R': gonion_R,
        'chin_tip': chin_tip,
        'nasion': nasion, 'mid_brow': mid_brow,
        'hairline_C': hairline_C, 'apex': apex,
        'brow_y': brow_y, 'chin_y': chin_y,
        'fh': fh, 'fw': fw,
        'left_x': left_x, 'right_x': right_x, 'mid_x': mid_x,
        'hairline_y': hairline_y, 'apex_y': apex_y, 'nose_base_y': nose_base_y,
        'angle': angle,
        'u_par': u_par, 'u_perp': u_perp,
        'skin_bgr': skin_bgr, 'bg_bgr': bg_bgr
    }


def get_color_palette(bgr, geom):
    """
    Returns sampled forehead skin tone and ambient background tone.
    """
    return geom['skin_bgr'], geom['bg_bgr']


# -------------------------------------------------------------------------
# STAGE 2.5: ANATOMICAL REGION SEGMENTATION MAP (PHASE 4.3)
# -------------------------------------------------------------------------

def build_anatomical_region_map(bgr, geom):
    """
    Phase 4.3: Constructs explicit, multi-region anatomical segmentation masks (Section 3):
      1. face_skin        2. forehead        3. scalp_head      4. hair
      5. left_eyebrow     6. right_eyebrow   7. left_eye        8. right_eye
      9. nose            10. mouth_lips     11. moustache_region 12. beard_jaw_region
     13. left_cheek      14. right_cheek    15. chin           16. ears
     17. neck            18. background
    """
    h, w = geom['h'], geom['w']
    fh, fw = geom['fh'], geom['fw']
    u_par, u_perp = geom['u_par'], geom['u_perp']

    chin = geom['chin']
    l_brow = geom['l_brow']
    r_brow = geom['r_brow']
    brows = geom['brows']
    l_eye = geom['l_eye']
    r_eye = geom['r_eye']
    eyes = geom['eyes']
    nose_tip = geom['nose_tip']
    lips = geom['lips']
    top_lip = geom['top_lip']
    bottom_lip = geom['bottom_lip']
    mouth_L = geom['mouth_L']
    mouth_R = geom['mouth_R']
    subnasale = geom['subnasale']
    chin_tip = geom['chin_tip']
    hairline_C = geom['hairline_C']
    apex = geom['apex']
    temple_L = geom['temple_L']
    temple_R = geom['temple_R']
    ear_L = geom['ear_L']
    ear_R = geom['ear_R']
    gonion_L = geom['gonion_L']
    gonion_R = geom['gonion_R']
    mid_brow = geom['mid_brow']
    nasion = geom['nasion']

    masks = {}

    def empty():
        return np.zeros((h, w), dtype=np.float32)

    # ---------------------------------------------------------------------
    # PRIMARY: BiSeNet ONNX Face Parser (19-Class Segmentation)
    # ---------------------------------------------------------------------
    seg = run_face_parser_segmentation(bgr)
    if seg is not None and np.any(np.isin(seg, [1, 17])):
        # Authoritative parser masks
        masks['hair'] = cv2.GaussianBlur((seg == 17).astype(np.float32), (5, 5), 1.0)
        masks['face_skin'] = cv2.GaussianBlur((seg == 1).astype(np.float32), (5, 5), 1.0)
        masks['left_eyebrow'] = cv2.GaussianBlur((seg == 2).astype(np.float32), (5, 5), 1.0)
        masks['right_eyebrow'] = cv2.GaussianBlur((seg == 3).astype(np.float32), (5, 5), 1.0)
        masks['left_eye'] = (seg == 4).astype(np.float32)
        masks['right_eye'] = (seg == 5).astype(np.float32)
        masks['nose'] = cv2.GaussianBlur((seg == 10).astype(np.float32), (5, 5), 1.0)
        masks['mouth_lips'] = np.isin(seg, [11, 12, 13]).astype(np.float32)
        masks['ears'] = cv2.GaussianBlur(np.isin(seg, [7, 8]).astype(np.float32), (5, 5), 1.0)
        masks['neck'] = cv2.GaussianBlur((seg == 14).astype(np.float32), (5, 5), 1.0)
        masks['background'] = cv2.GaussianBlur((seg == 0).astype(np.float32), (5, 5), 1.0)

        # Anatomical sub-regions derived with landmark guidance
        brow_y = float(mid_brow[1])
        forehead_mask = ((seg == 1) & (np.arange(h)[:, None] < (brow_y + 0.05 * fh))).astype(np.float32)
        masks['forehead'] = cv2.GaussianBlur(forehead_mask, (7, 7), 1.5)
        masks['scalp_head'] = masks['hair'].copy()

        # Moustache space (strictly excluding lips)
        m = empty()
        moust_poly = np.array([
            subnasale - (0.01 * fh) * u_perp,
            mouth_L + (0.02 * fh) * u_perp,
            geom['top_lip'][3] - (0.005 * fh) * u_perp,
            mouth_R + (0.02 * fh) * u_perp
        ], dtype=np.float32)
        cv2.fillPoly(m, [cv2.convexHull(np.int32(moust_poly))], 1.0)
        m[masks['mouth_lips'] > 0.1] = 0.0
        masks['moustache_region'] = cv2.GaussianBlur(m, (5, 5), 1.2)

        # Chin
        m = empty()
        chin_center = chin_tip - 0.06 * fh * u_perp
        cv2.circle(m, (int(chin_center[0]), int(chin_center[1])), int(0.12 * fw), 1.0, -1)
        m[masks['mouth_lips'] > 0.1] = 0.0
        masks['chin'] = cv2.GaussianBlur(m, (11, 11), 3.0)

        # Beard / jawline
        m = empty()
        jaw_pts = chin[3:14]
        sub_jaw = [pt + 0.08 * fh * u_perp for pt in jaw_pts]
        beard_poly = np.vstack([jaw_pts, sub_jaw[::-1]])
        cv2.fillPoly(m, [np.int32(beard_poly)], 1.0)
        m[masks['mouth_lips'] > 0.1] = 0.0
        masks['beard_jaw_region'] = cv2.GaussianBlur(m, (11, 11), 3.5)

        # Cheeks
        l_eye_bot = l_eye[4]
        cheek_L_poly = np.array([
            l_eye_bot + 0.04 * fh * u_perp,
            subnasale - 0.04 * fw * u_par,
            mouth_L,
            gonion_L + 0.02 * fh * u_perp,
            temple_L + 0.05 * fh * u_perp
        ], dtype=np.float32)
        m = empty()
        cv2.fillPoly(m, [cv2.convexHull(np.int32(cheek_L_poly))], 1.0)
        m[masks['left_eye'] > 0.3] = 0.0
        m[masks['nose'] > 0.3] = 0.0
        m[masks['mouth_lips'] > 0.3] = 0.0
        masks['left_cheek'] = cv2.GaussianBlur(m, (15, 15), 4.0)

        r_eye_bot = r_eye[4]
        cheek_R_poly = np.array([
            r_eye_bot + 0.04 * fh * u_perp,
            subnasale + 0.04 * fw * u_par,
            mouth_R,
            gonion_R + 0.02 * fh * u_perp,
            temple_R + 0.05 * fh * u_perp
        ], dtype=np.float32)
        m = empty()
        cv2.fillPoly(m, [cv2.convexHull(np.int32(cheek_R_poly))], 1.0)
        m[masks['right_eye'] > 0.3] = 0.0
        m[masks['nose'] > 0.3] = 0.0
        m[masks['mouth_lips'] > 0.3] = 0.0
        masks['right_cheek'] = cv2.GaussianBlur(m, (15, 15), 4.0)

        masks['skin'] = masks['face_skin']
        return masks

    # ---------------------------------------------------------------------
    # FALLBACK: Anatomical Landmark Geometry Spline Mapper (Phase 4.3)
    # ---------------------------------------------------------------------

    # 1. Left Eyebrow
    m = empty()
    cv2.fillPoly(m, [cv2.convexHull(np.int32(l_brow))], 1.0)
    masks['left_eyebrow'] = cv2.GaussianBlur(m, (5, 5), 1.0)

    # 2. Right Eyebrow
    m = empty()
    cv2.fillPoly(m, [cv2.convexHull(np.int32(r_brow))], 1.0)
    masks['right_eyebrow'] = cv2.GaussianBlur(m, (5, 5), 1.0)

    # 3. Left Eye
    m = empty()
    cv2.fillPoly(m, [cv2.convexHull(np.int32(l_eye))], 1.0)
    masks['left_eye'] = m

    # 4. Right Eye
    m = empty()
    cv2.fillPoly(m, [cv2.convexHull(np.int32(r_eye))], 1.0)
    masks['right_eye'] = m

    # 5. Nose
    m = empty()
    nose_poly = np.vstack([nasion, nose_tip])
    cv2.fillPoly(m, [cv2.convexHull(np.int32(nose_poly))], 1.0)
    masks['nose'] = cv2.GaussianBlur(m, (7, 7), 1.5)

    # 6. Mouth / Lips
    m = empty()
    cv2.fillPoly(m, [cv2.convexHull(np.int32(lips))], 1.0)
    masks['mouth_lips'] = m

    # Hairline arc for forehead & scalp
    hl_curve = []
    for deg in np.linspace(0, 180, 25):
        rad = np.radians(deg)
        pt = hairline_C + (0.46 * fw * np.cos(rad)) * u_par - (0.04 * fh * np.sin(rad)) * u_perp
        hl_curve.append(pt)
    hl_curve = np.array(hl_curve, dtype=np.float32)

    # 7. Forehead (between brows and hairline)
    m = empty()
    brow_curve = brows[np.argsort(brows[:, 0])]
    fh_poly = np.vstack([brow_curve, hl_curve[::-1]])
    cv2.fillPoly(m, [np.int32(fh_poly)], 1.0)
    m[masks['left_eyebrow'] > 0.4] = 0.0
    m[masks['right_eyebrow'] > 0.4] = 0.0
    masks['forehead'] = cv2.GaussianBlur(m, (9, 9), 2.0)

    # 8. Scalp / Head (cranial vault above hairline)
    m = empty()
    scalp_arc = []
    for deg in np.linspace(15, 165, 25):
        rad = np.radians(deg)
        pt = hairline_C + (0.48 * fw * np.cos(rad)) * u_par - (0.42 * fh * np.sin(rad)) * u_perp
        scalp_arc.append(pt)
    scalp_arc = np.array(scalp_arc, dtype=np.float32)
    scalp_poly = np.vstack([hl_curve, scalp_arc[::-1]])
    cv2.fillPoly(m, [np.int32(scalp_poly)], 1.0)
    masks['scalp_head'] = cv2.GaussianBlur(m, (11, 11), 3.0)

    # 9. Hair (overall hair envelope)
    m = empty()
    top_poly = np.array([
        [0, 0], [w, 0],
        [w, int(temple_R[1] + 0.15 * fh)],
        [int(temple_R[0] + 0.25 * fw), int(temple_R[1] + 0.15 * fh)],
        [int(hairline_C[0] + 0.46 * fw * u_par[0]), int(hairline_C[1] + 0.46 * fw * u_par[1])],
        [int(hairline_C[0] - 0.46 * fw * u_par[0]), int(hairline_C[1] - 0.46 * fw * u_par[1])],
        [int(temple_L[0] - 0.25 * fw), int(temple_L[1] + 0.15 * fh)],
        [0, int(temple_L[1] + 0.15 * fh)]
    ], dtype=np.int32)
    cv2.fillPoly(m, [top_poly], 1.0)
    face_poly = np.vstack([chin, hl_curve])
    cv2.fillPoly(m, [np.int32(face_poly)], 0.0)
    masks['hair'] = cv2.GaussianBlur(m, (15, 15), 5.0)

    # 10. Moustache Region (philtrum space)
    m = empty()
    moust_poly = np.array([
        subnasale - (0.01 * fh) * u_perp,
        mouth_L + (0.02 * fh) * u_perp,
        geom['top_lip'][3] - (0.005 * fh) * u_perp,
        mouth_R + (0.02 * fh) * u_perp
    ], dtype=np.float32)
    cv2.fillPoly(m, [cv2.convexHull(np.int32(moust_poly))], 1.0)
    m[masks['mouth_lips'] > 0.3] = 0.0
    masks['moustache_region'] = cv2.GaussianBlur(m, (5, 5), 1.2)

    # 11. Chin (mental prominence)
    m = empty()
    chin_center = chin_tip - 0.06 * fh * u_perp
    cv2.circle(m, (int(chin_center[0]), int(chin_center[1])), int(0.12 * fw), 1.0, -1)
    m[masks['mouth_lips'] > 0.3] = 0.0
    masks['chin'] = cv2.GaussianBlur(m, (11, 11), 3.0)

    # 12. Beard / Jaw Region
    m = empty()
    jaw_pts = chin[3:14]
    sub_jaw = [pt + 0.08 * fh * u_perp for pt in jaw_pts]
    beard_poly = np.vstack([jaw_pts, sub_jaw[::-1]])
    cv2.fillPoly(m, [np.int32(beard_poly)], 1.0)
    m[masks['mouth_lips'] > 0.3] = 0.0
    masks['beard_jaw_region'] = cv2.GaussianBlur(m, (11, 11), 3.5)

    # 13. Left Cheek
    m = empty()
    l_eye_bot = l_eye[4]
    cheek_L_poly = np.array([
        l_eye_bot + 0.04 * fh * u_perp,
        subnasale - 0.04 * fw * u_par,
        mouth_L,
        gonion_L + 0.02 * fh * u_perp,
        temple_L + 0.05 * fh * u_perp
    ], dtype=np.float32)
    cv2.fillPoly(m, [cv2.convexHull(np.int32(cheek_L_poly))], 1.0)
    m[masks['left_eye'] > 0.3] = 0.0
    m[masks['nose'] > 0.3] = 0.0
    m[masks['mouth_lips'] > 0.3] = 0.0
    masks['left_cheek'] = cv2.GaussianBlur(m, (15, 15), 4.0)

    # 14. Right Cheek
    m = empty()
    r_eye_bot = r_eye[4]
    cheek_R_poly = np.array([
        r_eye_bot + 0.04 * fh * u_perp,
        subnasale + 0.04 * fw * u_par,
        mouth_R,
        gonion_R + 0.02 * fh * u_perp,
        temple_R + 0.05 * fh * u_perp
    ], dtype=np.float32)
    cv2.fillPoly(m, [cv2.convexHull(np.int32(cheek_R_poly))], 1.0)
    m[masks['right_eye'] > 0.3] = 0.0
    m[masks['nose'] > 0.3] = 0.0
    m[masks['mouth_lips'] > 0.3] = 0.0
    masks['right_cheek'] = cv2.GaussianBlur(m, (15, 15), 4.0)

    # 15. Ears
    m = empty()
    cv2.circle(m, (int(ear_L[0]), int(ear_L[1])), int(0.10 * fw), 1.0, -1)
    cv2.circle(m, (int(ear_R[0]), int(ear_R[1])), int(0.10 * fw), 1.0, -1)
    masks['ears'] = cv2.GaussianBlur(m, (11, 11), 3.0)

    # 16. Neck
    m = empty()
    neck_pts = [pt + 0.25 * fh * u_perp for pt in chin[4:13]]
    neck_poly = np.vstack([chin[4:13], neck_pts[::-1]])
    cv2.fillPoly(m, [np.int32(neck_poly)], 1.0)
    masks['neck'] = cv2.GaussianBlur(m, (15, 15), 4.5)

    # 17. Face Skin
    m = np.clip(
        masks['forehead'] + masks['left_cheek'] + masks['right_cheek'] +
        masks['chin'] + masks['beard_jaw_region'] + masks['moustache_region'],
        0.0, 1.0
    )
    m[masks['left_eye'] > 0.2] = 0.0
    m[masks['right_eye'] > 0.2] = 0.0
    m[masks['left_eyebrow'] > 0.2] = 0.0
    m[masks['right_eyebrow'] > 0.2] = 0.0
    m[masks['mouth_lips'] > 0.2] = 0.0
    masks['face_skin'] = cv2.GaussianBlur(m, (9, 9), 2.0)

    # 18. Background
    m = np.ones((h, w), dtype=np.float32)
    m[masks['hair'] > 0.3] = 0.0
    m[masks['scalp_head'] > 0.3] = 0.0
    m[masks['face_skin'] > 0.3] = 0.0
    m[masks['neck'] > 0.3] = 0.0
    masks['background'] = cv2.GaussianBlur(m, (15, 15), 5.0)
    masks['skin'] = masks['face_skin']
    return masks


# -------------------------------------------------------------------------
# STAGE 3: WARPING ENGINE (PIECEWISE AFFINE VIA DELAUNAY)
# -------------------------------------------------------------------------

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append((int(t2[i][0] - r2[0]), int(t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 4), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])

    if img1_rect.shape[0] == 0 or img1_rect.shape[1] == 0 or size[0] == 0 or size[1] == 0:
        return

    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    y1, y2 = r2[1], r2[1] + r2[3]
    x1, x2 = r2[0], r2[0] + r2[2]
    if y1 < 0 or x1 < 0 or y2 > img2.shape[0] or x2 > img2.shape[1]:
        return

    img2[y1:y2, x1:x2] = img2[y1:y2, x1:x2] * (1 - mask) + img2_rect * mask


def piecewise_affine_warp(src_img, src_pts, dst_pts, out_shape):
    """
    Fast CPU Delaunay piecewise affine warping (~40ms).
    Triangulates destination geometry to ensure seamless, continuous coverage.
    """
    out_img = np.zeros(out_shape, dtype=src_img.dtype)
    tri = scipy.spatial.Delaunay(dst_pts)
    for simplex in tri.simplices:
        t_src = [src_pts[simplex[0]], src_pts[simplex[1]], src_pts[simplex[2]]]
        t_dst = [dst_pts[simplex[0]], dst_pts[simplex[1]], dst_pts[simplex[2]]]
        warp_triangle(src_img, out_img, t_src, t_dst)
    return out_img


def decontaminate_asset_fringe(asset, is_dark_hair=True):
    """
    Eliminates dark/black halos and white/gray fringe pixels from RGBA assets.
    Propagates pure interior color into semi-transparent border pixels.
    """
    bgr = asset[:, :, :3].copy()
    alpha = asset[:, :, 3].astype(np.float32) / 255.0

    if is_dark_hair:
        core_mask = (alpha > 0.70) & (np.mean(bgr, axis=2) < 130)
    else:
        core_mask = (alpha > 0.70)

    if np.sum(core_mask) == 0:
        core_mask = (alpha > 0.70)

    if is_dark_hair:
        bad_fringe = (alpha > 0.02) & ((alpha < 0.85) | (np.mean(bgr, axis=2) > 115) | (np.mean(bgr, axis=2) < 15))
    else:
        bad_fringe = (alpha > 0.02) & (alpha < 0.85)

    inpaint_mask = bad_fringe.astype(np.uint8) * 255
    if np.sum(core_mask) > 100 and np.sum(inpaint_mask) > 0:
        clean_bgr = cv2.inpaint(bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    else:
        clean_bgr = bgr.copy()

    a_norm = np.where(alpha < 0.04, 0.0, alpha)
    a_norm = np.clip((a_norm - 0.04) / 0.96, 0.0, 1.0)
    a_clean = a_norm * a_norm * (3.0 - 2.0 * a_norm)

    return np.dstack([clean_bgr, (a_clean * 255.0).astype(np.uint8)])


# -------------------------------------------------------------------------
# STAGE 4: CLEAN SHAVEN RECONSTRUCTION
# -------------------------------------------------------------------------

def clean_shave_beard_if_needed(canvas, geom, facial_hair, skin_bgr, regions=None):
    """
    Phase 4.3: Clean Shaven Reconstruction.
    When 'Clean Shaven' is selected, detects dense facial hair across chin, jaw, and moustache.
    Uses anatomical region masks (moustache_region, beard_jaw_region, chin) to suppress hair
    and reconstruct smooth skin texture while strictly protecting lip vermilion.
    """
    if facial_hair != 'Clean Shaven':
        return canvas

    h, w = geom['h'], geom['w']
    chin = geom['chin']
    fh, fw = geom['fh'], geom['fw']
    u_par = geom['u_par']
    u_perp = geom['u_perp']
    subnasale = geom['subnasale']

    chin_patch = canvas[int(geom['chin_tip'][1] - 0.22 * fh):int(geom['chin_tip'][1] - 0.04 * fh),
                        max(0, int(geom['chin_tip'][0] - 0.18 * fw)):min(w, int(geom['chin_tip'][0] + 0.18 * fw))]
    if chin_patch.size == 0:
        return canvas

    chin_gray = cv2.cvtColor(chin_patch, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.mean(chin_gray < 50)
    p10 = np.percentile(chin_gray, 10)
    if dark_ratio < 0.15 and p10 > 45:
        # Subject is already clean shaven
        return canvas

    # Target shave region from anatomical masks if available, else polygon
    if regions is not None and 'beard_jaw_region' in regions:
        shave_float = np.clip(regions['beard_jaw_region'] + regions['chin'] + regions['moustache_region'], 0.0, 1.0)
        shave_mask = (shave_float > 0.15).astype(np.uint8) * 255
    else:
        shave_mask = np.zeros((h, w), dtype=np.uint8)
        jaw_pts = chin[0:17]
        neck_pts = []
        for pt in jaw_pts[3:14]:
            neck_pts.append(pt + (0.10 * fh) * u_perp)
        neck_pts = np.array(neck_pts[::-1], dtype=np.float32)
        cheek_L = geom['mouth_L'] + (0.04 * fh) * u_perp - (0.08 * fw) * u_par
        cheek_R = geom['mouth_R'] + (0.04 * fh) * u_perp + (0.08 * fw) * u_par
        philtrum_top = subnasale - (0.04 * fh) * u_perp
        poly = np.vstack([jaw_pts[:2], [cheek_L], [philtrum_top], [cheek_R], jaw_pts[14:], neck_pts])
        cv2.fillPoly(shave_mask, [np.int32(poly)], 255)

    # Protect lips vermilion
    lip_hull = cv2.convexHull(np.int32(geom['lips']))
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lip_hull], 255)
    shave_mask[lip_mask > 0] = 0

    # Sample left and right cheek patches for spatially balanced skin tone
    cheek_L = canvas[int(geom['pupil_L'][1] + 0.15*fh):int(geom['pupil_L'][1] + 0.35*fh),
                     int(geom['pupil_L'][0] - 0.12*fw):int(geom['pupil_L'][0] + 0.12*fw)]
    cheek_R = canvas[int(geom['pupil_R'][1] + 0.15*fh):int(geom['pupil_R'][1] + 0.35*fh),
                     int(geom['pupil_R'][0] - 0.12*fw):int(geom['pupil_R'][0] + 0.12*fw)]

    skin_L = np.median(cheek_L, axis=(0, 1)) if cheek_L.size > 0 else skin_bgr
    skin_R = np.median(cheek_R, axis=(0, 1)) if cheek_R.size > 0 else skin_bgr

    lab_L = cv2.cvtColor(np.uint8([[skin_L]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab_R = cv2.cvtColor(np.uint8([[skin_R]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    # Spatially varying skin tone across face width
    span_x = max(10.0, geom['pupil_R'][0] - geom['pupil_L'][0])
    ref_lab_field = np.zeros((h, w, 3), dtype=np.float32)
    for col_x in range(w):
        wt_R = np.clip((col_x - geom['pupil_L'][0]) / span_x, 0.0, 1.0)
        ref_lab_field[:, col_x, :] = (1.0 - wt_R) * lab_L + wt_R * lab_R

    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
    bp = (shave_mask > 0)
    dark_hair = bp & (lab[:, :, 0] < ref_lab_field[:, :, 0] * 0.88)

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    inpaint_mask[dark_hair] = 255
    clean_bgr = cv2.inpaint(canvas, inpaint_mask, 7, cv2.INPAINT_TELEA)

    clean_lab = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    clean_lab[:, :, 0][bp] = 0.35 * clean_lab[:, :, 0][bp] + 0.65 * ref_lab_field[:, :, 0][bp]
    clean_lab[:, :, 1][bp] = 0.20 * clean_lab[:, :, 1][bp] + 0.80 * ref_lab_field[:, :, 1][bp]
    clean_lab[:, :, 2][bp] = 0.20 * clean_lab[:, :, 2][bp] + 0.80 * ref_lab_field[:, :, 2][bp]

    smoothed_bgr = cv2.cvtColor(np.clip(clean_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    smoothed = cv2.bilateralFilter(smoothed_bgr, 7, 25, 25)

    skin_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    texture_hp = (skin_gray.astype(np.float32) - cv2.GaussianBlur(skin_gray.astype(np.float32), (5, 5), 1.0))
    textured = np.clip(smoothed.astype(np.float32) + texture_hp[:, :, np.newaxis] * 0.25, 0, 255).astype(np.uint8)

    mask_f = cv2.GaussianBlur(shave_mask.astype(np.float32) / 255.0, (15, 15), 5.0)
    mf3 = np.dstack([mask_f] * 3)
    return (textured.astype(np.float32) * mf3 + canvas.astype(np.float32) * (1.0 - mf3)).astype(np.uint8)


# -------------------------------------------------------------------------
# STAGE 5: ORIGINAL HAIR SUPPRESSION
# -------------------------------------------------------------------------

def suppress_original_hair(canvas, geom, hair_style, headwear, bg_bgr, skin_bgr=None, regions=None):
    """
    Phase 4.6: Precise Semantic Hair Suppression.
    Uses authoritative BiSeNet parser hair mask to completely suppress subject's original
    hair into a smooth, gradient-matched background plate.
    - If Keep Original: original hair is 100% preserved.
    - If replacing hairstyle (Short, Long, Straight, Wavy, Curly, Bald):
      1. Obtains precise hair segmentation (regions['hair'] minus face skin/brows/eyes).
      2. Dilates mask by 9px to capture outer strand boundaries and suppress tall pompadours/curls.
      3. Inpaints and blends with clean background plate.
      4. For Bald: also blends cranial dome with skin tone.
      5. Forehead, eyebrows, eyes, and facial skin are strictly protected.
    """
    if hair_style == 'Keep Original':
        return canvas

    if skin_bgr is None:
        skin_bgr = np.array([160.0, 180.0, 205.0], dtype=np.float32)

    h, w = geom['h'], geom['w']
    chin = geom['chin']
    hairline_C = geom['hairline_C']
    fw = geom['fw']
    fh = geom['fh']
    u_par = geom['u_par']
    u_perp = geom['u_perp']

    # Sample true background from outer canvas margins
    col_L = np.median(canvas[:, :15], axis=1).astype(np.float32)
    col_R = np.median(canvas[:, -15:], axis=1).astype(np.float32)
    bg_plate = np.zeros_like(canvas, dtype=np.float32)
    for col_x in range(w):
        wt_R = float(col_x) / float(max(1, w - 1))
        bg_plate[:, col_x] = (1.0 - wt_R) * col_L + wt_R * col_R

    if regions is not None and 'hair' in regions and np.sum(regions['hair'] > 0.20) > 100:
        hair_m = regions['hair'].copy()
        if 'face_skin' in regions:
            hair_m[regions['face_skin'] > 0.30] = 0.0
        if 'left_eyebrow' in regions:
            hair_m[regions['left_eyebrow'] > 0.25] = 0.0
        if 'right_eyebrow' in regions:
            hair_m[regions['right_eyebrow'] > 0.25] = 0.0
        if 'left_eye' in regions:
            hair_m[regions['left_eye'] > 0.25] = 0.0
        if 'right_eye' in regions:
            hair_m[regions['right_eye'] > 0.25] = 0.0

        clean_hair_bin = (hair_m > 0.20).astype(np.uint8)
        if hair_style == 'Bald':
            scalp_skin = np.zeros_like(canvas, dtype=np.float32)
            for r in range(h):
                frac = np.clip((float(r) - (hairline_C[1] - 0.25 * fh)) / (0.35 * fh), 0.70, 1.0)
                scalp_skin[r, :] = skin_bgr * frac
            hair_blur = cv2.GaussianBlur(clean_hair_bin.astype(np.float32), (11, 11), 3.0)[:, :, np.newaxis]
            return (canvas.astype(np.float32) * (1.0 - hair_blur) + scalp_skin * hair_blur).astype(np.uint8)
        dil_hair = cv2.dilate(clean_hair_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        suppress_mask = cv2.GaussianBlur(dil_hair.astype(np.float32), (15, 15), 4.0)[:, :, np.newaxis]
    else:
        # Fallback to cranial landmark geometry
        face_mask = np.zeros((h, w), dtype=np.uint8)
        forehead_curve = []
        for deg in np.linspace(0, 180, 25):
            rad = np.radians(deg)
            px = int(hairline_C[0] + (fw * 0.46) * np.cos(rad) * u_par[0] + (fh * 0.02) * np.sin(rad) * u_perp[0])
            py = int(hairline_C[1] + (fw * 0.46) * np.cos(rad) * u_par[1] + (fh * 0.02) * np.sin(rad) * u_perp[1])
            forehead_curve.append([px, py])
        forehead_curve = np.array(forehead_curve, dtype=np.int32)
        face_poly = np.vstack([chin, forehead_curve]).astype(np.int32)
        cv2.fillPoly(face_mask, [face_poly], 255)

        suppress_m = np.zeros((h, w), dtype=np.float32)
        top_poly = np.array([
            [0, 0], [w, 0],
            [w, int(geom['temple_R'][1] + 0.12 * fh)],
            [int(geom['temple_R'][0] + 0.25 * fw), int(geom['temple_R'][1] + 0.12 * fh)],
            [int(hairline_C[0] + 0.46 * fw * u_par[0]), int(hairline_C[1] + 0.46 * fw * u_par[1])],
            [int(hairline_C[0] - 0.46 * fw * u_par[0]), int(hairline_C[1] - 0.46 * fw * u_par[1])],
            [int(geom['temple_L'][0] - 0.25 * fw), int(geom['temple_L'][1] + 0.12 * fh)],
            [0, int(geom['temple_L'][1] + 0.12 * fh)]
        ], dtype=np.int32)
        cv2.fillPoly(suppress_m, [top_poly], 1.0)
        suppress_m[face_mask > 0] = 0.0
        suppress_mask = cv2.GaussianBlur(suppress_m, (15, 15), 5.0)[:, :, np.newaxis]

    return (canvas.astype(np.float32) * (1.0 - suppress_mask) + bg_plate * suppress_mask).astype(np.uint8)


# -------------------------------------------------------------------------
# STAGE 6: SCALP & BALD CRANIAL VAULT RECONSTRUCTION
# -------------------------------------------------------------------------

def render_bald_appearance(canvas, geom, skin_bgr, bg_bgr):
    """
    Phase 4.2.2: Renders realistic bald appearance:
      1. Dense contour mapping of photographic cranial scalp asset.
      2. Natural cranial vault geometry oriented to subject's head roll angle theta.
      3. LAB color and luminance adaptation matching subject's exact forehead skin tone.
      4. Seamless exponential blend into upper forehead at mid-brow.
      5. Elimination of residual hair fragments and artificial dome boundaries.
    """
    bald_path = get_asset_path('bald.png')
    if not os.path.exists(bald_path):
        return canvas

    bald_asset = cv2.imread(bald_path, cv2.IMREAD_UNCHANGED)
    if bald_asset is None:
        return canvas

    h, w = geom['h'], geom['w']
    fw = geom['fw']
    fh = geom['fh']
    u_par = geom['u_par']
    u_perp = geom['u_perp']
    hairline_C = geom['hairline_C']
    apex = geom['apex']
    mid_brow = geom['mid_brow']

    b_bgr = bald_asset[:, :, :3]
    b_alpha = bald_asset[:, :, 3].astype(np.float32) / 255.0
    b_lab = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[skin_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    # Color match in LAB space
    core = b_alpha > 0.3
    cur_L = np.median(b_lab[:, :, 0][core])
    cur_A = np.median(b_lab[:, :, 1][core])
    cur_B = np.median(b_lab[:, :, 2][core])

    b_lab[:, :, 0] = np.clip(b_lab[:, :, 0] * (skin_lab[0] / max(1.0, cur_L)), 10.0, 240.0)
    b_lab[:, :, 1] = np.clip(b_lab[:, :, 1] + (skin_lab[1] - cur_A), 0, 255)
    b_lab[:, :, 2] = np.clip(b_lab[:, :, 2] + (skin_lab[2] - cur_B), 0, 255)
    matched_bgr = cv2.cvtColor(b_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    matched_asset = np.dstack([matched_bgr, bald_asset[:, :, 3]])

    # Dense contour sampling along perimeter of bald asset (avoids chord clipping)
    cnts, _ = cv2.findContours((bald_asset[:, :, 3] > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return canvas
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    indices = np.linspace(0, len(c) - 1, 48, dtype=int)
    src_contour = c[indices].astype(np.float32)

    src_cx, src_cy = 512.0, 530.0
    dst_cx = hairline_C[0] - 0.06 * fh * u_perp[0]
    dst_cy = hairline_C[1] - 0.06 * fh * u_perp[1]
    target_rx = fw * 0.54
    target_ry = fh * 0.46

    dst_contour = []
    for pt in src_contour:
        adx = (pt[0] - src_cx) / 415.0
        ady = (pt[1] - src_cy) / 415.0
        dst_x = dst_cx + (adx * target_rx) * u_par[0] + (ady * target_ry) * u_perp[0]
        dst_y = dst_cy + (adx * target_rx) * u_par[1] + (ady * target_ry) * u_perp[1]
        dst_contour.append([dst_x, dst_y])
    dst_contour = np.array(dst_contour, dtype=np.float32)

    # Interior anchors
    src_interior = np.array([[512, 530], [512, 250], [512, 800], [300, 530], [724, 530]], dtype=np.float32)
    dst_interior = np.array([
        [dst_cx, dst_cy],
        [dst_cx - 0.28 * target_ry * u_perp[0], dst_cy - 0.28 * target_ry * u_perp[1]],
        mid_brow - 0.02 * fh * u_perp,
        [dst_cx - 0.50 * target_rx * u_par[0], dst_cy - 0.50 * target_rx * u_par[1]],
        [dst_cx + 0.50 * target_rx * u_par[0], dst_cy + 0.50 * target_rx * u_par[1]]
    ], dtype=np.float32)

    # Outer envelope corners
    src_env = np.array([[0, 0], [1024, 0], [0, 1024], [1024, 1024]], dtype=np.float32)
    dst_env = np.array([
        apex - 0.40 * fh * u_perp - 0.90 * fw * u_par,
        apex - 0.40 * fh * u_perp + 0.90 * fw * u_par,
        mid_brow + 0.40 * fh * u_perp - 0.90 * fw * u_par,
        mid_brow + 0.40 * fh * u_perp + 0.90 * fw * u_par
    ], dtype=np.float32)

    src_pts = np.vstack([src_contour, src_interior, src_env])
    dst_pts = np.vstack([dst_contour, dst_interior, dst_env])

    warped = piecewise_affine_warp(matched_asset, src_pts, dst_pts, (h, w, 4))
    w_bgr = warped[:, :, :3]
    w_alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # Seamless linear fade into upper forehead between hairline and mid-brow
    mb_y = int(mid_brow[1])
    hl_y = int(hairline_C[1])
    fade_top = hl_y - int(0.02 * fh)
    fade_bot = mb_y - int(0.01 * fh)
    for r in range(max(0, fade_top), min(h, fade_bot)):
        factor = 1.0 - float(r - fade_top) / float(max(1, fade_bot - fade_top))
        w_alpha[r, :] *= factor
    w_alpha[fade_bot:, :] = 0.0

    w_alpha = cv2.GaussianBlur(w_alpha, (5, 5), 1.2)[:, :, np.newaxis]

    # Harmonize with subject skin pore grain
    skin_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture_hp = (skin_gray - cv2.GaussianBlur(skin_gray, (5, 5), 1.0))[:, :, np.newaxis]
    textured_bgr = np.clip(w_bgr.astype(np.float32) + texture_hp * 0.20, 0, 255)

    return (textured_bgr * w_alpha + canvas.astype(np.float32) * (1.0 - w_alpha)).astype(np.uint8)


# -------------------------------------------------------------------------
# STAGE 7: GEOMETRY-AWARE HAIRSTYLE WARPING
# -------------------------------------------------------------------------

def render_selected_hairstyle(canvas, geom, hair_style, is_cap_worn=False):
    """
    Phase 4.6: Photographic Hairstyle Synthesis with Anatomical Alignment.
    Features:
      - High-resolution photographic assets from static/variation_assets_photo/
      - Person-specific 2D affine warping conforming to head roll angle theta
      - Independent width (sx) and height (sy) scaling matching skull dimensions
      - Anatomical anchoring to Trichion hairline center (and nasion for Curly bangs)
      - Individual ocular, nasal, oral, and brow protection (no cheek cutting)
      - Micro-follicle grain at hairline forehead transition zone
      - Local LAB color & ambient lighting harmonization (source face untouched)
      - Cap occlusion: neatly layers under cap crown when headwear is selected
    """
    if hair_style in ['Keep Original', 'Bald']:
        return canvas

    fname_map = {
        'Short': 'short_hair.png',
        'Long': 'long_hair.png',
        'Straight': 'straight_hair.png',
        'Wavy': 'wavy_hair.png',
        'Curly': 'curly_hair.png'
    }
    asset_file = fname_map.get(hair_style)
    if not asset_file:
        return canvas

    asset_path = get_asset_path(asset_file)
    if not os.path.exists(asset_path):
        return canvas

    asset = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
    if asset is None:
        return canvas

    asset_clean = decontaminate_asset_fringe(asset, is_dark_hair=True)

    h, w = geom['h'], geom['w']
    fw = geom['fw']
    fh = geom['fh']
    angle = geom['angle']
    u_par = geom['u_par']
    u_perp = geom['u_perp']
    hairline_C = geom['hairline_C']
    chin_tip = geom['chin_tip']

    DONORS = {
        'Short': {'anchor': (512.0, 210.0), 'fw': 340.0, 'fh': 300.0, 'hairline_off': 0.0},
        'Curly': {'anchor': (512.0, 385.0), 'fw': 380.0, 'fh': 320.0, 'hairline_off': 0.12},
        'Long': {'anchor': (512.0, 135.0), 'fw': 270.0, 'fh': 290.0, 'hairline_off': -0.02},
        'Straight': {'anchor': (512.0, 175.0), 'fw': 330.0, 'fh': 310.0, 'hairline_off': -0.01},
        'Wavy': {'anchor': (512.0, 160.0), 'fw': 290.0, 'fh': 300.0, 'hairline_off': -0.02}
    }
    d = DONORS.get(hair_style, DONORS['Short'])

    # Independent scale factors matching skull proportions
    sx = (fw / d['fw']) * (0.95 if hair_style == 'Short' else 1.0)
    sy = (fh / d['fh']) * (0.95 if hair_style == 'Short' else 1.0)

    # Anatomical target anchor
    if hair_style == 'Curly':
        target_anchor = geom['nasion'] - 0.12 * fh * u_perp
    else:
        target_anchor = hairline_C + d['hairline_off'] * fh * u_perp

    dx, dy = d['anchor']
    tx, ty = target_anchor

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = cos_a * sx
    M[0, 1] = -sin_a * sy
    M[0, 2] = tx - (dx * M[0, 0] + dy * M[0, 1])

    M[1, 0] = sin_a * sx
    M[1, 1] = cos_a * sy
    M[1, 2] = ty - (dx * M[1, 0] + dy * M[1, 1])

    warped = cv2.warpAffine(asset_clean, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    w_bgr = warped[:, :, :3]
    w_alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # If cap is worn, occlude upper scalp hair under cap crown
    if is_cap_worn:
        cap_visor_y = int(hairline_C[1] + 0.02 * fh * u_perp[1])
        x_min_cap = max(0, int(hairline_C[0] - 0.35 * fw))
        x_max_cap = min(w, int(hairline_C[0] + 0.35 * fw))
        for r in range(0, min(h, cap_visor_y)):
            w_alpha[r, x_min_cap:x_max_cap] *= 0.08

    # Feature protection: protect left_eye, right_eye, nose_tip, lips, brows INDIVIDUALLY
    # (Never form a single convex hull connecting eyes to mouth across cheeks)
    feature_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(feature_mask, [cv2.convexHull(np.int32(geom['l_eye']))], 1.0)
    cv2.fillPoly(feature_mask, [cv2.convexHull(np.int32(geom['r_eye']))], 1.0)
    cv2.fillPoly(feature_mask, [cv2.convexHull(np.int32(geom['nose_tip']))], 1.0)
    cv2.fillPoly(feature_mask, [cv2.convexHull(np.int32(geom['lips']))], 1.0)
    cv2.fillPoly(feature_mask, [cv2.convexHull(np.int32(geom['brows']))], 1.0)
    feature_mask = cv2.dilate(feature_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    feature_mask = cv2.GaussianBlur(feature_mask, (7, 7), 1.5)
    clean_alpha = w_alpha * (1.0 - feature_mask)

    # Hairline Forehead Transition Zone: micro-follicle grain
    hl_y = int(target_anchor[1])
    band_top = max(0, int(hl_y - 0.05 * fh))
    band_bot = min(h, int(hl_y + 0.03 * fh))
    if band_bot > band_top:
        np.random.seed(42)
        grain = np.random.normal(1.0, 0.10, (band_bot - band_top, w)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (5, 5), 1.2)
        clean_alpha[band_top:band_bot, :] = np.clip(clean_alpha[band_top:band_bot, :] * grain, 0.0, 1.0)

    # Local LAB color & lighting harmonization (local adjustment on hair asset only)
    w_lab = cv2.cvtColor(w_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[geom['skin_bgr']]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[geom['bg_bgr']]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    ambient_A = 0.55 * skin_lab[1] + 0.45 * bg_lab[1]
    ambient_B = 0.55 * skin_lab[2] + 0.45 * bg_lab[2]
    w_lab[:, :, 1] = np.clip(w_lab[:, :, 1] + (ambient_A - 128.0) * 0.16, 0, 255)
    w_lab[:, :, 2] = np.clip(w_lab[:, :, 2] + (ambient_B - 128.0) * 0.16, 0, 255)
    harmonized_bgr = cv2.cvtColor(np.clip(w_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    ca3 = np.dstack([clean_alpha] * 3)
    return (harmonized_bgr.astype(np.float32) * ca3 + canvas.astype(np.float32) * (1.0 - ca3)).astype(np.uint8)


# -------------------------------------------------------------------------
# STAGE 8: FACIAL HAIR
# -------------------------------------------------------------------------

def render_selected_facial_hair(canvas, geom, facial_hair, regions=None):
    """
    Phase 4.6: Anatomically Aligned Facial Hair Synthesis.
      Clean Shaven: Handled in Stage 4
      Light / Heavy Stubble: Directional follicle grain along jawline & chin (preserved as visually acceptable)
      Moustache: Philtrum-aligned, upper lip curvature conforming, vermilion-protected
      Full Beard: Mandibular jaw contour conforming, mouth centered, sideburns tapered into ears
      Beard + Moustache: Harmonized combination
    """
    if facial_hair in ['Keep Original', 'Clean Shaven']:
        return canvas

    h, w = geom['h'], geom['w']
    chin = geom['chin']
    fh, fw = geom['fh'], geom['fw']
    u_par = geom['u_par']
    u_perp = geom['u_perp']
    skin_bgr = geom['skin_bgr']

    if facial_hair in ['Light Stubble', 'Heavy Stubble']:
        stubble_mask = np.zeros((h, w), dtype=np.float32)
        beard_pts = chin[2:15]
        top_y = geom['subnasale'] + (0.04 * fh) * u_perp
        poly = np.vstack([beard_pts, [top_y + (0.20 * fw) * u_par, top_y - (0.20 * fw) * u_par]])
        cv2.fillPoly(stubble_mask, [np.int32(poly)], 1.0)

        lip_hull = cv2.convexHull(np.int32(geom['lips']))
        cv2.fillPoly(stubble_mask, [lip_hull], 0.0)
        stubble_mask = cv2.GaussianBlur(stubble_mask, (15, 15), 4.5)[:, :, np.newaxis]

        np.random.seed(42)
        noise = np.random.normal(0, 10.0, (h, w, 1)).astype(np.float32)
        alpha_val = 0.35 if facial_hair == 'Light Stubble' else 0.55
        darkened = np.clip(canvas.astype(np.float32) * (0.80 if facial_hair == 'Light Stubble' else 0.65) + noise, 10.0, 240.0)
        canvas = (canvas.astype(np.float32) * (1.0 - stubble_mask * alpha_val) + darkened * (stubble_mask * alpha_val)).astype(np.uint8)

    if facial_hair in ['Moustache', 'Beard + Moustache']:
        m_asset = cv2.imread(get_asset_path('moustache.png'), cv2.IMREAD_UNCHANGED)
        if m_asset is not None:
            m_clean = decontaminate_asset_fringe(m_asset, is_dark_hair=True)
            mouth_c = (geom['top_lip'][3] + geom['bottom_lip'][3]) / 2.0
            subnasale = geom['subnasale']

            target_w = int(fw * 0.58)
            scale = target_w / float(m_clean.shape[1])
            target_h = int(m_clean.shape[0] * scale)

            resized = cv2.resize(m_clean, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            rot_mat = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), -np.degrees(geom['angle']), 1.0)
            rotated = cv2.warpAffine(resized, rot_mat, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            anchor_x = int(subnasale[0] - target_w / 2.0)
            anchor_y = int(subnasale[1] + 0.01 * fh - 310 * scale)

            bx1 = max(0, anchor_x)
            by1 = max(0, anchor_y)
            bx2 = min(w, anchor_x + target_w)
            by2 = min(h, anchor_y + target_h)

            ax1 = bx1 - anchor_x
            ay1 = by1 - anchor_y
            ax2 = ax1 + (bx2 - bx1)
            ay2 = ay1 + (by2 - by1)

            if bx2 > bx1 and by2 > by1:
                crop_m = rotated[ay1:ay2, ax1:ax2]
                m_bgr = crop_m[:, :, :3].astype(np.float32)
                m_alpha = (crop_m[:, :, 3].astype(np.float32) / 255.0)

                # Density gradient: thicker at philtrum, softer at corners
                philtrum_x = geom['philtrum'][0]
                for c in range(bx1, bx2):
                    dx = abs(c - philtrum_x) / (0.40 * fw)
                    falloff = np.clip(1.0 - 0.30 * (dx ** 1.5), 0.50, 1.0)
                    m_alpha[:, c - bx1] *= falloff

                # Protect top lip vermilion
                lip_hull = cv2.convexHull(np.int32(geom['top_lip']))
                lip_mask = np.zeros((h, w), dtype=np.float32)
                cv2.fillPoly(lip_mask, [lip_hull], 1.0)
                if regions is not None and 'mouth_lips' in regions:
                    lip_mask = np.maximum(lip_mask, regions['mouth_lips'])
                lip_mask = cv2.GaussianBlur(lip_mask, (7, 7), 2.0)
                m_alpha *= (1.0 - lip_mask[by1:by2, bx1:bx2])

                # Dark fringe tinting to eliminate white halo
                fringe = (m_alpha > 0.02) & (m_alpha < 0.90)
                dark_tone = np.array([25.0, 22.0, 20.0], dtype=np.float32)
                m_bgr[fringe] = m_bgr[fringe] * m_alpha[fringe, np.newaxis] + dark_tone * (1.0 - m_alpha[fringe, np.newaxis])

                ma3 = np.dstack([m_alpha] * 3)
                canvas[by1:by2, bx1:bx2] = (m_bgr * ma3 + canvas[by1:by2, bx1:bx2].astype(np.float32) * (1.0 - ma3)).astype(np.uint8)

    if facial_hair in ['Full Beard', 'Beard + Moustache']:
        chin_patch = canvas[int(geom['chin_tip'][1] - 0.22 * fh):int(geom['chin_tip'][1] - 0.04 * fh),
                            max(0, int(geom['chin_tip'][0] - 0.18 * fw)):min(w, int(geom['chin_tip'][0] + 0.18 * fw))]
        chin_gray = cv2.cvtColor(chin_patch, cv2.COLOR_BGR2GRAY) if chin_patch.size > 0 else np.array([120])
        already_bearded = (np.mean(chin_gray < 50) > 0.15 and np.percentile(chin_gray, 10) < 45)

        if not already_bearded:
            b_asset = cv2.imread(get_asset_path('full_beard.png'), cv2.IMREAD_UNCHANGED)
            if b_asset is not None:
                b_clean = decontaminate_asset_fringe(b_asset, is_dark_hair=True)
                mouth_c = (geom['top_lip'][3] + geom['bottom_lip'][3]) / 2.0
                ear_L = geom['ear_L']
                ear_R = geom['ear_R']

                target_w = int(fw * 1.15)
                scale = target_w / 810.0
                target_h = int(1024 * scale)

                resized = cv2.resize(b_clean, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                rot_mat = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), -np.degrees(geom['angle']), 1.0)
                rotated = cv2.warpAffine(resized, rot_mat, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                anchor_x = int(mouth_c[0] - target_w / 2.0)
                anchor_y = int(mouth_c[1] - 400 * scale)

                bx1 = max(0, anchor_x)
                by1 = max(0, anchor_y)
                bx2 = min(w, anchor_x + target_w)
                by2 = min(h, anchor_y + target_h)

                ax1 = bx1 - anchor_x
                ay1 = by1 - anchor_y
                ax2 = ax1 + (bx2 - bx1)
                ay2 = ay1 + (by2 - by1)

                if bx2 > bx1 and by2 > by1:
                    crop_b = rotated[ay1:ay2, ax1:ax2]
                    b_bgr = crop_b[:, :, :3].astype(np.float32)
                    b_alpha = (crop_b[:, :, 3].astype(np.float32) / 255.0)

                    # Eliminate halos on fringe pixels:
                    fringe = (b_alpha > 0.02) & (b_alpha < 0.92)
                    dark_tone = np.array([25.0, 22.0, 20.0], dtype=np.float32)
                    b_bgr[fringe] = b_bgr[fringe] * b_alpha[fringe, np.newaxis] + dark_tone * (1.0 - b_alpha[fringe, np.newaxis])

                    # Taper sideburns softly into ears
                    ear_y_local = int((ear_L[1] + ear_R[1]) / 2.0 - by1)
                    if 0 < ear_y_local < b_alpha.shape[0]:
                        for r in range(0, min(ear_y_local, b_alpha.shape[0])):
                            factor = float(r) / float(ear_y_local)
                            b_alpha[r, :] *= (factor ** 1.5)

                    # Feather upper cheek border into skin
                    subnasale_y_local = int(geom['subnasale'][1] - by1)
                    if 0 < subnasale_y_local < b_alpha.shape[0]:
                        for r in range(0, min(subnasale_y_local, b_alpha.shape[0])):
                            b_alpha[r, :] *= min(1.0, float(r) / float(max(1, subnasale_y_local)))

                    # Protect lips & mouth cavity
                    lip_hull = cv2.convexHull(np.int32(geom['lips']))
                    lip_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.fillPoly(lip_mask, [lip_hull], 1.0)
                    if regions is not None and 'mouth_lips' in regions:
                        lip_mask = np.maximum(lip_mask, regions['mouth_lips'])
                    lip_mask = cv2.GaussianBlur(lip_mask, (7, 7), 2.0)
                    b_alpha *= (1.0 - lip_mask[by1:by2, bx1:bx2])

                    # Adaptive luminance matching natural eyebrow tone
                    brow_pts = geom['brows'].astype(int)
                    brow_patch = canvas[np.min(brow_pts[:, 1]):np.max(brow_pts[:, 1])+1, np.min(brow_pts[:, 0]):np.max(brow_pts[:, 0])+1]
                    brow_gray = cv2.cvtColor(brow_patch, cv2.COLOR_BGR2GRAY) if brow_patch.size > 0 else np.array([45])
                    brow_lum = float(np.percentile(brow_gray, 20))
                    lum_factor = np.clip(brow_lum / 65.0, 0.50, 1.30)
                    b_bgr = np.clip(b_bgr * lum_factor, 0, 255)

                    ba3 = np.dstack([b_alpha] * 3)
                    canvas[by1:by2, bx1:bx2] = (b_bgr * ba3 + canvas[by1:by2, bx1:bx2].astype(np.float32) * (1.0 - ba3)).astype(np.uint8)

    return canvas


# -------------------------------------------------------------------------
# STAGE 9: HEADWEAR
# -------------------------------------------------------------------------

def render_selected_headwear(canvas, geom, headwear, regions=None):
    """
    Phase 4.3: Anatomically Positioned Headwear Synthesis.
      Cap: Sits strictly on cranial vault above brows, with forward visor pitch,
           eyebrow & ocular protection, and subtle forehead contact shadow.
      Turban: Wraps organically around cranial vault, parietal bulges, and temporal contour,
              conforming to head roll without obscuring eyes, eyebrows, nose, or lips.
    """
    if headwear in ['None', None]:
        return canvas

    h, w = geom['h'], geom['w']
    fw = geom['fw']
    fh = geom['fh']
    u_par = geom['u_par']
    u_perp = geom['u_perp']
    hairline_C = geom['hairline_C']
    apex = geom['apex']
    temple_L = geom['temple_L']
    temple_R = geom['temple_R']
    mid_brow = geom['mid_brow']

    angle = np.degrees(geom['angle'])
    if headwear == 'Cap':
        cap_asset = cv2.imread(get_asset_path('cap.png'), cv2.IMREAD_UNCHANGED)
        if cap_asset is not None:
            cap_clean = decontaminate_asset_fringe(cap_asset, is_dark_hair=False)
            th, tw = cap_clean.shape[:2]

            target_w = int(fw * 1.48)
            scale = target_w / float(tw)
            target_h = int(th * scale)

            resized = cv2.resize(cap_clean, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), -angle, 1.0)
            rot = cv2.warpAffine(resized, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            center_x = int((temple_L[0] + temple_R[0]) / 2.0)
            pos_y = int(mid_brow[1] - 0.05 * fh - target_h * 0.83)
            pos_x = int(center_x - target_w / 2.0)

            # Subtle contact shadow under visor on upper forehead
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(shadow_mask, (center_x, int(mid_brow[1] - 0.04 * fh)),
                        (int(fw * 0.40), int(fh * 0.020)), int(angle), 0, 180, 0.15, -1)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (9, 9), 2.5)
            comp_canvas = canvas.copy().astype(np.float32) * (1.0 - shadow_mask[:, :, np.newaxis])

            # Eyebrow / eye protection
            brow_eye_pts = np.vstack([geom['brows'], geom['eyes']])
            hull = cv2.convexHull(np.int32(brow_eye_pts))
            prot_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(prot_mask, [hull], 1.0)
            prot_mask = cv2.GaussianBlur(prot_mask, (11, 11), 3.0)

            for r in range(target_h):
                cy = pos_y + r
                if 0 <= cy < h:
                    for c in range(target_w):
                        cx = pos_x + c
                        if 0 <= cx < w:
                            alpha = (rot[r, c, 3] / 255.0) * (1.0 - prot_mask[cy, cx])
                            if alpha > 0.01:
                                comp_canvas[cy, cx] = rot[r, c, :3] * alpha + comp_canvas[cy, cx] * (1.0 - alpha)

            canvas = np.clip(comp_canvas, 0, 255).astype(np.uint8)

    elif headwear == 'Turban':
        turb_asset = cv2.imread(get_asset_path('turban.png'), cv2.IMREAD_UNCHANGED)
        if turb_asset is not None:
            turb_clean = decontaminate_asset_fringe(turb_asset, is_dark_hair=False)
            th, tw = turb_clean.shape[:2]

            target_w = int(fw * 1.55)
            scale = target_w / float(tw)
            target_h = int(th * scale)

            resized = cv2.resize(turb_clean, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), -angle, 1.0)
            rot = cv2.warpAffine(resized, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            center_x = int((temple_L[0] + temple_R[0]) / 2.0)
            pos_y = int(hairline_C[1] + 0.03 * fh - 555.0 * scale)
            pos_x = int(center_x - target_w / 2.0)

            # Subtle contact shadow under turban band on upper forehead
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(shadow_mask, (center_x, int(hairline_C[1] + 0.04 * fh)),
                        (int(fw * 0.44), int(fh * 0.020)), int(angle), 0, 180, 0.15, -1)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (9, 9), 2.5)
            comp_canvas = canvas.copy().astype(np.float32) * (1.0 - shadow_mask[:, :, np.newaxis])

            # Eyebrow / eye protection
            brow_eye_pts = np.vstack([geom['brows'], geom['eyes']])
            hull = cv2.convexHull(np.int32(brow_eye_pts))
            prot_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(prot_mask, [hull], 1.0)
            prot_mask = cv2.GaussianBlur(prot_mask, (5, 5), 1.5)

            for r in range(target_h):
                cy = pos_y + r
                if 0 <= cy < h:
                    for c in range(target_w):
                        cx = pos_x + c
                        if 0 <= cx < w:
                            alpha = (rot[r, c, 3] / 255.0) * (1.0 - prot_mask[cy, cx])
                            if alpha > 0.01:
                                comp_canvas[cy, cx] = rot[r, c, :3] * alpha + comp_canvas[cy, cx] * (1.0 - alpha)

            canvas = np.clip(comp_canvas, 0, 255).astype(np.uint8)

    return canvas


# -------------------------------------------------------------------------
# STAGE 10: AGE ADJUSTMENT (PHASE 4.3 REBUILD)
# -------------------------------------------------------------------------

def apply_age_adjustment(canvas, geom, age_delta, regions=None, current_age=25):
    """
    Phase 4.4: Learned Face Re-Aging Engine using FRAN ONNX model.
    Falls back gracefully to Phase 4.3 anatomical rhytid/dermal engine.
    """
    if age_delta == 0:
        return canvas

    if regions is None:
        regions = build_anatomical_region_map(canvas, geom)

    h, w = geom['h'], geom['w']
    fh, fw = geom['fh'], geom['fw']

    # ---------------------------------------------------------------------
    # PRIMARY: Learned Face Re-Aging Network (FRAN ONNX)
    # ---------------------------------------------------------------------
    fran_sess = get_fran_session()
    if fran_sess is not None:
        try:
            target_age = max(10, min(90, current_age + age_delta))
            chin_tip = geom['chin_tip']
            mid_brow = geom['mid_brow']
            nasion = geom['nasion']

            center_y = int((mid_brow[1] + chin_tip[1]) / 2.0)
            center_x = int(nasion[0])
            half_dim = int(max(fh, fw) * 0.85)

            y1 = max(0, center_y - int(half_dim * 1.15))
            y2 = min(h, center_y + int(half_dim * 0.95))
            x1 = max(0, center_x - half_dim)
            x2 = min(w, center_x + half_dim)

            crop = canvas[y1:y2, x1:x2].copy()
            ch_h, ch_w = crop.shape[:2]

            if ch_h >= 30 and ch_w >= 30:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_512 = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                img_norm = crop_512.astype(np.float32) / 255.0

                src_ch = np.full((512, 512, 1), current_age / 100.0, dtype=np.float32)
                tgt_ch = np.full((512, 512, 1), target_age / 100.0, dtype=np.float32)

                inp = np.concatenate([img_norm, src_ch, tgt_ch], axis=-1)
                inp = np.transpose(inp, (2, 0, 1))
                inp = np.expand_dims(inp, 0)

                inp_name = fran_sess.get_inputs()[0].name
                out_name = fran_sess.get_outputs()[0].name
                outs = fran_sess.run([out_name], {inp_name: inp})

                delta_512 = outs[0][0].transpose((1, 2, 0)) # (512, 512, 3) in RGB
                reaged_rgb_512 = np.clip(img_norm + delta_512 * 1.15, 0.0, 1.0)
                reaged_bgr_512 = cv2.cvtColor((reaged_rgb_512 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                reaged_crop = cv2.resize(reaged_bgr_512, (ch_w, ch_h), interpolation=cv2.INTER_LINEAR)

                # Spatial falloff boundary mask from FRAN assets
                mask512_path = os.path.join(os.path.dirname(__file__), 'model', 'face_reaging', 'mask512.jpg')
                if os.path.exists(mask512_path):
                    m512 = cv2.imread(mask512_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    spatial_falloff = cv2.resize(m512, (ch_w, ch_h), interpolation=cv2.INTER_LINEAR)
                else:
                    spatial_falloff = np.ones((ch_h, ch_w), dtype=np.float32)

                # Precision masked compositing
                skin_crop = regions['face_skin'][y1:y2, x1:x2].copy()
                if 'neck' in regions:
                    skin_crop = np.clip(skin_crop + regions['neck'][y1:y2, x1:x2], 0.0, 1.0)

                # Protect eyes, inner mouth, lips, and eyebrows
                protect = (regions['left_eye'][y1:y2, x1:x2] +
                           regions['right_eye'][y1:y2, x1:x2] +
                           regions['mouth_lips'][y1:y2, x1:x2] +
                           regions['left_eyebrow'][y1:y2, x1:x2] * 0.7 +
                           regions['right_eyebrow'][y1:y2, x1:x2] * 0.7)

                protect_dilated = cv2.dilate((protect > 0.15).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                skin_crop[protect_dilated > 0] = 0.0
                if 'hair' in regions:
                    skin_crop[regions['hair'][y1:y2, x1:x2] > 0.25] = 0.0
                if 'background' in regions:
                    skin_crop[regions['background'][y1:y2, x1:x2] > 0.3] = 0.0

                combined_mask = spatial_falloff * skin_crop
                feathered_mask = cv2.GaussianBlur(combined_mask, (15, 15), 5.0)[:, :, np.newaxis]

                # Composite learned re-aged crop
                crop_float = crop.astype(np.float32)
                aged_crop = (crop_float * (1.0 - feathered_mask) + reaged_crop.astype(np.float32) * feathered_mask).astype(np.uint8)

                out_canvas = canvas.copy()
                out_canvas[y1:y2, x1:x2] = aged_crop
                return out_canvas
        except Exception as e:
            logger.warning(f"Error during FRAN re-aging inference: {e}")
    # Fallback to procedural age adjustment if FRAN is unavailable
    return _apply_procedural_age_adjustment(canvas, geom, age_delta, regions=regions)


def _apply_procedural_age_adjustment(canvas, geom, age_delta, regions=None):
    """
    Phase 4.3 Anatomical Procedural Age Engine (Fallback & Comparison Baseline).
    """
    if age_delta == 0:
        return canvas

    if regions is None:
        regions = build_anatomical_region_map(canvas, geom)

    h, w = geom['h'], geom['w']
    fh, fw = geom['fh'], geom['fw']
    u_par, u_perp = geom['u_par'], geom['u_perp']
    mid_brow = geom['mid_brow']
    hairline_C = geom['hairline_C']
    subnasale = geom['subnasale']
    mouth_L = geom['mouth_L']
    mouth_R = geom['mouth_R']
    pupil_L = geom['pupil_L']
    pupil_R = geom['pupil_R']

    out_canvas = canvas.copy().astype(np.float32)

    if age_delta > 0:
        intensity = float(age_delta) / 20.0  # 0.25 to 1.0
        crease_layer = np.zeros((h, w), dtype=np.float32)
        highlight_layer = np.zeros((h, w), dtype=np.float32)

        # 1. Forehead Creases (3 undulating horizontal anatomical rhytids)
        forehead_span = float(np.linalg.norm(hairline_C - mid_brow))
        if forehead_span < 15.0:
            forehead_span = 0.32 * fh

        num_creases = 3 if age_delta >= 10 else 2
        for i in range(num_creases):
            t = 0.25 + i * 0.24
            center_pt = mid_brow - (t * forehead_span) * u_perp
            curve_pts = []
            for s in np.linspace(-0.36 * fw, 0.36 * fw, 50):
                undulation = (np.sin(s * 0.07 + i * 1.2) * 2.2 + np.cos(s * 0.12) * 1.0) * (fh * 0.012)
                pt = center_pt + s * u_par + undulation * u_perp
                curve_pts.append(pt)
            curve_pts = np.array(curve_pts, dtype=np.int32)
            cv2.polylines(crease_layer, [curve_pts], False, 28.0 * intensity, thickness=max(1, int(2.4 * intensity)))
            hi_pts = np.array([pt - (0.012 * fh) * u_perp for pt in curve_pts], dtype=np.int32)
            cv2.polylines(highlight_layer, [hi_pts], False, 10.0 * intensity, thickness=1)

        # 2. Nasolabial Folds (deepening grooves from alar crease down past oral commissures)
        nl_start_L = subnasale - (0.07 * fw) * u_par - (0.02 * fh) * u_perp
        nl_end_L = mouth_L - (0.05 * fw) * u_par + (0.05 * fh) * u_perp
        pts_nl_L = []
        for a in np.linspace(0.0, 1.0, 30):
            mid_curve = np.sin(a * np.pi) * (0.035 * fw)
            pt = (1.0 - a) * nl_start_L + a * nl_end_L - mid_curve * u_par
            pts_nl_L.append(pt)
        pts_nl_L = np.array(pts_nl_L, dtype=np.int32)
        cv2.polylines(crease_layer, [pts_nl_L], False, 32.0 * intensity, thickness=max(1, int(2.6 * intensity)))

        nl_start_R = subnasale + (0.07 * fw) * u_par - (0.02 * fh) * u_perp
        nl_end_R = mouth_R + (0.05 * fw) * u_par + (0.05 * fh) * u_perp
        pts_nl_R = []
        for a in np.linspace(0.0, 1.0, 30):
            mid_curve = np.sin(a * np.pi) * (0.035 * fw)
            pt = (1.0 - a) * nl_start_R + a * nl_end_R + mid_curve * u_par
            pts_nl_R.append(pt)
        pts_nl_R = np.array(pts_nl_R, dtype=np.int32)
        cv2.polylines(crease_layer, [pts_nl_R], False, 32.0 * intensity, thickness=max(1, int(2.6 * intensity)))

        # 3. Crow's Feet (lateral radiating periorbital lines)
        eye_L_outer = geom['l_eye'][0]
        eye_R_outer = geom['r_eye'][3]
        for rad_deg in [-25, 0, 25]:
            rad = np.radians(rad_deg)
            dir_L = -np.cos(rad) * u_par + np.sin(rad) * u_perp
            end_L = eye_L_outer + (0.12 * fw) * dir_L
            cv2.line(crease_layer, tuple(eye_L_outer.astype(int)), tuple(end_L.astype(int)), 20.0 * intensity, thickness=max(1, int(1.8 * intensity)))

            dir_R = np.cos(rad) * u_par + np.sin(rad) * u_perp
            end_R = eye_R_outer + (0.12 * fw) * dir_R
            cv2.line(crease_layer, tuple(eye_R_outer.astype(int)), tuple(end_R.astype(int)), 20.0 * intensity, thickness=max(1, int(1.8 * intensity)))

        # 4. Under-Eye Tear Troughs
        cv2.ellipse(crease_layer, (int(pupil_L[0]), int(pupil_L[1] + 0.075 * fh)), (int(0.12 * fw), int(0.025 * fh)), int(np.degrees(geom['angle'])), 10, 170, 18.0 * intensity, 2)
        cv2.ellipse(crease_layer, (int(pupil_R[0]), int(pupil_R[1] + 0.075 * fh)), (int(0.12 * fw), int(0.025 * fh)), int(np.degrees(geom['angle'])), 10, 170, 18.0 * intensity, 2)

        crease_layer = cv2.GaussianBlur(crease_layer, (5, 5), 1.3)
        highlight_layer = cv2.GaussianBlur(highlight_layer, (5, 5), 1.2)

        # Confine creases to face skin mask
        skin_envelope = regions['face_skin']
        crease_layer *= skin_envelope
        highlight_layer *= skin_envelope

        for c in range(3):
            out_canvas[:, :, c] = np.clip(out_canvas[:, :, c] - crease_layer + highlight_layer, 0.0, 255.0)

        # 5. Dermal micro-texture roughness & age pores
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = gray - cv2.GaussianBlur(gray, (5, 5), 1.0)
        coarse_texture = hp[:, :, np.newaxis] * (0.45 * intensity) * skin_envelope[:, :, np.newaxis]
        out_canvas = np.clip(out_canvas + coarse_texture, 0.0, 255.0)

    else:
        # Negative Age (-20..-5): Youthful rejuvenation
        intensity = float(abs(age_delta)) / 20.0

        # Bilateral dermal smoothing confined strictly to face_skin (preserves eyes, lips, nose contours)
        smooth = cv2.bilateralFilter(canvas, 11, 60, 60).astype(np.float32)
        skin_mask = regions['face_skin'][:, :, np.newaxis]
        weight = 0.65 * intensity
        out_canvas = out_canvas * (1.0 - skin_mask * weight) + smooth * (skin_mask * weight)

        # Under-eye brightening: remove tear troughs and fatigue shadows
        under_eye_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(under_eye_mask, (int(pupil_L[0]), int(pupil_L[1] + 0.075 * fh)), (int(0.14 * fw), int(0.035 * fh)), int(np.degrees(geom['angle'])), 0, 180, 1.0, -1)
        cv2.ellipse(under_eye_mask, (int(pupil_R[0]), int(pupil_R[1] + 0.075 * fh)), (int(0.14 * fw), int(0.035 * fh)), int(np.degrees(geom['angle'])), 0, 180, 1.0, -1)
        under_eye_mask = cv2.GaussianBlur(under_eye_mask, (11, 11), 3.0)[:, :, np.newaxis]
        out_canvas = np.clip(out_canvas + (16.0 * intensity) * under_eye_mask, 0.0, 255.0)

    return out_canvas.astype(np.uint8)


# -------------------------------------------------------------------------
# STAGE 11: CONFIGURATION PARSING
# -------------------------------------------------------------------------

def parse_and_validate_config(config_dict=None):
    """
    Validates user appearance settings against supported options.
    """
    cfg = DEFAULT_APPEARANCE_CONFIG.copy()
    if not config_dict:
        return cfg

    hair = config_dict.get('hair_style') or config_dict.get('hair')
    if hair in SUPPORTED_HAIR_STYLES:
        cfg['hair_style'] = hair

    fh = config_dict.get('facial_hair')
    if fh in SUPPORTED_FACIAL_HAIR:
        cfg['facial_hair'] = fh

    # Current age input (Phase 4.4)
    try:
        cur_age = int(config_dict.get('current_age', 25))
        cfg['current_age'] = max(10, min(90, cur_age))
    except (ValueError, TypeError):
        cfg['current_age'] = 25

    try:
        age = int(config_dict.get('age_delta', 0))
        if age in VALID_AGE_DELTAS:
            cfg['age_delta'] = age
        else:
            closest_age = min(VALID_AGE_DELTAS, key=lambda x: abs(x - age))
            cfg['age_delta'] = closest_age
    except (ValueError, TypeError):
        cfg['age_delta'] = 0

    cfg['target_age'] = max(10, min(90, cfg['current_age'] + cfg['age_delta']))

    hw = config_dict.get('headwear')
    if hw in SUPPORTED_HEADWEAR:
        cfg['headwear'] = hw

    return cfg


def format_age_label(age_delta, current_age=25):
    target_age = max(10, min(90, current_age + age_delta))
    if age_delta == 0:
        return f"Current Age (~{current_age} yrs)"
    elif age_delta > 0:
        return f"+{age_delta} Yrs (Approx. target age: ~{target_age} yrs)"
    else:
        return f"{age_delta} Yrs (Approx. target age: ~{target_age} yrs)"


# -------------------------------------------------------------------------
# STAGE 12: SINGLE APPEARANCE SYNTHESIS PIPELINE
# -------------------------------------------------------------------------

def generate_single_appearance_variation(base_bgr, geom, config):
    """
    Phase 4.4: Synthesizes exactly one appearance variation matching user configuration.
    Anatomical region pipeline:
      1. Anatomical region map generation (BiSeNet 19-class parser + landmark guidance)
      2. Original beard suppression (Clean Shave if requested)
      3. Original hair suppression into background plate (using authoritative hair mask)
      4. Learned face re-aging (FRAN ONNX with procedural fallback)
      5. Hairstyle / Bald scalp synthesis
      6. Facial hair synthesis (strictly preserving lips)
      7. Headwear (seated on cranial vault)
      8. Final lighting and texture harmonization
    """
    skin_bgr, bg_bgr = get_color_palette(base_bgr, geom)

    hair_style = config.get('hair_style', 'Keep Original')
    facial_hair = config.get('facial_hair', 'Clean Shaven')
    current_age = int(config.get('current_age', 25))
    age_delta = int(config.get('age_delta', 0))
    headwear = config.get('headwear', 'None')

    # Step 1: Explicit 18-region anatomical segmentation map (BiSeNet parser + landmark geometry)
    regions = build_anatomical_region_map(base_bgr, geom)

    # Step 2: Clean Shaven: remove beard and moustache if present
    canvas = clean_shave_beard_if_needed(base_bgr.copy(), geom, facial_hair, skin_bgr, regions=regions)

    # Step 3: Suppress original hair cleanly into background plate using authoritative hair mask
    canvas = suppress_original_hair(canvas, geom, hair_style, headwear, bg_bgr, skin_bgr=skin_bgr, regions=regions)

    # Step 4: Learned Face Re-Aging (FRAN ONNX with procedural fallback)
    canvas = apply_age_adjustment(canvas, geom, age_delta, regions=regions, current_age=current_age)

    # Step 5: Scalp or Hairstyle
    is_cap = (headwear in ['Cap', 'Turban'])
    if hair_style == 'Bald':
        canvas = render_bald_appearance(canvas, geom, skin_bgr, bg_bgr)
    else:
        canvas = render_selected_hairstyle(canvas, geom, hair_style, is_cap_worn=is_cap)

    # Step 6: Facial hair (if not Clean Shaven)
    canvas = render_selected_facial_hair(canvas, geom, facial_hair, regions=regions)

    # Step 7: Headwear
    canvas = render_selected_headwear(canvas, geom, headwear, regions=regions)

    return canvas


# -------------------------------------------------------------------------
# STAGE 13: MAIN GENERATOR ENTRY POINT (PHASE 4.2 SINGLE OUTPUT ONLY)
# -------------------------------------------------------------------------

def generate_face_variations(
    base_face_path,
    output_dir=None,
    custom_settings=None,
    count=1,
    include_suggestions=False,
    session_id=None,
    **kwargs
):
    """
    Main entry point for generating appearance variations (Phase 4.2 / 4.6).
    Produces EXACTLY ONE PRIMARY VARIATION matching user settings.
    No suggested variations or random alternatives are generated.
    Supports flexible signatures:
      1. generate_face_variations(path_or_bgr, output_dir, custom_settings=...)
      2. generate_face_variations(path_or_bgr, config_dict, session_id=...)
    """
    start_time = time.perf_counter()

    if isinstance(output_dir, dict):
        if custom_settings is None:
            custom_settings = output_dir
        if session_id:
            output_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_variations', f'session_{session_id}')
        else:
            output_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_variations', 'session_default')
    elif output_dir is None:
        if session_id:
            output_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_variations', f'session_{session_id}')
        else:
            output_dir = os.path.join(os.path.dirname(__file__), 'static', 'generated_variations', 'session_default')

    os.makedirs(output_dir, exist_ok=True)

    base_bgr = load_and_standardize_image(base_face_path)
    h, w = base_bgr.shape[:2]

    # Detect landmarks & extract geometry
    geom = extract_face_geometry(base_bgr)
    if geom is None:
        raise ValueError("No face detected in base image. Please upload a clear frontal face photograph.")

    # Parse primary configuration
    primary_cfg = parse_and_validate_config(custom_settings)

    uid = uuid.uuid4().hex[:8]

    # Exactly ONE Primary Variation
    primary_filename = f"variation_primary_{uid}.png"
    primary_filepath = os.path.join(output_dir, primary_filename)
    primary_img = generate_single_appearance_variation(base_bgr, geom, primary_cfg)
    cv2.imwrite(primary_filepath, primary_img)

    base_fname = os.path.basename(base_face_path) if isinstance(base_face_path, str) else "input_face.png"

    primary_entry = {
        'variation_id': f"primary_{uid}",
        'variation_number': 1,
        'is_primary': True,
        'filename': primary_filename,
        'label': 'PRIMARY APPEARANCE VARIATION',
        'hair': primary_cfg['hair_style'],
        'facial_hair': primary_cfg['facial_hair'],
        'current_age': primary_cfg['current_age'],
        'target_age': primary_cfg['target_age'],
        'age_delta': primary_cfg['age_delta'],
        'age': format_age_label(primary_cfg['age_delta'], primary_cfg['current_age']),
        'headwear': primary_cfg['headwear'],
        'summary': f"{primary_cfg['hair_style']} Hair, {primary_cfg['facial_hair']}, {format_age_label(primary_cfg['age_delta'], primary_cfg['current_age'])}, {primary_cfg['headwear']}"
    }

    elapsed_time = round(time.perf_counter() - start_time, 3)

    return {
        'success': True,
        'batch_id': uid,
        'base_face_filename': base_fname,
        'primary_variation': primary_entry,
        'suggested_variations': [],
        'total_variations': 1,
        'variations': [primary_entry],
        'processing_time': elapsed_time,
        'dimensions': [w, h],
        'landmarks_used': True
    }
