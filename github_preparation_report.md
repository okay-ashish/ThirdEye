# ThirdEye — Public GitHub Repository Preparation & Security Audit Report

**Audit Target:** `C:\ThirdEye`
**Remote Target:** `https://github.com/okay-ashish/ThirdEye.git` (Public)
**Audit Completed:** October 2026
**Push Execution:** **NOT PERFORMED (Strictly Halted for User Review)**

---

## 1. Executive Summary & Readiness Assessment

An exhaustive security, portability, licensing, and hygiene audit has been performed on the `C:\ThirdEye` project tree to prepare it for deployment as a professional, public GitHub repository.

### Key Readiness Metrics:
- **Repository Cleanliness:** Total tracked candidate files reduced from 5,416 unvetted items down to **323 clean, portable production assets and tests**.
- **Candidate Repository Size:** **94.30 MB** (fits comfortably within GitHub guidelines; no file exceeds 30.04 MB).
- **Credentials & Secret Exposure:** **Zero** active credentials or secrets exist in the candidate set. All database credentials and session secrets have been decoupled into environment variables (`.env.example` provided).
- **Biometric & Personal Data Protection:** Live database dump (`thirdeye_db.sql`) containing password hashes and biometric encodings is strictly excluded. A sanitized `thirdeye_db.example.sql` template has been provisioned.
- **Portability:** Machine-specific Windows user paths (`C:\Users\<USERNAME>\...`) and local environment paths were removed from launchers (`run.bat`, `run.ps1`) and runtime code.
- **Regression Testing:** **100% Pass Rate (95 of 95 tests passed)** across `test_phase6.py`, `test_phase5.py`, and `test_phase4_6.py`.
- **Git Push Status:** **No push performed. No remote commits created.**

---

## 2. Security & Credential Audit

### Scan Scope
A recursive pattern and keyword scan was executed across all `.py`, `.html`, `.js`, `.json`, `.sql`, `.md`, `.bat`, `.ps1`, `.txt`, and config files targeting passwords, API tokens, database URIs, JWTs, and private keys.

### Findings & Remediation Summary

| Location | Prior State | Action Taken & Remediation | Public Status |
|---|---|---|---|
| `app.py:29` | Hardcoded MySQL URI containing username `root` and active database password | Decoupled to `os.environ` via `THIRDEYE_DB_*` variables with dynamic URL-encoding. Added automated local `.env` loader. | **RESOLVED — REDACTED** |
| `app.py:28` | Static secret key string | Moved to `os.environ.get('THIRDEYE_SECRET_KEY')` with secure fallback. | **RESOLVED — REDACTED** |
| `app.py:118` | Hardcoded default admin password in CLI `init-db` command | Updated to read `THIRDEYE_ADMIN_PASSWORD` from environment. | **RESOLVED — REDACTED** |
| `create_admin.py:14` | Hardcoded default administrator password | Refactored to accept CLI arguments or `THIRDEYE_ADMIN_PASSWORD`. | **RESOLVED — REDACTED** |
| `create_user.py:32` | Hardcoded test credentials | Refactored to accept CLI arguments or `THIRDEYE_NEW_*` environment variables. | **RESOLVED — REDACTED** |
| `templates/index.html:23` | Plaintext default administrator credentials | Replaced with sanitized guidance on using configured credentials. | **RESOLVED — REDACTED** |
| `thirdeye_db.sql` | Contains live bcrypt password hashes and real face encodings | Excluded from Git in `.gitignore`. Replaced with `thirdeye_db.example.sql`. | **RESOLVED — EXCLUDED** |
| `.env` | Created locally with current machine credentials to ensure zero local regression | Explicitly ignored in `.gitignore`. Guaranteed never to enter Git. | **RESOLVED — EXCLUDED** |
| `test_phase*.py` | Synthetic placeholder strings in isolated test fixtures | Confirmed purely artificial test fixtures; safe for test verification. | **VERIFIED SAFE** |

---

## 3. Biometric & Privacy Data Audit

### Biometric Encodings
- **Finding:** The live MySQL dump `thirdeye_db.sql` contained 128-dimensional floating-point biometric facial embeddings.
- **Remediation:** `thirdeye_db.sql` is strictly excluded from version control via `.gitignore`. A sanitized `thirdeye_db.example.sql` was authored containing only the table schema and a synthetic 128-dimension zero-vector demo record.

### Reference Photographic Assets (`static/person_db/`)
- **Finding:** Contains 5 portrait images used as local reference test fixtures: `trump.jpg`, `obama.jpg`, `test_person.jpg`, `images.jpeg`, `images_2.jpeg`.
- **Classification & Action:** Excluded from the staged public repository via `.gitignore` (`static/person_db/*` with `!static/person_db/README.md`). Authored `static/person_db/README.md` detailing directory purpose and privacy policies. The physical files remain intact on the local disk for application and testing workflows.


---

## 4. Large File Audit (Top 30 Files)

GitHub imposes a **100 MB hard limit** per file (immediate push rejection) and a **50 MB warning threshold**.

### Top 30 Files Across Entire Local Workspace

| Rank | Size (MB) | File Path | Flag | Classification | Action Taken |
|---|---|---|---|---|---|
| 1 | 207.73 MB | `model/sketch_to_face/kunal_model.h5` | **>100MB CRITICAL** | Legacy GAN Model | Excluded in `.gitignore` (Obsolete) |
| 2 | 118.48 MB | `model/face_reaging/face_reaging.onnx` | **>100MB CRITICAL** | FRAN ONNX Model | Excluded in `.gitignore`; Documented in README |
| 3 | 112.12 MB | `model/sketch_to_face/rithanya_model.h5`| **>100MB CRITICAL** | Legacy Keras GAN | Excluded in `.gitignore` (Incompatible) |
| 4 | 95.08 MB | `.venv/.../shape_predictor_68_face_landmarks.dat` | **>90MB WARNING** | Local Environment | Excluded in `.gitignore` (`.venv/`) |
| 5 | 82.30 MB | `.venv/.../cv2.pyd` | **>50MB LARGE** | Local Environment | Excluded in `.gitignore` (`.venv/`) |
| 6 | 50.74 MB | `model/face_parsing/resnet18.onnx` | **>50MB LARGE** | Benchmark Parser | Excluded in `.gitignore` (Superseded) |
| 7 | 38.65 MB | `model/sketch_to_face/mishafakhar_model.h5`| Normal (<50MB) | Legacy U-Net | Excluded in `.gitignore` |
| 8 | 34.21 MB | `.venv/.../libopenblas.dll` | Normal (<50MB) | Local Environment | Excluded in `.gitignore` (`.venv/`) |
| 9 | **30.04 MB**| `model/sketch_to_face/cyclegan_model.pt` | **Normal (<50MB)**| **Production Model** | **KEEP — Tracked in Git** |
| 10 | 29.45 MB | `.venv/.../opencv_videoio_ffmpeg.dll` | Normal (<50MB) | Local Environment | Excluded in `.gitignore` (`.venv/`) |
| 11 | **26.28 MB**| `model/face_parsing/efficientnet_b0.onnx`| **Normal (<50MB)**| **Production Model** | **KEEP — Tracked in Git** |
| 12 | 21.43 MB | `.venv/.../dlib_face_recognition_resnet.dat` | Normal (<50MB) | Local Environment | Excluded in `.gitignore` (`.venv/`) |
| 13 | 7.83 MB | `static/benchmark_eval/scratch_fran_test.png`| Normal (<50MB) | Benchmark Output | Excluded in `.gitignore` |
| 14 | **4.73 MB** | `static/img/bgforensic3.png` | **Normal (<50MB)**| **Branding Graphic** | **KEEP — Tracked in Git** |
| 15 | 2.67 MB | `static/uploads/test_bench_sk.jpg` | Normal (<50MB) | Test Upload | Excluded in `.gitignore` |
| 16 | 2.29 MB | `static/benchmark_eval/p46_sheet5_multi.png`| Normal (<50MB) | Benchmark Output | Excluded in `.gitignore` |
| 17 | 2.14 MB | `static/variation_assets_photo/full_beard.png`| Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 18 | 2.13 MB | `static/variation_assets_photo/wavy_hair.png` | Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 19 | 2.05 MB | `scratch_clean_wigs/wavy_hair.png` | Normal (<50MB) | Scratch File | Excluded in `.gitignore` |
| 20 | 1.95 MB | `static/variation_assets_photo/curly_hair.png`| Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 21 | 1.92 MB | `static/variation_assets_photo/long_hair.png` | Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 22 | 1.91 MB | `static/variation_assets_photo/short_hair.png`| Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 23 | 1.81 MB | `static/variation_assets_photo/turban.png` | Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 24 | 1.78 MB | `static/variation_assets_photo/straight_hair.png`| Normal (<50MB)| Photo Overlay | Excluded in `.gitignore` |
| 25 | **1.72 MB** | `static/img/bgadas.png` | **Normal (<50MB)**| **Branding Graphic** | **KEEP — Tracked in Git** |
| 26 | **1.70 MB** | `static/variation_assets/wavy_hair.png` | **Normal (<50MB)**| **Sketch Overlay** | **KEEP — Tracked in Git** |
| 27 | **1.63 MB** | `static/variation_assets/bald.png` | **Normal (<50MB)**| **Sketch Overlay** | **KEEP — Tracked in Git** |
| 28 | **1.58 MB** | `static/variation_assets/curly_hair.png` | **Normal (<50MB)**| **Sketch Overlay** | **KEEP — Tracked in Git** |
| 29 | **1.58 MB** | `static/variation_assets/curly_hair_pure.png`| **Normal (<50MB)**| **Sketch Overlay** | **KEEP — Tracked in Git** |
| 30 | **1.57 MB** | `static/variation_assets/cap.png` | **Normal (<50MB)**| **Sketch Overlay** | **KEEP — Tracked in Git** |

### Candidate Set Size Statistics:
- **Total Tracked Files:** 323
- **Total Candidate Size:** 94.30 MB
- **Largest Tracked File:** `cyclegan_model.pt` (30.04 MB)
- **Files > 50 MB in Candidate Set:** **0**
- **Files > 100 MB in Candidate Set:** **0**

---

## 5. Model Inventory & Large Model Strategy

| Model Name | Size | Architecture | Purpose | License | In Candidate Set? | Strategy |
|---|---|---|---|---|---|---|
| `cyclegan_model.pt` | 30.04 MB | PyTorch TorchScript | Sketch-to-photo face reconstruction | Research Open-Source | **YES (Tracked)** | Direct commit (<50MB) |
| `efficientnet_b0.onnx` | 26.28 MB | BiSeNet (EfficientNet-B0) | 19-class semantic face parsing | MIT License | **YES (Tracked)** | Direct commit (<50MB) |
| `mask1024.jpg` | 0.20 MB | Grayscale Mask | Facial skin blending boundary | Project Internal | **YES (Tracked)** | Direct commit |
| `mask512.jpg` | 0.01 MB | Grayscale Mask | Standard facial skin mask | Project Internal | **YES (Tracked)** | Direct commit |
| `face_reaging.onnx` | 118.48 MB | Disney Research FRAN | Deep facial age transformation | Non-Commercial Research | **NO (Excluded)** | Documented in `model/face_reaging/README.md`. Optional download. Procedural fallback active. |
| `kunal_model.h5` | 207.73 MB | Keras GAN | Phase 3 evaluation benchmark | N/A | **NO (Excluded)** | Obsolete model; excluded. |
| `rithanya_model.h5` | 112.12 MB | Keras Functional GAN | Phase 3.1 evaluation benchmark | N/A | **NO (Excluded)** | Incompatible with Keras 3; excluded. |
| `resnet18.onnx` | 50.74 MB | BiSeNet ResNet-18 | Phase 4.4 benchmark parser | MIT License | **NO (Excluded)** | Superseded by EfficientNet-B0; excluded. |

---

## 6. Third-Party Licenses & Attribution Table

| Component | Source / Provider | License | Public Repo Permitted? | Attribution Required? | Redistribution Notes |
|---|---|---|---|---|---|
| **Flask** | Pallets Projects | BSD-3-Clause | Yes | Yes | Included in LICENSE notice |
| **Flask-SQLAlchemy** | Pallets Projects | BSD-3-Clause | Yes | Yes | Included in LICENSE notice |
| **Flask-Bcrypt** | Max Countryman | BSD-3-Clause | Yes | Yes | Included in LICENSE notice |
| **OpenCV** | OpenCV.org | Apache 2.0 | Yes | Yes | Standard open-source library |
| **dlib** | Davis King | Boost Software License 1.0 | Yes | Yes | Permissive |
| **face_recognition** | Adam Geitgey | MIT License | Yes | Yes | Permissive attribution |
| **face_recognition_models** | Davis King | CC0 1.0 Universal | Yes | No | Public Domain models |
| **PyTorch** | PyTorch Foundation | Modified BSD | Yes | Yes | Permissive |
| **ONNX Runtime** | Microsoft | MIT License | Yes | Yes | Permissive |
| **ReportLab** | ReportLab Inc. | BSD-style | Yes | Yes | Open source PDF generator |
| **BiSeNet ONNX** | Mrkomiljon/face-parsing | MIT License | Yes | Yes | Model weight included with attribution |
| **FRAN Model** | Disney Research | Non-Commercial Research | Yes (Guide Only) | Yes | Weight excluded; setup documented |
| **Bootstrap 5** | Twitter / Bootstrap Team | MIT License | Yes | Yes | MIT License preserved |
| **Fabric.js** | FabricJS Team | MIT License | Yes | Yes | MIT License preserved |

---

## 7. Photographic Assets Review (`static/variation_assets_photo/`)

- **Finding:** Contains 10 high-resolution transparent appearance overlays (`short_hair.png`, `long_hair.png`, etc., totaling 16.7 MB).
- **License / Provenance Status:** Derived from experimental AI generation runs during local development.
- **Action Taken:** In accordance with Section 17, photographic PNGs are **excluded from the public repository** via `.gitignore`.
- **Preservation & Fallback:** `static/variation_assets_photo/README.md` was created to explain manual asset placement. The variation engine (`face_variations.py`) automatically falls back to the clean line-drawn sketch assets in `static/variation_assets/` whenever photographic PNGs are absent.

---

## 8. Database & SQL Audit

- **Live Database Dump (`thirdeye_db.sql`):** Contains real user hashes and suspect encodings. **Excluded from Git via `.gitignore`.** Preserved locally on disk.
- **Public Safe Template (`thirdeye_db.example.sql`):** Authored with complete DDL schema (`users`, `persons`) and a synthetic placeholder demo record. Safe for public distribution.

---

## 9. Requirements Audit (`requirements.txt`)

- **Prior Issue:** Previous `requirements.txt` was missing essential deep learning and PDF generation runtimes (`torch`, `onnxruntime`, `reportlab`, `scipy`, `pypdf`, `python-dotenv`).
- **Remediation:** Updated with pinned minimum versions verified against Python 3.11.x without including unrelated Conda environment bloat.

---

## 10. Core Documentation Status

- **`README.md`:** Comprehensive, professional, covering all 21 required sections (Overview, Objectives, Architecture, AI Vision Components, Hardware, Installation, Models, Workflow, Storage, Testing, Limitations, Licenses, and Forensic Disclaimers).
- **`LICENSE`:** MIT License for ThirdEye source code with explicit third-party attribution declarations.
- **`.gitignore`:** Comprehensive exclusions covering Python bytecode, `.venv`, `.vscode`, `.env`, `thirdeye_db.sql`, `scratch_*.py`, benchmark dumps, oversized weights, and runtime session directories.
- **`.env.example`:** Clean configuration template with placeholders.

---

## 11. Storage & Runtime Lifecycle Audit

| Directory | Item Count | Total Size | Lifecycle Policy | Git Status |
|---|---|---|---|---|
| `static/generated_faces/` | 3 files | 0.28 MB | Session-scoped; purged on logout & 6-hr expiry | Ignored (`.gitkeep` tracked) |
| `static/generated_variations/` | 3 files | 1.31 MB | Session-scoped; purged on logout & 6-hr expiry | Ignored (`.gitkeep` tracked) |
| `static/generated_posters/` | 64 files | 4.78 MB | Session-scoped; purged on logout & 6-hr expiry | Ignored (`.gitkeep` tracked) |
| `static/uploads/` | 6 files | 4.55 MB | Session-scoped; temporary canvas uploads | Ignored (`.gitkeep` tracked) |
| `static/benchmark_eval/` | 380 files | 222.74 MB | Development milestone visual evaluations | Ignored (Excluded) |
| `prototype_eval/` | 18 files | 12.95 MB | Early experimental prototype outputs | Ignored (Excluded) |

---

## 12. Automated Regression Test Results

All regression suites were executed against the updated codebase:

1. **`test_phase6.py` (Poster Constructor & Full Regression):**
   - **Result:** `Ran 28 tests in 11.673s — OK` (28/28 passed).
   - Gates verified: Route loading, 5 poster purposes, dynamic fields, PDF generation, A4 sizing, session-scoped storage, print layout mode, logout cleanup, and backward regression across Phases 2, 3, 3.1, 4, 4.6, and 5.
2. **`test_phase5.py` & `test_phase4_6.py` (Photo-to-Sketch, Biometrics & Appearance Variations):**
   - **Result:** `Ran 67 tests in 212.181s — OK` (67/67 passed).
3. **Total Verified Tests:** **95 tests ran, 95 tests passed (100% pass rate).**

---

## 13. Git Status & Remote Configuration

- **Remote Status:** Configured to `https://github.com/okay-ashish/ThirdEye.git`
- **Branch:** `main`
- **Candidate Files in Scope:** 323 files
- **Push Execution:** **NOT PERFORMED** (Halted as mandated)

---

## 14. MANUAL ACTIONS REQUIRED BEFORE FIRST GITHUB PUSH

Before you initiate the first `git push` to your public GitHub repository, perform the following manual steps:

1. **Inspect Git Status:**
   Run the following in your terminal to review all candidate files:
   ```powershell
   git status --short
   ```
2. **Review Environment Settings:**
   Ensure your local `.env` contains your active MySQL password and that `.env` is NOT listed in `git status` (it should remain untracked).
3. **Stage Candidate Files:**
   When you are satisfied with the candidate files:
   ```powershell
   git add .
   ```
4. **Perform First Git Commit:**
   ```powershell
   git commit -m "Initial commit: ThirdEye AI-assisted forensic facial analysis & reconstruction system"
   ```
5. **Push to GitHub:**
   Once verified, push to your remote repository:
   ```powershell
   git push -u origin main
   ```
