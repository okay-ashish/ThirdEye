# ThirdEye Project — GitHub Preparation File Audit & Classification

**Audit Target:** `C:\ThirdEye`
**Target Repository:** `https://github.com/okay-ashish/ThirdEye.git` (Public)
**Audit Date:** October 2026
**Status:** Audit Completed — Local Ready

---

## 1. Classification Methodology

Every item across the `C:\ThirdEye` workspace is categorized into one of six distinct lifecycle states for public repository deployment:

- **KEEP (A - Must Commit / B - Optional):** Permanent production source code, configuration templates, public assets, templates, models under 50MB, and automated test suites.
- **IGNORE (C - Must Ignore):** Ephemeral artifacts, bytecode, environment files, runtime directories, OS metadata, and local development benchmarks.
- **LOCAL ONLY (D):** Development scratch scripts, private local configurations (`.env`), live local database dumps (`thirdeye_db.sql`), and historical milestones preserved on disk.
- **REMOVE (E):** Disposable temporary artifacts and obsolete files that should not exist in version control.
- **SPECIAL STORAGE (F):** Large model weights (>100MB or exceeding Git limits) requiring Git LFS or external model repository hosting.
- **THIRD-PARTY (G):** External vendor libraries, pre-trained neural network weights, and frameworks requiring explicit licensing attribution.

---

## 2. Master File Classification Table

| Path / Pattern | Classification | File Type | Size / Count | Notes & Action |
|---|---|---|---|---|
| `app.py` | **KEEP** | Source Code | 46 KB | Core Flask application controller and REST API routes. Hardcoded secrets removed. |
| `models.py` | **KEEP** | Source Code | 1.2 KB | SQLAlchemy database models (`User`, `Person`). |
| `utils.py` | **KEEP** | Source Code | 7.3 KB | Biometric encoding, matching distance, and photo-to-sketch algorithms. |
| `face_reconstruction.py` | **KEEP** | Source Code | 22 KB | CycleGAN TorchScript wrapper and skin tone blending. |
| `face_variations.py` | **KEEP** | Source Code | 84 KB | BiSeNet face parsing, landmark warping, and appearance variation synthesis. |
| `poster_generator.py` | **KEEP** | Source Code | 37 KB | ReportLab vector DIN A4 poster builder. |
| `session_cleanup.py` | **KEEP** | Source Code | 12 KB | Per-session directory isolation, stale session garbage collection, and logout scrubbing. |
| `create_admin.py` | **KEEP** | Source / CLI | 1.1 KB | CLI admin user provisioner (updated to support environment variables). |
| `create_user.py` | **KEEP** | Source / CLI | 1.3 KB | CLI user account provisioner (updated to support CLI arguments and environment variables). |
| `templates/*.html` (12 files) | **KEEP** | HTML5 / Jinja2 | ~215 KB | Clean Jinja2 frontend view templates (sanitized credentials). |
| `static/css/*` | **KEEP** | Stylesheets | ~45 KB | Custom responsive UI styling tokens and glassmorphism themes. |
| `static/js/*` | **KEEP** | Client Scripts | ~35 KB | Frontend interactivity and asynchronous fetch controllers. |
| `static/img/*` | **KEEP** | Graphics | 6.5 MB | Permanent branding graphics (`bgadas.png`, `bgforensic3.png`). |
| `static/sketch_parts/*` | **KEEP** | Assets | 3.6 MB (176 files) | Forensic composite sketch feature components (eyebrows, eyes, lips, nose, hair, shapes). |
| `static/variation_assets/*` | **KEEP** | Assets | 14.7 MB (11 files) | Clean vector sketch appearance overlays (hair, beards, headwear). |
| `static/person_db/README.md` | **KEEP** | Documentation | ~1.1 KB | Explains suspect database usage, local testing, and privacy guidelines. |
| `static/person_db/*.jpg, *.jpeg` | **LOCAL ONLY / IGNORE** | Reference Data | 3.2 MB (5 files) | Reference portrait photos. Preserved locally for tests, excluded from Git. |
| `model/sketch_to_face/cyclegan_model.pt` | **KEEP / THIRD-PARTY** | Model Weight | 30.04 MB | TorchScript sketch-to-photo generator (< 50MB, fits within GitHub limits). |
| `model/face_parsing/efficientnet_b0.onnx` | **KEEP / THIRD-PARTY** | Model Weight | 26.28 MB | BiSeNet 19-class semantic face parser (< 50MB, fits within GitHub limits). |
| `model/face_reaging/mask1024.jpg` | **KEEP** | Model Asset | 0.20 MB | Facial skin blending mask for re-aging transitions. |
| `model/face_reaging/mask512.jpg` | **KEEP** | Model Asset | 0.01 MB | Facial skin blending mask. |
| `model/*/README.md` (3 files) | **KEEP** | Documentation | ~4 KB | Model provenance, architecture, and external download setup instructions. |
| `static/variation_assets_photo/README.md` | **KEEP** | Documentation | ~1.5 KB | Explains photographic asset provenance, fallback to sketch assets, and manual placement. |
| `requirements.txt` | **KEEP** | Configuration | 0.5 KB | Verified production dependencies pinned for Python 3.11.x. |
| `README.md` | **KEEP** | Documentation | ~11 KB | Comprehensive, professional public GitHub documentation with all required sections. |
| `LICENSE` | **KEEP** | Legal | 2.2 KB | MIT License for ThirdEye source code + third-party attribution notice. |
| `.gitignore` | **KEEP** | Configuration | 2.5 KB | Comprehensive rules excluding secrets, environments, huge models, benchmarks, and scratch files. |
| `.env.example` | **KEEP** | Configuration | 0.8 KB | Sanitized environment variable template with placeholders only. |
| `thirdeye_db.example.sql` | **KEEP** | Schema | 2.1 KB | Public-safe database DDL and synthetic sample demo records (no private data or passwords). |
| `run.bat`, `run.ps1` | **KEEP** | Launcher | ~0.7 KB | Portable Windows batch & PowerShell launchers without hardcoded machine paths. |
| `test_phase2.py` - `test_phase6.py` | **KEEP** | Automated Tests | ~180 KB | Core regression test suites verifying computer vision, biometrics, and poster engines. |
| `github_preparation_audit.md` | **KEEP** | Documentation | ~8 KB | Comprehensive file classification and repository preparation audit. |
| `github_preparation_report.md` | **KEEP** | Documentation | ~12 KB | Full security, storage, credential, large file, and readiness report. |
| `.env` | **LOCAL ONLY / IGNORE** | Environment | ~0.2 KB | Local environment configuration with live credentials. Ignored by Git. |
| `thirdeye_db.sql` | **LOCAL ONLY / IGNORE** | Database Dump | 12.8 KB | Live database dump containing bcrypt password hashes and real face encodings. Ignored by Git. |
| `model/face_reaging/face_reaging.onnx` | **SPECIAL STORAGE / IGNORE** | Model Weight | 118.48 MB | Disney Research FRAN ONNX weight. Exceeds GitHub 100MB hard limit. Excluded from Git. Documented in README. |
| `model/sketch_to_face/kunal_model.h5` | **REMOVE / IGNORE** | Legacy Model | 207.73 MB | Legacy Keras GAN from Phase 3 evaluation. Exceeds GitHub 100MB limit. Not used by app. Excluded from Git. |
| `model/sketch_to_face/rithanya_model.h5`| **REMOVE / IGNORE** | Legacy Model | 112.12 MB | Legacy Keras GAN from Phase 3.1 evaluation. Incompatible with Keras 3. Excluded from Git. |
| `model/sketch_to_face/mishafakhar_model.h5`| **REMOVE / IGNORE** | Legacy Model | 38.65 MB | Legacy U-Net model from Phase 3.1 evaluation. Not used by app. Excluded from Git. |
| `model/face_parsing/resnet18.onnx` | **REMOVE / IGNORE** | Benchmark Model| 50.74 MB | Phase 4.4 benchmark parser model. Superseded by efficientnet_b0.onnx. Excluded from Git. |
| `model/chennai.csv`, `model/chennai.ipynb` | **REMOVE / IGNORE** | Unrelated Data | 0.11 MB | Unrelated local crime statistics dataset & notebook. Excluded from Git. |
| `static/variation_assets_photo/*.png` | **LOCAL ONLY / IGNORE** | Assets | 16.7 MB (10 files) | High-res photographic appearance PNGs. Excluded from public Git due to provenance review. |
| `static/generated_faces/*` | **IGNORE** | Runtime Data | 0.28 MB | Session-scoped temporary face reconstruction outputs. Excluded via `.gitignore`. |
| `static/generated_variations/*`| **IGNORE** | Runtime Data | 1.31 MB | Session-scoped temporary appearance variations. Excluded via `.gitignore`. |
| `static/generated_posters/*` | **IGNORE** | Runtime Data | 4.78 MB | Session-scoped temporary PDF flyers. Excluded via `.gitignore`. |
| `static/uploads/*` | **IGNORE** | Runtime Data | 4.55 MB | Session-scoped temporary upload sketches and crops. Excluded via `.gitignore`. |
| `static/benchmark_eval/*` | **IGNORE** | Dev Artifacts | 222.7 MB (380 files) | Phase 4.x / 5 evaluation dumps and visual comparison sheets. Excluded via `.gitignore`. |
| `prototype_eval/*` | **IGNORE** | Dev Artifacts | 12.95 MB (18 files) | Early prototype test outputs. Excluded via `.gitignore`. |
| `static/test_variations*` | **IGNORE** | Dev Artifacts | 89.9 MB (205 files) | Phase 4.2 / 4.3 visual test outputs. Excluded via `.gitignore`. |
| `scratch_clean_wigs/*` | **IGNORE** | Dev Artifacts | 8.7 MB (5 files) | Temporary wig cleaning test files. Excluded via `.gitignore`. |
| `static/sketch_parts_original_backup/*`| **LOCAL ONLY / IGNORE** | Backup | 3.5 MB (176 files) | Raw unrefined sketch part backups. Preserved locally, excluded from Git. |
| `scratch_*.py` (63 scripts) | **LOCAL ONLY / IGNORE** | Dev Scripts | ~380 KB | Milestone experiment and debugging scripts. Preserved locally, excluded from Git. |
| `phase*.md` (12 reports) | **LOCAL ONLY / IGNORE** | Dev Reports | ~230 KB | Internal development quality reports containing local paths. Preserved locally, excluded from Git. |
| `generate_phase*.py`, `run_phase*.py` | **LOCAL ONLY / IGNORE** | Dev Scripts | ~45 KB | Benchmark generators for internal reports. Preserved locally, excluded from Git. |
| `prepare_photo_assets.py` etc. | **LOCAL ONLY / IGNORE** | Dev Scripts | ~30 KB | Offline asset extraction scripts containing local paths. Preserved locally, excluded from Git. |
| `.venv/`, `__pycache__/`, `.vscode/` | **IGNORE** | Environment | ~900 MB (5000+ files) | Virtual environments, compiled bytecode, and IDE settings. Excluded via `.gitignore`. |

---

## 3. Candidate Repository Footprint Summary

- **Total Files Tracked for Git:** 323 files
- **Total Repository Size:** ~94.30 MB
- **Files Exceeding 100 MB:** 0 (Clean)
- **Files Exceeding 50 MB:** 0 (Clean)
- **Hardcoded Secret Credentials in Candidate Set:** 0 (Clean)
- **Hardcoded Personal Windows Paths in Candidate Set:** 0 (Clean)
- **Test Suite Pass Rate:** 100% (95/95 automated regression tests passed)
