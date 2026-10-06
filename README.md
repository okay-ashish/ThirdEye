# ThirdEye

**AI-Assisted Forensic Computer Vision System for Facial Analysis & Reconstruction**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask_3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Inference: CPU](https://img.shields.io/badge/Hardware-CPU_Optimized-orange.svg)]()

---

## Overview

**ThirdEye** is an AI-assisted forensic computer vision system for composite sketch construction, computer-generated facial reconstruction, face recognition, probable appearance variation, photo-to-sketch conversion, and public-information poster generation.

Designed for investigative assistance and forensic research, ThirdEye provides an integrated, modular environment that bridges composite sketch generation with deep neural networks and classical geometric image processing. The platform operates on standard consumer hardware without requiring dedicated GPU acceleration.

---

## Project Objectives

1. **Digital Facial Composite Synthesis:** Provide an interactive, canvas-based composite sketch construction tool allowing law enforcement artists or witnesses to assemble facial features dynamically.
2. **Computer-Generated Facial Reconstruction:** Translate grayscale or line-art composite sketches into realistic photorealistic appearance hypotheses using generative domain adaptation.
3. **Probable Appearance Variations:** Simulate appearance changesâ€”including hairstyles, facial hair, longitudinal age progression, and headwearâ€”while anchoring anatomical landmarks and identity geometry.
4. **Photo-to-Sketch Forensic Inversion:** Convert photographic evidence or surveillance stills into standardized pencil sketches for database matching and feature analysis.
5. **Biometric Face Recognition & Verification:** Extract 128-dimensional facial embeddings to identify and rank candidate matches from a structured suspect database.
6. **Forensic Poster & Flyer Construction:** Generate printable, high-resolution A4 public-display missing person notices and wanted flyers with legal/forensic disclaimers and exportable vector PDFs.

---

## Features

- **Interactive Sketch Constructor:** HTML5 Canvas-based tool with 176+ interchangeable facial components (eyes, eyebrows, nose, mouth, facial shape, hair, and accessories).
- **Deep Generative Face Reconstruction:** TorchScript CycleGAN model translating 256x256 sketches to candidate photographic renderings with tunable skin tone adaptation.
- **Hybrid Appearance Variation Engine:**
  - 5 Distinct Hairstyle Overlays: Short, Long, Straight, Wavy, and Curly hair.
  - Facial Hair Manipulation: Clean-shaven bilateral pixel shaving, full beard, and moustache overlays.
  - Cranial Headwear: Baseball caps and turbans aligned to cranial landmarks.
  - Longitudinal Age Transformation: Learned progression/regression via FRAN neural network with procedural fallback.
- **Biometric Matching Engine:** 68-point landmark detection and 128-dimensional Euclidean distance comparison with configurable verification thresholds.
- **Automated Photo-to-Sketch Converter:** Dual-pipeline engine supporting adaptive pencil sketch edge-filtering and gradient extraction.
- **Print-Ready Poster Constructor:** ReportLab-driven vector PDF generator adhering to DIN A4 dimensions across 5 investigative templates with automated layout balancing and print preview.
- **Session-Scoped Security & Lifecycle:** Automatic per-session isolated storage with proactive stale-session pruning and secure cleanup upon user logout.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend Framework** | Python 3.11, Flask 3.x | Application server, routing, REST endpoints |
| **Database & ORM** | MySQL 8.x, Flask-SQLAlchemy | Relational storage for users, suspect records, and embeddings |
| **Authentication & Security** | Flask-Bcrypt | Password hashing, role-based access control (Admin / Artist) |
| **Computer Vision Core** | OpenCV (cv2), Pillow, NumPy | Image processing, Delaunay warping, alpha blending, inpainting |
| **Biometrics & Landmarks** | dlib, `face_recognition` | 68-point facial landmark extraction, 128-D embedding generation |
| **Deep Learning Runtimes** | PyTorch (TorchScript), ONNX Runtime | CPU inference for CycleGAN, BiSeNet parsing, and FRAN re-aging |
| **Document Generation** | ReportLab, PyPDF | High-resolution DIN A4 PDF document composition |
| **Frontend Architecture** | Vanilla JavaScript, HTML5 Canvas, Bootstrap 5 | Interactive layout, canvas manipulation, responsive UI |

---

## System Architecture

```
                                  +---------------------------------------+
                                  |         ThirdEye Web Client           |
                                  |  (HTML5 Canvas / Bootstrap 5 / JS)    |
                                  +-------------------+-------------------+
                                                      |
                                             HTTP / REST Endpoints
                                                      |
                                  +-------------------v-------------------+
                                  |         Flask Application Core        |
                                  |     (app.py - Routing & Sessions)     |
                                  +----+--------------+---------------+---+
                                       |              |               |
       +-------------------------------+              |               +-------------------------------+
       |                                              |                                               |
+------v---------------------+          +-------------v------------+                   +--------------v--------------+
|   Forensic Vision Core     |          |   Data & Model Services  |                   |   Publishing & Cleanup      |
+----------------------------+          +--------------------------+                   +-----------------------------+
| * Face Reconstruction      |          | * MySQL DB (SQLAlchemy)  |                   | * Poster Generator          |
|   (CycleGAN TorchScript)   |          | * BiSeNet Parser (ONNX)  |                   |   (ReportLab PDF Engine)    |
| * Face Variations Engine   |          | * FRAN Age Re-Aging      |                   | * Session Cleanup Service   |
|   (BiSeNet + Delaunay)     |          | * Face Recognition Core  |                   |   (LRU / Logout Scrubbing)  |
| * Photo-to-Sketch Engine   |          |   (dlib 128-D Embeddings)|                   +-----------------------------+
+----------------------------+          +--------------------------+
```

---

## Core AI / Computer Vision Components

### 1. CycleGAN Sketch-to-Photo Generator
- **Location:** `model/sketch_to_face/cyclegan_model.pt` (30.04 MB)
- **Role:** Generates an initial photorealistic face hypothesis from a 2-D composite sketch.
- **Execution:** CPU inference via PyTorch JIT.

### 2. BiSeNet Semantic Face Parser
- **Location:** `model/face_parsing/efficientnet_b0.onnx` (26.28 MB)
- **Role:** Computes 19-class semantic segmentation masks (CelebAMask-HQ standard) to isolate hair, facial skin, eyebrows, and background for seamless composite blending.
- **Execution:** ONNX Runtime (`CPUExecutionProvider`).

### 3. FRAN (Face Re-Aging Network)
- **Location:** `model/face_reaging/face_reaging.onnx` (Optional 118.48 MB download)
- **Role:** Simulates age progression (+5 to +20 years) and age regression (-5 to -20 years) via learned pixel offsets.
- **Fallback:** If weight is absent, automatically applies procedural geometric landmark-anchored skin contrast adjustments.

### 4. 68-Point Facial Landmark & Embedding Engine
- **Role:** Detects anthropometric facial coordinates to align hairstyles, beard boundaries, and headwear, and extracts 128-dimensional biometric embeddings.

---

## Project Structure

```
C:\ThirdEye/
â”œâ”€â”€ .env.example                     # Environment template for local configuration
â”œâ”€â”€ .gitignore                        # Comprehensive Git ignore rules
â”œâ”€â”€ LICENSE                          # MIT License with third-party notices
â”œâ”€â”€ README.md                        # Project documentation
â”œâ”€â”€ requirements.txt                 # Pinned production dependencies (Python 3.11)
â”œâ”€â”€ run.bat                          # Portable Windows CMD launcher
â”œâ”€â”€ run.ps1                          # Portable Windows PowerShell launcher
â”œâ”€â”€ thirdeye_db.example.sql          # Safe sample database schema & seed data
â”‚
â”œâ”€â”€ app.py                           # Main Flask server, routes, and API endpoints
â”œâ”€â”€ models.py                        # SQLAlchemy database models (User, Person)
â”œâ”€â”€ utils.py                         # Biometric encodings & photo-to-sketch logic
â”œâ”€â”€ face_reconstruction.py           # CycleGAN reconstruction & skin tone blending
â”œâ”€â”€ face_variations.py               # BiSeNet parsing, warping, and variations
â”œâ”€â”€ poster_generator.py              # ReportLab A4 vector poster builder
â”œâ”€â”€ session_cleanup.py               # Session-scoped artifact management
â”œâ”€â”€ create_admin.py                  # CLI utility to provision administrator
â”œâ”€â”€ create_user.py                   # CLI utility to provision users
â”‚
â”œâ”€â”€ model/                           # Pretrained model weights
â”‚   â”œâ”€â”€ face_parsing/
â”‚   â”‚   â”œâ”€â”€ efficientnet_b0.onnx     # 26.28 MB BiSeNet ONNX parser
â”‚   â”‚   â””â”€â”€ README.md
â”‚   â”œâ”€â”€ face_reaging/
â”‚   â”‚   â”œâ”€â”€ mask1024.jpg             # High-res skin blending mask
â”‚   â”‚   â”œâ”€â”€ mask512.jpg              # Standard skin blending mask
â”‚   â”‚   â””â”€â”€ README.md                # FRAN setup & download guide
â”‚   â””â”€â”€ sketch_to_face/
â”‚       â”œâ”€â”€ cyclegan_model.pt        # 30.04 MB TorchScript CycleGAN generator
â”‚       â””â”€â”€ README.md
â”‚
â”œâ”€â”€ static/                          # Static assets and runtime directories
â”‚   â”œâ”€â”€ css/                         # Application stylesheets
â”‚   â”œâ”€â”€ js/                          # Client-side scripts
â”‚   â”œâ”€â”€ img/                         # Application branding graphics
â”‚   â”œâ”€â”€ person_db/                   # Sample reference images for recognition
â”‚   â”œâ”€â”€ sketch_parts/                # 176+ facial feature canvas PNG components
â”‚   â”œâ”€â”€ variation_assets/            # Clean line-art appearance variation overlays
â”‚   â”œâ”€â”€ variation_assets_photo/      # Photographic assets directory (README)
â”‚   â”œâ”€â”€ vendor/                      # Bootstrap, Fabric.js, icons
â”‚   â”œâ”€â”€ generated_faces/             # Runtime temporary faces (session-scoped)
â”‚   â”œâ”€â”€ generated_variations/        # Runtime temporary variations (session-scoped)
â”‚   â”œâ”€â”€ generated_posters/           # Runtime generated PDF flyers (session-scoped)
â”‚   â””â”€â”€ uploads/                     # Runtime temporary uploads (session-scoped)
â”‚
â”œâ”€â”€ templates/                       # Jinja2 HTML5 presentation templates
â”‚   â”œâ”€â”€ layout.html                  # Master application shell
â”‚   â”œâ”€â”€ login.html                   # Authentication view
â”‚   â”œâ”€â”€ register.html                # User registration view
â”‚   â”œâ”€â”€ home.html                    # Main dashboard
â”‚   â”œâ”€â”€ sketch.html                  # Interactive Sketch Constructor
â”‚   â”œâ”€â”€ reconstructed_face.html      # Face reconstruction preview
â”‚   â”œâ”€â”€ face_variations.html         # Appearance variation generation view
â”‚   â”œâ”€â”€ photo_to_sketch.html         # Photo-to-sketch converter
â”‚   â”œâ”€â”€ recognition.html             # Suspect identification results
â”‚   â”œâ”€â”€ create_poster.html           # Investigative Poster Constructor
â”‚   â””â”€â”€ admin_dashboard.html         # Administrative user management
â”‚
â””â”€â”€ test_phase*.py                   # Automated regression test suites (Phase 2 to 6)
```

---

## Requirements

### Operating System & Environment
- **Operating System:** Windows 10 / 11 (or Linux / macOS with appropriate dlib compilation)
- **Python Version:** **Python 3.11.x** (64-bit)
- **Database Server:** MySQL 8.0+ or MariaDB 10.4+

### Hardware Specifications
- **CPU:** Standard Intel / AMD multi-core processor (Intel Core i3 / i5 or equivalent)
- **RAM:** 8 GB system RAM recommended (minimum 4 GB)
- **GPU:** **None required.** All neural network inference pipelines (TorchScript, ONNX Runtime) are CPU-optimized.
- **Disk Space:** ~500 MB for core repository and models (~1 GB with full dependencies).

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/okay-ashish/ThirdEye.git
cd ThirdEye
```

### 2. Create and Activate Virtual Environment
Using Conda:
```bash
conda create -n ThirdEye311 python=3.11 -y
conda activate ThirdEye311
```
Or using standard Python `venv`:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on dlib installation on Windows:** If compiling dlib from source fails, install the prebuilt wheel via `pip install dlib-bin` or use conda: `conda install -c conda-forge dlib`.

---

## Environment Variables

Copy the provided `.env.example` file to create your local `.env`:
```powershell
copy .env.example .env
```

Configure your local `.env` settings:
```ini
# Flask Security Key
THIRDEYE_SECRET_KEY=replace-with-a-secure-random-secret

# MySQL Database Connection
THIRDEYE_DB_HOST=localhost
THIRDEYE_DB_PORT=3306
THIRDEYE_DB_USER=root
THIRDEYE_DB_PASSWORD=your_mysql_password
THIRDEYE_DB_NAME=thirdeye_db
```

---

## Database Setup

1. Start your local MySQL service.
2. Initialize the database schema and sample records using `thirdeye_db.example.sql`:
   ```bash
   mysql -u root -p < thirdeye_db.example.sql
   ```
3. Create the initial administrative user:
   ```bash
   python create_admin.py
   ```
   *(Credentials can be configured via THIRDEYE_ADMIN_USER and THIRDEYE_ADMIN_PASSWORD environment variables or CLI arguments).*

---

## Model Setup

ThirdEye ships with the core lightweight models pre-packaged:
1. **BiSeNet EfficientNet-B0 Face Parser:** Located in `model/face_parsing/efficientnet_b0.onnx` (Included, 26 MB).
2. **CycleGAN Sketch Generator:** Located in `model/sketch_to_face/cyclegan_model.pt` (Included, 30 MB).
3. **FRAN Face Re-Aging Network (Optional):** Due to GitHub file size limits, the 118 MB `face_reaging.onnx` is excluded from Git.
   - If desired, download `face_reaging.onnx` following instructions in [`model/face_reaging/README.md`](model/face_reaging/README.md).
   - If omitted, ThirdEye automatically operates in procedural aging mode without error.

---

## Running the Application

### Option A: Using Launcher Scripts
In PowerShell:
```powershell
.\run.ps1
```
In Command Prompt:
```cmd
run.bat
```

### Option B: Using Python Directly
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Feature Workflow

1. **Authentication:** Log in as an Administrator or Artist.
2. **Sketch Constructor (`/sketch`):** Assemble composite features on canvas, adjust scales/positions, and click *Construct Face*.
3. **Face Reconstruction (`/construct-face`):** Review the synthesized photorealistic appearance hypothesis and select preferred skin tone palette.
4. **Appearance Variations (`/variations`):** Explore alternative hairstyles, facial hair, headwear, and age progression (+/- 20 years).
5. **Database Identification (`/recognition`):** Match the reconstructed face or sketch against registered suspect profiles with similarity distance scoring.
6. **Poster Construction (`/create-poster`):** Generate a print-ready DIN A4 public information flyer combining suspect images, demographic details, and statutory disclaimers.

---

## Storage and Temporary Files

ThirdEye implements strict data boundary policies:
- **Session-Scoped Directories:** Generated faces, variations, and posters are isolated inside per-session subdirectories (`session_<session_id>`).
- **Logout Cleansing:** Logging out immediately purges all generated files, cached crops, and temporary uploads associated with that session.
- **Automated Stale Directory Pruning:** A background garbage collection cycle periodically purges unauthenticated directories older than 6 hours.
- **Repository Safety:** All runtime generated artifacts (`static/generated_*`, `static/uploads/*`) are excluded from version control via `.gitignore`.

---

## Testing

Comprehensive test suites are included to verify functionality across each architectural phase:

```bash
# Run Phase 6 poster generation tests (28 test gates)
python -m unittest test_phase6.py -v

# Run Phase 5 photo-to-sketch & recognition tests
python -m unittest test_phase5.py -v

# Run Phase 4.6 appearance variation tests
python -m unittest test_phase4_6.py -v
```

---

## Limitations

- **Computer-Generated Hypotheses:** Synthesized faces are statistical reconstructions and must not be interpreted as actual surveillance photographs.
- **Illumination & Pose Sensitivity:** Face recognition and landmark detection require reasonably frontal facial alignment (within $\pm 25^\circ$ yaw).
- **Resolution Boundary:** Canvas composites operate at $256 \times 256$ to $1024 \times 1024$ resolution.
- **Lighting Invariance:** Variational hair and accessory overlays use alpha matting and inpainting; extreme ambient lighting variations may exhibit minor edge boundary artifacts.

---

## Third-Party Models and Assets

| Component | Source / Repository | Purpose | License |
|---|---|---|---|
| **CycleGAN Generator** | PyTorch TorchScript compilation | Sketch-to-photo domain translation | Open Research License |
| **BiSeNet EfficientNet-B0** | [Mrkomiljon/face-parsing](https://github.com/Mrkomiljon/face-parsing) | 19-class semantic face parsing | MIT License |
| **FRAN Architecture** | [Glat0s/face_reaging-onnx](https://github.com/Glat0s/face_reaging-onnx) | Learned longitudinal face re-aging | Disney Research (Research Only) |
| **face_recognition** | [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) | 128-D facial feature embeddings | MIT License |
| **dlib Models** | Davis King (dlib) | 68-point facial landmark predictor | Boost 1.0 / CC0 Public Domain |
| **Bootstrap 5** | Twitter / Bootstrap Team | Front-end UI responsive framework | MIT License |
| **Fabric.js** | FabricJS Team | HTML5 interactive canvas engine | MIT License |

---

## License

The ThirdEye project source code is released under the [MIT License](LICENSE).
Pretrained neural network weights and third-party libraries retain their original respective licenses.

---

## Disclaimer

> **IMPORTANT FORENSIC & ETHICAL NOTICE**
> ThirdEye produces computer-generated facial reconstructions and probable appearance variations exclusively for experimental, research, and investigative assistance purposes. Generated appearances are mathematical hypotheses and **must not** be treated as confirmed current photographs, definitive forensic evidence, or biometric ground truth in legal proceedings.
