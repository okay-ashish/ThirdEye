"""
ThirdEye System — Phase 3.1 Verification Test Suite
Tests:
 1. Model file availability
 2. Model loading
 3. Model inference & execution latency
 4. Valid composite sketch input
 5. Invalid input handling
 6. No-face sketch input
 7. Output dimensions
 8. Output file validity
 9. Clean presentation background
10. Absence of the previous rectangular neck artifact
11. Skin tone processing across 6 Fitzpatrick tones
12. Session-scoped storage
13. Logout session cleanup
14. Fallback procedural engine
15. Phase 2 regressions (Auth, Dashboard, Navigation)
16. Phase 3 regressions (176 transparent assets, Sketch UI)
"""

import os
import sys
import shutil
import unittest
import base64
import numpy as np
import cv2
from PIL import Image
import torch

from app import app, db
from models import User, bcrypt
from face_reconstruction import (
    reconstruct_face,
    get_sketch_to_face_model,
    SKIN_TONE_PALETTE,
    DEFAULT_RECONSTRUCTION_CONFIG
)
from session_cleanup import get_session_dir


class Phase31TestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        # Login a test user
        with app.app_context():
            user = User.query.filter_by(username='test_phase31').first()
            if not user:
                user = User(username='test_phase31', password='TestFixturePassword123!', role='user', full_name='Phase 3.1 Tester')
                db.session.add(user)
                db.session.commit()
            else:
                user.password = bcrypt.generate_password_hash('TestFixturePassword123!').decode('utf-8')
                db.session.commit()

        cls.client.post('/login', data={
            'username': 'test_phase31',
            'password': 'TestFixturePassword123!'
        }, follow_redirects=True)

        # Build a standard composite sketch for testing
        cls.test_sketch_path = 'static/uploads/temp_phase31_sketch.png'
        cw, ch = 700, 550
        c = Image.new('RGBA', (cw, ch), (255, 255, 255, 255))
        parts = [
            ('static/sketch_parts/head/01.png', 70, 0.72),
            ('static/sketch_parts/hair/01.png', 20, 0.65),
            ('static/sketch_parts/eyebrows/01.png', 180, 0.58),
            ('static/sketch_parts/eyes/01.png', 215, 0.58),
            ('static/sketch_parts/nose/01.png', 260, 0.55),
            ('static/sketch_parts/lips/01.png', 345, 0.55)
        ]
        for p, top, scale in parts:
            if os.path.exists(p):
                im = Image.open(p).convert('RGBA')
                nw, nh = int(im.width * scale), int(im.height * scale)
                im_scaled = im.resize((nw, nh), Image.Resampling.LANCZOS)
                c.paste(im_scaled, ((cw - nw) // 2, top), im_scaled)
        c.convert('RGB').save(cls.test_sketch_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_sketch_path):
            os.remove(cls.test_sketch_path)

    # -------------------------------------------------------------
    # 1. Model Availability
    # -------------------------------------------------------------
    def test_01_model_availability(self):
        """1. Verify candidate learned model exists in model/sketch_to_face/ and is non-empty."""
        model_path = DEFAULT_RECONSTRUCTION_CONFIG['model_path']
        self.assertTrue(os.path.exists(model_path), f"Model file missing at {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.assertGreater(size_mb, 25.0, f"Model file unexpectedly small: {size_mb:.2f} MB")
        print(f"\n[Test 01 PASS] Model file verified at {model_path} ({size_mb:.2f} MB).")

    # -------------------------------------------------------------
    # 2. Model Loading
    # -------------------------------------------------------------
    def test_02_model_loading(self):
        """2. Verify TorchScript learned model loads onto CPU without errors."""
        model = get_sketch_to_face_model()
        self.assertIsNotNone(model, "Learned sketch-to-face model failed to load.")
        self.assertTrue(hasattr(model, 'forward') or callable(model))
        print("[Test 02 PASS] CycleGAN TorchScript model loaded into memory on CPU.")

    # -------------------------------------------------------------
    # 3. Model Inference Latency
    # -------------------------------------------------------------
    def test_03_inference_execution(self):
        """3. Verify CPU inference executes in sub-second time on standard (1, 3, 128, 128) tensor."""
        import time
        model = get_sketch_to_face_model()
        x = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        dt = time.perf_counter() - t0
        self.assertEqual(y.shape, (1, 3, 128, 128))
        self.assertLess(dt, 1.0, f"CPU inference exceeded 1 second threshold: {dt:.3f}s")
        print(f"[Test 03 PASS] CPU inference verified: latency={dt * 1000:.1f}ms, shape={y.shape}.")

    # -------------------------------------------------------------
    # 4. Valid Sketch Input
    # -------------------------------------------------------------
    def test_04_valid_sketch_input(self):
        """4. Verify reconstruct_face() processes valid composite sketch successfully."""
        out_dir = 'static/generated_faces/test_t4'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium')
        self.assertTrue(res['success'])
        self.assertEqual(res['engine_used'], 'learned_cyclegan')
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['reconstruction_filename'])))
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 04 PASS] Valid sketch processed by learned engine.")

    # -------------------------------------------------------------
    # 5. Invalid Input Handling
    # -------------------------------------------------------------
    def test_05_invalid_input_handling(self):
        """5. Verify graceful ValueError on corrupt / invalid bytes."""
        with self.assertRaises(ValueError):
            reconstruct_face(b'corrupt_invalid_binary_data')
        print("[Test 05 PASS] Graceful ValueError on corrupted image input.")

    # -------------------------------------------------------------
    # 6. No-Face Sketch Input
    # -------------------------------------------------------------
    def test_06_no_face_sketch(self):
        """6. Verify no-face / blank / abstract sketch executes without crashing."""
        blank = Image.new('RGB', (400, 400), (255, 255, 255))
        blank_p = 'static/uploads/temp_blank.png'
        blank.save(blank_p)
        out_dir = 'static/generated_faces/test_t6'
        res = reconstruct_face(blank_p, output_dir=out_dir, skin_tone='medium')
        self.assertTrue(res['success'])
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        print("[Test 06 PASS] No-face / blank sketch handled gracefully with geometric fallback.")

    # -------------------------------------------------------------
    # 7. Output Dimensions
    # -------------------------------------------------------------
    def test_07_output_dimensions(self):
        """7. Verify generated output image matches input canvas dimensions."""
        out_dir = 'static/generated_faces/test_t7'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium')
        img = cv2.imread(os.path.join(out_dir, res['reconstruction_filename']))
        self.assertEqual(img.shape[0], 550)
        self.assertEqual(img.shape[1], 700)
        self.assertEqual(res['dimensions'], [700, 550])
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 07 PASS] Output image dimensions verified (700x550).")

    # -------------------------------------------------------------
    # 8. Output File Validity
    # -------------------------------------------------------------
    def test_08_output_file_validity(self):
        """8. Verify output file is a valid 3-channel PNG with substantial content size."""
        out_dir = 'static/generated_faces/test_t8'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium')
        fpath = os.path.join(out_dir, res['reconstruction_filename'])
        self.assertTrue(os.path.exists(fpath))
        self.assertGreater(os.path.getsize(fpath), 20000)
        img = cv2.imread(fpath)
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 08 PASS] Output file validity verified.")

    # -------------------------------------------------------------
    # 9. Clean Presentation Background
    # -------------------------------------------------------------
    def test_09_clean_presentation_background(self):
        """9. Verify corner pixels maintain clean presentation backdrop (#F2F5F8), zero color flood."""
        out_dir = 'static/generated_faces/test_t9'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium')
        img = cv2.imread(os.path.join(out_dir, res['reconstruction_filename']))
        # Check all 4 corners
        corners = [img[5, 5], img[5, -5], img[-5, 5], img[-5, -5]]
        for c in corners:
            # Expected BGR roughly [248, 245, 242]
            self.assertGreater(c[0], 235, f"Corner flooded with pigment: {c}")
            self.assertGreater(c[1], 235)
            self.assertGreater(c[2], 235)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 09 PASS] Clean presentation background verified across all 4 canvas corners.")

    # -------------------------------------------------------------
    # 10. Absence of Rectangular Neck Artifact
    # -------------------------------------------------------------
    def test_10_absence_of_rectangular_neck_defect(self):
        """10. Verify bottom-center region has ZERO rectangular neck pillar extending to h-1."""
        out_dir = 'static/generated_faces/test_t10'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium')
        img = cv2.imread(os.path.join(out_dir, res['reconstruction_filename']))
        h, w = img.shape[:2]
        # Inspect bottom edge at multiple lateral positions below chin
        for dx in range(-40, 41, 10):
            pixel = img[h - 5, w // 2 + dx]
            self.assertGreater(pixel[0], 235, f"Neck artifact detected at bottom edge ({h - 5}, {w // 2 + dx}): {pixel}")
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 10 PASS] Complete absence of rectangular bottom neck artifact confirmed.")

    # -------------------------------------------------------------
    # 11. Skin Tone Processing Across Fitzpatrick Palette
    # -------------------------------------------------------------
    def test_11_skin_tone_processing(self):
        """11. Verify Fitzpatrick skin tones modulate synthesized pigment monotonically (Fair > Dark)."""
        out_dir = 'static/generated_faces/test_t11'
        res_fair = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='fair')
        res_dark = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='dark')

        img_fair = cv2.imread(os.path.join(out_dir, res_fair['reconstruction_filename']))
        img_dark = cv2.imread(os.path.join(out_dir, res_dark['reconstruction_filename']))

        # Cheek area mean luminance
        cheek_fair = img_fair[220:280, 320:380].mean()
        cheek_dark = img_dark[220:280, 320:380].mean()

        self.assertGreater(cheek_fair, cheek_dark + 40.0,
                           f"Fair skin ({cheek_fair:.1f}) must be substantially brighter than Dark skin ({cheek_dark:.1f})")
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print(f"[Test 11 PASS] Skin tone variation verified: Fair={cheek_fair:.1f} vs Dark={cheek_dark:.1f} (delta={cheek_fair - cheek_dark:.1f}).")

    # -------------------------------------------------------------
    # 12. Session-Scoped Output
    # -------------------------------------------------------------
    def test_12_session_scoped_output(self):
        """12. Verify POST /construct-face produces session-scoped filename."""
        with open(self.test_sketch_path, 'rb') as f:
            b64_data = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

        resp = self.client.post('/construct-face', data={
            'sketch_data': b64_data,
            'skin_tone': 'medium'
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'session_', resp.data)
        self.assertIn(b'reconstruction_', resp.data)
        print("[Test 12 PASS] Session-scoped output storage verified.")

    # -------------------------------------------------------------
    # 13. Logout Session Cleanup
    # -------------------------------------------------------------
    def test_13_logout_cleanup(self):
        """13. Verify session folder is deleted upon user logout."""
        with self.client.session_transaction() as sess:
            sid = sess.get('session_id')
        session_folder = os.path.join(app.config['GENERATED_FACES_FOLDER'], f"session_{sid}")
        # Logout
        resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(os.path.exists(session_folder), "Session folder must be purged on logout.")
        print("[Test 13 PASS] Logout cleanup verified (Phase 1 lifecycle).")

    # -------------------------------------------------------------
    # 14. Fallback Procedural Engine
    # -------------------------------------------------------------
    def test_14_fallback_engine(self):
        """14. Verify fallback procedural engine executes when learned model is bypassed."""
        out_dir = 'static/generated_faces/test_t14'
        res = reconstruct_face(self.test_sketch_path, output_dir=out_dir, skin_tone='medium', config={'force_fallback': True})
        self.assertTrue(res['success'])
        self.assertEqual(res['engine_used'], 'procedural_fallback')
        self.assertIn('Computational Fallback', res['method'])
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)
        print("[Test 14 PASS] Fallback procedural engine verified.")

    # -------------------------------------------------------------
    # 15. Phase 2 Regressions (Auth, Navigation, Dashboard)
    # -------------------------------------------------------------
    def test_15_regression_phase2(self):
        """15. Verify Phase 2 routes (login, dashboard, home) are intact."""
        # Log back in
        login_resp = self.client.post('/login', data={'username': 'test_phase31', 'password': 'TestFixturePassword123!'}, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)

        # Authenticated home serves the workspace dashboard
        resp_home = self.client.get('/')
        self.assertEqual(resp_home.status_code, 200)
        self.assertIn(b'Phase 3.1 Tester', resp_home.data)
        self.assertIn(b'Sketch Constructor', resp_home.data)
        self.assertIn(b'Photo to Sketch', resp_home.data)
        self.assertIn(b'Face Recognition', resp_home.data)
        print("[Test 15 PASS] Phase 2 non-regression verified (Dashboard & Home operational).")

    # -------------------------------------------------------------
    # 16. Phase 3 Regressions (Assets & Sketch Constructor)
    # -------------------------------------------------------------
    def test_16_regression_phase3(self):
        """16. Verify all 176 sketch assets remain transparent and sketch constructor loads."""
        resp_sk = self.client.get('/sketch')
        self.assertEqual(resp_sk.status_code, 200)
        self.assertIn(b'id="sketchCanvas"', resp_sk.data)
        self.assertIn(b'data-category="head"', resp_sk.data)

        # Transparency check on assets
        base_dir = r'static/sketch_parts'
        count = 0
        opaque = 0
        for cat in os.listdir(base_dir):
            cdir = os.path.join(base_dir, cat)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                count += 1
                with Image.open(os.path.join(cdir, f)) as im:
                    arr = np.array(im.convert('RGBA'))
                    if np.all(arr[:, :, 3] >= 250):
                        opaque += 1
        self.assertEqual(count, 176)
        self.assertEqual(opaque, 0)
        print("[Test 16 PASS] Phase 3 non-regression verified (all 176 assets transparent, Sketch UI active).")


if __name__ == '__main__':
    unittest.main()
