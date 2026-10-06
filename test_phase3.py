import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image

from app import app, db
from models import User, bcrypt
from face_reconstruction import reconstruct_face, SKIN_TONE_PALETTE
from session_cleanup import get_session_dir

class Phase3TestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Login a test user
        with app.app_context():
            user = User.query.filter_by(username='test_phase3').first()
            if not user:
                user = User(username='test_phase3', password='TestFixturePassword123!', role='user', full_name='Phase 3 Tester')
                db.session.add(user)
                db.session.commit()
            else:
                user.password = bcrypt.generate_password_hash('TestFixturePassword123!').decode('utf-8')
                db.session.commit()

        self.client.post('/login', data={
            'username': 'test_phase3',
            'password': 'TestFixturePassword123!'
        }, follow_redirects=True)

    def test_01_all_176_assets_are_transparent(self):
        """A1 & A2: Verify all 176 sketch assets under static/sketch_parts/ have transparency (0 opaque)"""
        base_dir = r'static/sketch_parts'
        categories = sorted(os.listdir(base_dir))
        total_count = 0
        opaque_count = 0

        for cat in categories:
            cat_dir = os.path.join(base_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for f in os.listdir(cat_dir):
                fpath = os.path.join(cat_dir, f)
                total_count += 1
                with Image.open(fpath) as im:
                    arr = np.array(im.convert('RGBA'))
                    if np.all(arr[:, :, 3] >= 250):
                        opaque_count += 1

        self.assertEqual(total_count, 176, f"Expected 176 assets, found {total_count}")
        self.assertEqual(opaque_count, 0, f"Found {opaque_count} opaque assets with white backgrounds remaining!")
        print(f"\n[Test 01 PASS] Verified all {total_count} assets in static/sketch_parts/ are transparent PNGs (0 opaque).")

    def test_02_sketch_page_loads_with_categories_and_swatches(self):
        """A3, A4, B2: Verify Sketch Constructor page loads with category-first workflow and skin tone palette"""
        resp = self.client.get('/sketch')
        self.assertEqual(resp.status_code, 200)

        # Check Category tabs
        self.assertIn(b'data-category="head"', resp.data)
        self.assertIn(b'data-category="hair"', resp.data)
        self.assertIn(b'data-category="eyebrows"', resp.data)
        self.assertIn(b'data-category="eyes"', resp.data)
        self.assertIn(b'data-category="nose"', resp.data)
        self.assertIn(b'data-category="lips"', resp.data)
        self.assertIn(b'data-category="mustach"', resp.data)
        self.assertIn(b'data-category="more"', resp.data)

        # Check Skin Tone Swatches
        for tone_id in SKIN_TONE_PALETTE:
            self.assertIn(f'value="{tone_id}"'.encode('utf-8'), resp.data)

        # Check Canvas & Action Buttons
        self.assertIn(b'id="sketchCanvas"', resp.data)
        self.assertIn(b'id="constructFaceBtn"', resp.data)
        self.assertIn(b'id="layerControlsBar"', resp.data)
        self.assertIn(b'id="bringForwardBtn"', resp.data)
        self.assertIn(b'id="sendBackwardBtn"', resp.data)
        print("[Test 02 PASS] Sketch Constructor page rendered with category-first tabs and skin swatches.")

    def test_03_reconstruction_pipeline_with_composite_sketch(self):
        """B3-B9: Verify face reconstruction on composite sketch produces 3D shading, depth, and full coverage"""
        # Create a sample composite sketch
        canvas = Image.new('RGBA', (700, 520), (255, 255, 255, 255))
        head = Image.open('static/sketch_parts/head/01.png').resize((270, 400))
        canvas.paste(head, (215, 70), head)
        hair = Image.open('static/sketch_parts/hair/01.png').resize((360, 340))
        canvas.paste(hair, (170, 20), hair)
        eyebrows = Image.open('static/sketch_parts/eyebrows/01.png').resize((170, 32))
        canvas.paste(eyebrows, (265, 180), eyebrows)
        eyes = Image.open('static/sketch_parts/eyes/01.png').resize((170, 52))
        canvas.paste(eyes, (265, 215), eyes)
        nose = Image.open('static/sketch_parts/nose/01.png').resize((60, 75))
        canvas.paste(nose, (320, 260), nose)
        lips = Image.open('static/sketch_parts/lips/01.png').resize((95, 42))
        canvas.paste(lips, (302, 345), lips)

        tmp_out = 'static/uploads/temp_test_comp.png'
        canvas.save(tmp_out)

        out_dir = 'static/generated_faces/test_session'
        os.makedirs(out_dir, exist_ok=True)

        result = reconstruct_face(tmp_out, output_dir=out_dir, skin_tone='medium')
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['reconstruction_filename'])
        self.assertIsNotNone(result['sketch_filename'])
        self.assertEqual(result['skin_tone'], 'medium')

        # Read reconstructed image and verify qualities
        recon_path = os.path.join(out_dir, result['reconstruction_filename'])
        self.assertTrue(os.path.exists(recon_path))
        img = cv2.imread(recon_path)
        h, w, c = img.shape
        self.assertEqual(c, 3)

        # 1. Background isolation check: corner pixels must be clean background (#F2F5F8), NOT flooded with skin color
        corner = img[5, 5]
        # Skin base medium is roughly B=142, G=178, R=222. Background is B~248, G~245, R~242.
        self.assertGreater(corner[0], 230, "Background corner must remain light neutral, not tinted with skin color")

        # 2. Face coverage check: center of face (nose tip / cheeks) must have skin tone color
        face_center = img[h // 2, w // 2]
        self.assertLess(face_center[0], 215, "Face center must have active skin pigment")

        # 3. 3D shading check: standard deviation of luminance across face must be non-zero (proving depth, not flat polygon)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_crop = gray[150:400, 200:500]
        self.assertGreater(float(np.std(face_crop)), 15.0, "Face region must possess 3D shading variation and depth cues")

        # Clean up
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)

        print("[Test 03 PASS] Reconstruction pipeline generated 3D shaded facial volume with clean background.")

    def test_04_skin_tone_variation_affects_output(self):
        """B2: Verify that changing the skin tone parameter directly alters the synthesized pigment"""
        canvas = Image.new('RGBA', (400, 400), (255, 255, 255, 255))
        head = Image.open('static/sketch_parts/head/01.png').resize((200, 280))
        canvas.paste(head, (100, 60), head)
        tmp_sk = 'static/uploads/temp_skin_test.png'
        canvas.save(tmp_sk)

        out_dir = 'static/generated_faces/test_skin_session'
        os.makedirs(out_dir, exist_ok=True)

        res_fair = reconstruct_face(tmp_sk, output_dir=out_dir, skin_tone='fair')
        res_dark = reconstruct_face(tmp_sk, output_dir=out_dir, skin_tone='dark')

        img_fair = cv2.imread(os.path.join(out_dir, res_fair['reconstruction_filename']))
        img_dark = cv2.imread(os.path.join(out_dir, res_dark['reconstruction_filename']))

        # Cheek pixel comparison: Fair must be visibly brighter than Dark
        cheek_fair = img_fair[200, 200].mean()
        cheek_dark = img_dark[200, 200].mean()

        self.assertGreater(cheek_fair, cheek_dark + 40.0, "Fair skin tone must be significantly lighter than dark skin tone")

        # Clean up
        if os.path.exists(tmp_sk):
            os.remove(tmp_sk)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)

        print(f"[Test 04 PASS] Skin tone selection verified: Fair mean={cheek_fair:.1f}, Dark mean={cheek_dark:.1f}")

    def test_05_construct_face_post_route_end_to_end(self):
        """A10, B14, B17: Verify POST /construct-face creates session-scoped files and renders reconstructed_face.html"""
        # Create a simple valid base64 image data URL
        canvas = Image.new('RGB', (300, 300), (255, 255, 255))
        import io, base64
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        resp = self.client.post('/construct-face', data={
            'sketch_data': data_url,
            'skin_tone': 'olive'
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'COMPUTER-GENERATED FACIAL RECONSTRUCTION', resp.data)
        self.assertIn(b'Mandatory Forensic', resp.data)
        self.assertIn(b'Tan / Olive', resp.data)
        self.assertIn(b'CREATE FACE VARIATIONS', resp.data)
        self.assertIn(b'session_', resp.data)  # Confirms session-scoped storage prefix
        print("[Test 05 PASS] POST /construct-face executed end-to-end with session-scoped storage.")

    def test_06_error_handling_empty_input(self):
        """B16: Verify graceful error handling when empty sketch data is submitted"""
        resp = self.client.post('/construct-face', data={
            'sketch_data': ''
        }, follow_redirects=True)

        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'No sketch input provided', resp.data)
        print("[Test 06 PASS] Graceful flash error on empty sketch submission.")

    def test_07_logout_cleanup_removes_reconstructed_artifacts(self):
        """B17: Verify Phase 1 session cleanup removes generated reconstruction files on logout"""
        # First generate a reconstruction via POST
        canvas = Image.new('RGB', (200, 200), (255, 255, 255))
        import io, base64
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        self.client.post('/construct-face', data={
            'sketch_data': data_url,
            'skin_tone': 'medium'
        })

        # Verify session directory exists
        with self.client.session_transaction() as sess:
            session_id = sess.get('session_id')

        self.assertIsNotNone(session_id)
        session_faces = os.path.join(app.config['GENERATED_FACES_FOLDER'], f"session_{session_id}")
        self.assertTrue(os.path.exists(session_faces))

        # Logout
        resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        # Verify session folder was removed
        self.assertFalse(os.path.exists(session_faces), "Session-scoped generated faces folder must be deleted on logout")
        print("[Test 07 PASS] Phase 1 storage lifecycle intact: session artifacts deleted on logout.")

if __name__ == '__main__':
    unittest.main()
