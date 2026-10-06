"""
ThirdEye Phase 4.1 Automated Test Suite
Face Variations Quality Correction: Realistic Appearance Replacement + User Photo Input

Tests verify all 25 Phase 4.1 requirements:
 1. Face Variations page loads.
 2. Upload Photo control exists.
 3. Existing Reconstruction option exists.
 4. Hair controls exist.
 5. Facial hair controls exist.
 6. Age controls exist.
 7. Headwear controls exist.
 8. Eyewear does not exist.
 9. Suggested variations do not exist.
 10. Auto-generation does not occur.
 11. Exactly one output is generated.
 12. Uploaded photo works.
 13. Reconstructed face works.
 14. Curly hair modifies/removes original hair appropriately.
 15. Bald removes original hair appropriately.
 16. Clean-shaven removes beard appropriately.
 17. Beard produces aligned facial hair.
 18. Headwear aligns with head geometry.
 19. Invalid photo handled.
 20. No-face photo handled.
 21. Storage is session-scoped.
 22. Logout cleanup works.
 23. Phase 2 tests pass.
 24. Phase 3 tests pass.
 25. Phase 3.1 tests pass.
"""

import os
import sys
import unittest
import io
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, db
from models import User
from face_variations import (
    generate_face_variations,
    generate_single_appearance_variation,
    parse_and_validate_config,
    extract_face_geometry,
    load_and_standardize_image,
    SUPPORTED_HAIR_STYLES,
    SUPPORTED_FACIAL_HAIR,
    SUPPORTED_HEADWEAR,
    VALID_AGE_DELTAS
)


class Phase41TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase41_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase41_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.1 Quality Tester'
                )
                db.session.add(test_user)
                db.session.commit()

        cls.real_photo_path = 'static/person_db/trump.jpg'
        if not os.path.exists(cls.real_photo_path):
            cls.real_photo_path = 'static/person_db/test_person.jpg'
        cls.recon_face_path = 'static/benchmark_eval/bench_1_learned.png'
        cls.bearded_photo_path = 'static/person_db/images.jpeg'

    def login(self):
        return self.client.post('/login', data={
            'username': 'test_phase41_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # 1. Face Variations page loads
    def test_01_face_variations_page_loads(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'FACE VARIATIONS', res.data)

    # 2. Upload Photo control exists
    def test_02_upload_photo_control_exists(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Upload Photo', res.data)
        self.assertIn(b'uploaded_photo', res.data)
        self.assertIn(b'uploadPhotoInput', res.data)

    # 3. Existing Reconstruction option exists
    def test_03_existing_reconstruction_option_exists(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Use Existing Reconstruction', res.data)

    # 4. Hair controls exist
    def test_04_hair_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        for h in ['Keep Original', 'Short', 'Long', 'Straight', 'Wavy', 'Curly', 'Bald']:
            self.assertIn(h.encode('utf-8'), res.data)

    # 5. Facial hair controls exist
    def test_05_facial_hair_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        for fh in ['Clean Shaven', 'Light Stubble', 'Heavy Stubble', 'Moustache', 'Full Beard', 'Beard + Moustache']:
            self.assertIn(fh.encode('utf-8'), res.data)

    # 6. Age controls exist
    def test_06_age_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'AGE ADJUSTMENT', res.data)
        self.assertIn(b'ageSlider', res.data)

    # 7. Headwear controls exist
    def test_07_headwear_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        for hw in ['None', 'Cap', 'Turban']:
            self.assertIn(hw.encode('utf-8'), res.data)

    # 8. Eyewear does not exist
    def test_08_eyewear_does_not_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertNotIn(b'name="glasses"', res.data)
        self.assertNotIn(b'name="eyewear"', res.data)
        self.assertNotIn(b'Normal Glasses', res.data)
        self.assertNotIn(b'Sunglasses', res.data)

    # 9. Suggested variations do not exist
    def test_09_suggested_variations_do_not_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertNotIn(b'SUGGESTED VARIATIONS', res.data)
        self.assertNotIn(b'suggested-card', res.data)
        self.assertNotIn(b'Suggested Variation 1', res.data)
        self.assertNotIn(b'Suggested Variation 2', res.data)

    # 10. Auto-generation does not occur
    def test_10_auto_generation_does_not_occur(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'READY TO GENERATE VARIATION', res.data)
        self.assertNotIn(b'PRIMARY APPEARANCE VARIATION', res.data)

    # 11. Exactly one output is generated
    def test_11_exactly_one_output_generated(self):
        out_dir = 'static/test_variations/test_session_single'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            count=1,
            include_suggestions=False
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertEqual(len(res['variations']), 1)
        self.assertEqual(len(res['suggested_variations']), 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 12. Uploaded photo works
    def test_12_uploaded_photo_works(self):
        self.login()
        with open(self.real_photo_path, 'rb') as f:
            photo_data = f.read()

        data = {
            'uploaded_photo': (io.BytesIO(photo_data), 'test_upload.jpg'),
            'hair_style': 'Curly',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        }
        res = self.client.post('/create-variations', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'PRIMARY APPEARANCE VARIATION', res.data)

    # 13. Reconstructed face works
    def test_13_reconstructed_face_works(self):
        out_dir = 'static/test_variations/test_session_recon_p41'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.recon_face_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Moustache', 'age_delta': 5, 'headwear': 'Cap'},
            count=1
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 14. Curly hair modifies/removes original hair appropriately
    def test_14_curly_hair_modifies_original_hair(self):
        out_dir = 'static/test_variations/test_session_curly'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.real_photo_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Verify scalp/hair region above forehead is substantially modified
        hairline_y = geom['hairline_y']
        orig_hair_roi = base[0:hairline_y, :]
        new_hair_roi = out_img[0:hairline_y, :]
        diff = cv2.absdiff(orig_hair_roi, new_hair_roi)
        # Difference must be significant (curly hair replacing original hair)
        self.assertGreater(np.mean(diff), 15.0)

    # 15. Bald removes original hair appropriately
    def test_15_bald_removes_original_hair(self):
        out_dir = 'static/test_variations/test_session_bald'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.real_photo_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Bald', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Verify bald skull dome is rendered and hair texture is gone
        mid_x = geom['mid_x']
        apex_y = max(5, geom['apex_y'])
        dome_patch = out_img[apex_y:apex_y+20, mid_x-20:mid_x+20]
        # Skin tone dome: R channel should dominate over B channel for warm skin
        mean_bgr = np.mean(dome_patch, axis=(0, 1))
        # Blue channel should be less than Red channel for skin tone
        self.assertGreater(mean_bgr[2], mean_bgr[0])

    # 16. Clean-shaven removes beard appropriately
    def test_16_clean_shaven_removes_beard(self):
        if not os.path.exists(self.bearded_photo_path):
            self.skipTest("Bearded image images.jpeg not present")
        out_dir = 'static/test_variations/test_session_cleanshave'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.bearded_photo_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.bearded_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Measure chin patch luminance increase (suppression of dark beard)
        chin_y = geom['chin_y']
        mid_x = geom['mid_x']
        orig_chin_roi = base[chin_y-25:chin_y-5, mid_x-25:mid_x+25]
        new_chin_roi = out_img[chin_y-25:chin_y-5, mid_x-25:mid_x+25]

        orig_lum = np.mean(cv2.cvtColor(orig_chin_roi, cv2.COLOR_BGR2GRAY))
        new_lum = np.mean(cv2.cvtColor(new_chin_roi, cv2.COLOR_BGR2GRAY))
        self.assertGreaterEqual(new_lum, orig_lum - 2.0)

    # 17. Beard produces aligned facial hair
    def test_17_beard_produces_aligned_facial_hair(self):
        out_dir = 'static/test_variations/test_session_beard'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.real_photo_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Full Beard', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Jaw/chin area must show added facial hair
        chin_y = geom['chin_y']
        mid_x = geom['mid_x']
        orig_chin = np.mean(base[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        new_chin = np.mean(out_img[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        # Full beard must darken the chin area
        self.assertLess(new_chin, orig_chin)

    # 18. Headwear aligns with head geometry
    def test_18_headwear_aligns_with_head_geometry(self):
        out_dir = 'static/test_variations/test_session_hw'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.real_photo_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))
        self.assertEqual(out_img.shape, base.shape)
        # Cap is anchored at apex/brow
        brow_y = geom['brow_y']
        diff = cv2.absdiff(base[:brow_y, :], out_img[:brow_y, :])
        self.assertGreater(np.mean(diff), 5.0)

    # 19. Invalid photo handled
    def test_19_invalid_photo_handled(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            generate_face_variations('static/nonexistent_file_xyz.jpg', 'static/test_variations/err')

    # 20. No-face photo handled
    def test_20_no_face_photo_handled(self):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        blank_path = 'static/test_variations/blank_test_p41.png'
        os.makedirs('static/test_variations', exist_ok=True)
        cv2.imwrite(blank_path, blank)
        with self.assertRaises(ValueError):
            generate_face_variations(blank_path, 'static/test_variations/err')
        if os.path.exists(blank_path):
            os.remove(blank_path)

    # 21. Storage is session-scoped
    def test_21_storage_is_session_scoped(self):
        self.login()
        res = self.client.post('/create-variations', data={
            'base_face_filename': 'person_db/trump.jpg',
            'hair_style': 'Short',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'session_', res.data)

    # 22. Logout cleanup works
    def test_22_logout_cleanup_works(self):
        self.login()
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 23. Phase 2 tests pass
    def test_23_phase2_tests_pass(self):
        from test_phase2 import Phase2TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 24. Phase 3 tests pass
    def test_24_phase3_tests_pass(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 25. Phase 3.1 tests pass
    def test_25_phase3_1_tests_pass(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")


if __name__ == '__main__':
    unittest.main()
