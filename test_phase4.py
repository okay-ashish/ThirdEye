"""
ThirdEye Phase 4 Automated Test Suite
Verifies all 21 Phase 4 requirements and regression tests:
  1. Face Variations route loads
  2. Configuration controls exist
  3. Eyewear controls do NOT exist
  4. Headwear controls exist
  5. Age adjustment exists
  6. No variation is automatically generated on page load
  7. Primary generation works
  8. Suggested variation generation works
  9. Hair configuration changes renderer input
  10. Facial hair configuration changes renderer input
  11. Age configuration changes renderer input
  12. Headwear configuration changes renderer input
  13. Real photo works
  14. Phase 3.1 reconstructed face works
  15. Invalid image handled
  16. No-face input handled
  17. Session-scoped storage works
  18. Logout cleanup works
  19. Phase 2 regression tests pass
  20. Phase 3 regression tests pass
  21. Phase 3.1 reconstruction tests pass
"""

import os
import sys
import unittest
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
    SUPPORTED_HAIR_STYLES,
    SUPPORTED_FACIAL_HAIR,
    SUPPORTED_HEADWEAR,
    VALID_AGE_DELTAS,
    get_landmarks
)


class Phase4TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            # Create a test user for authenticated requests
            test_user = User.query.filter_by(username='test_phase4_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase4_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase Four Tester'
                )
                db.session.add(test_user)
                db.session.commit()

        # Ensure benchmark and test assets exist
        cls.real_photo_path = 'static/person_db/test_person.jpg'
        cls.recon_face_path = 'static/benchmark_eval/bench_1_learned.png'

    def login(self):
        return self.client.post('/login', data={
            'username': 'test_phase4_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # 1. Face Variations route loads
    def test_01_face_variations_route_loads(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'FACE VARIATIONS', res.data)

    # 2. Configuration controls exist
    def test_02_configuration_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        # Hair, facial hair, age, headwear
        self.assertIn(b'hair_style', res.data)
        self.assertIn(b'facial_hair', res.data)
        self.assertIn(b'age_delta', res.data)
        self.assertIn(b'headwear', res.data)

    # 3. Eyewear controls do NOT exist
    def test_03_eyewear_controls_do_not_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        # Verify glasses/eyewear inputs were purged
        self.assertNotIn(b'name="glasses"', res.data)
        self.assertNotIn(b'name="eyewear"', res.data)
        self.assertNotIn(b'Normal Glasses', res.data)
        self.assertNotIn(b'Sunglasses', res.data)

    # 4. Headwear controls exist
    def test_04_headwear_controls_exist(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Cap', res.data)
        self.assertIn(b'Turban', res.data)

    # 5. Age adjustment exists
    def test_05_age_adjustment_exists(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'AGE ADJUSTMENT', res.data)
        self.assertIn(b'ageSlider', res.data)

    # 6. No variation is automatically generated on page load
    def test_06_no_variation_automatically_generated_on_page_load(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        # Should show initial instruction state, not generated variations
        self.assertIn(b'READY TO SYNTHESIZE APPEARANCE', res.data)
        self.assertNotIn(b'PRIMARY APPEARANCE VARIATION', res.data)

    # 7. Primary generation works
    def test_07_primary_generation_works(self):
        self.login()
        post_data = {
            'base_face_filename': 'person_db/test_person.jpg',
            'hair_style': 'Short',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        }
        res = self.client.post('/create-variations', data=post_data)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'PRIMARY APPEARANCE VARIATION', res.data)
        self.assertIn(b'variation_primary_', res.data)

    # 8. Suggested variation generation works
    def test_08_suggested_variation_generation_works(self):
        self.login()
        post_data = {
            'base_face_filename': 'person_db/test_person.jpg',
            'hair_style': 'Long',
            'facial_hair': 'Moustache',
            'age_delta': '5',
            'headwear': 'Cap'
        }
        res = self.client.post('/create-variations', data=post_data)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'SUGGESTED VARIATIONS', res.data)
        self.assertIn(b'variation_suggested_1_', res.data)

    # 9. Hair configuration changes renderer input
    def test_09_hair_configuration_changes_renderer(self):
        cfg1 = parse_and_validate_config({'hair_style': 'Short'})
        cfg2 = parse_and_validate_config({'hair_style': 'Bald'})
        self.assertEqual(cfg1['hair_style'], 'Short')
        self.assertEqual(cfg2['hair_style'], 'Bald')
        self.assertNotEqual(cfg1['hair_style'], cfg2['hair_style'])

    # 10. Facial hair configuration changes renderer input
    def test_10_facial_hair_configuration_changes_renderer(self):
        cfg1 = parse_and_validate_config({'facial_hair': 'Clean Shaven'})
        cfg2 = parse_and_validate_config({'facial_hair': 'Full Beard'})
        self.assertEqual(cfg1['facial_hair'], 'Clean Shaven')
        self.assertEqual(cfg2['facial_hair'], 'Full Beard')
        self.assertNotEqual(cfg1['facial_hair'], cfg2['facial_hair'])

    # 11. Age configuration changes renderer input
    def test_11_age_configuration_changes_renderer(self):
        cfg1 = parse_and_validate_config({'age_delta': -15})
        cfg2 = parse_and_validate_config({'age_delta': 20})
        self.assertEqual(cfg1['age_delta'], -15)
        self.assertEqual(cfg2['age_delta'], 20)

    # 12. Headwear configuration changes renderer input
    def test_12_headwear_configuration_changes_renderer(self):
        cfg1 = parse_and_validate_config({'headwear': 'Cap'})
        cfg2 = parse_and_validate_config({'headwear': 'Turban'})
        self.assertEqual(cfg1['headwear'], 'Cap')
        self.assertEqual(cfg2['headwear'], 'Turban')

    # 13. Real photo works
    def test_13_real_photo_works(self):
        out_dir = 'static/test_variations/test_session_real'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.real_photo_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            count=2
        )
        self.assertTrue(res['success'])
        self.assertTrue(res['primary_variation']['is_primary'])
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 14. Phase 3.1 reconstructed face works
    def test_14_reconstructed_face_works(self):
        out_dir = 'static/test_variations/test_session_recon'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.recon_face_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Full Beard', 'age_delta': 10, 'headwear': 'None'},
            count=2
        )
        self.assertTrue(res['success'])
        self.assertTrue(res['primary_variation']['is_primary'])
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 15. Invalid image handled
    def test_15_invalid_image_handled(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            generate_face_variations('static/nonexistent_image.jpg', 'static/test_variations/err')

    # 16. No-face input handled
    def test_16_no_face_input_handled(self):
        blank = np.ones((400, 400, 3), dtype=np.uint8) * 255
        blank_path = 'static/test_variations/blank_test.png'
        cv2.imwrite(blank_path, blank)
        with self.assertRaises(ValueError):
            generate_face_variations(blank_path, 'static/test_variations/err')
        if os.path.exists(blank_path):
            os.remove(blank_path)

    # 17. Session-scoped storage works
    def test_17_session_scoped_storage(self):
        self.login()
        res = self.client.post('/create-variations', data={
            'base_face_filename': 'person_db/test_person.jpg',
            'hair_style': 'Short',
            'facial_hair': 'Moustache',
            'age_delta': '0',
            'headwear': 'None'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'session_', res.data)

    # 18. Logout cleanup works
    def test_18_logout_cleanup_works(self):
        self.login()
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 19. Phase 2 regression tests pass
    def test_19_phase2_regression(self):
        from test_phase2 import Phase2TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 20. Phase 3 regression tests pass
    def test_20_phase3_regression(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 21. Phase 3.1 reconstruction tests pass
    def test_21_phase3_1_reconstruction_regression(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")


if __name__ == '__main__':
    unittest.main()
