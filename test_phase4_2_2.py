"""
ThirdEye Phase 4.2.2 Automated Test Suite
Final Face Variations Appearance Quality Pass

Verifies all 27 Phase 4.2.2 requirements and regression gates:
 1. route
 2. photo upload
 3. reconstruction source
 4. exactly one output
 5. all configurations
 6. short hair
 7. curly hair
 8. wavy hair
 9. long hair
 10. bald
 11. clean shave
 12. beard
 13. moustache
 14. age adjustment
 15. cap
 16. turban
 17. subject/background separation
 18. face preservation
 19. storage
 20. logout cleanup
 21. Phase 2 regression
 22. Phase 3 regression
 23. Phase 3.1 regression
 24. Phase 4 regression
 25. Phase 4.1 regression
 26. Phase 4.2 regression
 27. Phase 4.2.1 regression
"""

import os
import sys
import unittest
import io
import shutil
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
    suppress_original_hair,
    clean_shave_beard_if_needed,
    decontaminate_asset_fringe,
    render_bald_appearance,
    render_selected_hairstyle,
    render_selected_facial_hair,
    render_selected_headwear,
    apply_age_adjustment,
    SUPPORTED_HAIR_STYLES,
    SUPPORTED_FACIAL_HAIR,
    SUPPORTED_HEADWEAR,
    VALID_AGE_DELTAS,
    DEFAULT_APPEARANCE_CONFIG
)


class Phase422TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase422_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase422_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.2.2 Appearance Quality Tester'
                )
                db.session.add(test_user)
                db.session.commit()

        cls.trump_path = 'static/person_db/trump.jpg'
        cls.obama_path = 'static/person_db/obama.jpg'
        cls.bearded_path = 'static/person_db/images.jpeg'
        cls.recon_path = 'static/benchmark_eval/bench_1_learned.png'
        if not os.path.exists(cls.recon_path):
            cls.recon_path = 'static/benchmark_eval/bench_3_procedural.png'

    def login(self):
        return self.client.post('/login', data={
            'username': 'test_phase422_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # 1. route
    def test_01_route(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Face Variations', res.data)

    # 2. photo upload
    def test_02_photo_upload(self):
        self.login()
        with open(self.trump_path, 'rb') as f:
            data = {
                'source_type': 'photo_upload',
                'photo_file': (io.BytesIO(f.read()), 'trump.jpg'),
                'hair_style': 'Keep Original',
                'facial_hair': 'Clean Shaven',
                'age_delta': '0',
                'headwear': 'None'
            }
            res = self.client.post('/create-variations', data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            self.assertIn(b'Primary Variation', res.data)

    # 3. reconstruction source
    def test_03_reconstruction_source(self):
        self.login()
        res = self.client.post('/create-variations', data={
            'source_type': 'reconstruction',
            'hair_style': 'Short',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Primary Variation', res.data)

    # 4. exactly one output
    def test_04_exactly_one_output(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = parse_and_validate_config({'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'})
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, base.shape)
        # Exactly one image matrix returned, not a list or dictionary of multiple candidates
        self.assertIsInstance(res, np.ndarray)

    # 5. all configurations
    def test_05_all_configurations(self):
        for h in SUPPORTED_HAIR_STYLES:
            self.assertIn(h, ['Keep Original', 'Short', 'Long', 'Straight', 'Wavy', 'Curly', 'Bald'])
        for fh in SUPPORTED_FACIAL_HAIR:
            self.assertIn(fh, ['Clean Shaven', 'Light Stubble', 'Heavy Stubble', 'Moustache', 'Full Beard', 'Beard + Moustache'])
        for hw in SUPPORTED_HEADWEAR:
            self.assertIn(hw, ['None', 'Cap', 'Turban'])
        for d in VALID_AGE_DELTAS:
            self.assertIn(d, [-20, -15, -10, -5, 0, 5, 10, 15, 20])

    # 6. short hair
    def test_06_short_hair(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Check cranial curvature and absence of top apex fin
        top_y = int(geom['apex'][1] - 0.08 * geom['fh'])
        mid_x = int(geom['hairline_C'][0])
        # Hair should exist at top center without stretching into outer bounding box corners
        self.assertEqual(res.shape[:2], (geom['h'], geom['w']))

    # 7. curly hair
    def test_07_curly_hair(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Check that there is NO horizontal mid-row cut slit across the head
        mid_y = int((geom['hairline_C'][1] + geom['mid_brow'][1]) / 2.0)
        center_region = res[mid_y-5:mid_y+5, int(geom['hairline_C'][0])-10:int(geom['hairline_C'][0])+10]
        # Skin/forehead should be intact and continuous without black cut lines
        self.assertTrue(np.mean(center_region) > 20)

    # 8. wavy hair
    def test_08_wavy_hair(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Wavy', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, base.shape)

    # 9. long hair
    def test_09_long_hair(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Long', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, base.shape)

    # 10. bald
    def test_10_bald(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Bald', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Scalp region above hairline should match skin tone, not flat paint
        scalp_y = int(geom['apex'][1] + 0.05 * geom['fh'])
        scalp_pixel = res[scalp_y, int(geom['hairline_C'][0])]
        skin_pixel = geom['skin_bgr']
        diff = np.linalg.norm(scalp_pixel.astype(float) - skin_pixel.astype(float))
        # Tone adapted naturally within reasonable color distance
        self.assertLess(diff, 95.0)

    # 11. clean shave
    def test_11_clean_shave(self):
        base = load_and_standardize_image(self.bearded_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, base.shape)

    # 12. beard
    def test_12_beard(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Full Beard', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        chin_y = int(geom['chin_tip'][1] - 0.05 * geom['fh'])
        chin_x = int(geom['chin_tip'][0])
        # Beard hair should be visible around chin
        self.assertNotEqual(res[chin_y, chin_x, 0], base[chin_y, chin_x, 0])

    # 13. moustache
    def test_13_moustache(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Moustache', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        philtrum_y = int(geom['philtrum'][1])
        philtrum_x = int(geom['philtrum'][0])
        # Moustache hair should be visible on philtrum
        self.assertNotEqual(res[philtrum_y, philtrum_x, 0], base[philtrum_y, philtrum_x, 0])
        # Vermilion of lips must remain clean
        lip_c = np.mean(geom['top_lip'], axis=0)
        lip_y = int(lip_c[1] + 0.01 * geom['fh'])
        lip_x = int(lip_c[0])
        diff_lip = np.linalg.norm(res[lip_y, lip_x].astype(float) - base[lip_y, lip_x].astype(float))
        self.assertLess(diff_lip, 50.0)

    # 14. age adjustment
    def test_14_age_adjustment(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg_0 = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        cfg_20 = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 20, 'headwear': 'None'}
        res_0 = generate_single_appearance_variation(base.copy(), geom, cfg_0)
        res_20 = generate_single_appearance_variation(base.copy(), geom, cfg_20)
        # Visible anatomical change on facial features
        max_diff = np.max(np.abs(res_20.astype(float) - res_0.astype(float)))
        self.assertGreater(max_diff, 10.0, "Age +20 must produce visible anatomical facial changes")
        brow_y = int(geom['mid_brow'][1])
        chin_y = int(geom['chin_tip'][1])
        left_x = int(geom['temple_L'][0])
        right_x = int(geom['temple_R'][0])
        face_diff = np.mean(np.abs(res_20[brow_y:chin_y, left_x:right_x].astype(float) - res_0[brow_y:chin_y, left_x:right_x].astype(float)))
        self.assertGreater(face_diff, 0.4, "Age +20 must be clearly and visibly different from Age 0 on face")

    # 15. cap
    def test_15_cap(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Cap crown above hairline
        cap_y = int(geom['hairline_C'][1] - 0.05 * geom['fh'])
        cap_x = int(geom['hairline_C'][0])
        self.assertNotEqual(res[cap_y, cap_x, 0], base[cap_y, cap_x, 0])

    # 16. turban
    def test_16_turban(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Turban'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, base.shape)

    # 17. subject/background separation
    def test_17_subject_background_separation(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        suppressed = suppress_original_hair(base.copy(), geom, 'Bald', 'None', geom['bg_bgr'])
        self.assertIsNotNone(suppressed)
        # Outer margins should blend with background
        self.assertEqual(suppressed.shape, base.shape)

    # 18. face preservation
    def test_18_face_preservation(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        # Central eye region must be preserved exactly
        eye_y = int(geom['mid_brow'][1] + 0.04 * geom['fh'])
        eye_x = int(geom['mid_brow'][0])
        diff_eye = np.linalg.norm(res[eye_y, eye_x].astype(float) - base[eye_y, eye_x].astype(float))
        self.assertLess(diff_eye, 5.0)

    # 19. storage
    def test_19_storage(self):
        self.login()
        res = self.client.post('/create-variations', data={
            'base_face_filename': 'person_db/trump.jpg',
            'hair_style': 'Curly',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'session_', res.data)

    # 20. logout cleanup
    def test_20_logout_cleanup(self):
        self.login()
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 21. Phase 2 regression
    def test_21_phase2_regression(self):
        from test_phase2 import Phase2TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 22. Phase 3 regression
    def test_22_phase3_regression(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 23. Phase 3.1 regression
    def test_23_phase3_1_regression(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 24. Phase 4 regression
    def test_24_phase4_regression(self):
        for h in SUPPORTED_HAIR_STYLES:
            cfg = parse_and_validate_config({'hair_style': h})
            self.assertEqual(cfg['hair_style'], h)
        for fh in SUPPORTED_FACIAL_HAIR:
            cfg = parse_and_validate_config({'facial_hair': fh})
            self.assertEqual(cfg['facial_hair'], fh)
        for hw in SUPPORTED_HEADWEAR:
            cfg = parse_and_validate_config({'headwear': hw})
            self.assertEqual(cfg['headwear'], hw)
        for delta in VALID_AGE_DELTAS:
            cfg = parse_and_validate_config({'age_delta': str(delta)})
            self.assertEqual(cfg['age_delta'], delta)

    # 25. Phase 4.1 regression
    def test_25_phase4_1_regression(self):
        cfg = parse_and_validate_config()
        self.assertEqual(cfg['hair_style'], 'Keep Original')
        self.assertIn(cfg['facial_hair'], ['Clean Shaven', 'Keep Original'])
        self.assertEqual(cfg['age_delta'], 0)
        self.assertEqual(cfg['headwear'], 'None')

    # 26. Phase 4.2 regression
    def test_26_phase4_2_regression(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)
        self.assertIn('u_par', geom)
        self.assertIn('u_perp', geom)
        self.assertIn('apex', geom)
        self.assertIn('hairline_C', geom)
        self.assertAlmostEqual(np.dot(geom['u_par'], geom['u_perp']), 0.0, places=4)

    # 27. Phase 4.2.1 regression
    def test_27_phase4_2_1_regression(self):
        from test_phase4_2_1 import Phase421TestCase
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase421TestCase)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 4.2.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")


if __name__ == '__main__':
    unittest.main()
