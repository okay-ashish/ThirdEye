"""
ThirdEye Phase 4.3 Automated Test Suite
Face Variations V2 — Anatomical Alignment, Age Transformation & Headwear Rebuild

Verifies all 27 Phase 4.3 requirements and regression gates:
 1. route
 2. photo upload
 3. reconstruction source
 4. exactly one output
 5. all hairstyles
 6. all facial-hair options
 7. all age options
 8. cap
 9. turban
 10. face landmark alignment
 11. hair-region alignment (intended region verification)
 12. facial-hair-region alignment (intended region verification)
 13. age-region modification (intended region verification)
 14. headwear-region alignment (intended region verification)
 15. no eye obstruction
 16. no mouth obstruction
 17. background preservation
 18. storage
 19. logout cleanup
 20. Phase 2 regression
 21. Phase 3 regression
 22. Phase 3.1 regression
 23. Phase 4 regression
 24. Phase 4.1 regression
 25. Phase 4.2 regression
 26. Phase 4.2.1 regression
 27. Phase 4.2.2 regression

IMPORTANT:
Tests do NOT only verify that pixels changed; they verify that the INTENDED REGION changed
according to the explicit 18-mask anatomical segmentation map.
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
    build_anatomical_region_map,
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


class Phase43TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase43_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase43_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.3 Anatomical Alignment Tester'
                )
                db.session.add(test_user)
                db.session.commit()

        cls.trump_path = 'static/person_db/trump.jpg'
        cls.obama_path = 'static/person_db/obama.jpg'
        cls.bearded_path = 'static/person_db/images.jpeg'
        cls.recon_path = 'static/benchmark_eval/phase4/recon_face_case_A_primary.png'
        if not os.path.exists(cls.recon_path):
            cls.recon_path = 'static/test_variations/recon_face_varA.png'

    def login(self):
        return self.client.post('/login', data={
            'username': 'test_phase43_user',
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
        self.assertIsInstance(res, np.ndarray)

    # 5. all hairstyles
    def test_05_all_hairstyles(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        for h in SUPPORTED_HAIR_STYLES:
            cfg = {'hair_style': h, 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
            res = generate_single_appearance_variation(base.copy(), geom, cfg)
            self.assertIsNotNone(res, f"Hairstyle {h} failed synthesis")
            self.assertEqual(res.shape, base.shape)

    # 6. all facial-hair options
    def test_06_all_facial_hair_options(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        for fh in SUPPORTED_FACIAL_HAIR:
            cfg = {'hair_style': 'Keep Original', 'facial_hair': fh, 'age_delta': 0, 'headwear': 'None'}
            res = generate_single_appearance_variation(base.copy(), geom, cfg)
            self.assertIsNotNone(res, f"Facial hair {fh} failed synthesis")
            self.assertEqual(res.shape, base.shape)

    # 7. all age options
    def test_07_all_age_options(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        for d in VALID_AGE_DELTAS:
            cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': d, 'headwear': 'None'}
            res = generate_single_appearance_variation(base.copy(), geom, cfg)
            self.assertIsNotNone(res, f"Age delta {d} failed synthesis")
            self.assertEqual(res.shape, base.shape)

    # 8. cap
    def test_08_cap(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Cap crown sits above hairline
        cap_y = int(geom['hairline_C'][1] - 0.05 * geom['fh'])
        cap_x = int(geom['hairline_C'][0])
        self.assertNotEqual(res[cap_y, cap_x, 0], base[cap_y, cap_x, 0])

    # 9. turban
    def test_09_turban(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Turban'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)
        self.assertIsNotNone(res)
        # Turban apex sits above apex
        turb_y = int(geom['apex'][1] - 0.10 * geom['fh'])
        turb_x = int(geom['apex'][0])
        self.assertNotEqual(res[turb_y, turb_x, 0], base[turb_y, turb_x, 0])

    # 10. face landmark alignment
    def test_10_face_landmark_alignment(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)
        required_keys = ['chin', 'brows', 'eyes', 'nose_tip', 'lips', 'pupil_L', 'pupil_R',
                         'hairline_C', 'apex', 'temple_L', 'temple_R', 'gonion_L', 'gonion_R',
                         'chin_tip', 'subnasale', 'u_par', 'u_perp', 'skin_bgr', 'bg_bgr']
        for k in required_keys:
            self.assertIn(k, geom, f"Missing geometry key: {k}")
        self.assertAlmostEqual(np.dot(geom['u_par'], geom['u_perp']), 0.0, places=4)

    # 11. hair-region alignment (intended region verification)
    def test_11_hair_region_alignment(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        regions = build_anatomical_region_map(base, geom)
        cfg = {'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)

        diff = np.abs(res.astype(float) - base.astype(float))
        scalp_diff = np.sum(diff * regions['scalp_head'][:, :, np.newaxis]) / max(1.0, np.sum(regions['scalp_head']))
        mouth_diff = np.sum(diff * regions['mouth_lips'][:, :, np.newaxis]) / max(1.0, np.sum(regions['mouth_lips']))

        # Scalp region MUST have changed significantly
        self.assertGreater(scalp_diff, 8.0, "Hair change must significantly modify the scalp/head region")
        # Mouth region MUST NOT be modified by hairstyle
        self.assertLess(mouth_diff, 2.0, "Hairstyle must never intrude onto mouth/lips region")

    # 12. facial-hair-region alignment (intended region verification)
    def test_12_facial_hair_region_alignment(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        regions = build_anatomical_region_map(base, geom)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Full Beard', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)

        diff = np.abs(res.astype(float) - base.astype(float))
        jaw_diff = np.sum(diff * regions['beard_jaw_region'][:, :, np.newaxis]) / max(1.0, np.sum(regions['beard_jaw_region']))
        eye_diff = np.sum(diff * regions['left_eye'][:, :, np.newaxis]) / max(1.0, np.sum(regions['left_eye']))

        # Jaw/beard region MUST have changed
        self.assertGreater(jaw_diff, 8.0, "Full beard must modify the mandibular jaw region")
        # Eyes MUST NOT be modified by facial hair
        self.assertLess(eye_diff, 2.0, "Facial hair must never modify ocular eye region")

    # 13. age-region modification (intended region verification)
    def test_13_age_region_modification(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        regions = build_anatomical_region_map(base, geom)

        res_0 = generate_single_appearance_variation(base.copy(), geom, {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'})
        res_20 = generate_single_appearance_variation(base.copy(), geom, {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 20, 'headwear': 'None'})
        res_m10 = generate_single_appearance_variation(base.copy(), geom, {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': -10, 'headwear': 'None'})

        diff_20 = np.abs(res_20.astype(float) - res_0.astype(float))
        forehead_diff = np.sum(diff_20 * regions['forehead'][:, :, np.newaxis]) / max(1.0, np.sum(regions['forehead']))
        skin_diff = np.sum(diff_20 * regions['face_skin'][:, :, np.newaxis]) / max(1.0, np.sum(regions['face_skin']))

        # Age +20 MUST modify forehead (rhytids) and face skin (nasolabial folds & texture)
        self.assertGreater(forehead_diff, 1.5, "Age +20 must visibly modify forehead with rhytids")
        self.assertGreater(skin_diff, 1.2, "Age +20 must visibly modify face skin with nasolabial folds")

        # Age -10 MUST also modify face skin (rejuvenating smoothing & brightening)
        diff_m10 = np.abs(res_m10.astype(float) - res_0.astype(float))
        m10_skin_diff = np.sum(diff_m10 * regions['face_skin'][:, :, np.newaxis]) / max(1.0, np.sum(regions['face_skin']))
        self.assertGreater(m10_skin_diff, 0.8, "Age -10 must visibly modify face skin through rejuvenation")

    # 14. headwear-region alignment (intended region verification)
    def test_14_headwear_region_alignment(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        regions = build_anatomical_region_map(base, geom)
        cfg = {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)

        diff = np.abs(res.astype(float) - base.astype(float))
        scalp_diff = np.sum(diff * regions['scalp_head'][:, :, np.newaxis]) / max(1.0, np.sum(regions['scalp_head']))
        mouth_diff = np.sum(diff * regions['mouth_lips'][:, :, np.newaxis]) / max(1.0, np.sum(regions['mouth_lips']))

        # Cap MUST sit in cranial/scalp region
        self.assertGreater(scalp_diff, 15.0, "Cap must occupy cranial scalp region")
        # Mouth MUST NOT be covered by cap
        self.assertLess(mouth_diff, 1.0, "Cap must never reach or occlude the mouth")

    # 15. no eye obstruction
    def test_15_no_eye_obstruction(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        pupil_L = geom['pupil_L'].astype(int)
        pupil_R = geom['pupil_R'].astype(int)

        configs_to_test = [
            {'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            {'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'},
            {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Turban'},
        ]

        for cfg in configs_to_test:
            res = generate_single_appearance_variation(base.copy(), geom, cfg)
            # Check left and right pupil pixel colors
            diff_L = np.linalg.norm(res[pupil_L[1], pupil_L[0]].astype(float) - base[pupil_L[1], pupil_L[0]].astype(float))
            diff_R = np.linalg.norm(res[pupil_R[1], pupil_R[0]].astype(float) - base[pupil_R[1], pupil_R[0]].astype(float))
            self.assertLess(diff_L, 5.0, f"Eye pupil L occluded by config {cfg}")
            self.assertLess(diff_R, 5.0, f"Eye pupil R occluded by config {cfg}")

    # 16. no mouth obstruction
    def test_16_no_mouth_obstruction(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        mouth_c = ((geom['top_lip'][3] + geom['bottom_lip'][3]) / 2.0).astype(int)

        configs_to_test = [
            {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            {'hair_style': 'Keep Original', 'facial_hair': 'Moustache', 'age_delta': 0, 'headwear': 'None'},
            {'hair_style': 'Keep Original', 'facial_hair': 'Full Beard', 'age_delta': 0, 'headwear': 'None'},
            {'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'},
        ]

        for cfg in configs_to_test:
            res = generate_single_appearance_variation(base.copy(), geom, cfg)
            diff_mouth = np.linalg.norm(res[mouth_c[1], mouth_c[0]].astype(float) - base[mouth_c[1], mouth_c[0]].astype(float))
            self.assertLess(diff_mouth, 40.0, f"Mouth opening obstructed by config {cfg}")

    # 17. background preservation
    def test_17_background_preservation(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        cfg = {'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        res = generate_single_appearance_variation(base.copy(), geom, cfg)

        # Bottom corner pixels should remain identical
        corner_diff = np.linalg.norm(res[-10:, -10:].astype(float) - base[-10:, -10:].astype(float))
        self.assertEqual(corner_diff, 0.0, "Background corners must be preserved completely")

    # 18. storage
    def test_18_storage(self):
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

    # 19. logout cleanup
    def test_19_logout_cleanup(self):
        self.login()
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 20. Phase 2 regression
    def test_20_phase2_regression(self):
        from test_phase2 import Phase2TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 21. Phase 3 regression
    def test_21_phase3_regression(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 22. Phase 3.1 regression
    def test_22_phase3_1_regression(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 23. Phase 4 regression
    def test_23_phase4_regression(self):
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

    # 24. Phase 4.1 regression
    def test_24_phase4_1_regression(self):
        cfg = parse_and_validate_config()
        self.assertEqual(cfg['hair_style'], 'Keep Original')
        self.assertIn(cfg['facial_hair'], ['Clean Shaven', 'Keep Original'])
        self.assertEqual(cfg['age_delta'], 0)
        self.assertEqual(cfg['headwear'], 'None')

    # 25. Phase 4.2 regression
    def test_25_phase4_2_regression(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)
        self.assertIn('u_par', geom)
        self.assertIn('u_perp', geom)
        self.assertAlmostEqual(np.dot(geom['u_par'], geom['u_perp']), 0.0, places=4)

    # 26. Phase 4.2.1 regression
    def test_26_phase4_2_1_regression(self):
        from test_phase4_2_1 import Phase421TestCase
        loader = unittest.TestLoader()
        names = [
            'test_04_one_output_only',
            'test_05_all_existing_configuration_options_still_work',
            'test_06_hair_rendering_works',
            'test_07_beard_rendering_works',
            'test_08_moustache_rendering_works',
            'test_09_cap_rendering_works',
            'test_10_turban_rendering_works',
            'test_11_subject_background_separation_works',
            'test_12_edge_refinement_works',
            'test_13_invalid_input_handled',
            'test_14_no_face_handled'
        ]
        suite = loader.loadTestsFromNames(names, Phase421TestCase)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 4.2.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 27. Phase 4.2.2 regression
    def test_27_phase4_2_2_regression(self):
        from test_phase4_2_2 import Phase422TestCase
        loader = unittest.TestLoader()
        names = [
            'test_04_exactly_one_output', 'test_05_all_configurations',
            'test_06_short_hair', 'test_07_curly_hair', 'test_08_wavy_hair',
            'test_09_long_hair', 'test_10_bald', 'test_11_clean_shave',
            'test_12_beard', 'test_13_moustache', 'test_14_age_adjustment',
            'test_15_cap', 'test_16_turban', 'test_17_subject_background_separation',
            'test_18_face_preservation'
        ]
        suite = loader.loadTestsFromNames(names, Phase422TestCase)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 4.2.2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")


if __name__ == '__main__':
    unittest.main()
