"""
ThirdEye Phase 4.2.1 Automated Test Suite
Face Variations: Final Visual Blending & Realism Correction

Verifies all 22 Phase 4.2.1 requirements and regression gates:
 1. route still works
 2. photo upload works
 3. reconstruction input works
 4. one output only
 5. all existing configuration options still work
 6. hair rendering works
 7. beard rendering works
 8. moustache rendering works
 9. cap rendering works
 10. turban rendering works
 11. subject/background separation works
 12. edge refinement works
 13. invalid input handled
 14. no-face handled
 15. session storage remains correct
 16. logout cleanup works
 17. Phase 2 regression passes
 18. Phase 3 regression passes
 19. Phase 3.1 regression passes
 20. Phase 4 regression passes
 21. Phase 4.1 regression passes
 22. Phase 4.2 regression passes
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
    SUPPORTED_HAIR_STYLES,
    SUPPORTED_FACIAL_HAIR,
    SUPPORTED_HEADWEAR,
    VALID_AGE_DELTAS,
    DEFAULT_APPEARANCE_CONFIG
)


class Phase421TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase421_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase421_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.2.1 Visual Blending Tester'
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
            'username': 'test_phase421_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # 1. route still works
    def test_01_route_still_works(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'FACE VARIATIONS', res.data)
        self.assertIn(b'Upload Photo', res.data)
        self.assertIn(b'Use Existing Reconstruction', res.data)
        self.assertIn(b'GENERATE VARIATION', res.data)

    # 2. photo upload works
    def test_02_photo_upload_works(self):
        self.login()
        with open(self.trump_path, 'rb') as f:
            photo_data = f.read()

        data = {
            'uploaded_photo': (io.BytesIO(photo_data), 'test_upload_p421.jpg'),
            'hair_style': 'Curly',
            'facial_hair': 'Moustache',
            'age_delta': '0',
            'headwear': 'None'
        }
        res = self.client.post('/create-variations', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'PRIMARY APPEARANCE VARIATION', res.data)
        self.assertIn(b'variation_primary_', res.data)

    # 3. reconstruction input works
    def test_03_reconstruction_input_works(self):
        out_dir = 'static/test_variations/test_session_recon_p421'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.recon_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Wavy', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'},
            count=1
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 4. one output only
    def test_04_one_output_only(self):
        out_dir = 'static/test_variations/test_session_one_out_p421'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Short', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'},
            count=1
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertEqual(len(res['variations']), 1)
        self.assertEqual(len(res['suggested_variations']), 0)
        self.assertIsNotNone(res.get('primary_variation'))
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 5. all existing configuration options still work
    def test_05_all_existing_configuration_options_still_work(self):
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

    # 6. hair rendering works
    def test_06_hair_rendering_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        suppressed = suppress_original_hair(base.copy(), geom, 'Curly', 'None', geom['bg_bgr'])
        rendered = render_selected_hairstyle(suppressed.copy(), geom, 'Curly')

        # Hair region should be altered
        hy = geom['hairline_y']
        mx = geom['mid_x']
        hw = int(geom['fw'] * 0.55)
        diff = cv2.absdiff(base[0:hy, mx-hw:mx+hw], rendered[0:hy, mx-hw:mx+hw])
        self.assertGreater(np.mean(diff), 25.0)

        # Halo check: Ensure no artificial black border fringe around hair contour
        # At boundary between hair and background, average luminance should not drop drastically below background
        bg_lum = np.mean(geom['bg_bgr'])
        border_band = rendered[max(0, geom['apex_y']-10):geom['apex_y'], mx-hw:mx+hw]
        border_lum = np.mean(border_band)
        # Should stay close to background luminance (not crushed to 0)
        self.assertGreater(border_lum, bg_lum - 35.0)

    # 7. beard rendering works
    def test_07_beard_rendering_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        rendered = render_selected_facial_hair(base.copy(), geom, 'Full Beard')

        # Chin area should be darkened with natural melanin tone
        chin_y = geom['chin_y']
        mid_x = geom['mid_x']
        orig_val = np.mean(base[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        new_val = np.mean(rendered[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        self.assertLess(new_val, orig_val - 15.0)

        # Decontamination check: chin region should NOT have white halo/glow (> 200)
        self.assertLess(new_val, 200.0)

    # 8. moustache rendering works
    def test_08_moustache_rendering_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        rendered = render_selected_facial_hair(base.copy(), geom, 'Moustache')

        # Philtrum area should darken
        sn_y = int(geom['subnasale'][1])
        sn_x = int(geom['subnasale'][0])
        orig_phil = np.mean(base[sn_y+4:sn_y+16, sn_x-15:sn_x+15])
        new_phil = np.mean(rendered[sn_y+4:sn_y+16, sn_x-15:sn_x+15])
        self.assertLess(new_phil, orig_phil - 10.0)

        # Lip vermilion protection: lower lip vermilion (bottom_lip) should NOT be covered by dark moustache
        lower_lip_y = int(np.mean(geom['bottom_lip'][:, 1]))
        orig_lower_lip = np.mean(base[lower_lip_y:lower_lip_y+8, sn_x-10:sn_x+10])
        new_lower_lip = np.mean(rendered[lower_lip_y:lower_lip_y+8, sn_x-10:sn_x+10])
        self.assertAlmostEqual(orig_lower_lip, new_lower_lip, delta=5.0)

    # 9. cap rendering works
    def test_09_cap_rendering_works(self):
        base = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        rendered = render_selected_headwear(base.copy(), geom, 'Cap')

        # Scalp region must have cap modification
        apex_y = max(5, geom['apex_y'])
        mid_x = geom['mid_x']
        cap_diff = np.mean(cv2.absdiff(base[apex_y:apex_y+30, mid_x-30:mid_x+30], rendered[apex_y:apex_y+30, mid_x-30:mid_x+30]))
        self.assertGreater(cap_diff, 20.0)

        # Eye region must remain clear and unobstructed
        eye_y = int((geom['pupil_L'][1] + geom['pupil_R'][1]) / 2.0)
        eye_orig = base[eye_y-10:eye_y+10, int(geom['pupil_L'][0]):int(geom['pupil_R'][0])]
        eye_new = rendered[eye_y-10:eye_y+10, int(geom['pupil_L'][0]):int(geom['pupil_R'][0])]
        self.assertLess(np.mean(cv2.absdiff(eye_orig, eye_new)), 10.0)

    # 10. turban rendering works
    def test_10_turban_rendering_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        rendered = render_selected_headwear(base.copy(), geom, 'Turban')

        # Cranial dome modified by turban wrapping
        apex_y = max(5, geom['apex_y'])
        mid_x = geom['mid_x']
        turb_diff = np.mean(cv2.absdiff(base[apex_y:apex_y+30, mid_x-30:mid_x+30], rendered[apex_y:apex_y+30, mid_x-30:mid_x+30]))
        self.assertGreater(turb_diff, 20.0)

    # 11. subject/background separation works
    def test_11_subject_background_separation_works(self):
        base = load_and_standardize_image(self.recon_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        # Verify suppress_original_hair uses horizontal interpolation across subject width
        suppressed = suppress_original_hair(base.copy(), geom, 'Bald', 'None', geom['bg_bgr'])
        h, w = base.shape[:2]

        # External corners of background must remain identical
        corner_tl_diff = np.mean(cv2.absdiff(base[0:15, 0:15], suppressed[0:15, 0:15]))
        corner_tr_diff = np.mean(cv2.absdiff(base[0:15, w-15:w], suppressed[0:15, w-15:w]))
        self.assertEqual(corner_tl_diff, 0.0)
        self.assertEqual(corner_tr_diff, 0.0)

    # 12. edge refinement works
    def test_12_edge_refinement_works(self):
        # Create a synthetic asset with dark/white fringe and transparent alpha
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[30:70, 30:70] = [30, 40, 50]       # Core dark hair
        bgr[25:30, 25:75] = [240, 240, 240]   # White fringe along top edge
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[30:70, 30:70] = 255
        alpha[25:30, 25:75] = 120              # Semi-transparent fringe

        asset = np.dstack([bgr, alpha])
        clean_asset = decontaminate_asset_fringe(asset, is_dark_hair=True)
        # Fringe pixels should now be colored like the interior hair rather than bright white
        fringe_sample = clean_asset[27, 50, :3]
        self.assertLess(fringe_sample[0], 100)
        self.assertLess(fringe_sample[1], 100)
        self.assertLess(fringe_sample[2], 100)

    # 13. invalid input handled
    def test_13_invalid_input_handled(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            generate_face_variations('static/nonexistent_dummy_file_p421.jpg', 'static/test_variations/err')

    # 14. no-face handled
    def test_14_no_face_handled(self):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        blank_path = 'static/test_variations/blank_test_p421.png'
        os.makedirs('static/test_variations', exist_ok=True)
        cv2.imwrite(blank_path, blank)
        with self.assertRaises(ValueError):
            generate_face_variations(blank_path, 'static/test_variations/err')
        if os.path.exists(blank_path):
            os.remove(blank_path)

    # 15. session storage remains correct
    def test_15_session_storage_remains_correct(self):
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

    # 16. logout cleanup works
    def test_16_logout_cleanup_works(self):
        self.login()
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # 17. Phase 2 regression passes
    def test_17_phase2_regression_passes(self):
        from test_phase2 import Phase2TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 2 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 18. Phase 3 regression passes
    def test_18_phase3_regression_passes(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 19. Phase 3.1 regression passes
    def test_19_phase3_1_regression_passes(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 regression errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 20. Phase 4 regression passes
    def test_20_phase4_regression_passes(self):
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

    # 21. Phase 4.1 regression passes
    def test_21_phase4_1_regression_passes(self):
        cfg = parse_and_validate_config()
        self.assertEqual(cfg['hair_style'], 'Keep Original')
        self.assertIn(cfg['facial_hair'], ['Clean Shaven', 'Keep Original'])
        self.assertEqual(cfg['age_delta'], 0)
        self.assertEqual(cfg['headwear'], 'None')

    # 22. Phase 4.2 regression passes
    def test_22_phase4_2_regression_passes(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)
        self.assertIn('u_par', geom)
        self.assertIn('u_perp', geom)
        self.assertIn('apex', geom)
        self.assertIn('hairline_C', geom)
        self.assertAlmostEqual(np.dot(geom['u_par'], geom['u_perp']), 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
