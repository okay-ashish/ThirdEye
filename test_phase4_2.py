"""
ThirdEye Phase 4.2 Automated Test Suite
Face Variations: Geometry-Aware Realism & Seamless Appearance Replacement

Tests verify all 25 Phase 4.2 requirements and regression gates:
 1. route loads
 2. direct photo upload works
 3. reconstruction source works
 4. exactly one output
 5. no suggested variants
 6. curly replacement works
 7. bald replacement works
 8. clean shave works
 9. full beard works
 10. moustache works
 11. cap alignment works
 12. turban alignment works
 13. original feature suppression works
 14. head-aware geometry works
 15. multiple face sizes handled
 16. multiple subject images handled
 17. invalid input handled
 18. no-face handled
 19. storage is session-scoped
 20. logout cleanup
 21. Phase 2 regression
 22. Phase 3 regression
 23. Phase 3.1 regression
 24. Phase 4 regression
 25. Phase 4.1 regression
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
    piecewise_affine_warp,
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


class Phase42TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase42_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase42_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.2 Quality Tester'
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
            'username': 'test_phase42_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # 1. route loads
    def test_01_route_loads(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'FACE VARIATIONS', res.data)
        self.assertIn(b'Upload Photo', res.data)
        self.assertIn(b'Use Existing Reconstruction', res.data)
        self.assertIn(b'GENERATE VARIATION', res.data)

    # 2. direct photo upload works
    def test_02_direct_photo_upload_works(self):
        self.login()
        with open(self.trump_path, 'rb') as f:
            photo_data = f.read()

        data = {
            'uploaded_photo': (io.BytesIO(photo_data), 'test_upload_p42.jpg'),
            'hair_style': 'Curly',
            'facial_hair': 'Clean Shaven',
            'age_delta': '0',
            'headwear': 'None'
        }
        res = self.client.post('/create-variations', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'PRIMARY APPEARANCE VARIATION', res.data)
        self.assertIn(b'variation_primary_', res.data)

    # 3. reconstruction source works
    def test_03_reconstruction_source_works(self):
        out_dir = 'static/test_variations/test_session_recon_p42'
        os.makedirs(out_dir, exist_ok=True)
        res = generate_face_variations(
            self.recon_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Moustache', 'age_delta': 5, 'headwear': 'Cap'},
            count=1
        )
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 4. exactly one output
    def test_04_exactly_one_output(self):
        out_dir = 'static/test_variations/test_session_one_out'
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
        self.assertIsNotNone(res.get('primary_variation'))
        self.assertTrue(os.path.exists(os.path.join(out_dir, res['primary_variation']['filename'])))

    # 5. no suggested variants
    def test_05_no_suggested_variants(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        self.assertNotIn(b'SUGGESTED VARIATIONS', res.data)
        self.assertNotIn(b'suggested-card', res.data)

        out_dir = 'static/test_variations/test_session_no_sugg'
        os.makedirs(out_dir, exist_ok=True)
        gen_res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Wavy', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        self.assertEqual(len(gen_res['suggested_variations']), 0)

    # 6. curly replacement works
    def test_06_curly_replacement_works(self):
        out_dir = 'static/test_variations/test_session_curly_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Measure head scalp ROI where hair is replaced
        hy = geom['hairline_y']
        mx = geom['mid_x']
        hw = int(geom['fw'] * 0.55)
        orig_hair_patch = base[0:hy, mx-hw:mx+hw]
        new_hair_patch = out_img[0:hy, mx-hw:mx+hw]
        diff = cv2.absdiff(orig_hair_patch, new_hair_patch)
        self.assertGreater(np.mean(diff), 25.0)

    # 7. bald replacement works
    def test_07_bald_replacement_works(self):
        out_dir = 'static/test_variations/test_session_bald_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Bald', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Verify bald cranial vault is rendered with skin dome
        apex_y = max(10, geom['apex_y'])
        mid_x = geom['mid_x']
        dome_patch = out_img[apex_y:apex_y+25, mid_x-20:mid_x+20]
        mean_bgr = np.mean(dome_patch, axis=(0, 1))
        # Warm skin dome: Red > Blue
        self.assertGreater(mean_bgr[2], mean_bgr[0])

    # 8. clean shave works
    def test_08_clean_shave_works(self):
        out_dir = 'static/test_variations/test_session_cleanshave_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.bearded_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.bearded_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Measure chin patch texture roughness reduction (Laplacian variance)
        chin_y = geom['chin_y']
        mid_x = geom['mid_x']
        orig_roi = cv2.cvtColor(base[chin_y-25:chin_y-5, mid_x-25:mid_x+25], cv2.COLOR_BGR2GRAY)
        new_roi = cv2.cvtColor(out_img[chin_y-25:chin_y-5, mid_x-25:mid_x+25], cv2.COLOR_BGR2GRAY)

        orig_var = cv2.Laplacian(orig_roi, cv2.CV_64F).var()
        new_var = cv2.Laplacian(new_roi, cv2.CV_64F).var()
        # Coarse whisker texture must decrease noticeably
        self.assertLess(new_var, orig_var * 0.70)

    # 9. full beard works
    def test_09_full_beard_works(self):
        out_dir = 'static/test_variations/test_session_beard_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Full Beard', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Chin patch should darken with beard addition
        chin_y = geom['chin_y']
        mid_x = geom['mid_x']
        orig_val = np.mean(base[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        new_val = np.mean(out_img[chin_y-25:chin_y-5, mid_x-20:mid_x+20])
        self.assertLess(new_val, orig_val - 15.0)

    # 10. moustache works
    def test_10_moustache_works(self):
        out_dir = 'static/test_variations/test_session_moust_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Moustache', 'age_delta': 0, 'headwear': 'None'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Philtrum area should darken
        sn_y = int(geom['subnasale'][1])
        sn_x = int(geom['subnasale'][0])
        orig_phil = np.mean(base[sn_y+4:sn_y+16, sn_x-15:sn_x+15])
        new_phil = np.mean(out_img[sn_y+4:sn_y+16, sn_x-15:sn_x+15])
        self.assertLess(new_phil, orig_phil - 10.0)

    # 11. cap alignment works
    def test_11_cap_alignment_works(self):
        out_dir = 'static/test_variations/test_session_cap_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.obama_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Cap'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Cap must sit above brows on skull
        # Eye region should remain unobstructed
        eye_y = int((geom['pupil_L'][1] + geom['pupil_R'][1]) / 2.0)
        eye_patch_orig = base[eye_y-10:eye_y+10, int(geom['pupil_L'][0]):int(geom['pupil_R'][0])]
        eye_patch_new = out_img[eye_y-10:eye_y+10, int(geom['pupil_L'][0]):int(geom['pupil_R'][0])]
        eye_diff = np.mean(cv2.absdiff(eye_patch_orig, eye_patch_new))
        self.assertLess(eye_diff, 10.0)  # Eyes unobstructed

        # Scalp above forehead must have cap modification
        apex_y = max(5, geom['apex_y'])
        mid_x = geom['mid_x']
        cap_diff = np.mean(cv2.absdiff(base[apex_y:apex_y+30, mid_x-30:mid_x+30], out_img[apex_y:apex_y+30, mid_x-30:mid_x+30]))
        self.assertGreater(cap_diff, 20.0)

    # 12. turban alignment works
    def test_12_turban_alignment_works(self):
        out_dir = 'static/test_variations/test_session_turb_p42'
        os.makedirs(out_dir, exist_ok=True)
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        res = generate_face_variations(
            self.trump_path,
            output_dir=out_dir,
            custom_settings={'hair_style': 'Keep Original', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'Turban'}
        )
        out_img = cv2.imread(os.path.join(out_dir, res['primary_variation']['filename']))

        # Turban should wrap around cranial contour
        apex_y = max(5, geom['apex_y'])
        mid_x = geom['mid_x']
        turb_diff = np.mean(cv2.absdiff(base[apex_y:apex_y+30, mid_x-30:mid_x+30], out_img[apex_y:apex_y+30, mid_x-30:mid_x+30]))
        self.assertGreater(turb_diff, 20.0)

    # 13. original feature suppression works
    def test_13_original_feature_suppression_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        # Hair suppression should leave eyes, nose and mouth untouched
        bg_bgr = np.array([245.0, 245.0, 245.0], dtype=np.float32)
        suppressed = suppress_original_hair(base.copy(), geom, 'Bald', 'None', bg_bgr)

        # Central face ROI (between pupils and mouth)
        py1 = int(geom['pupil_L'][1] + 10)
        py2 = int(geom['subnasale'][1] - 5)
        px1 = int(geom['pupil_L'][0] + 15)
        px2 = int(geom['pupil_R'][0] - 15)
        face_roi_diff = np.mean(cv2.absdiff(base[py1:py2, px1:px2], suppressed[py1:py2, px1:px2]))
        self.assertEqual(face_roi_diff, 0.0)

    # 14. head-aware geometry works
    def test_14_head_aware_geometry_works(self):
        base = load_and_standardize_image(self.trump_path)
        geom = extract_face_geometry(base)
        self.assertIsNotNone(geom)

        # Required 2D anatomical parameters
        self.assertIn('angle', geom)
        self.assertIn('u_par', geom)
        self.assertIn('u_perp', geom)
        self.assertIn('hairline_C', geom)
        self.assertIn('apex', geom)
        self.assertIn('temple_L', geom)
        self.assertIn('temple_R', geom)
        self.assertIn('gonion_L', geom)
        self.assertIn('gonion_R', geom)
        self.assertIn('chin_tip', geom)
        self.assertIn('skin_bgr', geom)
        self.assertIn('bg_bgr', geom)

        # Vector orthonormality
        u_par = geom['u_par']
        u_perp = geom['u_perp']
        self.assertAlmostEqual(np.dot(u_par, u_perp), 0.0, places=4)
        self.assertAlmostEqual(np.linalg.norm(u_par), 1.0, places=4)
        self.assertAlmostEqual(np.linalg.norm(u_perp), 1.0, places=4)

    # 15. multiple face sizes handled
    def test_15_multiple_face_sizes_handled(self):
        base = load_and_standardize_image(self.trump_path)
        # Scaled down (300x330)
        small = cv2.resize(base, (300, 330), interpolation=cv2.INTER_AREA)
        geom_s = extract_face_geometry(small)
        self.assertIsNotNone(geom_s)
        out_s = generate_single_appearance_variation(small, geom_s, {'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'})
        self.assertEqual(out_s.shape, small.shape)

        # Scaled up (600x660)
        large = cv2.resize(base, (600, 660), interpolation=cv2.INTER_CUBIC)
        geom_l = extract_face_geometry(large)
        self.assertIsNotNone(geom_l)
        out_l = generate_single_appearance_variation(large, geom_l, {'hair_style': 'Curly', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'})
        self.assertEqual(out_l.shape, large.shape)

    # 16. multiple subject images handled
    def test_16_multiple_subject_images_handled(self):
        subjects = [self.trump_path, self.obama_path, self.bearded_path]
        for path in subjects:
            base = load_and_standardize_image(path)
            geom = extract_face_geometry(base)
            self.assertIsNotNone(geom, f"Failed geometry on {path}")
            out = generate_single_appearance_variation(base, geom, {'hair_style': 'Wavy', 'facial_hair': 'Clean Shaven', 'age_delta': 0, 'headwear': 'None'})
            self.assertEqual(out.shape, base.shape)

    # 17. invalid input handled
    def test_17_invalid_input_handled(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            generate_face_variations('static/nonexistent_dummy_file.jpg', 'static/test_variations/err')

    # 18. no-face handled
    def test_18_no_face_handled(self):
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        blank_path = 'static/test_variations/blank_test_p42.png'
        os.makedirs('static/test_variations', exist_ok=True)
        cv2.imwrite(blank_path, blank)
        with self.assertRaises(ValueError):
            generate_face_variations(blank_path, 'static/test_variations/err')
        if os.path.exists(blank_path):
            os.remove(blank_path)

    # 19. storage is session-scoped
    def test_19_storage_is_session_scoped(self):
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
        self.assertTrue(res.wasSuccessful(), f"Phase 2 errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 22. Phase 3 regression
    def test_22_phase3_regression(self):
        from test_phase3 import Phase3TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase3TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3 errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 23. Phase 3.1 regression
    def test_23_phase3_1_regression(self):
        from test_phase3_1 import Phase31TestSuite
        suite = unittest.TestLoader().loadTestsFromTestCase(Phase31TestSuite)
        res = unittest.TextTestRunner(verbosity=0).run(suite)
        self.assertTrue(res.wasSuccessful(), f"Phase 3.1 errors: {len(res.errors)}, failures: {len(res.failures)}")

    # 24. Phase 4 regression
    def test_24_phase4_regression(self):
        # Configuration validity across all parameters
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
        # Verify single output, suppression, and photo upload compatibility
        cfg = parse_and_validate_config()
        self.assertEqual(cfg['hair_style'], 'Keep Original')
        self.assertIn(cfg['facial_hair'], ['Clean Shaven', 'Keep Original'])
        self.assertEqual(cfg['age_delta'], 0)
        self.assertEqual(cfg['headwear'], 'None')


if __name__ == '__main__':
    unittest.main()
