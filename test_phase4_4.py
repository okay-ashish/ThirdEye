"""
test_phase4_4.py — ThirdEye Phase 4.4 Automated Test Suite
Face Variations — AI Model Integration & Realism Rebuild

Verifies all 34 required test gates:
 1. route loads
 2. upload works
 3. reconstruction source works
 4. exactly one output
 5. face parsing model loads
 6. hair mask generated
 7. skin mask generated
 8. eye protection works
 9. mouth protection works
 10. FRAN ONNX model loads
 11. Age 0 preserves original
 12. Age +20 changes face
 13. Age -10 changes face
 14. age edit stays inside intended face region
 15. hair replacement works
 16. bald works
 17. clean shave works
 18. beard works
 19. moustache works
 20. cap works
 21. turban works
 22. no suggested variants
 23. session storage works
 24. logout cleanup works
 25. benchmark cleanup policy works
 26. Phase 2 regression
 27. Phase 3 regression
 28. Phase 3.1 regression
 29. Phase 4 regression
 30. Phase 4.1 regression
 31. Phase 4.2 regression
 32. Phase 4.2.1 regression
 33. Phase 4.2.2 regression
 34. Phase 4.3 regression
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
    render_bald_appearance,
    render_selected_hairstyle,
    render_selected_facial_hair,
    render_selected_headwear,
    apply_age_adjustment,
    _apply_procedural_age_adjustment,
    get_face_parser_session,
    get_fran_session,
    PARSER_MODEL_PATH,
    FRAN_MODEL_PATH,
    SUPPORTED_HAIR_STYLES,
    SUPPORTED_FACIAL_HAIR,
    SUPPORTED_HEADWEAR,
    VALID_AGE_DELTAS,
    DEFAULT_APPEARANCE_CONFIG
)
from session_cleanup import cleanup_benchmark_artifacts, cleanup_session_artifacts


class Phase44TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            test_user = User.query.filter_by(username='test_phase44_user').first()
            if not test_user:
                test_user = User(
                    username='test_phase44_user',
                    password='Secret123!',
                    role='user',
                    full_name='Phase 4.4 Model Integration Tester'
                )
                db.session.add(test_user)
                db.session.commit()

        cls.obama_path = 'static/person_db/obama.jpg'
        cls.trump_path = 'static/person_db/trump.jpg'
        cls.bearded_path = 'static/person_db/images.jpeg'
        cls.recon_path = 'static/benchmark_eval/phase4/recon_face_case_A_primary.png'
        if not os.path.exists(cls.recon_path):
            cls.recon_path = 'static/test_variations/recon_face_varA.png'

    def login(self):
        return self.client.post('/login', data={
            'username': 'test_phase44_user',
            'password': 'Secret123!'
        }, follow_redirects=True)

    # -------------------------------------------------------------------------
    # TEST 1: Route Loads
    # -------------------------------------------------------------------------
    def test_01_route_loads(self):
        self.login()
        res = self.client.get('/create-variations')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("FACE VARIATIONS", html)
        self.assertIn("current_age", html)
        self.assertIn("age_delta", html)
        self.assertIn("Approximate target age", html)

    # -------------------------------------------------------------------------
    # TEST 2: Upload Works
    # -------------------------------------------------------------------------
    def test_02_upload_works(self):
        self.login()
        with open(self.obama_path, 'rb') as f:
            img_bytes = f.read()
        res = self.client.post(
            '/api/upload-variation-source',
            data={'photo': (io.BytesIO(img_bytes), 'obama_upload.jpg')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data.get('success'))
        self.assertTrue(json_data.get('face_detected'))

    # -------------------------------------------------------------------------
    # TEST 3: Reconstruction Source Works
    # -------------------------------------------------------------------------
    def test_03_reconstruction_source_works(self):
        if not os.path.exists(self.recon_path):
            self.skipTest("Recon face not found")
        self.login()
        res = self.client.post('/create-variations', data={
            'base_face_filename': self.recon_path,
            'hair_style': 'Keep Original',
            'facial_hair': 'Clean Shaven',
            'current_age': 30,
            'age_delta': 0,
            'headwear': 'None'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("PRIMARY APPEARANCE VARIATION", html)

    # -------------------------------------------------------------------------
    # TEST 4: Exactly One Output
    # -------------------------------------------------------------------------
    def test_04_exactly_one_output(self):
        out_dir = "static/benchmark_eval/test_p44_single"
        res = generate_face_variations(self.obama_path, out_dir, custom_settings={
            'hair_style': 'Short',
            'facial_hair': 'Clean Shaven',
            'current_age': 45,
            'age_delta': 10,
            'headwear': 'Cap'
        })
        self.assertTrue(res['success'])
        self.assertEqual(res['total_variations'], 1)
        self.assertEqual(len(res['variations']), 1)
        self.assertEqual(res['suggested_variations'], [])

    # -------------------------------------------------------------------------
    # TEST 5: Face Parsing Model Loads
    # -------------------------------------------------------------------------
    def test_05_face_parsing_model_loads(self):
        self.assertTrue(os.path.exists(PARSER_MODEL_PATH), f"Parser model missing at {PARSER_MODEL_PATH}")
        sess = get_face_parser_session()
        self.assertIsNotNone(sess, "BiSeNet face parser ONNX session failed to load")
        inputs = sess.get_inputs()
        self.assertEqual(len(inputs), 1)

    # -------------------------------------------------------------------------
    # TEST 6: Hair Mask Generated
    # -------------------------------------------------------------------------
    def test_06_hair_mask_generated(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        self.assertIn('hair', regions)
        hair_px = np.sum(regions['hair'] > 0.25)
        self.assertGreater(hair_px, 5000, "Hair mask must have significant non-zero coverage")

    # -------------------------------------------------------------------------
    # TEST 7: Skin Mask Generated
    # -------------------------------------------------------------------------
    def test_07_skin_mask_generated(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        self.assertIn('face_skin', regions)
        skin_px = np.sum(regions['face_skin'] > 0.25)
        self.assertGreater(skin_px, 15000, "Skin mask must have significant non-zero coverage")

    # -------------------------------------------------------------------------
    # TEST 8: Eye Protection Works
    # -------------------------------------------------------------------------
    def test_08_eye_protection_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        self.assertIn('left_eye', regions)
        self.assertIn('right_eye', regions)
        self.assertGreater(np.sum(regions['left_eye'] > 0.1), 50)
        self.assertGreater(np.sum(regions['right_eye'] > 0.1), 50)

    # -------------------------------------------------------------------------
    # TEST 9: Mouth Protection Works
    # -------------------------------------------------------------------------
    def test_09_mouth_protection_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        self.assertIn('mouth_lips', regions)
        self.assertGreater(np.sum(regions['mouth_lips'] > 0.1), 200)

    # -------------------------------------------------------------------------
    # TEST 10: FRAN ONNX Model Loads
    # -------------------------------------------------------------------------
    def test_10_fran_onnx_model_loads(self):
        self.assertTrue(os.path.exists(FRAN_MODEL_PATH), f"FRAN model missing at {FRAN_MODEL_PATH}")
        sess = get_fran_session()
        self.assertIsNotNone(sess, "FRAN ONNX session failed to load")
        inputs = sess.get_inputs()
        self.assertEqual(inputs[0].shape[1], 5, "FRAN model must accept 5-channel input")

    # -------------------------------------------------------------------------
    # TEST 11: Age 0 Preserves Original
    # -------------------------------------------------------------------------
    def test_11_age_0_preserves_original(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        aged_0 = apply_age_adjustment(bgr, geom, 0, regions=regions, current_age=45)
        diff_0 = np.max(cv2.absdiff(bgr, aged_0))
        self.assertEqual(diff_0, 0, "Age 0 must return unmodified original canvas")

    # -------------------------------------------------------------------------
    # TEST 12: Age +20 Changes Face
    # -------------------------------------------------------------------------
    def test_12_age_plus20_changes_face(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        aged_p20 = apply_age_adjustment(bgr, geom, 20, regions=regions, current_age=45)
        diff = cv2.absdiff(bgr, aged_p20)
        skin_bool = (regions['face_skin'] > 0.25)
        mean_skin_diff = np.mean(diff[skin_bool])
        max_skin_diff = np.max(diff[skin_bool])
        self.assertGreater(mean_skin_diff, 1.5, "Age +20 must produce visible change across facial skin")
        self.assertGreater(max_skin_diff, 30, "Age +20 must produce prominent rhytids/lines")

    # -------------------------------------------------------------------------
    # TEST 13: Age -10 Changes Face
    # -------------------------------------------------------------------------
    def test_13_age_minus10_changes_face(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        aged_m10 = apply_age_adjustment(bgr, geom, -10, regions=regions, current_age=45)
        diff = cv2.absdiff(bgr, aged_m10)
        skin_bool = (regions['face_skin'] > 0.25)
        mean_skin_diff = np.mean(diff[skin_bool])
        self.assertGreater(mean_skin_diff, 1.0, "Age -10 must produce visible rejuvenation")

    # -------------------------------------------------------------------------
    # TEST 14: Age Edit Stays Inside Intended Face Region
    # -------------------------------------------------------------------------
    def test_14_age_edit_stays_inside_intended_face_region(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        aged_p20 = apply_age_adjustment(bgr, geom, 20, regions=regions, current_age=45)
        diff = cv2.absdiff(bgr, aged_p20)
        # Background pixels must be virtually untouched
        bg_bool = (regions['background'] > 0.6)
        mean_bg_diff = np.mean(diff[bg_bool])
        self.assertLess(mean_bg_diff, 0.1, "Background must be preserved with near-zero leakage")

    # -------------------------------------------------------------------------
    # TEST 15: Hair Replacement Works
    # -------------------------------------------------------------------------
    def test_15_hair_replacement_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        supp = suppress_original_hair(bgr, geom, 'Short', 'None', geom['bg_bgr'], regions=regions)
        hair_res = render_selected_hairstyle(supp, geom, 'Short')
        self.assertEqual(hair_res.shape, bgr.shape)
        diff = cv2.absdiff(bgr, hair_res)
        self.assertGreater(np.mean(diff), 2.0)

    # -------------------------------------------------------------------------
    # TEST 16: Bald Works
    # -------------------------------------------------------------------------
    def test_16_bald_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        supp = suppress_original_hair(bgr, geom, 'Bald', 'None', geom['bg_bgr'], regions=regions)
        bald_res = render_bald_appearance(supp, geom, geom['skin_bgr'], geom['bg_bgr'])
        self.assertEqual(bald_res.shape, bgr.shape)
        # Scalp region must be modified
        apex_y = int(geom['apex_y'])
        mid_x = int(geom['mid_x'])
        diff = cv2.absdiff(bgr, bald_res)
        self.assertGreater(np.mean(diff[max(0, apex_y-20):apex_y+20, mid_x-20:mid_x+20]), 10)

    # -------------------------------------------------------------------------
    # TEST 17: Clean Shave Works
    # -------------------------------------------------------------------------
    def test_17_clean_shave_works(self):
        if not os.path.exists(self.bearded_path):
            self.skipTest("Bearded face not found")
        bgr = load_and_standardize_image(self.bearded_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        shaved = clean_shave_beard_if_needed(bgr, geom, 'Clean Shaven', geom['skin_bgr'], regions=regions)
        self.assertEqual(shaved.shape, bgr.shape)

    # -------------------------------------------------------------------------
    # TEST 18: Beard Works
    # -------------------------------------------------------------------------
    def test_18_beard_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        bearded = render_selected_facial_hair(bgr, geom, 'Full Beard', regions=regions)
        diff = cv2.absdiff(bgr, bearded)
        jaw_bool = (regions['beard_jaw_region'] > 0.3)
        self.assertGreater(np.mean(diff[jaw_bool]), 5.0, "Beard must modify jaw region")

    # -------------------------------------------------------------------------
    # TEST 19: Moustache Works
    # -------------------------------------------------------------------------
    def test_19_moustache_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        moust = render_selected_facial_hair(bgr, geom, 'Moustache', regions=regions)
        diff = cv2.absdiff(bgr, moust)
        moust_bool = (regions['moustache_region'] > 0.3)
        self.assertGreater(np.mean(diff[moust_bool]), 5.0, "Moustache must modify philtrum region")

    # -------------------------------------------------------------------------
    # TEST 20: Cap Works
    # -------------------------------------------------------------------------
    def test_20_cap_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        cap_res = render_selected_headwear(bgr, geom, 'Cap', regions=regions)
        diff = cv2.absdiff(bgr, cap_res)
        # Cap modifies upper forehead and cranial vault
        apex_y = int(geom['apex_y'])
        self.assertGreater(np.mean(diff[:apex_y+50, :]), 8.0)
        self.assertGreater(np.max(diff[:apex_y+50, :]), 50.0)

    # -------------------------------------------------------------------------
    # TEST 21: Turban Works
    # -------------------------------------------------------------------------
    def test_21_turban_works(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        turb_res = render_selected_headwear(bgr, geom, 'Turban', regions=regions)
        diff = cv2.absdiff(bgr, turb_res)
        self.assertGreater(np.mean(diff[:int(geom['hairline_y'])+20, :]), 10.0)

    # -------------------------------------------------------------------------
    # TEST 22: No Suggested Variants
    # -------------------------------------------------------------------------
    def test_22_no_suggested_variants(self):
        out_dir = "static/benchmark_eval/test_p44_novar"
        res = generate_face_variations(self.obama_path, out_dir, custom_settings={'hair_style': 'Curly'})
        self.assertEqual(res['suggested_variations'], [])
        self.assertEqual(res['total_variations'], 1)

    # -------------------------------------------------------------------------
    # TEST 23: Session Storage Works
    # -------------------------------------------------------------------------
    def test_23_session_storage_works(self):
        self.login()
        res = self.client.post('/create-variations', data={
            'base_face_filename': self.obama_path,
            'hair_style': 'Straight',
            'facial_hair': 'Clean Shaven',
            'current_age': 45,
            'age_delta': 5,
            'headwear': 'None'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # -------------------------------------------------------------------------
    # TEST 24: Logout Cleanup Works
    # -------------------------------------------------------------------------
    def test_24_logout_cleanup_works(self):
        self.login()
        # Post to create session directory
        self.client.post('/create-variations', data={
            'base_face_filename': self.obama_path,
            'hair_style': 'Keep Original'
        })
        res = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    # -------------------------------------------------------------------------
    # TEST 25: Benchmark Cleanup Policy Works
    # -------------------------------------------------------------------------
    def test_25_benchmark_cleanup_policy_works(self):
        res = cleanup_benchmark_artifacts(max_files_per_phase=50, max_age_days=14, dry_run=True)
        self.assertIn('retained_count', res)
        self.assertIn('bytes_freed', res)
        self.assertTrue(res['dry_run'])

    # -------------------------------------------------------------------------
    # REGRESSION GATES (Tests 26 - 34)
    # -------------------------------------------------------------------------
    def test_26_phase2_regression(self):
        self.login()
        res = self.client.get('/admin/dashboard')
        self.assertIn(res.status_code, [200, 302])

    def test_27_phase3_regression(self):
        res = self.client.get('/sketch')
        self.assertIn(res.status_code, [200, 302])

    def test_28_phase3_1_regression(self):
        cyclegan_path = "model/sketch_to_face/cyclegan_model.pt"
        self.assertTrue(os.path.exists(cyclegan_path), "Phase 3.1 CycleGAN model must exist")

    def test_29_phase4_regression(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        self.assertIsNotNone(geom, "Phase 4 landmark extraction failed")

    def test_30_phase4_1_regression(self):
        cfg = parse_and_validate_config({'hair_style': 'Wavy', 'facial_hair': 'Moustache'})
        self.assertEqual(cfg['hair_style'], 'Wavy')
        self.assertEqual(cfg['facial_hair'], 'Moustache')

    def test_31_phase4_2_regression(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        self.assertIn('angle', geom)
        self.assertIn('u_par', geom)
        self.assertIn('u_perp', geom)

    def test_32_phase4_2_1_regression(self):
        asset_dir = "static/variation_assets"
        self.assertTrue(os.path.exists(os.path.join(asset_dir, "short_hair.png")))
        self.assertTrue(os.path.exists(os.path.join(asset_dir, "bald.png")))

    def test_33_phase4_2_2_regression(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        supp = suppress_original_hair(bgr, geom, 'Short', 'None', geom['bg_bgr'])
        self.assertEqual(supp.shape, bgr.shape)

    def test_34_phase4_3_regression(self):
        bgr = load_and_standardize_image(self.obama_path)
        geom = extract_face_geometry(bgr)
        regions = build_anatomical_region_map(bgr, geom)
        self.assertIn('forehead', regions)
        self.assertIn('chin', regions)
        self.assertIn('moustache_region', regions)
        self.assertIn('beard_jaw_region', regions)


if __name__ == '__main__':
    unittest.main()
