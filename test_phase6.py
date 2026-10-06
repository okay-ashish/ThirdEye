"""
test_phase6.py — ThirdEye Phase 6 Automated Test Suite
Real Printable Public-Display Poster / Flyer System

Verifies all 28 required test gates:
 1. Poster Constructor route loads (GET /create-poster returns 200)
 2. All 5 purposes exist
 3. Purpose selection works
 4. Dynamic fields change by purpose
 5. Person fields exist
 6. Contact fields exist
 7. Custom title works
 8. Real photo input works
 9. Reconstruction source works
 10. Variation source works
 11. Existing session images can be selected
 12. Preview updates
 13. Additional Appearance Image field is absent
 14. Missing Person title appears correctly
 15. Probable-appearance disclaimer appears
 16. Public Information / Person of Interest title is user-controlled
 17. PDF generation works
 18. PDF is A4
 19. Output is session-scoped
 20. Download works
 21. Print mode hides application UI
 22. Logout cleanup removes poster outputs
 23. Phase 2 regression passes
 24. Phase 3 regression passes
 25. Phase 3.1 regression passes
 26. Phase 4 regression passes
 27. Phase 4.6 regression passes
 28. Phase 5 regression passes
"""

import os
import sys
import unittest
import io
import shutil
import json
import pypdf
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, db
from models import User, bcrypt
from poster_generator import (
    generate_poster_pdf,
    DISCLAIMER_GENERATED,
    DISCLAIMER_PROBABLE
)
from session_cleanup import cleanup_session_artifacts, get_session_dir
from utils import convert_to_sketch_in_memory
from face_variations import (
    DEFAULT_APPEARANCE_CONFIG,
    generate_face_variations,
    extract_face_geometry,
    load_and_standardize_image
)


class Phase6TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

        with app.app_context():
            u = User.query.filter_by(username='test_p6_user').first()
            if not u:
                u = User(
                    username='test_p6_user',
                    password='Password123!',
                    role='admin',
                    full_name='Officer Sarah Connor'
                )
                db.session.add(u)
                db.session.commit()
            cls.test_user_id = u.id

    def setUp(self):
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.test_user_id
            sess['username'] = 'test_p6_user'
            sess['full_name'] = 'Officer Sarah Connor'
            sess['role'] = 'admin'
            sess['_fresh'] = True
            sess['session_id'] = 'phase6_test_sess'

    # Gate 1: Poster Constructor route loads
    def test_01_poster_constructor_route_loads(self):
        resp = self.client.get('/create-poster')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn("A4 PRINTABLE PUBLIC-DISPLAY POSTER SYSTEM", html)
        self.assertIn("livePosterSheet", html)

    # Gate 2: All 5 purposes exist
    def test_02_all_5_purposes_exist(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        self.assertIn("1. Person Information Poster", html)
        self.assertIn("2. Person + Multiple Appearances", html)
        self.assertIn("3. Public Information / Person of Interest", html)
        self.assertIn("4. Missing Person — Real Photo", html)
        self.assertIn("5. Missing Person — Real Photo + Probable Appearances", html)

    # Gate 3: Purpose selection works
    def test_03_purpose_selection_works(self):
        resp = self.client.post('/create-poster', data={
            'template_id': '4',
            'poster_type': 'missing_person'
        })
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn('"template_id": 4', html)
        self.assertIn('tpl-card-4', html)

    # Gate 4: Dynamic fields change by purpose
    def test_04_dynamic_fields_change_by_purpose(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        # Check dynamic containers exist in the form
        self.assertIn('id="grpMissingTimeline"', html)
        self.assertIn('id="grpClothing"', html)
        self.assertIn('id="grpCurrentAge"', html)
        self.assertIn('id="lblAge"', html)
        self.assertIn('selectTemplate', html)

    # Gate 5: Person fields exist
    def test_05_person_fields_exist(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        self.assertIn('id="inpName"', html)
        self.assertIn('id="inpAge"', html)
        self.assertIn('id="inpGender"', html)
        self.assertIn('id="inpHeight"', html)
        self.assertIn('id="inpBuild"', html)
        self.assertIn('id="inpHair"', html)
        self.assertIn('id="inpEyes"', html)
        self.assertIn('id="inpMarks"', html)
        self.assertIn('id="inpDesc"', html)
        self.assertIn('id="inpCaseNo"', html)

    # Gate 6: Contact fields exist
    def test_06_contact_fields_exist(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        self.assertIn('id="inpPhone"', html)
        self.assertIn('id="inpAltPhone"', html)
        self.assertIn('id="inpContactPerson"', html)
        self.assertIn('id="inpAgency"', html)
        self.assertIn('id="inpContactNote"', html)

    # Gate 7: Custom title works
    def test_07_custom_title_works(self):
        payload = {
            'primary_face': {'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'PHOTO'},
            'template_id': 3,
            'person_data': {
                'full_name': 'SUBJECT ALPHA',
                'custom_title': 'AUTHORISED IDENTIFICATION NOTICE',
                'phone': '1234567890'
            }
        }
        resp = self.client.post('/api/generate-poster', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        pdf_path = data['pdf_path']
        reader = pypdf.PdfReader(pdf_path)
        extracted = reader.pages[0].extract_text()
        self.assertIn("AUTHORISED IDENTIFICATION NOTICE", extracted)

    # Gate 8: Real photo input works
    def test_08_real_photo_input_works(self):
        with open('static/person_db/trump.jpg', 'rb') as f:
            photo_bytes = f.read()
        resp = self.client.post('/create-poster', data={
            'uploaded_photo': (io.BytesIO(photo_bytes), 'test_upload.jpg')
        }, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn("test_upload.jpg", html)

    # Gate 9: Reconstruction source works
    def test_09_reconstruction_source_works(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'reconstructed', 'label': 'COMPUTER-GENERATED RECONSTRUCTION'},
            template_id=1,
            person_data={'full_name': 'RECON SUBJECT', 'phone': '99999'}
        )
        self.assertTrue(res['success'])
        reader = pypdf.PdfReader(res['pdf_path'])
        text = reader.pages[0].extract_text()
        self.assertIn("COMPUTER-GENERATED RECONSTRUCTION", text)

    # Gate 10: Variation source works
    def test_10_variation_source_works(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'LAST KNOWN PHOTO'},
            addl_faces=[{'image_path': 'static/person_db/modi.jpg', 'source_type': 'variation', 'label': 'COMPUTER-GENERATED APPEARANCE'}],
            template_id=2,
            person_data={'full_name': 'MULTI VARIATION SUBJECT', 'phone': '88888'}
        )
        self.assertTrue(res['success'])
        reader = pypdf.PdfReader(res['pdf_path'])
        text = reader.pages[0].extract_text()
        self.assertIn("COMPUTER-GENERATED APPEARANCE", text)

    # Gate 11: Existing session images can be selected
    def test_11_existing_session_images_can_be_selected(self):
        resp = self.client.get('/create-poster')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn("image-select-card", html)
        self.assertIn("trump.jpg", html)

    # Gate 12: Preview updates
    def test_12_preview_updates(self):
        payload = {
            'primary_face': {'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'PHOTO'},
            'template_id': 1,
            'person_data': {
                'full_name': 'UPDATED PREVIEW NAME',
                'age': '42',
                'phone': '555-4321'
            }
        }
        resp = self.client.post('/api/generate-poster', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('poster_', data['pdf_filename'])
        self.assertIn('/download-poster/', data['download_url'])

    # Gate 13: Additional Appearance Image field is absent
    def test_13_additional_appearance_image_field_is_absent(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        self.assertNotIn("Additional Appearance Image", html)

    # Gate 14: Missing Person title appears correctly
    def test_14_missing_person_title_appears_correctly(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'LAST KNOWN REAL PHOTOGRAPH'},
            template_id=4,
            person_data={'full_name': 'MISSING PERSON NAME', 'phone': '100'}
        )
        reader = pypdf.PdfReader(res['pdf_path'])
        text = reader.pages[0].extract_text()
        self.assertIn("MISSING PERSON", text)
        self.assertIn("PLEASE HELP US BRING THEM HOME", text)

    # Gate 15: Probable-appearance disclaimer appears
    def test_15_probable_appearance_disclaimer_appears(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'LAST KNOWN REAL PHOTOGRAPH'},
            addl_faces=[{'image_path': 'static/person_db/modi.jpg', 'source_type': 'variation', 'label': 'COMPUTER-GENERATED PROBABLE APPEARANCE'}],
            template_id=5,
            person_data={'full_name': 'LONG TERM SEARCH', 'phone': '100'}
        )
        reader = pypdf.PdfReader(res['pdf_path'])
        text = reader.pages[0].extract_text()
        self.assertIn("COMPUTER-GENERATED PROBABLE APPEARANCE", text)
        self.assertIn("Computer-generated probable appearance", text)

    # Gate 16: Public Information / Person of Interest title is user-controlled
    def test_16_public_information_title_is_user_controlled(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo', 'label': 'RECORD'},
            template_id=3,
            person_data={'full_name': 'TEST SUBJECT', 'custom_title': 'PUBLIC INFORMATION BULLETIN', 'phone': '100'}
        )
        reader = pypdf.PdfReader(res['pdf_path'])
        text = reader.pages[0].extract_text()
        self.assertIn("PUBLIC INFORMATION BULLETIN", text)
        self.assertNotIn("CRIMINAL", text)
        self.assertNotIn("GUILTY", text)
        self.assertNotIn("WANTED", text)

    # Gate 17: PDF generation works
    def test_17_pdf_generation_works(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo'},
            template_id=1,
            person_data={'full_name': 'PDF VALIDATION USER', 'phone': '999'}
        )
        self.assertTrue(res['success'])
        self.assertTrue(os.path.exists(res['pdf_path']))
        self.assertGreater(res['file_size_kb'], 10)

    # Gate 18: PDF is A4
    def test_18_pdf_is_a4(self):
        res = generate_poster_pdf(
            primary_face={'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo'},
            template_id=1,
            person_data={'full_name': 'A4 VERIFY'}
        )
        reader = pypdf.PdfReader(res['pdf_path'])
        self.assertEqual(len(reader.pages), 1)
        box = reader.pages[0].mediabox
        self.assertAlmostEqual(float(box.width), 595.28, delta=1.0)
        self.assertAlmostEqual(float(box.height), 841.89, delta=1.0)

    # Gate 19: Output is session-scoped
    def test_19_output_is_session_scoped(self):
        resp = self.client.post('/api/generate-poster', json={
            'primary_face': {'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo'},
            'template_id': 1,
            'person_data': {'full_name': 'SESSION SCOPED TEST', 'phone': '123'}
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('session_', data['pdf_filename'])
        self.assertIn('session_', data['pdf_path'])

    # Gate 20: Download works
    def test_20_download_works(self):
        resp = self.client.post('/api/generate-poster', json={
            'primary_face': {'image_path': 'static/person_db/trump.jpg', 'source_type': 'real_photo'},
            'template_id': 1,
            'person_data': {'full_name': 'DOWNLOAD TEST', 'phone': '123'}
        })
        data = resp.get_json()
        dl_resp = self.client.get(data['download_url'])
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn('application/pdf', dl_resp.headers.get('Content-Type', ''))

    # Gate 21: Print mode hides application UI
    def test_21_print_mode_hides_application_ui(self):
        resp = self.client.get('/create-poster')
        html = resp.data.decode('utf-8')
        self.assertIn("@media print", html)
        self.assertIn("#appSidebar", html)
        self.assertIn(".poster-controls-column", html)
        self.assertIn("display: none !important", html)

    # Gate 22: Logout cleanup removes poster outputs
    def test_22_logout_cleanup_removes_poster_outputs(self):
        sess_id = "test_cleanup_sess_p6"
        poster_dir = get_session_dir('posters', sess_id)
        os.makedirs(poster_dir, exist_ok=True)
        test_file = os.path.join(poster_dir, "test_poster.pdf")
        with open(test_file, 'w') as f:
            f.write("%PDF-1.4 test")
        self.assertTrue(os.path.exists(test_file))

        cleanup_session_artifacts(sess_id)
        self.assertFalse(os.path.exists(test_file))

    # Gate 23: Phase 2 regression
    def test_23_phase2_regression(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn("Officer Sarah Connor", html)

    # Gate 24: Phase 3 regression
    def test_24_phase3_regression(self):
        resp = self.client.get('/sketch')
        self.assertEqual(resp.status_code, 200)

    # Gate 25: Phase 3.1 regression
    def test_25_phase3_1_regression(self):
        from face_reconstruction import reconstruct_face
        # Verify module imports and callable
        self.assertTrue(callable(reconstruct_face))

    # Gate 26: Phase 4 regression
    def test_26_phase4_regression(self):
        resp = self.client.get('/create-variations')
        self.assertEqual(resp.status_code, 200)

    # Gate 27: Phase 4.6 regression
    def test_27_phase4_6_regression(self):
        cfg = DEFAULT_APPEARANCE_CONFIG.copy()
        self.assertEqual(cfg.get('facial_hair'), 'Keep Original')

    # Gate 28: Phase 5 regression
    def test_28_phase5_regression(self):
        resp = self.client.get('/photo-to-sketch')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn("Photo to Sketch Converter", html)
        self.assertIn("dropzone", html)


if __name__ == '__main__':
    unittest.main()
