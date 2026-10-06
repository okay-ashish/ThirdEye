import os
import sys
import unittest
from app import app, db
from models import User, Person, bcrypt

TEST_FIXTURE_PASS = 'TestFixturePassword123!'

class Phase2TestSuite(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_01_environment_packages(self):
        """Verify package versions against Phase 2 requirements"""
        import numpy
        import tensorflow
        import cv2
        import dlib
        import face_recognition
        import PIL
        import flask
        import flask_sqlalchemy
        import flask_bcrypt
        import mysql.connector
        import reportlab

        print("\n--- PACKAGE ENVIRONMENT VERIFICATION ---")
        print(f"Python: {sys.version.split()[0]}")
        print(f"NumPy: {numpy.__version__}")
        print(f"TensorFlow: {tensorflow.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        print(f"dlib: {dlib.__version__}")
        print(f"face_recognition: {face_recognition.__version__}")
        print(f"Pillow: {PIL.__version__}")
        print(f"Flask: {flask.__version__}")
        print(f"Flask-SQLAlchemy: {flask_sqlalchemy.__version__}")
        print(f"Flask-Bcrypt: {flask_bcrypt.__version__}")
        print(f"mysql-connector-python: {mysql.connector.__version__}")
        print(f"ReportLab: {reportlab.__version__}")

        self.assertEqual(sys.version_info[:2], (3, 11))
        self.assertTrue(numpy.__version__.startswith('1.26'))

    def test_02_registration_with_fullname(self):
        """Test A: New user registration with full_name"""
        with app.app_context():
            existing = User.query.filter_by(username='test_new').first()
            if existing:
                db.session.delete(existing)
                db.session.commit()

        response = self.client.post('/register', data={
            'full_name': 'Test User',
            'username': 'test_new',
            'password': TEST_FIXTURE_PASS,
            'confirm_password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Account created successfully', response.data)

        with app.app_context():
            user = User.query.filter_by(username='test_new').first()
            self.assertIsNotNone(user)
            self.assertEqual(user.full_name, 'Test User')
            self.assertTrue(user.check_password(TEST_FIXTURE_PASS))
            self.assertFalse(user.password == TEST_FIXTURE_PASS)

    def test_03_login_with_fullname_greeting(self):
        """Test B: Login and session full_name greeting"""
        response = self.client.post('/login', data={
            'username': 'test_new',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        # Check flash message contains full name
        self.assertIn(b'Welcome, Test User', response.data)
        # Check dashboard greeting
        self.assertIn(b'Welcome, Test User', response.data)
        self.assertIn(b'Your computer vision workspace', response.data)

        # Check sidebar contains user's full name
        self.assertIn(b'Test User', response.data)
        # Check sidebar links are present
        self.assertIn(b'Sketch Constructor', response.data)
        self.assertIn(b'Photo to Sketch', response.data)
        self.assertIn(b'Face Recognition', response.data)
        self.assertIn(b'Face Variations', response.data)
        self.assertIn(b'Create Poster', response.data)
        self.assertIn(b'Logout', response.data)

    def test_04_logout_clears_session(self):
        """Test C: Logout clears session and artifacts"""
        self.client.post('/login', data={
            'username': 'test_new',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'You have been logged out', response.data)
        self.assertIn(b'Get Started', response.data)
        self.assertIn(b'Vision Intelligence', response.data)

    def test_05_existing_user_null_fullname_fallback(self):
        """Test D: Existing user with NULL full_name falls back safely to username"""
        with app.app_context():
            user = User.query.filter_by(username='test_null_name').first()
            if not user:
                user = User(username='test_null_name', password=TEST_FIXTURE_PASS, role='user', full_name=None)
                db.session.add(user)
                db.session.commit()
            else:
                user.password = bcrypt.generate_password_hash(TEST_FIXTURE_PASS).decode('utf-8')
                user.full_name = None
                db.session.commit()

        response = self.client.post('/login', data={
            'username': 'test_null_name',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome, test_null_name', response.data)
        self.assertIn(b'test_null_name', response.data)

    def test_06_admin_access_and_protection(self):
        """Test E: Admin access and protection"""
        # Regular user trying to access admin dashboard
        self.client.post('/login', data={
            'username': 'test_new',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        resp = self.client.get('/admin/dashboard', follow_redirects=True)
        self.assertIn(b'You do not have permission', resp.data)

        self.client.get('/logout')

        # Admin user login
        with app.app_context():
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                admin_user = User(username='admin', password=TEST_FIXTURE_PASS, role='admin', full_name='System Administrator')
                db.session.add(admin_user)
            else:
                admin_user.password = bcrypt.generate_password_hash(TEST_FIXTURE_PASS).decode('utf-8')
            db.session.commit()

        resp_admin = self.client.post('/login', data={
            'username': 'admin',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        self.assertEqual(resp_admin.status_code, 200)
        self.assertIn(b'Admin Control Panel', resp_admin.data)
        self.assertIn(b'Admin Dashboard', resp_admin.data)

    def test_07_navigation_and_functional_routes(self):
        """Verify all functional routes open properly with new layout"""
        self.client.post('/login', data={
            'username': 'test_new',
            'password': TEST_FIXTURE_PASS
        }, follow_redirects=True)

        routes = [
            ('/', 200),
            ('/sketch', 200),
            ('/photo-to-sketch', 200),
            ('/recognition', 200),
            ('/create-variations', [200, 302]),
            ('/create-poster', 200),
            ('/construct-face', 302),
        ]

        for route, expected_status in routes:
            resp = self.client.get(route)
            allowed = expected_status if isinstance(expected_status, list) else [expected_status]
            self.assertIn(resp.status_code, allowed, f"Route {route} returned unexpected status {resp.status_code}")

            # If 200, verify new layout integration
            if resp.status_code == 200:
                self.assertIn(b'app-sidebar', resp.data, f"Sidebar missing on route {route}")
                self.assertIn(b'toast-container-custom', resp.data, f"Toast container missing on route {route}")
                self.assertIn(b'2026 ThirdEye', resp.data, f"2026 footer missing on route {route}")
            print(f"Verified route {route} status {resp.status_code} with new layout.")

    def test_08_public_landing_page(self):
        """Verify public landing page elements and wording"""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'ThirdEye', resp.data)
        self.assertIn(b'Why ThirdEye Matters', resp.data)
        self.assertIn(b'Core System Features', resp.data)
        self.assertIn(b'Our Purpose', resp.data)
        self.assertIn(b'SKETCH CONSTRUCTOR', resp.data)
        self.assertIn(b'PHOTO TO SKETCH', resp.data)
        self.assertIn(b'CONSTRUCT FACE', resp.data)
        self.assertIn(b'FACE VARIATIONS', resp.data)
        self.assertIn(b'FACE RECOGNITION', resp.data)
        self.assertIn(b'POSTER CONSTRUCTOR', resp.data)
        self.assertIn(b'2026 ThirdEye', resp.data)

    def test_09_auth_pages_no_video_no_hardcoded_values(self):
        """Verify login and register pages have pure CSS backgrounds, no video, no hardcoded values"""
        resp_login = self.client.get('/login')
        self.assertEqual(resp_login.status_code, 200)
        self.assertNotIn(b'registrationbg.mp4', resp_login.data)
        self.assertNotIn(b'<video', resp_login.data)
        self.assertNotIn(b'value="username"', resp_login.data)
        self.assertNotIn(b'value="password"', resp_login.data)
        self.assertIn(b'autocomplete="username"', resp_login.data)
        self.assertIn(b'autocomplete="current-password"', resp_login.data)

        resp_register = self.client.get('/register')
        self.assertEqual(resp_register.status_code, 200)
        self.assertNotIn(b'registrationbg.mp4', resp_register.data)
        self.assertNotIn(b'<video', resp_register.data)
        self.assertIn(b'name="full_name"', resp_register.data)
        self.assertIn(b'autocomplete="name"', resp_register.data)
        self.assertIn(b'autocomplete="new-password"', resp_register.data)

if __name__ == '__main__':
    unittest.main()
