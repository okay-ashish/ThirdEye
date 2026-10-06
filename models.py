from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user') # 'admin' or 'user'

    def __init__(self, username, password, role='user', full_name=None):
        self.username = username
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
        self.role = role
        self.full_name = full_name

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class Person(db.Model):
    __tablename__ = 'persons'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # Renamed 'metadata' to 'person_details' to avoid keyword conflict
    person_details = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    face_encoding = db.Column(db.Text, nullable=False)
