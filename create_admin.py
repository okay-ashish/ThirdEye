import os
import sys
from app import app
from models import db, User

def add_admin_user(username=None, password=None):
    """Creates the admin user if it doesn't exist."""
    admin_username = username or os.environ.get('THIRDEYE_ADMIN_USER', 'admin')
    admin_password = password or os.environ.get('THIRDEYE_ADMIN_PASSWORD')

    if not admin_password:
        print("ERROR: Admin password is required.")
        print("Provide via command-line argument: python create_admin.py [username] <password>")
        print("or set the THIRDEYE_ADMIN_PASSWORD environment variable.")
        sys.exit(1)

    with app.app_context():
        # Check if the admin user already exists
        if User.query.filter_by(username=admin_username).first():
            print(f"Admin user '{admin_username}' already exists.")
            return

        # Create the admin user
        # The User model's __init__ method handles the hashing
        admin_user = User(username=admin_username, password=admin_password, role='admin', full_name='System Administrator')

        db.session.add(admin_user)
        db.session.commit()
        print(f"Admin user '{admin_username}' created successfully!")

if __name__ == '__main__':
    u = sys.argv[1] if len(sys.argv) > 1 else None
    p = sys.argv[2] if len(sys.argv) > 2 else None
    add_admin_user(u, p)
