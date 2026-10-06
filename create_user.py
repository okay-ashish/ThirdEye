# create_user.py
import os
import sys
from app import app
from models import db, User

def add_user(username, password, role='user', full_name=None):
    """
    Creates a new user with a securely hashed password.
    """
    with app.app_context():
        # Check if the user already exists to avoid errors
        if User.query.filter_by(username=username).first():
            print(f"User '{username}' already exists.")
            return

        # Create a new User object.
        # The __init__ method in the User model automatically handles password hashing.
        new_user = User(username=username, password=password, role=role, full_name=full_name)

        # Add the new user to the database session
        db.session.add(new_user)
        db.session.commit()

        print(f"User '{username}' with role '{role}' created successfully!")
        print("You can now log in with this account.")

if __name__ == '__main__':
    # CLI arguments or environment variables
    new_username = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('THIRDEYE_NEW_USER')
    new_password = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('THIRDEYE_NEW_PASS')
    new_role = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('THIRDEYE_NEW_ROLE', 'user')

    if not new_username or not new_password:
        print("ERROR: Username and password are required.")
        print("Usage: python create_user.py <username> <password> [role]")
        print("   or set THIRDEYE_NEW_USER and THIRDEYE_NEW_PASS environment variables.")
        sys.exit(1)

    # Call the function to add the user
    add_user(new_username, new_password, new_role)
