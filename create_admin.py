import os
import bcrypt
from app import app, db, User

def create_admin():
    with app.app_context():
        # Check if admin already exists
        admin = User.query.filter_by(role='admin').first()
        if admin:
            print(f"Admin already exists: {admin.email}")
            return

        # Create new admin
        email = "admin@eduguide.com"
        password = "admin123"
        hashed_pwd = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        new_admin = User(
            name="Platform Admin",
            email=email,
            password_hash=hashed_pwd,
            role="admin"
        )
        
        db.session.add(new_admin)
        db.session.commit()
        print(f"Admin account created successfully!")
        print(f"Email: {email}")
        print(f"Password: {password}")

if __name__ == "__main__":
    create_admin()
