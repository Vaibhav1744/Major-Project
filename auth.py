import sys
sys.path.append('./libs')
import bcrypt
from database import users_collection

def hash_password(password):
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user."""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def register_user(email, username, password, preferences=None):
    """Register a new user."""
    if preferences is None:
        preferences = {}
    
    # Check if email or username already exists
    if users_collection.find_one({"email": email}):
        return False, "Email already registered"
    
    if users_collection.find_one({"username": username}):
        return False, "Username already taken"
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Create user document
    user = {
        "email": email,
        "username": username,
        "password": hashed_password,
        "preferences": preferences
    }
    
    # Insert user into database
    users_collection.insert_one(user)
    return True, "Registration successful"

def login_user(username_or_email, password):
    """Authenticate a user and return user data if successful."""
    # Find user by username or email
    user = users_collection.find_one({
        "$or": [
            {"username": username_or_email},
            {"email": username_or_email}
        ]
    })
    
    if not user:
        return False, "Invalid username or email", None
    
    # Verify password
    if not verify_password(user["password"], password):
        return False, "Incorrect password", None
    
    # Return user data (excluding password)
    user_data = {k: v for k, v in user.items() if k != 'password'}
    return True, "Login successful", user_data