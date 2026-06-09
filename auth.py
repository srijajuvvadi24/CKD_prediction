import pandas as pd
import os
import re
import hashlib

FILE = "users.csv"

# 🔹 Load users
def load_users():
    if not os.path.exists(FILE):
        return pd.DataFrame(columns=["username", "password"])
    return pd.read_csv(FILE)

# 🔹 Username validation (only alphabets)
def valid_username(username):
    return username.isalpha()

# 🔹 Password validation
def valid_password(password):
    if len(password) < 6:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[!@#$%^&*]", password):
        return False
    return True

# 🔹 Hash password (security)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 🔹 Save user
def save_user(username, password):

    # Validation
    if not valid_username(username):
        return "Username must contain only alphabets ❌"

    if not valid_password(password):
        return "Password must contain uppercase, lowercase, number, special character (!@#$%^&*), and be at least 6 characters ❌"

    df = load_users()

    # Check duplicate user
    if username in df["username"].values:
        return "Username already exists ❌"

    # Hash password
    password = hash_password(password)

    df = pd.concat([df, pd.DataFrame([[username, password]], columns=["username", "password"])])
    df.to_csv(FILE, index=False)

    return "Account Created Successfully ✅"

# 🔹 Check login
def check_user(username, password):
    df = load_users()
    password = hash_password(password)
    return ((df["username"] == username) & (df["password"] == password)).any()
