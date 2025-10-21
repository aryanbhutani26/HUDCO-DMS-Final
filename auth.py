"""
Simple authentication system for HUDCO Document Management
"""
import sqlite3
import hashlib
import streamlit as st
from pathlib import Path

class AuthManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the users database with default admin"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create default admin if not exists
        admin_hash = self.hash_password("admin123")
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role) 
            VALUES (?, ?, ?)
        ''', ("admin", admin_hash, "admin"))
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username, password):
        """Authenticate user and return role if successful"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute(
            "SELECT role FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def register_user(self, username, password, role="user"):
        """Register a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def get_all_users(self):
        """Get all users (admin only)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        conn.close()
        
        return users

def require_auth():
    """Streamlit authentication decorator"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
    
    if not st.session_state.authenticated:
        show_login()
        return False
    
    return True

def show_login():
    """Display login form"""
    st.title("🔐 HUDCO Document Management - Login")
    
    auth_manager = AuthManager()
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", type="primary"):
                if username and password:
                    role = auth_manager.authenticate(username, password)
                    if role:
                        st.session_state.authenticated = True
                        st.session_state.user_role = role
                        st.session_state.username = username
                        st.success(f"Welcome {username}! ({role})")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                else:
                    st.error("Please enter both username and password")
        
        with col2:
            st.info("**Demo Credentials:**\n- Admin: `admin` / `admin123`")
    
    with tab2:
        st.subheader("Register New User")
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        
        if st.button("Register"):
            if new_username and new_password:
                if auth_manager.register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please enter both username and password")

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()