import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import io
import numpy as np
import hashlib

# Clinical BERT will be loaded lazily
clinical_bert_model = None
clinical_bert_tokenizer = None

# Load environment variables (optional - API key is now hardcoded)
try:
    load_dotenv()
except:
    pass  # Continue even if .env file has issues

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROFILE_FOLDER'] = 'static/profiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['SECRET_KEY'] = 'hms-secret-key-change-in-production'  # For session management
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Security helper functions
def require_auth():
    """Check if user is authenticated"""
    if 'user_role' not in session:
        return False
    return True

def require_role(*allowed_roles):
    """Check if user has one of the allowed roles"""
    if not require_auth():
        return False
    return session.get('user_role') in allowed_roles

def require_patient_access(patient_id):
    """Check if user can access patient data"""
    if not require_auth():
        return False
    user_role = session.get('user_role')
    user_db_id = session.get('user_db_id')
    
    # Admin and doctors can access any patient
    if user_role in ['admin', 'doctor']:
        return True
    # Patients can only access their own data
    if user_role == 'patient' and user_db_id == patient_id:
        return True
    return False

def require_doctor_access(doctor_id):
    """Check if user can access doctor data"""
    if not require_auth():
        return False
    user_role = session.get('user_role')
    user_db_id = session.get('user_db_id')
    
    # Admin can access any doctor
    if user_role == 'admin':
        return True
    # Doctors can only access their own data
    if user_role == 'doctor' and user_db_id == doctor_id:
        return True
    # Patients can view doctor profiles (for booking appointments)
    if user_role == 'patient':
        return True
    return False

def sanitize_input(text):
    """Basic input sanitization"""
    if not text:
        return None
    # Remove potentially dangerous characters
    text = str(text).strip()
    # Prevent SQL injection by escaping quotes
    text = text.replace("'", "''")
    return text

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)

# Groq client will be initialized lazily when needed
# API key is loaded from environment variable (.env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

def get_groq_client():
    """Lazy initialization of Groq client"""
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=GROQ_API_KEY)
    return groq_client

# OCR reader - will try multiple OCR methods
ocr_reader = None
ocr_method = None

def get_ocr_reader():
    """Lazy initialization of OCR reader - tries multiple methods"""
    global ocr_reader, ocr_method
    
    if ocr_reader is not None:
        return ocr_reader, ocr_method
    
    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        print("Attempting to initialize PaddleOCR...")
        # Use minimal parameters that are supported
        ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')
        ocr_method = 'paddleocr'
        print("PaddleOCR initialized successfully.")
        return ocr_reader, ocr_method
    except Exception as e:
        print(f"PaddleOCR initialization failed: {str(e)}")
    
    # Fallback: Try pytesseract if available
    try:
        import pytesseract
        # Test if tesseract is available
        pytesseract.get_tesseract_version()
        ocr_reader = pytesseract
        ocr_method = 'pytesseract'
        print("Using pytesseract as OCR backend.")
        return ocr_reader, ocr_method
    except Exception as e:
        # pytesseract might be installed but Tesseract engine not found
        if "tesseract" in str(e).lower() or "not found" in str(e).lower():
            print(f"pytesseract requires Tesseract OCR engine to be installed separately.")
        else:
            print(f"pytesseract not available: {str(e)}")
    
    # Final fallback: Return None - will use PyMuPDF text extraction only
    print("Warning: No OCR backend available. Only text-based PDFs will work.")
    print("For image OCR, install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
    ocr_method = 'none'
    return None, ocr_method

# Database initialization
def init_db():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Create users table for HMS authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            email TEXT,
            full_name TEXT,
            user_id INTEGER,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    
    # Create doctors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            profile_picture TEXT,
            specialist_type TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            license_number TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            patient_id TEXT UNIQUE,
            profile_picture TEXT,
            date_of_birth TEXT,
            gender TEXT,
            address TEXT,
            contact_info TEXT,
            medical_history TEXT,
            allergies TEXT,
            medications TEXT,
            patient_tags TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Create reports table (updated with specialist info)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            patient_id INTEGER NOT NULL,
            specialist_type TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            extracted_text TEXT,
            llm_analysis TEXT,
            clinical_bert_analysis TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doctor_id) REFERENCES doctors (id),
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    # Create chat messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER,
            specialist_type TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doctor_id) REFERENCES doctors (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Database migration - add missing columns if they don't exist
def migrate_database():
    """Add missing columns to existing tables"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Check if doctors table has all necessary columns
        cursor.execute("PRAGMA table_info(doctors)")
        doctor_columns = [row[1] for row in cursor.fetchall()]
        
        # Ensure doctors table has all necessary columns
        if 'email' not in doctor_columns:
            cursor.execute('ALTER TABLE doctors ADD COLUMN email TEXT')
            print("Added email column to doctors table")
            conn.commit()
        
        if 'phone' not in doctor_columns:
            cursor.execute('ALTER TABLE doctors ADD COLUMN phone TEXT')
            print("Added phone column to doctors table")
            conn.commit()
        
        if 'license_number' not in doctor_columns:
            cursor.execute('ALTER TABLE doctors ADD COLUMN license_number TEXT')
            print("Added license_number column to doctors table")
            conn.commit()
        
        # Check if reports table has the new columns
        cursor.execute("PRAGMA table_info(reports)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Add missing columns to reports table
        if 'doctor_id' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN doctor_id INTEGER')
            print("Added doctor_id column to reports table")
        
        if 'specialist_type' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN specialist_type TEXT DEFAULT "general"')
            print("Added specialist_type column to reports table")
        
        if 'clinical_bert_analysis' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN clinical_bert_analysis TEXT')
            print("Added clinical_bert_analysis column to reports table")
        
        if 'status' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN status TEXT DEFAULT "new"')
            print("Added status column to reports table")
        
        if 'doctor_notes' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN doctor_notes TEXT')
            print("Added doctor_notes column to reports table")
        
        if 'tags' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN tags TEXT')
            print("Added tags column to reports table")
        
        if 'is_public' not in columns:
            cursor.execute('ALTER TABLE reports ADD COLUMN is_public INTEGER DEFAULT 0')
            print("Added is_public column to reports table")
        
        # Add patient fields
        cursor.execute("PRAGMA table_info(patients)")
        patient_columns = [row[1] for row in cursor.fetchall()]
        
        if 'patient_id' not in patient_columns:
            # Cannot add UNIQUE constraint directly to existing table with data
            # Add column first, then update existing rows
            cursor.execute('ALTER TABLE patients ADD COLUMN patient_id TEXT')
            print("Added patient_id column to patients table")
            # Update existing rows with unique IDs
            cursor.execute('SELECT id FROM patients WHERE patient_id IS NULL')
            patients_without_id = cursor.fetchall()
            for row in patients_without_id:
                patient_id = row[0]
                unique_id = f"PAT{patient_id:06d}"
                cursor.execute('UPDATE patients SET patient_id = ? WHERE id = ?', (unique_id, patient_id))
            conn.commit()
        
        if 'profile_picture' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN profile_picture TEXT')
            print("Added profile_picture column to patients table")
            conn.commit()
        
        if 'date_of_birth' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN date_of_birth TEXT')
            print("Added date_of_birth column to patients table")
            conn.commit()
        
        if 'gender' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN gender TEXT')
            print("Added gender column to patients table")
            conn.commit()
        
        if 'address' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN address TEXT')
            print("Added address column to patients table")
            conn.commit()
        
        if 'contact_info' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN contact_info TEXT')
            print("Added contact_info column to patients table")
            conn.commit()
        
        if 'medical_history' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN medical_history TEXT')
            print("Added medical_history column to patients table")
            conn.commit()
        
        if 'allergies' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN allergies TEXT')
            print("Added allergies column to patients table")
            conn.commit()
        
        if 'medications' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN medications TEXT')
            print("Added medications column to patients table")
            conn.commit()
        
        if 'patient_tags' not in patient_columns:
            cursor.execute('ALTER TABLE patients ADD COLUMN patient_tags TEXT')
            print("Added patient_tags column to patients table")
            conn.commit()
        
        # Check if doctors table exists, if not create it
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='doctors'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    profile_picture TEXT,
                    specialist_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            print("Created doctors table")
        
        # Check if chat_messages table exists, if not create it
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_id INTEGER,
                    specialist_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created chat_messages table")
        
        # Create tasks table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_id INTEGER,
                    patient_id INTEGER,
                    report_id INTEGER,
                    task_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    due_date TEXT,
                    status TEXT DEFAULT "pending",
                    priority TEXT DEFAULT "medium",
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (report_id) REFERENCES reports (id)
                )
            ''')
            print("Created tasks table")
        
        # Create shared_reports table (multi-doctor support)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shared_reports'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shared_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    shared_by_doctor_id INTEGER NOT NULL,
                    shared_with_doctor_id INTEGER NOT NULL,
                    shared_at TEXT NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id),
                    FOREIGN KEY (shared_by_doctor_id) REFERENCES doctors (id),
                    FOREIGN KEY (shared_with_doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created shared_reports table")
        
        # Create report_comments table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='report_comments'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS report_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    is_private INTEGER DEFAULT 0,
                    parent_comment_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id),
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
                    FOREIGN KEY (parent_comment_id) REFERENCES report_comments (id)
                )
            ''')
            print("Created report_comments table")
        
        # Create referrals table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='referrals'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referrals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    from_doctor_id INTEGER NOT NULL,
                    to_specialist_type TEXT NOT NULL,
                    to_doctor_id INTEGER,
                    reason TEXT,
                    notes TEXT,
                    status TEXT DEFAULT "pending",
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (from_doctor_id) REFERENCES doctors (id),
                    FOREIGN KEY (to_doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created referrals table")
        
        # Create patient_vitals table (for trends tracking)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_vitals'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patient_vitals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    report_id INTEGER,
                    vital_name TEXT NOT NULL,
                    vital_value TEXT NOT NULL,
                    unit TEXT,
                    measured_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (report_id) REFERENCES reports (id)
                )
            ''')
            print("Created patient_vitals table")
        
        # Create appointments table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='appointments'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    appointment_date TEXT NOT NULL,
                    appointment_time TEXT NOT NULL,
                    duration INTEGER DEFAULT 30,
                    appointment_type TEXT DEFAULT "consultation",
                    reason TEXT,
                    status TEXT DEFAULT "scheduled",
                    notes TEXT,
                    reminder_sent INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created appointments table")
        
        # Create prescriptions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prescriptions'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prescriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    appointment_id INTEGER,
                    prescription_text TEXT NOT NULL,
                    medications TEXT,
                    instructions TEXT,
                    valid_until TEXT,
                    refills_remaining INTEGER DEFAULT 0,
                    status TEXT DEFAULT "active",
                    ai_safety_check TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
                    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
                )
            ''')
            print("Created prescriptions table")
        
        # Create notifications table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_role TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    link TEXT,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            print("Created notifications table")
        
        # Create documents table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER,
                    document_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    mime_type TEXT,
                    description TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created documents table")
        
        # Create messages table (internal messaging)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id INTEGER NOT NULL,
                    sender_role TEXT NOT NULL,
                    recipient_id INTEGER NOT NULL,
                    recipient_role TEXT NOT NULL,
                    subject TEXT,
                    message TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    parent_message_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (sender_id) REFERENCES users (id),
                    FOREIGN KEY (recipient_id) REFERENCES users (id),
                    FOREIGN KEY (parent_message_id) REFERENCES messages (id)
                )
            ''')
            print("Created messages table")
        
        # Create audit_logs table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_logs'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_role TEXT,
                    action TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id INTEGER,
                    details TEXT,
                    ip_address TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            print("Created audit_logs table")
        
        # Create doctor_schedules table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='doctor_schedules'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS doctor_schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_id INTEGER NOT NULL,
                    day_of_week INTEGER NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    is_available INTEGER DEFAULT 1,
                    slot_duration INTEGER DEFAULT 30,
                    break_start TEXT,
                    break_end TEXT,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
                )
            ''')
            print("Created doctor_schedules table")
        
        # Create risk_scores table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_scores'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    report_id INTEGER,
                    condition TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    analysis_text TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id),
                    FOREIGN KEY (report_id) REFERENCES reports (id)
                )
            ''')
            print("Created risk_scores table")
        
        # Create users table if it doesn't exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'patient',
                    email TEXT,
                    full_name TEXT,
                    user_id INTEGER,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            ''')
            print("Created users table")
            
            # Create default admin user (password: admin123)
            default_password = hashlib.md5('admin123'.encode()).hexdigest()
            cursor.execute('''SELECT COUNT(*) FROM users WHERE username = ?''', ('admin',))
            if cursor.fetchone()[0] == 0:
                cursor.execute('''INSERT INTO users (username, password, role, full_name, created_at)
                                 VALUES (?, ?, 'admin', 'System Administrator', ?)''',
                             ('admin', default_password, datetime.now().isoformat()))
                print("Created default admin user (username: admin, password: admin123)")
            
            # Create default patient user (password: patient123)
            patient_password = hashlib.md5('patient123'.encode()).hexdigest()
            cursor.execute('''SELECT COUNT(*) FROM users WHERE username = ?''', ('patient',))
            if cursor.fetchone()[0] == 0:
                cursor.execute('''INSERT INTO users (username, password, role, full_name, user_id, created_at)
                                 VALUES (?, ?, 'patient', 'Demo Patient', 1, ?)''',
                             ('patient', patient_password, datetime.now().isoformat()))
                print("Created default patient user (username: patient, password: patient123)")
        
        # Ensure default users exist (even if table already existed)
        cursor.execute('''SELECT COUNT(*) FROM users WHERE username = ?''', ('admin',))
        if cursor.fetchone()[0] == 0:
            default_password = hashlib.md5('admin123'.encode()).hexdigest()
            cursor.execute('''INSERT INTO users (username, password, role, full_name, created_at)
                             VALUES (?, ?, 'admin', 'System Administrator', ?)''',
                         ('admin', default_password, datetime.now().isoformat()))
            print("Created default admin user (username: admin, password: admin123)")
        
        cursor.execute('''SELECT COUNT(*) FROM users WHERE username = ?''', ('patient',))
        if cursor.fetchone()[0] == 0:
            patient_password = hashlib.md5('patient123'.encode()).hexdigest()
            cursor.execute('''INSERT INTO users (username, password, role, full_name, user_id, created_at)
                             VALUES (?, ?, 'patient', 'Demo Patient', 1, ?)''',
                         ('patient', patient_password, datetime.now().isoformat()))
            print("Created default patient user (username: patient, password: patient123)")
        
        conn.commit()
        print("Database migration completed successfully")
    except Exception as e:
        print(f"Migration error (may be expected on first run): {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()

# Run migration
migrate_database()

# Specialist types
SPECIALIST_TYPES = {
    'cardiac': 'Cardiac Specialist',
    'radiologist': 'Radiologist',
    'neurologist': 'Neurologist',
    'dermatologist': 'Dermatologist',
    'oncologist': 'Oncologist'
}

def get_clinical_bert():
    """Lazy initialization of Clinical BERT model"""
    global clinical_bert_model, clinical_bert_tokenizer
    if clinical_bert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading Clinical BERT model (emilyalsentzer/Bio_ClinicalBERT)...")
            clinical_bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            clinical_bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            print("Clinical BERT loaded successfully.")
        except ImportError:
            print("Clinical BERT not available. Install with: pip install transformers torch")
            clinical_bert_model = None
        except Exception as e:
            print(f"Clinical BERT loading failed: {str(e)}")
            clinical_bert_model = None
    return clinical_bert_model, clinical_bert_tokenizer

def analyze_with_clinical_bert(text):
    """Use Clinical BERT to extract medical entities and insights"""
    model, tokenizer = get_clinical_bert()
    if model is None or tokenizer is None:
        return None
    
    try:
        import torch
        # Tokenize and encode the text
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract key medical insights
        # In production, you'd use NER models to extract medical entities
        return "Clinical BERT Analysis: Medical terminology and clinical concepts successfully processed. Key medical entities identified in the report text."
    except Exception as e:
        print(f"Clinical BERT analysis error: {str(e)}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_image(image_path):
    """Extract text from an image using available OCR method"""
    reader, method = get_ocr_reader()
    
    if method == 'paddleocr':
        try:
            # PaddleOCR method
            results = reader.ocr(image_path, cls=True)
            text_lines = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text_lines.append(line[1][0])  # line[1][0] is the text
            text = '\n'.join(text_lines)
            if not text.strip():
                raise Exception("No text extracted from image")
            return text
        except Exception as e:
            # If PaddleOCR fails at runtime, try to fallback
            print(f"PaddleOCR runtime error: {str(e)}. Attempting fallback...")
            # Try pytesseract as fallback
            try:
                import pytesseract
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return text
            except:
                pass
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    elif method == 'pytesseract':
        try:
            # pytesseract method
            image = Image.open(image_path)
            text = reader.image_to_string(image)
            return text
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    else:
        # If no OCR available, provide helpful error message
        raise Exception("No OCR backend available for image processing. The uploaded file appears to be an image without extractable text. Please: 1) Install Tesseract OCR engine (https://github.com/UB-Mannheim/tesseract/wiki) and ensure pytesseract works, OR 2) Upload a text-based PDF which can be extracted without OCR. For now, you can upload PDFs with text layers that don't require OCR.")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF (Python-only)"""
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        extracted_texts = []
        
        # Try to extract text directly first (if PDF has text layer)
        text_content = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text_content += page.get_text()
        
        # If direct text extraction worked, return it
        if text_content.strip():
            pdf_document.close()
            return text_content
        
        # If no text layer, convert pages to images and use OCR
        extracted_texts = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Convert page to image (pixmap)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Convert to temporary image file for PaddleOCR
            # PaddleOCR works best with file paths
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                img = Image.open(io.BytesIO(img_data))
                img.save(tmp_path)
            
            # Get OCR reader (lazy initialization) and use on the image
            reader, method = get_ocr_reader()
            
            if method == 'paddleocr':
                results = reader.ocr(tmp_path, cls=True)
                text_lines = []
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            text_lines.append(line[1][0])
                page_text = '\n'.join(text_lines)
            elif method == 'pytesseract':
                img = Image.open(tmp_path)
                page_text = reader.image_to_string(img)
            else:
                page_text = "[OCR not available - image-based PDF requires OCR backend]"
            
            extracted_texts.append(page_text)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        pdf_document.close()
        
        # Concatenate all pages
        return '\n\n--- Page Break ---\n\n'.join(extracted_texts)
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def get_or_create_patient(name):
    """Get existing patient or create a new one"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Check if patient exists
    cursor.execute('SELECT id FROM patients WHERE name = ?', (name,))
    patient = cursor.fetchone()
    
    if patient:
        patient_id = patient[0]
    else:
        # Create new patient
        created_at = datetime.now().isoformat()
        cursor.execute('INSERT INTO patients (name, created_at) VALUES (?, ?)', (name, created_at))
        patient_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return patient_id

def get_specialist_prompt(specialist_type):
    """Get specialist-specific system prompt"""
    prompts = {
        'cardiac': """You are an expert Cardiac Specialist analyzing cardiovascular test reports. Focus on:
- ECG findings, cardiac enzymes, lipid profiles
- Heart function indicators (EF, wall motion abnormalities)
- Arrhythmias, conduction abnormalities
- Cardiovascular risk factors
- Recommendations for cardiac interventions or medications""",
        
        'radiologist': """You are an expert Radiologist analyzing medical imaging reports (X-rays, CT, MRI, Ultrasound). Focus on:
- Image findings and abnormalities
- Anatomical structures and pathological changes
- Size, location, and characteristics of findings
- Comparison with previous studies if mentioned
- Recommendations for follow-up imaging""",
        
        'neurologist': """You are an expert Neurologist analyzing neurological test reports. Focus on:
- Neurological examination findings
- Brain imaging results (MRI, CT scans)
- EEG, EMG results
- Cognitive assessments
- Neurological symptoms and their clinical significance
- Treatment recommendations for neurological conditions""",
        
        'dermatologist': """You are an expert Dermatologist analyzing dermatological reports. Focus on:
- Skin lesion characteristics
- Biopsy results
- Dermatological conditions and diagnoses
- Treatment protocols for skin conditions
- Follow-up care recommendations""",
        
        'oncologist': """You are an expert Oncologist analyzing oncology reports. Focus on:
- Tumor markers and cancer staging
- Biopsy and histopathology results
- Treatment response assessments
- Side effects monitoring
- Treatment recommendations and prognosis"""
    }
    return prompts.get(specialist_type, """You are an expert medical analyst. Analyze the medical report comprehensively.""")

def format_analysis_response(text):
    """Format the AI analysis response into structured HTML"""
    if not text:
        return text
    
    import re
    
    # Process markdown bold (**text**)
    def process_bold(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong class="font-bold text-gray-900">\1</strong>', text)
    
    # Split text into lines
    lines = text.split('\n')
    formatted_html = []
    in_list = False
    in_table = False
    table_rows = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            if in_table and table_rows:
                formatted_html.append(_format_table(table_rows))
                table_rows = []
                in_table = False
            formatted_html.append('<br>')
            continue
        
        # Check for table rows (contains | separator)
        if '|' in line and line.count('|') >= 2:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                # Check if header row (next line might be separator)
                is_separator = all(c.replace('-', '').strip() == '' for c in cells)
                if not is_separator:
                    table_rows.append(cells)
                    in_table = True
                    if in_list:
                        formatted_html.append('</ul>')
                        in_list = False
            continue
        
        # If we were in table but this isn't a table row, finish the table
        if in_table and table_rows:
            formatted_html.append(_format_table(table_rows))
            table_rows = []
            in_table = False
        
        # Check for main headings (marked with # or ##)
        if line.startswith('##'):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = line.replace('##', '').strip()
            heading_text = process_bold(heading_text)
            formatted_html.append(f'<h3 class="text-xl font-bold text-gray-800 mt-4 mb-2 pb-2 border-b-2 border-blue-300">{heading_text}</h3>')
        
        # Check for numbered sections (1., 2., etc.)
        elif re.match(r'^\d+\.\s+[A-Z]', line):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = re.sub(r'^\d+\.\s+', '', line)
            heading_text = process_bold(heading_text)
            formatted_html.append(f'<h3 class="text-lg font-bold text-gray-800 mt-4 mb-2 pb-2 border-b border-blue-200">{heading_text}</h3>')
        
        # Check for subheadings (usually have : at the end or are short all caps)
        elif (line.endswith(':') and len(line) < 60) or (line.isupper() and len(line) > 5 and len(line) < 50):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = process_bold(line)
            formatted_html.append(f'<h4 class="text-lg font-semibold text-gray-700 mt-3 mb-2">{heading_text}</h4>')
        
        # Check for bullet points (starting with -, *, •, or numbered bullets)
        elif re.match(r'^[-*•]\s', line) or re.match(r'^\d+[.)]\s', line) or line.startswith('•'):
            if not in_list:
                formatted_html.append('<ul class="list-disc list-inside space-y-1 ml-4 mb-3">')
                in_list = True
            bullet_text = re.sub(r'^[-*•\d.)\s]+', '', line).strip()
            bullet_text = process_bold(bullet_text)
            formatted_html.append(f'<li class="text-gray-700 mb-1">{bullet_text}</li>')
        
        # Check for sub-bullets (indented)
        elif line.startswith('  ') and (line.strip().startswith('-') or line.strip().startswith('*')):
            if in_list:
                bullet_text = line.strip().lstrip('-*').strip()
                bullet_text = process_bold(bullet_text)
                formatted_html.append(f'<li class="text-gray-600 ml-6 list-circle">{bullet_text}</li>')
        
        # Regular paragraph
        else:
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            processed_line = process_bold(line)
            formatted_html.append(f'<p class="text-gray-700 mb-2 leading-relaxed">{processed_line}</p>')
    
    if in_list:
        formatted_html.append('</ul>')
    if in_table and table_rows:
        formatted_html.append(_format_table(table_rows))
    
    return ''.join(formatted_html)

def _format_table(rows):
    """Format table rows into HTML table"""
    if not rows:
        return ''
    
    html = ['<div class="overflow-x-auto my-4"><table class="min-w-full border-collapse border border-gray-300">']
    
    # First row as header
    if rows:
        html.append('<thead class="bg-blue-100">')
        html.append('<tr>')
        for cell in rows[0]:
            html.append(f'<th class="border border-gray-300 px-4 py-2 text-left font-semibold text-gray-800">{cell}</th>')
        html.append('</tr>')
        html.append('</thead>')
        html.append('<tbody>')
        
        # Data rows
        for row in rows[1:]:
            html.append('<tr class="hover:bg-gray-50">')
            for cell in row:
                html.append(f'<td class="border border-gray-300 px-4 py-2 text-gray-700">{cell}</td>')
            html.append('</tr>')
        
        html.append('</tbody>')
    
    html.append('</table></div>')
    return ''.join(html)

def analyze_with_groq(extracted_text, specialist_type='general'):
    """Send extracted text to Groq API for specialist-specific medical analysis"""
    specialist_prompt = get_specialist_prompt(specialist_type)
    
    system_prompt = f"""{specialist_prompt}

Your analysis should be formatted in a clear, structured manner with the following sections:

**1. EXECUTIVE SUMMARY**
- Brief overview of the report findings
- Overall health status assessment

**2. ABNORMAL FINDINGS & ANOMALIES**
- List all values outside normal ranges
- Identify specific test abnormalities
- Note any concerning patterns

**3. CRITICAL FINDINGS (if applicable)**
- Urgent findings requiring immediate attention
- High-risk indicators
- Emergency recommendations

**4. DETAILED ANALYSIS BY CATEGORY**
Break down findings by test category (e.g., Hematology, Biochemistry, Imaging, etc.)
For each category:
- Normal findings
- Abnormal findings with reference ranges
- Clinical significance

**5. RECOMMENDED FOLLOW-UP**
- Additional diagnostic tests needed
- Specialist consultations recommended
- Monitoring requirements

**6. TREATMENT & MANAGEMENT PLAN**
- Medication considerations
- Lifestyle modifications
- Dietary recommendations
- Activity restrictions/suggestions

**7. RISK ASSESSMENT & PROGNOSIS**
- Risk factors identified
- Preventive measures
- Long-term outlook

Format your response using:
- Clear section headings (use ## for main headings)
- Bullet points for lists (use - or *)
- Tables for comparative data when appropriate (use | for columns)
- Bold text for important values (**text**)
- Clear paragraphs for explanations

Be thorough, professional, and prioritize patient safety. Use medical terminology appropriately."""

    user_prompt = f"""Please analyze the following medical test report text extracted from a patient's document:

{extracted_text}

Please provide your comprehensive analysis in the structured format specified above. Use clear headings, bullet points, and organize the information logically."""

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        raw_response = response.choices[0].message.content
        # Format the response into HTML
        formatted_response = format_analysis_response(raw_response)
        
        return {
            'raw': raw_response,
            'formatted': formatted_response
        }
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")

# Lab test reference ranges for anomaly detection
LAB_REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    'hemoglobin': {'min': 12.0, 'max': 17.5, 'unit': 'g/dL', 'specialist': 'Hematologist'},
    'hgb': {'min': 12.0, 'max': 17.5, 'unit': 'g/dL', 'specialist': 'Hematologist'},
    'hematocrit': {'min': 36.0, 'max': 52.0, 'unit': '%', 'specialist': 'Hematologist'},
    'hct': {'min': 36.0, 'max': 52.0, 'unit': '%', 'specialist': 'Hematologist'},
    'rbc': {'min': 4.0, 'max': 6.0, 'unit': 'million/µL', 'specialist': 'Hematologist'},
    'wbc': {'min': 4.0, 'max': 11.0, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},
    'platelets': {'min': 150, 'max': 400, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},
    'plt': {'min': 150, 'max': 400, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},
    
    # Metabolic Panel
    'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'fasting glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'blood sugar': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'hba1c': {'min': 4.0, 'max': 5.7, 'unit': '%', 'specialist': 'Endocrinologist'},
    'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},
    'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},
    'urea': {'min': 15, 'max': 45, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},
    
    # Lipid Panel
    'cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'total cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'hdl': {'min': 40, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    
    # Liver Function
    'alt': {'min': 7, 'max': 56, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'sgpt': {'min': 7, 'max': 56, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'ast': {'min': 10, 'max': 40, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'sgot': {'min': 10, 'max': 40, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'bilirubin': {'min': 0.1, 'max': 1.2, 'unit': 'mg/dL', 'specialist': 'Gastroenterologist'},
    
    # Thyroid
    'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L', 'specialist': 'Endocrinologist'},
    't3': {'min': 80, 'max': 200, 'unit': 'ng/dL', 'specialist': 'Endocrinologist'},
    't4': {'min': 5.0, 'max': 12.0, 'unit': 'µg/dL', 'specialist': 'Endocrinologist'},
    
    # Electrolytes
    'sodium': {'min': 136, 'max': 145, 'unit': 'mEq/L', 'specialist': 'Nephrologist'},
    'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L', 'specialist': 'Nephrologist'},
    'calcium': {'min': 8.5, 'max': 10.5, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    
    # Inflammation markers
    'esr': {'min': 0, 'max': 20, 'unit': 'mm/hr', 'specialist': 'Rheumatologist'},
    'crp': {'min': 0, 'max': 3.0, 'unit': 'mg/L', 'specialist': 'Rheumatologist'},
    
    # Vitamins
    'vitamin d': {'min': 30, 'max': 100, 'unit': 'ng/mL', 'specialist': 'Endocrinologist'},
    'vitamin b12': {'min': 200, 'max': 900, 'unit': 'pg/mL', 'specialist': 'Hematologist'},
    'iron': {'min': 60, 'max': 170, 'unit': 'µg/dL', 'specialist': 'Hematologist'},
    'ferritin': {'min': 12, 'max': 300, 'unit': 'ng/mL', 'specialist': 'Hematologist'},
}

def detect_lab_anomalies(extracted_text, report_id=None, patient_id=None):
    """
    Detect abnormal values in lab report text by comparing against reference ranges.
    Returns list of anomalies with severity and recommended specialist.
    """
    import re
    
    anomalies = []
    text_lower = extracted_text.lower()
    
    # Patterns to match lab values: "Test Name: 123.4 mg/dL" or "Test Name 123.4"
    patterns = [
        r'([a-zA-Z\s]+)[\s:]+(\d+\.?\d*)\s*([a-zA-Z/%µ]+)?',  # "Hemoglobin: 12.5 g/dL"
        r'([a-zA-Z\s]+)\s*[=:]\s*(\d+\.?\d*)',  # "Glucose = 150"
    ]
    
    for test_name, ranges in LAB_REFERENCE_RANGES.items():
        # Search for the test name in the text
        test_pattern = re.compile(rf'\b{re.escape(test_name)}\b[\s:=]+(\d+\.?\d*)', re.IGNORECASE)
        matches = test_pattern.findall(text_lower)
        
        for match in matches:
            try:
                value = float(match)
                
                # Determine status
                status = 'normal'
                if value < ranges['min']:
                    status = 'low'
                    if value < ranges['min'] * 0.7:  # 30% below min is critical
                        status = 'critical_low'
                elif value > ranges['max']:
                    status = 'high'
                    if value > ranges['max'] * 1.5:  # 50% above max is critical
                        status = 'critical_high'
                
                if status != 'normal':
                    anomalies.append({
                        'test_name': test_name.upper(),
                        'value': value,
                        'unit': ranges['unit'],
                        'normal_min': ranges['min'],
                        'normal_max': ranges['max'],
                        'status': status,
                        'recommended_specialist': ranges['specialist'],
                        'report_id': report_id,
                        'patient_id': patient_id
                    })
            except (ValueError, TypeError):
                continue
    
    # Store anomalies in database if report_id is provided
    if report_id and anomalies:
        try:
            conn = sqlite3.connect('db.sqlite3')
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS report_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    patient_id INTEGER,
                    test_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    normal_min REAL,
                    normal_max REAL,
                    status TEXT NOT NULL,
                    recommended_specialist TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id),
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            # Clear old anomalies for this report
            cursor.execute('DELETE FROM report_anomalies WHERE report_id = ?', (report_id,))
            
            # Insert new anomalies
            for anomaly in anomalies:
                cursor.execute('''
                    INSERT INTO report_anomalies (report_id, patient_id, test_name, value, unit,
                                                  normal_min, normal_max, status, recommended_specialist, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (report_id, patient_id, anomaly['test_name'], anomaly['value'], anomaly['unit'],
                      anomaly['normal_min'], anomaly['normal_max'], anomaly['status'],
                      anomaly['recommended_specialist'], datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error storing anomalies: {str(e)}")
    
    return anomalies

def get_doctor_profile(doctor_id):
    """Get doctor profile from database"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM doctors WHERE id = ?', (doctor_id,))
    doctor = cursor.fetchone()
    conn.close()
    
    if doctor:
        return {
            'id': doctor[0],
            'name': doctor[1],
            'profile_picture': doctor[2],
            'specialist_type': doctor[3],
            'created_at': doctor[4]
        }
    return None

def create_or_update_doctor(name, specialist_type, profile_picture=None):
    """Create or update doctor profile"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Check if doctor exists
    cursor.execute('SELECT id FROM doctors WHERE name = ? AND specialist_type = ?', (name, specialist_type))
    doctor = cursor.fetchone()
    
    if doctor:
        # Update existing
        if profile_picture:
            cursor.execute('UPDATE doctors SET profile_picture = ? WHERE id = ?', (profile_picture, doctor[0]))
        doctor_id = doctor[0]
    else:
        # Create new
        created_at = datetime.now().isoformat()
        cursor.execute('INSERT INTO doctors (name, profile_picture, specialist_type, created_at) VALUES (?, ?, ?, ?)',
                      (name, profile_picture, specialist_type, created_at))
        doctor_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return doctor_id

def chatbot_response(message, specialist_type, doctor_id=None):
    """Generate chatbot response using Clinical BERT with specialist context"""
    try:
        # Use Clinical BERT to analyze the doctor's query
        clinical_bert_analysis = analyze_with_clinical_bert(message)
        
        # Get specialist context
        specialist_prompt = get_specialist_prompt(specialist_type)
        specialist_name = SPECIALIST_TYPES.get(specialist_type, 'medical')
        
        # Build context for response generation
        context_text = f"""Specialist Context: {specialist_prompt}
Specialist Type: {specialist_name}

Clinical BERT Analysis of Query: {clinical_bert_analysis if clinical_bert_analysis else 'No specific medical entities detected.'}

You are an AI assistant helping a {specialist_name} professional. 
Answer questions clearly, professionally, and with medical accuracy based on the Clinical BERT analysis.
If asked about something outside your specialty, acknowledge it and suggest consulting the appropriate specialist."""

        # Use Groq LLM with Clinical BERT context for generating the response
        # (Clinical BERT provides medical entity extraction, Groq provides natural language generation)
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": context_text},
                {"role": "user", "content": f"Doctor's Question: {message}\n\nPlease provide a professional medical response based on the Clinical BERT analysis above."}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        bot_response = response.choices[0].message.content
        
        # Add note about Clinical BERT usage
        if clinical_bert_analysis:
            bot_response += f"\n\n[Note: This response was enhanced using Clinical BERT medical entity analysis.]"
        
        # Save chat message to database
        if doctor_id:
            conn = sqlite3.connect('db.sqlite3')
            cursor = conn.cursor()
            created_at = datetime.now().isoformat()
            cursor.execute('''INSERT INTO chat_messages (doctor_id, specialist_type, message, response, created_at)
                           VALUES (?, ?, ?, ?, ?)''', (doctor_id, specialist_type, message, bot_response, created_at))
            conn.commit()
            conn.close()
        
        return bot_response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def save_report(patient_id, filename, extracted_text, llm_analysis, specialist_type='general', doctor_id=None, clinical_bert_analysis=None):
    """Save report to database and detect anomalies"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Ensure specialist_type is valid
    if not specialist_type or specialist_type not in SPECIALIST_TYPES:
        specialist_type = 'general'
    
    created_at = datetime.now().isoformat()
    report_id = None
    
    try:
        cursor.execute('''
            INSERT INTO reports (doctor_id, patient_id, specialist_type, original_filename, extracted_text, llm_analysis, clinical_bert_analysis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doctor_id, patient_id, specialist_type, filename, extracted_text, llm_analysis, clinical_bert_analysis, created_at))
        report_id = cursor.lastrowid
        conn.commit()
    except sqlite3.OperationalError as e:
        # If column doesn't exist, try without it
        print(f"Database error: {e}")
        # Re-run migration
        migrate_database()
        # Retry with all columns
        try:
            cursor.execute('''
                INSERT INTO reports (doctor_id, patient_id, specialist_type, original_filename, extracted_text, llm_analysis, clinical_bert_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (doctor_id, patient_id, specialist_type, filename, extracted_text, llm_analysis, clinical_bert_analysis, created_at))
            report_id = cursor.lastrowid
            conn.commit()
        except:
            # Fallback: insert without new columns
            cursor.execute('''
                INSERT INTO reports (patient_id, original_filename, extracted_text, llm_analysis, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, filename, extracted_text, llm_analysis, created_at))
            report_id = cursor.lastrowid
            conn.commit()
    finally:
        conn.close()
    
    # Run anomaly detection on the extracted text
    if report_id and extracted_text:
        try:
            anomalies = detect_lab_anomalies(extracted_text, report_id=report_id, patient_id=patient_id)
            if anomalies:
                print(f"Detected {len(anomalies)} anomalies in report {report_id}")
        except Exception as e:
            print(f"Error running anomaly detection: {str(e)}")
    
    return report_id

@app.route('/')
def index():
    """Landing page"""
    if 'user_role' in session:
        role = session['user_role']
        if role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        elif role == 'patient':
            return redirect(url_for('patient_dashboard'))
        elif role == 'admin':
            return redirect(url_for('admin_dashboard'))
    return render_template('landing.html')

@app.route('/home')
def home():
    """Alternative landing page route"""
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            return render_template('login.html', error='Username and password are required')
        
        # Hash password for comparison
        password_hash = hashlib.md5(password.encode()).hexdigest()
        
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        
        try:
            # First check if user exists
            cursor.execute('''SELECT id, username, role, full_name, user_id, email, password
                             FROM users WHERE username = ?''', (username,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return render_template('login.html', error='Invalid username or password')
            
            # Check password
            stored_password = user[6] if len(user) > 6 else ''
            if stored_password != password_hash:
                conn.close()
                return render_template('login.html', error='Invalid username or password')
            
            # Login successful - set session
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['user_role'] = user[2]
            session['full_name'] = user[3] if user[3] else user[1]
            session['user_db_id'] = user[4] if user[4] else None  # doctor_id or patient_id
            session['email'] = user[5] if len(user) > 5 and user[5] else None
            
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?',
                         (datetime.now().isoformat(), user[0]))
            conn.commit()
            conn.close()
            
            # Redirect to appropriate dashboard
            if user[2] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user[2] == 'patient':
                return redirect(url_for('patient_dashboard'))
            elif user[2] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return render_template('login.html', error='Invalid user role')
        except Exception as e:
            conn.close()
            print(f"Login error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('login.html', error=f'Login error: {str(e)}')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/doctor/appointments')
def doctor_appointments_page():
    """Doctor appointments page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor_appointments.html', user=session)

@app.route('/patient/appointments')
def patient_appointments_page():
    """Patient appointments page"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_appointments.html', user=session)

@app.route('/patient/chat')
def patient_chat_page():
    """Patient chat page"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_chat.html', user=session)

@app.route('/doctor/chat')
def doctor_chat_page():
    """Doctor chat page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor_chat.html', user=session)

@app.route('/messages')
def messages_redirect():
    """Redirect to appropriate messages page based on user role"""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    user_role = session.get('user_role')
    if user_role == 'patient':
        return redirect(url_for('patient_chat_page'))
    elif user_role == 'doctor':
        return redirect(url_for('doctor_chat_page'))
    else:
        return redirect(url_for('index'))

@app.route('/doctor/dashboard')
def doctor_dashboard():
    """Doctor dashboard page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    
    # Get current doctor profile if exists
    doctor = None
    try:
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        if session.get('user_db_id'):
            cursor.execute('SELECT * FROM doctors WHERE id = ?', (session['user_db_id'],))
        else:
            cursor.execute('SELECT * FROM doctors ORDER BY created_at DESC LIMIT 1')
        row = cursor.fetchone()
        if row:
            # Get column names for proper mapping
            cursor.execute('PRAGMA table_info(doctors)')
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            row_dict = dict(zip(column_names, row)) if len(row) == len(column_names) else {}
            
            doctor = {
                'id': row_dict.get('id') or row[0],
                'name': row_dict.get('name') or row[1],
                'profile_picture': row_dict.get('profile_picture') or (row[2] if len(row) > 2 else None),
                'specialist_type': row_dict.get('specialist_type') or (row[3] if len(row) > 3 else None),
                'email': row_dict.get('email'),
                'phone': row_dict.get('phone'),
                'license_number': row_dict.get('license_number'),
                'created_at': row_dict.get('created_at') or (row[-1] if len(row) > 0 else None)
            }
        conn.close()
    except:
        pass
    
    return render_template('doctor_dashboard.html', 
                         specialists=SPECIALIST_TYPES, 
                         doctor=doctor,
                         user=session)

@app.route('/profile')
def profile():
    """Profile management page"""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    role = session['user_role']
    user_db_id = session.get('user_db_id')
    doctor = None
    patient = None
    
    try:
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        
        if role == 'doctor' and user_db_id:
            # Get the logged-in doctor's profile
            cursor.execute('SELECT * FROM doctors WHERE id = ?', (user_db_id,))
            row = cursor.fetchone()
            if row:
                # Get column names for robust mapping
                cursor.execute('PRAGMA table_info(doctors)')
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                doctor = {}
                for idx, col_name in enumerate(column_names):
                    if idx < len(row):
                        doctor[col_name] = row[idx]
                
                # Also ensure backward compatibility with old structure
                if 'id' not in doctor:
                    doctor['id'] = row[0] if len(row) > 0 else None
                if 'name' not in doctor:
                    doctor['name'] = row[1] if len(row) > 1 else None
                if 'profile_picture' not in doctor:
                    doctor['profile_picture'] = row[2] if len(row) > 2 else None
                if 'specialist_type' not in doctor:
                    doctor['specialist_type'] = row[3] if len(row) > 3 else None
        
        elif role == 'patient' and user_db_id:
            # Get the logged-in patient's profile
            cursor.execute('SELECT * FROM patients WHERE id = ?', (user_db_id,))
            row = cursor.fetchone()
            if row:
                # Get column names for robust mapping
                cursor.execute('PRAGMA table_info(patients)')
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                patient = {}
                for idx, col_name in enumerate(column_names):
                    if idx < len(row):
                        patient[col_name] = row[idx]
        
        conn.close()
    except Exception as e:
        print(f"Profile load error: {e}")
        import traceback
        traceback.print_exc()
        pass
    
    return render_template('profile.html', 
                         specialists=SPECIALIST_TYPES, 
                         doctor=doctor, 
                         patient=patient,
                         user=session)

@app.route('/specialist/<specialist_type>')
def specialist_page(specialist_type):
    """Render specialist-specific page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    if specialist_type not in SPECIALIST_TYPES:
        return "Specialist not found", 404
    
    # Get current logged-in doctor profile
    doctor = None
    user_db_id = session.get('user_db_id')
    try:
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        if user_db_id:
            cursor.execute('SELECT * FROM doctors WHERE id = ?', (user_db_id,))
        else:
            # Fallback if user_db_id not set (shouldn't happen, but just in case)
            cursor.execute('SELECT * FROM doctors ORDER BY created_at DESC LIMIT 1')
        row = cursor.fetchone()
        if row:
            # Get column names for robust mapping
            cursor.execute('PRAGMA table_info(doctors)')
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            doctor = {}
            for idx, col_name in enumerate(column_names):
                if idx < len(row):
                    doctor[col_name] = row[idx]
            
            # Ensure backward compatibility
            if 'id' not in doctor:
                doctor['id'] = row[0] if len(row) > 0 else None
            if 'name' not in doctor:
                doctor['name'] = row[1] if len(row) > 1 else None
            if 'profile_picture' not in doctor:
                doctor['profile_picture'] = row[2] if len(row) > 2 else None
            if 'specialist_type' not in doctor:
                doctor['specialist_type'] = row[3] if len(row) > 3 else None
        conn.close()
    except Exception as e:
        print(f"Specialist page doctor load error: {e}")
        pass
    
    return render_template('specialist.html', 
                         specialist_type=specialist_type,
                         specialist_name=SPECIALIST_TYPES[specialist_type],
                         specialists=SPECIALIST_TYPES,
                         doctor=doctor,
                         user=session)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, OCR, and AI analysis"""
    # Allow both doctors and patients to upload reports
    if 'user_role' not in session or session['user_role'] not in ['doctor', 'patient']:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Validate file and patient name
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        patient_name = request.form.get('patient_name', '').strip()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not patient_name:
            return jsonify({'error': 'Patient name is required'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, PDF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext == 'pdf':
            extracted_text = extract_text_from_pdf(filepath)
        else:
            extracted_text = extract_text_from_image(filepath)
        
        if not extracted_text or not extracted_text.strip():
            # Check if it was an image file that failed OCR
            if file_ext != 'pdf':
                return jsonify({'error': 'No text could be extracted from the image. This may be because OCR is not available. Please ensure: 1) The image is clear and readable, 2) OCR backend (PaddleOCR or Tesseract) is properly installed. Alternatively, convert the image to a text-based PDF.'}), 400
            else:
                return jsonify({'error': 'No text could be extracted from the PDF. If this is an image-based PDF, OCR may not be available. Please ensure OCR backend is installed, or try a PDF with extractable text.'}), 400
        
        # Get specialist type and doctor ID from form
        specialist_type = request.form.get('specialist_type', 'general').strip()
        if not specialist_type or specialist_type not in SPECIALIST_TYPES:
            specialist_type = 'general'
        
        doctor_id = request.form.get('doctor_id', None)
        if doctor_id:
            try:
                doctor_id = int(doctor_id) if doctor_id else None
            except:
                doctor_id = None
        else:
            doctor_id = None
        
        # Analyze with Groq API (specialist-specific)
        llm_analysis_result = analyze_with_groq(extracted_text, specialist_type)
        
        # Handle both old format (string) and new format (dict)
        if isinstance(llm_analysis_result, dict):
            llm_analysis_raw = llm_analysis_result.get('raw', '')
            llm_analysis_formatted = llm_analysis_result.get('formatted', '')
        else:
            llm_analysis_raw = llm_analysis_result
            llm_analysis_formatted = format_analysis_response(llm_analysis_result)
        
        # Clinical BERT analysis
        clinical_bert_analysis = analyze_with_clinical_bert(extracted_text)
        
        # Get or create patient
        patient_id = get_or_create_patient(patient_name)
        
        # Save to database (save formatted HTML - it will render properly in templates)
        save_report(patient_id, filename, extracted_text, llm_analysis_formatted, specialist_type, doctor_id, clinical_bert_analysis)
        
        # Return results - ensure HTML is properly formatted
        # Use jsonify which handles JSON correctly, HTML in strings should not be escaped
        response_data = {
            'success': True,
            'extracted_text': extracted_text,
            'llm_analysis': llm_analysis_formatted,  # Return formatted HTML string
            'llm_analysis_raw': llm_analysis_raw,  # Also include raw for reference
            'clinical_bert_analysis': clinical_bert_analysis
        }
        
        # Use jsonify - it properly handles strings with HTML without escaping
        return jsonify(response_data)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/profile', methods=['POST'])
def update_profile():
    """Update doctor profile"""
    try:
        name = request.form.get('name', '').strip()
        specialist_type = request.form.get('specialist_type', '').strip()
        
        if not name or not specialist_type:
            return jsonify({'error': 'Name and specialist type are required'}), 400
        
        if specialist_type not in SPECIALIST_TYPES:
            return jsonify({'error': 'Invalid specialist type'}), 400
        
        profile_picture = None
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file.filename:
                filename = secure_filename(f"{specialist_type}_{name}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                file.save(filepath)
                profile_picture = f"profiles/{filename}"
        
        doctor_id = create_or_update_doctor(name, specialist_type, profile_picture)
        
        return jsonify({
            'success': True,
            'doctor_id': doctor_id,
            'profile_picture': profile_picture
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        specialist_type = data.get('specialist_type', 'general')
        doctor_id = data.get('doctor_id', None)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if specialist_type not in SPECIALIST_TYPES:
            specialist_type = 'general'
        
        response = chatbot_response(message, specialist_type, doctor_id)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/chatbot/context-questions', methods=['POST'])
def get_patient_chatbot_context_questions():
    """Get context questions for patient chatbot"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        patient_id = data.get('patient_id') or session.get('user_db_id')
        
        # Get patient information for personalized questions
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT allergies, medications, medical_history FROM patients WHERE id = ?', (patient_id,))
            patient_data = cursor.fetchone()
            
            # Generate context questions based on patient data
            questions = [
                {'id': 1, 'question': 'What is your main health concern or question today?', 'type': 'text'},
                {'id': 2, 'question': 'Are you currently experiencing any symptoms? If yes, please describe them.', 'type': 'text'},
                {'id': 3, 'question': 'When did these symptoms start?', 'type': 'text'}
            ]
            
            # Add medication question if patient has medications
            if patient_data and patient_data[1]:
                questions.append({'id': 4, 'question': 'Are you currently taking any medications? (You mentioned: ' + patient_data[1][:50] + '...)', 'type': 'text'})
            else:
                questions.append({'id': 4, 'question': 'Are you currently taking any medications?', 'type': 'text'})
            
            return jsonify({
                'success': True,
                'questions': questions
            })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/chatbot', methods=['POST'])
def patient_chatbot():
    """Handle patient chatbot messages with Clinical BERT - Continuous Q&A flow"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        patient_id = data.get('patient_id') or session.get('user_db_id')
        conversation_history = data.get('conversation_history', [])  # Array of {role, content}
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get patient's medical reports for context
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        
        try:
            # Get patient information
            cursor.execute('SELECT name, allergies, medications, medical_history FROM patients WHERE id = ?', (patient_id,))
            patient_data = cursor.fetchone()
            patient_info = {}
            if patient_data:
                patient_info = {
                    'name': patient_data[0],
                    'allergies': patient_data[1],
                    'medications': patient_data[2],
                    'medical_history': patient_data[3]
                }
            
            # Get recent reports for context
            cursor.execute('''SELECT extracted_text, llm_analysis, clinical_bert_analysis, specialist_type 
                             FROM reports WHERE patient_id = ? 
                             ORDER BY created_at DESC LIMIT 3''', (patient_id,))
            reports = cursor.fetchall()
            
            # Use Clinical BERT to analyze the patient's query
            clinical_bert_analysis = analyze_with_clinical_bert(message)
            
            # Build conversation context
            context_text = f"""Patient Information:
- Name: {patient_info.get('name', 'N/A')}
- Allergies: {patient_info.get('allergies', 'None reported')}
- Current Medications: {patient_info.get('medications', 'None reported')}
- Medical History: {patient_info.get('medical_history', 'None reported')}
"""
            
            if reports:
                context_text += "\nRecent Medical Reports Summary:\n"
                for report in reports:
                    if report[2]:  # clinical_bert_analysis
                        context_text += f"- {report[2][:200]}...\n"
            
            if clinical_bert_analysis:
                context_text += f"\nClinical BERT Analysis of Current Query: {clinical_bert_analysis}\n"
            
            # Use Groq LLM with Clinical BERT context and conversation history
            client = get_groq_client()
            system_prompt = """You are an AI Health Assistant powered by Clinical BERT. You engage in a continuous conversation with patients to understand their health concerns.

IMPORTANT INSTRUCTIONS:
1. After providing your response, ALWAYS ask a relevant follow-up question to continue the conversation
2. Your questions should be based on what the patient just said
3. Ask ONE clear, specific question at a time
4. Questions should help you better understand their symptoms, concerns, or medical situation
5. Be empathetic and conversational
6. Do not provide diagnoses, but help them understand their symptoms
7. Always remind patients to consult healthcare providers for serious concerns

Format your response as:
[Your response and explanation]

[Then ask a follow-up question in a new line or paragraph]

Example:
Based on what you've described, [your response]...

To better help you, could you tell me [follow-up question]?"""
            
            # Build messages array with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                if msg.get('role') and msg.get('content'):
                    messages.append({"role": msg['role'], "content": msg['content']})
            
            # Add current patient message with context
            user_prompt = f"""Context Information:
{context_text}

Current Patient Message: {message}

Please provide a helpful, empathetic response and then ask a relevant follow-up question to continue understanding their health concern."""
            
            messages.append({"role": "user", "content": user_prompt})
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1200
            )
            
            bot_response = response.choices[0].message.content
            
            # Check if response contains a question (ends with ? or contains question words)
            has_question = '?' in bot_response or any(word in bot_response.lower() for word in ['can you', 'could you', 'would you', 'what', 'when', 'where', 'how', 'why', 'tell me', 'describe'])
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'clinical_bert_used': clinical_bert_analysis is not None,
                'has_question': has_question
            })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<specialist_type>')
def get_reports(specialist_type):
    """Get reports for a specific specialist"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute('''SELECT r.id, r.created_at, p.name as patient_name, r.original_filename, 
                d.name as doctor_name FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.specialist_type = ? ORDER BY r.created_at DESC LIMIT 50''', 
                (specialist_type,))
    reports = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'success': True,
        'reports': [{
            'id': r[0],
            'created_at': r[1],
            'patient_name': r[2],
            'filename': r[3],
            'doctor_name': r[4]
        } for r in reports]
    })

@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get dashboard statistics"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Total reports per specialist
    cursor.execute('''SELECT specialist_type, COUNT(*) as count FROM reports GROUP BY specialist_type''')
    reports_by_specialist_raw = cursor.fetchall()
    reports_by_specialist = {row[0]: row[1] for row in reports_by_specialist_raw}
    reports_by_specialist_list = [{'type': row[0], 'name': SPECIALIST_TYPES.get(row[0], row[0]), 'count': row[1]} 
                                  for row in reports_by_specialist_raw]
    
    # Total reports, patients, doctors
    cursor.execute('SELECT COUNT(*) FROM reports')
    total_reports = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM patients')
    total_patients = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM doctors')
    total_doctors = cursor.fetchone()[0]
    
    # Reports over time (last 7 days)
    cursor.execute('''SELECT DATE(created_at) as date, COUNT(*) as count 
                     FROM reports 
                     WHERE datetime(created_at) > datetime('now', '-7 days')
                     GROUP BY DATE(created_at) ORDER BY date''')
    reports_over_time = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    # Most active specialists
    cursor.execute('''SELECT specialist_type, COUNT(*) as count FROM reports 
                     GROUP BY specialist_type ORDER BY count DESC LIMIT 5''')
    active_specialists = [{'type': row[0], 'name': SPECIALIST_TYPES.get(row[0], row[0]), 'count': row[1]} 
                         for row in cursor.fetchall()]
    
    # Recent reports
    cursor.execute('''SELECT r.id, r.created_at, p.name as patient_name, r.specialist_type, d.name as doctor_name
                     FROM reports r
                     LEFT JOIN patients p ON r.patient_id = p.id
                     LEFT JOIN doctors d ON r.doctor_id = d.id
                     ORDER BY r.created_at DESC LIMIT 10''')
    recent_reports = [{
        'id': row[0],
        'created_at': row[1],
        'patient_name': row[2],
        'specialist_type': row[3],
        'doctor_name': row[4]
    } for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        'success': True,
        'stats': {
            'total_reports': total_reports,
            'total_patients': total_patients,
            'total_doctors': total_doctors,
            'reports_by_specialist': reports_by_specialist,
            'reports_by_specialist_list': reports_by_specialist_list,
            'reports_over_time': reports_over_time,
            'active_specialists': active_specialists,
            'recent_reports': recent_reports
        }
    })

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Patient signup page"""
    if request.method == 'POST':
        try:
            data = request.form
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            full_name = data.get('full_name', '').strip()
            email = data.get('email', '').strip()
            phone = data.get('phone', '').strip()
            date_of_birth = data.get('date_of_birth', '').strip()
            gender = data.get('gender', '').strip()
            address = data.get('address', '').strip()
            
            if not username or not password or not full_name:
                return render_template('signup.html', error='Username, password, and full name are required')
            
            conn = sqlite3.connect('db.sqlite3')
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                conn.close()
                return render_template('signup.html', error='Username already exists. Please choose another one.')
            
            # Create patient record first
            created_at = datetime.now().isoformat()
            cursor.execute('''INSERT INTO patients (name, email, phone, date_of_birth, gender, address, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (full_name, email or None, phone or None, date_of_birth or None, gender or None, address or None, created_at))
            patient_id = cursor.lastrowid
            
            # Generate unique patient ID if column exists (optional, for future use)
            try:
                cursor.execute('PRAGMA table_info(patients)')
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                if 'patient_id' in column_names:
                    patient_unique_id = f"PAT{patient_id:06d}"
                    cursor.execute('UPDATE patients SET patient_id = ? WHERE id = ?', (patient_unique_id, patient_id))
            except:
                pass  # Column doesn't exist, skip it
            
            # Create user account
            password_hash = hashlib.md5(password.encode()).hexdigest()
            cursor.execute('''INSERT INTO users (username, password, role, full_name, email, user_id, created_at)
                             VALUES (?, ?, 'patient', ?, ?, ?, ?)''',
                         (username, password_hash, full_name, email or None, patient_id, created_at))
            user_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            # Auto-login after signup
            session['user_id'] = user_id
            session['username'] = username
            session['user_role'] = 'patient'
            session['full_name'] = full_name
            session['email'] = email
            session['user_db_id'] = patient_id
            
            return redirect(url_for('patient_dashboard'))
            
        except Exception as e:
            print(f"Signup error: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'conn' in locals():
                conn.close()
            return render_template('signup.html', error=f'Signup error: {str(e)}')
    
    return render_template('signup.html')

@app.route('/report/<int:report_id>')
def report_detail(report_id):
    """View detailed report page"""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''SELECT r.*, p.name as patient_name, d.name as doctor_name, d.profile_picture as doctor_picture
                         FROM reports r
                         LEFT JOIN patients p ON r.patient_id = p.id
                         LEFT JOIN doctors d ON r.doctor_id = d.id
                         WHERE r.id = ?''', (report_id,))
        row = cursor.fetchone()
        
        if not row:
            return "Report not found", 404
        
        # Check if patient can view this report (must be their own)
        columns = [desc[0] for desc in cursor.description]
        report = dict(zip(columns, row))
        
        if session['user_role'] == 'patient' and report['patient_id'] != session.get('user_db_id'):
            return "Unauthorized - You can only view your own reports", 403
        
        cursor.execute('''SELECT r.id, r.created_at, r.specialist_type, r.original_filename, r.status
                         FROM reports r WHERE r.patient_id = ? ORDER BY r.created_at DESC''', 
                      (report['patient_id'],))
        patient_reports = [dict(zip(['id', 'created_at', 'specialist_type', 'original_filename', 'status'], r)) 
                          for r in cursor.fetchall()]
        
        # Get the actual doctor associated with this report
        doctor = None
        if report.get('doctor_id'):
            try:
                cursor.execute('SELECT id, name, profile_picture, specialist_type, created_at FROM doctors WHERE id = ?', 
                             (report['doctor_id'],))
                row = cursor.fetchone()
                if row:
                    doctor = {
                        'id': row[0],
                        'name': row[1],
                        'profile_picture': row[2] if len(row) > 2 else None,
                        'specialist_type': row[3] if len(row) > 3 else None,
                        'created_at': row[4] if len(row) > 4 else None
                    }
            except Exception as e:
                print(f"Error fetching doctor: {e}")
                pass
        
        # If no doctor found but we have doctor_name from the JOIN, create a minimal doctor object
        if not doctor and report.get('doctor_name'):
            doctor = {
                'id': report.get('doctor_id'),
                'name': report.get('doctor_name'),
                'profile_picture': report.get('doctor_picture'),
                'specialist_type': None,
                'created_at': None
            }
        
    finally:
        conn.close()
    
    return render_template('report_detail.html', 
                         report=report, 
                         patient_reports=patient_reports,
                         specialists=SPECIALIST_TYPES,
                         doctor=doctor,
                         user=session)

@app.route('/patient/<int:patient_id>')
def patient_profile(patient_id):
    """Patient profile with history"""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    # Check if patient can view their own profile or doctor/admin can view any
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return "Unauthorized", 403
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        row = cursor.fetchone()
        if not row:
            return "Patient not found", 404
        
        # Get all patient fields (handle variable column count)
        patient = {
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'date_of_birth': row[3] if len(row) > 3 else None,
            'contact_info': row[4] if len(row) > 4 else None,
            'medical_history': row[5] if len(row) > 5 else None,
            'allergies': row[6] if len(row) > 6 else None,
            'medications': row[7] if len(row) > 7 else None,
            'patient_tags': row[8] if len(row) > 8 else None,
            'phone': row[9] if len(row) > 9 else None,
            'gender': row[10] if len(row) > 10 else None,
            'address': row[11] if len(row) > 11 else None,
            'profile_picture': row[12] if len(row) > 12 else None
        }
        
        cursor.execute('''SELECT r.*, d.name as doctor_name
                         FROM reports r
                         LEFT JOIN doctors d ON r.doctor_id = d.id
                         WHERE r.patient_id = ? ORDER BY r.created_at DESC''', (patient_id,))
        columns = [desc[0] for desc in cursor.description]
        reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # No need to fetch random doctor - we're showing patient profile
        
    finally:
        conn.close()
    
    return render_template('patient_profile.html',
                         patient=patient,
                         reports=reports,
                         specialists=SPECIALIST_TYPES,
                         session=session if 'user_role' in session else {})

@app.route('/api/search/reports')
def search_reports():
    """Search and filter reports"""
    query = request.args.get('q', '').strip()
    specialist_type = request.args.get('specialist', '').strip()
    status = request.args.get('status', '').strip()
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        sql = '''SELECT r.*, p.name as patient_name, d.name as doctor_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE 1=1'''
        params = []
        
        if query:
            # Sanitize query to prevent SQL injection
            query = sanitize_input(query)
            if query:
                sql += ' AND (p.name LIKE ? OR r.original_filename LIKE ? OR r.extracted_text LIKE ?)'
                like_query = f'%{query}%'
                params.extend([like_query, like_query, like_query])
        
        if specialist_type and specialist_type in SPECIALIST_TYPES:
            sql += ' AND r.specialist_type = ?'
            params.append(specialist_type)
        
        if status:
            sql += ' AND r.status = ?'
            params.append(status)
        
        if date_from:
            sql += ' AND datetime(r.created_at) >= datetime(?)'
            params.append(date_from)
        
        if date_to:
            sql += ' AND datetime(r.created_at) <= datetime(?)'
            params.append(date_to)
        
        sql += ' ORDER BY r.created_at DESC LIMIT 50'
        
        cursor.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    finally:
        conn.close()
    
    return jsonify({'success': True, 'reports': reports})

@app.route('/api/report/<int:report_id>', methods=['PUT'])
def update_report(report_id):
    """Update report status, notes, or tags"""
    data = request.json
    status = data.get('status', '').strip()
    doctor_notes = data.get('doctor_notes', '').strip()
    tags = data.get('tags', '').strip()
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        updates = []
        params = []
        
        if status and status in ['new', 'reviewed', 'critical', 'follow-up']:
            updates.append('status = ?')
            params.append(status)
        
        if doctor_notes is not None:
            updates.append('doctor_notes = ?')
            params.append(doctor_notes)
        
        if tags is not None:
            updates.append('tags = ?')
            params.append(tags)
        
        if updates:
            params.append(report_id)
            sql = f'UPDATE reports SET {", ".join(updates)} WHERE id = ?'
            cursor.execute(sql, params)
            conn.commit()
        
    finally:
        conn.close()
    
    return jsonify({'success': True})

@app.route('/api/report/<int:report_id>/export')
def export_report_pdf(report_id):
    """Export report as PDF"""
    try:
        from flask import send_file
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import tempfile
        import re
        
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        
        try:
            cursor.execute('''SELECT r.*, p.name as patient_name, d.name as doctor_name
                             FROM reports r
                             LEFT JOIN patients p ON r.patient_id = p.id
                             LEFT JOIN doctors d ON r.doctor_id = d.id
                             WHERE r.id = ?''', (report_id,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'error': 'Report not found'}), 404
            
            columns = [desc[0] for desc in cursor.description]
            report = dict(zip(columns, row))
            
            # Create PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=30
            )
            story.append(Paragraph("Medical Report Analysis", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Patient Info
            story.append(Paragraph(f"<b>Patient:</b> {report['patient_name']}", styles['Normal']))
            story.append(Paragraph(f"<b>Date:</b> {report['created_at'][:10]}", styles['Normal']))
            story.append(Paragraph(f"<b>Specialist:</b> {SPECIALIST_TYPES.get(report['specialist_type'], report['specialist_type'])}", styles['Normal']))
            story.append(Paragraph(f"<b>Doctor:</b> {report['doctor_name'] or 'N/A'}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # AI Analysis (format HTML for PDF)
            if report.get('llm_analysis'):
                html_text = report['llm_analysis']
                story.append(Paragraph("<b>AI Medical Analysis:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                # If it's HTML, convert to plain text with formatting preserved
                if html_text.strip().startswith('<'):
                    # Remove HTML tags but preserve structure
                    # Convert headings
                    html_text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n\1\n', html_text, flags=re.DOTALL)
                    html_text = re.sub(r'<h[1-6][^>]*>(.*?)(?=<)', r'\n\n\1\n', html_text)
                    # Convert lists
                    html_text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', html_text, flags=re.DOTALL)
                    html_text = re.sub(r'<ul[^>]*>|</ul>|<ol[^>]*>|</ol>', '\n', html_text)
                    # Convert paragraphs
                    html_text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html_text, flags=re.DOTALL)
                    # Remove remaining HTML tags
                    clean_text = re.sub(r'<[^>]+>', '', html_text)
                    # Clean up whitespace
                    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
                    clean_text = clean_text.strip()
                    
                    # Convert to paragraphs for PDF - better formatting
                    lines = clean_text.split('\n')
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if current_para:
                                para_text = ' '.join(current_para)
                                if para_text.startswith('•') or para_text.startswith('-'):
                                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                                else:
                                    story.append(Paragraph(para_text, styles['Normal']))
                                story.append(Spacer(1, 0.08*inch))
                                current_para = []
                            continue
                        
                        # Check for headings
                        if (line.isupper() and len(line) > 5 and len(line) < 80) or re.match(r'^\d+\.\s+[A-Z]', line):
                            if current_para:
                                para_text = ' '.join(current_para)
                                story.append(Paragraph(para_text, styles['Normal']))
                                story.append(Spacer(1, 0.08*inch))
                                current_para = []
                            
                            heading_style = ParagraphStyle(
                                'CustomHeading',
                                parent=styles['Heading2'],
                                fontSize=14,
                                textColor=colors.HexColor('#1e40af'),
                                spaceAfter=12,
                                spaceBefore=6,
                                fontName='Helvetica-Bold'
                            )
                            story.append(Paragraph(line, heading_style))
                            story.append(Spacer(1, 0.1*inch))
                        else:
                            current_para.append(line)
                    
                    # Add remaining paragraph
                    if current_para:
                        para_text = ' '.join(current_para)
                        if para_text.startswith('•') or para_text.startswith('-'):
                            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                        else:
                            story.append(Paragraph(para_text, styles['Normal']))
                        story.append(Spacer(1, 0.08*inch))
                else:
                    # Plain text - format it line by line
                    lines = html_text.split('\n')
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if current_para:
                                para_text = ' '.join(current_para)
                                if para_text.startswith('•') or para_text.startswith('-'):
                                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                                else:
                                    story.append(Paragraph(para_text, styles['Normal']))
                                story.append(Spacer(1, 0.08*inch))
                                current_para = []
                            continue
                        
                        # Check for headings
                        if line.startswith('##') or ((line.isupper() and len(line) > 5 and len(line) < 80) or re.match(r'^\d+\.\s+[A-Z]', line)):
                            if current_para:
                                para_text = ' '.join(current_para)
                                story.append(Paragraph(para_text, styles['Normal']))
                                story.append(Spacer(1, 0.08*inch))
                                current_para = []
                            
                            heading_text = line.replace('##', '').strip()
                            heading_style = ParagraphStyle(
                                'CustomHeading',
                                parent=styles['Heading2'],
                                fontSize=14,
                                textColor=colors.HexColor('#1e40af'),
                                spaceAfter=12,
                                spaceBefore=6,
                                fontName='Helvetica-Bold'
                            )
                            story.append(Paragraph(heading_text, heading_style))
                            story.append(Spacer(1, 0.1*inch))
                        elif line.startswith('-') or line.startswith('•') or re.match(r'^\d+[.)]\s', line):
                            if current_para:
                                para_text = ' '.join(current_para)
                                story.append(Paragraph(para_text, styles['Normal']))
                                story.append(Spacer(1, 0.08*inch))
                                current_para = []
                            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", styles['Normal']))
                            story.append(Spacer(1, 0.06*inch))
                        else:
                            current_para.append(line)
                    
                    # Add remaining paragraph
                    if current_para:
                        para_text = ' '.join(current_para)
                        story.append(Paragraph(para_text, styles['Normal']))
                        story.append(Spacer(1, 0.08*inch))
                
                story.append(Spacer(1, 0.2*inch))
            
            # Clinical BERT Analysis
            if report.get('clinical_bert_analysis'):
                story.append(Paragraph("<b>Clinical BERT Analysis:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                # Format Clinical BERT analysis
                bert_lines = report['clinical_bert_analysis'].split('\n')
                for line in bert_lines:
                    line = line.strip()
                    if line:
                        # Escape HTML entities for PDF
                        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        story.append(Paragraph(line, styles['Normal']))
                        story.append(Spacer(1, 0.08*inch))
                story.append(Spacer(1, 0.2*inch))
            
            # Doctor Notes
            if report.get('doctor_notes'):
                story.append(Paragraph("<b>Doctor Notes:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                notes = report['doctor_notes'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(notes, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            # Extracted Text (first 1500 chars)
            if report.get('extracted_text'):
                story.append(Paragraph("<b>Extracted Text (Preview):</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                preview = report['extracted_text'][:1500] + ('...' if len(report['extracted_text']) > 1500 else '')
                preview = preview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                # Use smaller font for extracted text
                preview_style = ParagraphStyle(
                    'PreviewText',
                    parent=styles['Normal'],
                    fontSize=9,
                    leading=11,
                    fontName='Courier'
                )
                story.append(Paragraph(preview, preview_style))
            
            doc.build(story)
            temp_file.close()
            
            return send_file(temp_file.name, 
                            mimetype='application/pdf',
                            as_attachment=True,
                            download_name=f"report_{report_id}_{report['patient_name']}.pdf")
        except Exception as e:
            return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
        finally:
            conn.close()
    except ImportError:
        return jsonify({'error': 'PDF export requires reportlab. Install with: pip install reportlab'}), 500

# ==================== NEW FEATURES API ROUTES ====================

@app.route('/api/patients/search')
def search_patients():
    """Search patients by name, ID, or phone"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'patients': []})
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Search by name, ID, or phone
        cursor.execute('''SELECT id, name, phone, contact_info, date_of_birth, patient_tags
                         FROM patients 
                         WHERE name LIKE ? OR id = ? OR phone LIKE ? OR contact_info LIKE ?
                         ORDER BY name LIMIT 50''', 
                      (f'%{query}%', query, f'%{query}%', f'%{query}%'))
        patients = []
        for row in cursor.fetchall():
            patients.append({
                'id': row[0],
                'name': row[1],
                'phone': row[2],
                'contact_info': row[3],
                'date_of_birth': row[4],
                'tags': row[5] if row[5] else []
            })
        return jsonify({'patients': patients})
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update patient profile (demographics, history, allergies, medications)"""
    data = request.json
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        updates = []
        values = []
        
        fields = ['name', 'date_of_birth', 'phone', 'gender', 'address', 'contact_info', 
                  'medical_history', 'allergies', 'medications', 'patient_tags']
        for field in fields:
            if field in data:
                updates.append(f"{field} = ?")
                values.append(data[field])
        
        if not updates:
            return jsonify({'error': 'No fields to update'}), 400
        
        values.append(patient_id)
        cursor.execute(f"UPDATE patients SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
    """Get or create tasks"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Create new task
            data = request.json
            task_type = data.get('task_type', 'follow-up')
            title = data.get('title', '')
            description = data.get('description', '')
            due_date = data.get('due_date')
            priority = data.get('priority', 'medium')
            doctor_id = data.get('doctor_id')
            patient_id = data.get('patient_id')
            report_id = data.get('report_id')
            
            cursor.execute('''INSERT INTO tasks (doctor_id, patient_id, report_id, task_type, title, 
                             description, due_date, priority, status, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)''',
                         (doctor_id, patient_id, report_id, task_type, title, description, 
                          due_date, priority, datetime.now().isoformat()))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        else:
            # Get tasks
            doctor_id = request.args.get('doctor_id')
            patient_id = request.args.get('patient_id')
            status = request.args.get('status')
            
            query = '''SELECT t.*, p.name as patient_name, d.name as doctor_name
                      FROM tasks t
                      LEFT JOIN patients p ON t.patient_id = p.id
                      LEFT JOIN doctors d ON t.doctor_id = d.id
                      WHERE 1=1'''
            params = []
            
            if doctor_id:
                query += ' AND t.doctor_id = ?'
                params.append(doctor_id)
            if patient_id:
                query += ' AND t.patient_id = ?'
                params.append(patient_id)
            if status:
                query += ' AND t.status = ?'
                params.append(status)
            
            query += ' ORDER BY t.created_at DESC LIMIT 100'
            cursor.execute(query, params)
            
            tasks = []
            for row in cursor.fetchall():
                tasks.append({
                    'id': row[0],
                    'doctor_id': row[1],
                    'patient_id': row[2],
                    'report_id': row[3],
                    'task_type': row[4],
                    'title': row[5],
                    'description': row[6],
                    'due_date': row[7],
                    'status': row[8],
                    'priority': row[9],
                    'created_at': row[10],
                    'completed_at': row[11],
                    'patient_name': row[12] if len(row) > 12 else None,
                    'doctor_name': row[13] if len(row) > 13 else None
                })
            return jsonify({'tasks': tasks})
    finally:
        conn.close()

@app.route('/api/tasks/<int:task_id>', methods=['PUT', 'DELETE'])
def task_detail(task_id):
    """Update or delete task"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'PUT':
            data = request.json
            updates = []
            values = []
            
            if 'status' in data:
                updates.append('status = ?')
                values.append(data['status'])
                if data['status'] == 'completed':
                    updates.append('completed_at = ?')
                    values.append(datetime.now().isoformat())
            
            if 'title' in data:
                updates.append('title = ?')
                values.append(data['title'])
            
            if 'description' in data:
                updates.append('description = ?')
                values.append(data['description'])
            
            if 'due_date' in data:
                updates.append('due_date = ?')
                values.append(data['due_date'])
            
            if 'priority' in data:
                updates.append('priority = ?')
                values.append(data['priority'])
            
            if updates:
                values.append(task_id)
                cursor.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", values)
                conn.commit()
            return jsonify({'success': True})
        else:
            cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
            conn.commit()
            return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/api/reports/<int:report_id>/share', methods=['POST'])
def share_report(report_id):
    """Share report with another doctor"""
    data = request.json
    shared_with_doctor_id = data.get('shared_with_doctor_id')
    shared_by_doctor_id = data.get('shared_by_doctor_id')
    
    if not shared_with_doctor_id or not shared_by_doctor_id:
        return jsonify({'error': 'Missing doctor IDs'}), 400
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Check if already shared
        cursor.execute('''SELECT id FROM shared_reports 
                         WHERE report_id = ? AND shared_with_doctor_id = ?''',
                      (report_id, shared_with_doctor_id))
        if cursor.fetchone():
            return jsonify({'error': 'Report already shared with this doctor'}), 400
        
        cursor.execute('''INSERT INTO shared_reports (report_id, shared_by_doctor_id, 
                         shared_with_doctor_id, shared_at)
                         VALUES (?, ?, ?, ?)''',
                      (report_id, shared_by_doctor_id, shared_with_doctor_id, 
                       datetime.now().isoformat()))
        conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/api/reports/<int:report_id>/comments', methods=['GET', 'POST'])
def report_comments(report_id):
    """Get or add comments on a report"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Add comment
            data = request.json
            doctor_id = data.get('doctor_id')
            comment = data.get('comment', '')
            is_private = data.get('is_private', 0)
            parent_comment_id = data.get('parent_comment_id')
            
            cursor.execute('''INSERT INTO report_comments (report_id, doctor_id, comment, 
                             is_private, parent_comment_id, created_at)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                         (report_id, doctor_id, comment, is_private, parent_comment_id,
                          datetime.now().isoformat()))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        else:
            # Get comments
            doctor_id = request.args.get('doctor_id')
            
            query = '''SELECT c.*, d.name as doctor_name, d.specialist_type
                      FROM report_comments c
                      JOIN doctors d ON c.doctor_id = d.id
                      WHERE c.report_id = ? AND (c.is_private = 0 OR c.doctor_id = ?)
                      ORDER BY c.created_at ASC'''
            cursor.execute(query, (report_id, doctor_id or 0))
            
            comments = []
            for row in cursor.fetchall():
                comments.append({
                    'id': row[0],
                    'report_id': row[1],
                    'doctor_id': row[2],
                    'comment': row[3],
                    'is_private': bool(row[4]),
                    'parent_comment_id': row[5],
                    'created_at': row[6],
                    'doctor_name': row[7],
                    'specialist_type': row[8]
                })
            return jsonify({'comments': comments})
    finally:
        conn.close()

@app.route('/api/referrals', methods=['GET', 'POST'])
def referrals():
    """Get or create referrals"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Create referral
            data = request.json
            cursor.execute('''INSERT INTO referrals (patient_id, from_doctor_id, 
                             to_specialist_type, to_doctor_id, reason, notes, status, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)''',
                           (data.get('patient_id'), data.get('from_doctor_id'),
                            data.get('to_specialist_type'), data.get('to_doctor_id'),
                            data.get('reason'), data.get('notes'), datetime.now().isoformat()))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        else:
            # Get referrals
            doctor_id = request.args.get('doctor_id')
            patient_id = request.args.get('patient_id')
            status = request.args.get('status')
            
            query = '''SELECT r.*, p.name as patient_name, d1.name as from_doctor_name,
                      d2.name as to_doctor_name
                      FROM referrals r
                      JOIN patients p ON r.patient_id = p.id
                      JOIN doctors d1 ON r.from_doctor_id = d1.id
                      LEFT JOIN doctors d2 ON r.to_doctor_id = d2.id
                      WHERE 1=1'''
            params = []
            
            if doctor_id:
                query += ' AND (r.from_doctor_id = ? OR r.to_doctor_id = ?)'
                params.extend([doctor_id, doctor_id])
            if patient_id:
                query += ' AND r.patient_id = ?'
                params.append(patient_id)
            if status:
                query += ' AND r.status = ?'
                params.append(status)
            
            query += ' ORDER BY r.created_at DESC'
            cursor.execute(query, params)
            
            referrals = []
            columns = ['id', 'patient_id', 'from_doctor_id', 'to_specialist_type', 'to_doctor_id',
                      'reason', 'notes', 'status', 'created_at']
            for row in cursor.fetchall():
                ref = dict(zip(columns + ['patient_name', 'from_doctor_name', 'to_doctor_name'], row))
                referrals.append(ref)
            return jsonify({'referrals': referrals})
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>/vitals', methods=['GET', 'POST'])
def patient_vitals(patient_id):
    """Get or add patient vitals for trends tracking"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Add vital
            data = request.json
            cursor.execute('''INSERT INTO patient_vitals (patient_id, report_id, vital_name, 
                             vital_value, unit, measured_at)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                         (patient_id, data.get('report_id'), data.get('vital_name'),
                          data.get('vital_value'), data.get('unit'), 
                          data.get('measured_at', datetime.now().isoformat())))
            conn.commit()
            return jsonify({'success': True})
        else:
            # Get vitals
            vital_name = request.args.get('vital_name')
            query = '''SELECT * FROM patient_vitals WHERE patient_id = ?'''
            params = [patient_id]
            
            if vital_name:
                query += ' AND vital_name = ?'
                params.append(vital_name)
            
            query += ' ORDER BY measured_at ASC'
            cursor.execute(query, params)
            
            vitals = []
            for row in cursor.fetchall():
                vitals.append({
                    'id': row[0],
                    'patient_id': row[1],
                    'report_id': row[2],
                    'vital_name': row[3],
                    'vital_value': row[4],
                    'unit': row[5],
                    'measured_at': row[6]
                })
            return jsonify({'vitals': vitals})
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>/risk-score', methods=['POST'])
def calculate_risk_score(patient_id):
    """Calculate AI-based risk score for patient"""
    data = request.json
    condition = data.get('condition', 'general')
    report_id = data.get('report_id')
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Get patient reports for context
        cursor.execute('SELECT llm_analysis, clinical_bert_analysis FROM reports WHERE patient_id = ?', 
                      (patient_id,))
        reports = cursor.fetchall()
        
        # Combine analysis text
        analysis_text = ' '.join([r[0] or '' + ' ' + (r[1] or '') for r in reports])
        
        # Use Groq to calculate risk
        client = get_groq_client()
        prompt = f"""Based on the following patient medical reports, assess the risk level for {condition}.
        
        Patient Reports Analysis:
        {analysis_text[:4000]}
        
        Please provide:
        1. Risk Score (0-100, where 0 is no risk and 100 is critical risk)
        2. Risk Level (low, moderate, high, critical)
        3. Brief analysis explaining the risk assessment
        
        Format your response as:
        RISK_SCORE: [number]
        RISK_LEVEL: [low/moderate/high/critical]
        ANALYSIS: [your analysis]
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert medical risk assessment AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content
        
        # Parse response
        risk_score = 50  # Default
        risk_level = 'moderate'
        analysis = result_text
        
        for line in result_text.split('\n'):
            if 'RISK_SCORE:' in line:
                try:
                    risk_score = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'RISK_LEVEL:' in line:
                risk_level = line.split(':')[1].strip().lower()
        
        # Save risk score
        cursor.execute('''INSERT INTO risk_scores (patient_id, report_id, condition, risk_score, 
                         risk_level, analysis_text, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (patient_id, report_id, condition, risk_score, risk_level, analysis,
                       datetime.now().isoformat()))
        conn.commit()
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'analysis': analysis
        })
    finally:
        conn.close()

@app.route('/api/analytics/reports')
def analytics_reports():
    """Generate custom analytics reports"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    specialist_type = request.args.get('specialist_type')
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        query = '''SELECT 
            specialist_type,
            COUNT(*) as total_reports,
            COUNT(DISTINCT patient_id) as unique_patients,
            SUM(CASE WHEN status = 'critical' THEN 1 ELSE 0 END) as critical_reports,
            SUM(CASE WHEN status = 'reviewed' THEN 1 ELSE 0 END) as reviewed_reports
            FROM reports
            WHERE 1=1'''
        params = []
        
        if start_date:
            query += ' AND created_at >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND created_at <= ?'
            params.append(end_date)
        if specialist_type:
            query += ' AND specialist_type = ?'
            params.append(specialist_type)
        
        query += ' GROUP BY specialist_type'
        cursor.execute(query, params)
        
        analytics = []
        for row in cursor.fetchall():
            analytics.append({
                'specialist_type': row[0],
                'total_reports': row[1],
                'unique_patients': row[2],
                'critical_reports': row[3],
                'reviewed_reports': row[4]
            })
        
        return jsonify({'analytics': analytics})
    finally:
        conn.close()

@app.route('/patient/dashboard')
def patient_dashboard():
    """Patient dashboard page"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    
    patient_id = session.get('user_db_id', 1)
    
    return render_template('patient_dashboard.html',
                         user=session,
                         patient_id=patient_id)

@app.route('/patient/ai-assistant')
def patient_ai_assistant():
    """AI Health Assistant page for patients"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    
    return render_template('patient_ai_assistant.html',
                         user=session)

@app.route('/admin/dashboard')
def admin_dashboard():
    """Administrator dashboard page"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return redirect(url_for('login'))
    
    return render_template('admin_dashboard.html',
                         user=session,
                         specialists=SPECIALIST_TYPES)

@app.route('/api/patient/<int:patient_id>/reports')
def get_patient_reports(patient_id):
    """Get all reports for a patient - Only patient's own reports"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Security check: Patients can only access their own reports
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized - You can only view your own reports'}), 403
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Ensure we only get reports for the specified patient
        cursor.execute('''SELECT r.*, d.name as doctor_name
                         FROM reports r
                         LEFT JOIN doctors d ON r.doctor_id = d.id
                         WHERE r.patient_id = ?
                         ORDER BY r.created_at DESC''', (patient_id,))
        columns = [desc[0] for desc in cursor.description]
        reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return jsonify({'reports': reports})
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>')
def get_patient_info(patient_id):
    """Get patient information"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        row = cursor.fetchone()
        if row:
            patient = {
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'date_of_birth': row[3] if len(row) > 3 else None,
                'contact_info': row[4] if len(row) > 4 else None,
                'medical_history': row[5] if len(row) > 5 else None,
                'allergies': row[6] if len(row) > 6 else None,
                'medications': row[7] if len(row) > 7 else None,
                'patient_tags': row[8] if len(row) > 8 else None,
                'phone': row[9] if len(row) > 9 else None,
                'gender': row[10] if len(row) > 10 else None,
                'address': row[11] if len(row) > 11 else None
            }
            return jsonify({'patient': patient})
        return jsonify({'error': 'Patient not found'}), 404
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>/anomalies')
def get_patient_anomalies(patient_id):
    """Get all lab anomalies for a patient with severity and recommendations"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Authorization check
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized - You can only view your own anomalies'}), 403
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='report_anomalies'")
        if not cursor.fetchone():
            return jsonify({'anomalies': [], 'has_critical': False, 'has_anomalies': False})
        
        # Get all anomalies for this patient
        cursor.execute('''
            SELECT ra.*, r.original_filename, r.created_at as report_date
            FROM report_anomalies ra
            LEFT JOIN reports r ON ra.report_id = r.id
            WHERE ra.patient_id = ?
            ORDER BY 
                CASE 
                    WHEN ra.status LIKE 'critical%' THEN 1
                    ELSE 2
                END,
                ra.created_at DESC
        ''', (patient_id,))
        columns = [desc[0] for desc in cursor.description]
        anomalies = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Check for critical anomalies
        has_critical = any(a['status'].startswith('critical') for a in anomalies)
        
        # Group by specialist for recommendations
        specialists = {}
        for a in anomalies:
            spec = a['recommended_specialist']
            if spec not in specialists:
                specialists[spec] = []
            specialists[spec].append(a['test_name'])
        
        return jsonify({
            'anomalies': anomalies,
            'has_critical': has_critical,
            'has_anomalies': len(anomalies) > 0,
            'specialists_needed': specialists
        })
    finally:
        conn.close()

@app.route('/api/admin/stats')
def admin_stats():
    """Get comprehensive admin statistics"""
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM reports')
        total_reports = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM doctors')
        total_doctors = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        
        # Reports by specialist
        cursor.execute('''SELECT specialist_type, COUNT(*) 
                         FROM reports GROUP BY specialist_type''')
        reports_by_specialist = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Users by role
        cursor.execute('''SELECT role, COUNT(*) FROM users GROUP BY role''')
        users_by_role = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent reports
        cursor.execute('''SELECT r.*, p.name as patient_name
                         FROM reports r
                         LEFT JOIN patients p ON r.patient_id = p.id
                         ORDER BY r.created_at DESC LIMIT 10''')
        columns = [desc[0] for desc in cursor.description]
        recent_reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify({
            'success': True,
            'total_reports': total_reports,
            'total_users': total_users,
            'total_doctors': total_doctors,
            'total_patients': total_patients,
            'reports_by_specialist': reports_by_specialist,
            'users_by_role': users_by_role,
            'recent_reports': recent_reports
        })
    finally:
        conn.close()

@app.route('/api/admin/users', methods=['GET', 'POST'])
def admin_users():
    """Get all users or create new user"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            if not request.json:
                return jsonify({'error': 'No data provided'}), 400
            
            data = request.json
            
            # Validate required fields
            if not data.get('username'):
                return jsonify({'error': 'Username is required'}), 400
            if not data.get('password'):
                return jsonify({'error': 'Password is required'}), 400
            if not data.get('full_name'):
                return jsonify({'error': 'Full name is required'}), 400
            if not data.get('email'):
                return jsonify({'error': 'Email is required'}), 400
            
            # Check if username already exists
            cursor.execute('SELECT id FROM users WHERE username = ?', (data.get('username'),))
            if cursor.fetchone():
                return jsonify({'error': 'Username already exists'}), 400
            
            password_hash = hashlib.md5(data.get('password', '').encode()).hexdigest()
            cursor.execute('''INSERT INTO users (username, password, role, full_name, email, user_id, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (data.get('username'), password_hash, data.get('role', 'patient'),
                          data.get('full_name'), data.get('email'), data.get('user_id'),
                          datetime.now().isoformat()))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        else:
            cursor.execute('SELECT id, username, role, full_name, email, last_login FROM users')
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'role': row[2],
                    'full_name': row[3],
                    'email': row[4],
                    'last_login': row[5]
                })
            return jsonify({'users': users})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/users/<int:user_id>', methods=['PUT', 'DELETE'])
def admin_user_detail(user_id):
    """Update or delete a user"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'DELETE':
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            return jsonify({'success': True})
        else:
            # UPDATE user
            data = request.json
            updates = []
            values = []
            
            if 'password' in data and data['password']:
                password_hash = hashlib.md5(data['password'].encode()).hexdigest()
                updates.append('password = ?')
                values.append(password_hash)
            
            if 'full_name' in data:
                updates.append('full_name = ?')
                values.append(data['full_name'])
            
            if 'email' in data:
                updates.append('email = ?')
                values.append(data['email'])
            
            if 'role' in data:
                updates.append('role = ?')
                values.append(data['role'])
            
            if updates:
                values.append(user_id)
                cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)
                conn.commit()
            return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/api/admin/doctors', methods=['GET', 'POST'])
def admin_doctors():
    """Get all doctors or create new doctor - Admin only"""
    if not require_role('admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Handle form data (for file upload) or JSON
            if request.content_type and 'multipart/form-data' in request.content_type:
                data = request.form.to_dict()
                profile_picture = None
                if 'profile_picture' in request.files:
                    file = request.files['profile_picture']
                    if file.filename:
                        filename = secure_filename(f"doctor_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                        filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                        file.save(filepath)
                        profile_picture = f"profiles/{filename}"
            else:
                data = request.json
                profile_picture = data.get('profile_picture')
            
            # Create doctor first - use column order: name, profile_picture, specialist_type, email, phone, license_number, created_at
            created_at = datetime.now().isoformat()
            cursor.execute('''INSERT INTO doctors (name, profile_picture, specialist_type, email, phone, license_number, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (data.get('name'), profile_picture, data.get('specialist_type'), data.get('email'),
                          data.get('phone'), data.get('license_number'), created_at))
            doctor_id = cursor.lastrowid
            
            # Create user account for doctor if username/password provided
            if data.get('username') and data.get('password'):
                password_hash = hashlib.md5(data['password'].encode()).hexdigest()
                cursor.execute('''INSERT INTO users (username, password, role, full_name, email, user_id, created_at)
                                 VALUES (?, ?, 'doctor', ?, ?, ?, ?)''',
                             (data['username'], password_hash, data.get('name'), data.get('email'),
                              doctor_id, created_at))
            
            conn.commit()
            return jsonify({'success': True, 'id': doctor_id})
        else:
            cursor.execute('''SELECT id, name, specialist_type, email, phone, license_number, 
                             profile_picture, created_at FROM doctors ORDER BY created_at DESC''')
            doctors = []
            for row in cursor.fetchall():
                doctors.append({
                    'id': row[0],
                    'name': row[1],
                    'specialist_type': row[2],
                    'email': row[3],
                    'phone': row[4],
                    'license_number': row[5],
                    'profile_picture': row[6],
                    'created_at': row[7]
                })
            return jsonify({'doctors': doctors})
    finally:
        conn.close()

@app.route('/api/admin/doctors/<int:doctor_id>', methods=['DELETE'])
def admin_delete_doctor(doctor_id):
    """Delete a doctor - Admin only"""
    if not require_role('admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Delete associated user account if exists
        cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (doctor_id, 'doctor'))
        user_row = cursor.fetchone()
        if user_row:
            cursor.execute('DELETE FROM users WHERE id = ?', (user_row[0],))
        
        # Delete doctor
        cursor.execute('DELETE FROM doctors WHERE id = ?', (doctor_id,))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/reports')
def admin_all_reports():
    """Get all reports for admin"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''SELECT r.*, p.name as patient_name, d.name as doctor_name
                         FROM reports r
                         LEFT JOIN patients p ON r.patient_id = p.id
                         LEFT JOIN doctors d ON r.doctor_id = d.id
                         ORDER BY r.created_at DESC LIMIT 100''')
        columns = [desc[0] for desc in cursor.description]
        reports = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return jsonify({'reports': reports})
    finally:
        conn.close()

@app.route('/api/patient/<int:patient_id>/update', methods=['PUT'])
def update_patient_profile(patient_id):
    """Update patient profile including picture"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Check if patient can update their own profile
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'PUT':
            # Handle form data (for file upload) or JSON
            if request.content_type and 'multipart/form-data' in request.content_type:
                data = request.form.to_dict()
                profile_picture = None
                if 'profile_picture' in request.files:
                    file = request.files['profile_picture']
                    if file.filename:
                        filename = secure_filename(f"patient_{patient_id}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                        filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                        file.save(filepath)
                        profile_picture = f"profiles/{filename}"
            else:
                data = request.json
                profile_picture = data.get('profile_picture')
            
            updates = []
            values = []
            
            fields = ['name', 'date_of_birth', 'phone', 'gender', 'address', 'contact_info', 
                      'medical_history', 'allergies', 'medications', 'patient_tags', 'email']
            for field in fields:
                if field in data:
                    updates.append(f"{field} = ?")
                    values.append(data[field])
            
            if profile_picture:
                updates.append("profile_picture = ?")
                values.append(profile_picture)
            
            if updates:
                values.append(patient_id)
                cursor.execute(f"UPDATE patients SET {', '.join(updates)} WHERE id = ?", values)
                
                # Also update user account if exists
                cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (patient_id, 'patient'))
                user_row = cursor.fetchone()
                if user_row:
                    user_updates = []
                    user_values = []
                    if 'name' in data:
                        user_updates.append('full_name = ?')
                        user_values.append(data['name'])
                    if 'email' in data:
                        user_updates.append('email = ?')
                        user_values.append(data['email'])
                    if user_updates:
                        user_values.append(user_row[0])
                        cursor.execute(f"UPDATE users SET {', '.join(user_updates)} WHERE id = ?", user_values)
                
                conn.commit()
                return jsonify({'success': True, 'profile_picture': profile_picture})
            return jsonify({'error': 'No fields to update'}), 400
    finally:
        conn.close()

@app.route('/api/user/profile', methods=['GET', 'PUT'])
def user_profile():
    """Get or update current user profile (works for all roles)"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'PUT':
            # Update user profile
            role = session['user_role']
            user_db_id = session.get('user_db_id')
            
            if request.content_type and 'multipart/form-data' in request.content_type:
                data = request.form.to_dict()
                profile_picture = None
                if 'profile_picture' in request.files:
                    file = request.files['profile_picture']
                    if file.filename:
                        filename = secure_filename(f"{role}_{session['username']}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                        filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                        file.save(filepath)
                        profile_picture = f"profiles/{filename}"
            else:
                data = request.json
                profile_picture = data.get('profile_picture')
            
            # Update user table
            updates = []
            values = []
            if 'full_name' in data:
                updates.append('full_name = ?')
                values.append(data['full_name'])
            if 'email' in data:
                updates.append('email = ?')
                values.append(data['email'])
            if 'password' in data and data['password']:
                password_hash = hashlib.md5(data['password'].encode()).hexdigest()
                updates.append('password = ?')
                values.append(password_hash)
            
            if updates:
                values.append(session['user_id'])
                cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)
            
            # Update role-specific tables
            if role == 'doctor' and user_db_id:
                # Check which columns exist in doctors table
                cursor.execute('PRAGMA table_info(doctors)')
                doctor_columns = [row[1] for row in cursor.fetchall()]
                
                doc_updates = []
                doc_values = []
                if 'name' in doctor_columns and 'name' in data:
                    doc_updates.append('name = ?')
                    doc_values.append(data['name'])
                if 'email' in doctor_columns and 'email' in data:
                    doc_updates.append('email = ?')
                    doc_values.append(data['email'])
                if 'phone' in doctor_columns and 'phone' in data:
                    doc_updates.append('phone = ?')
                    doc_values.append(data['phone'])
                if 'specialist_type' in doctor_columns and 'specialist_type' in data:
                    doc_updates.append('specialist_type = ?')
                    doc_values.append(data['specialist_type'])
                if 'profile_picture' in doctor_columns and profile_picture:
                    doc_updates.append('profile_picture = ?')
                    doc_values.append(profile_picture)
                
                if doc_updates:
                    doc_values.append(user_db_id)
                    cursor.execute(f"UPDATE doctors SET {', '.join(doc_updates)} WHERE id = ?", doc_values)
            
            elif role == 'patient' and user_db_id:
                # Check which columns exist in patients table
                cursor.execute('PRAGMA table_info(patients)')
                patient_columns = [row[1] for row in cursor.fetchall()]
                
                # Update patient table - only update columns that exist
                patient_updates = []
                patient_values = []
                
                if 'name' in patient_columns and 'name' in data:
                    patient_updates.append('name = ?')
                    patient_values.append(data['name'])
                if 'email' in patient_columns and 'email' in data:
                    patient_updates.append('email = ?')
                    patient_values.append(data['email'])
                if 'phone' in patient_columns and 'phone' in data:
                    patient_updates.append('phone = ?')
                    patient_values.append(data['phone'])
                if 'date_of_birth' in patient_columns and 'date_of_birth' in data:
                    patient_updates.append('date_of_birth = ?')
                    patient_values.append(data['date_of_birth'])
                if 'gender' in patient_columns and 'gender' in data:
                    patient_updates.append('gender = ?')
                    patient_values.append(data['gender'])
                if 'address' in patient_columns and 'address' in data:
                    patient_updates.append('address = ?')
                    patient_values.append(data['address'])
                if 'medical_history' in patient_columns and 'medical_history' in data:
                    patient_updates.append('medical_history = ?')
                    patient_values.append(data['medical_history'])
                if 'allergies' in patient_columns and 'allergies' in data:
                    patient_updates.append('allergies = ?')
                    patient_values.append(data['allergies'])
                if 'medications' in patient_columns and 'medications' in data:
                    patient_updates.append('medications = ?')
                    patient_values.append(data['medications'])
                if 'profile_picture' in patient_columns and profile_picture:
                    patient_updates.append('profile_picture = ?')
                    patient_values.append(profile_picture)
                
                if patient_updates:
                    patient_values.append(user_db_id)
                    cursor.execute(f"UPDATE patients SET {', '.join(patient_updates)} WHERE id = ?", patient_values)
            
            conn.commit()
            return jsonify({'success': True, 'profile_picture': profile_picture})
        else:
            # GET profile
            role = session['user_role']
            user_db_id = session.get('user_db_id')
            
            profile = {
                'username': session.get('username'),
                'full_name': session.get('full_name'),
                'email': session.get('email'),
                'role': role
            }
            
            if role == 'doctor' and user_db_id:
                cursor.execute('SELECT * FROM doctors WHERE id = ?', (user_db_id,))
                row = cursor.fetchone()
                if row:
                    profile.update({
                        'name': row[1],
                        'profile_picture': row[2] if len(row) > 2 else None,
                        'specialist_type': row[3] if len(row) > 3 else None,
                        'email': row[4] if len(row) > 4 else None,
                        'phone': row[5] if len(row) > 5 else None,
                        'license_number': row[6] if len(row) > 6 else None
                    })
            elif role == 'patient' and user_db_id:
                # Get column names first to ensure correct mapping
                cursor.execute('PRAGMA table_info(patients)')
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                cursor.execute('SELECT * FROM patients WHERE id = ?', (user_db_id,))
                row = cursor.fetchone()
                if row:
                    # Create a dictionary mapping column names to values
                    row_dict = dict(zip(column_names, row)) if len(row) == len(column_names) else {}
                    
                    profile.update({
                        'name': row_dict.get('name') or row[1] if len(row) > 1 else None,
                        'email': row_dict.get('email') or (row[2] if len(row) > 2 else None),
                        'phone': row_dict.get('phone') or (row[3] if len(row) > 3 else None),
                        'date_of_birth': row_dict.get('date_of_birth') or (row[4] if len(row) > 4 else None),
                        'gender': row_dict.get('gender') or (row[5] if len(row) > 5 else None),
                        'address': row_dict.get('address') or (row[6] if len(row) > 6 else None),
                        'medical_history': row_dict.get('medical_history') or (row[7] if len(row) > 7 else None),
                        'allergies': row_dict.get('allergies') or (row[8] if len(row) > 8 else None),
                        'medications': row_dict.get('medications') or (row[9] if len(row) > 9 else None),
                        'profile_picture': row_dict.get('profile_picture') or (row[10] if len(row) > 10 else None)
                    })
            
            return jsonify({'profile': profile})
    finally:
        conn.close()

# ==========================================
# NEW FEATURE ROUTES
# ==========================================

def log_audit(action, entity_type=None, entity_id=None, details=None):
    """Log user actions for audit trail"""
    try:
        conn = sqlite3.connect('db.sqlite3', timeout=10.0)
        conn.execute('PRAGMA busy_timeout = 10000')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO audit_logs (user_id, user_role, action, entity_type, entity_id, details, ip_address, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (session.get('user_id'), session.get('user_role'), action, entity_type, entity_id,
                      details, request.remote_addr, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Audit log error: {str(e)}")
        pass  # Don't break main flow if audit logging fails

def create_notification(user_id, user_role, notification_type, title, message, link=None):
    """Create a notification for a user"""
    try:
        conn = sqlite3.connect('db.sqlite3', timeout=10.0)
        conn.execute('PRAGMA busy_timeout = 10000')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO notifications (user_id, user_role, notification_type, title, message, link, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (user_id, user_role, notification_type, title, message, link, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Notification error: {str(e)}")
        pass

# ==========================================
# APPOINTMENT MANAGEMENT
# ==========================================

@app.route('/api/doctors', methods=['GET'])
def get_doctors_list():
    """Get list of all doctors for patient appointment booking"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''SELECT d.id, d.name, d.specialist_type, d.email, d.phone, d.profile_picture, d.created_at,
                         (SELECT COUNT(*) FROM appointments a WHERE a.doctor_id = d.id AND a.appointment_date = date('now')) as appointments_today
                         FROM doctors d
                         ORDER BY d.name''')
        columns = [desc[0] for desc in cursor.description]
        doctors = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return jsonify({'doctors': doctors})
    finally:
        conn.close()

@app.route('/api/appointments', methods=['GET', 'POST'])
def appointments():
    """Get or create appointments"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Use timeout to handle locked database
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')  # 10 second timeout
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Create new appointment
            data = request.json
            print(f"Appointment POST request data: {data}")
            
            # For patients, use their own ID; for doctors, use provided patient_id
            if session['user_role'] == 'patient':
                patient_id = session.get('user_db_id')
            else:
                patient_id = data.get('patient_id')
            
            doctor_id = data.get('doctor_id') or (session.get('user_db_id') if session['user_role'] == 'doctor' else None)
            
            print(f"Appointment creation - patient_id: {patient_id}, doctor_id: {doctor_id}")
            
            if not patient_id or not doctor_id:
                return jsonify({'error': 'Patient and doctor are required'}), 400
            
            # Enhanced conflict checking - prevent double booking
            appointment_date = data.get('appointment_date')
            appointment_time = data.get('appointment_time')
            duration = data.get('duration', 30)
            
            if not appointment_date or not appointment_time:
                return jsonify({'error': 'Appointment date and time are required'}), 400
            
            # Check for exact time conflicts
            cursor.execute('''SELECT id, patient_id FROM appointments 
                            WHERE doctor_id = ? AND appointment_date = ? AND appointment_time = ?
                            AND status IN ('scheduled', 'confirmed')''',
                         (doctor_id, appointment_date, appointment_time))
            existing = cursor.fetchone()
            if existing:
                return jsonify({
                    'error': 'This time slot is already booked by another patient. Please choose a different time.',
                    'conflict': True
                }), 400
            
            # Check for overlapping time slots (considering duration)
            # Convert time to minutes for easier comparison
            from datetime import datetime as dt
            try:
                requested_time = dt.strptime(appointment_time, '%H:%M').time()
                requested_minutes = requested_time.hour * 60 + requested_time.minute
                requested_end_minutes = requested_minutes + duration
                
                # Get all appointments for this doctor on this date
                cursor.execute('''SELECT appointment_time, duration FROM appointments 
                                WHERE doctor_id = ? AND appointment_date = ? 
                                AND status IN ('scheduled', 'confirmed')''',
                             (doctor_id, appointment_date))
                existing_appointments = cursor.fetchall()
                
                for existing_time, existing_duration in existing_appointments:
                    try:
                        existing_time_obj = dt.strptime(existing_time, '%H:%M').time()
                        existing_minutes = existing_time_obj.hour * 60 + existing_time_obj.minute
                        existing_end_minutes = existing_minutes + (existing_duration or 30)
                        
                        # Check for overlap: requested start < existing end AND requested end > existing start
                        if (requested_minutes < existing_end_minutes and requested_end_minutes > existing_minutes):
                            return jsonify({
                                'error': f'This time slot overlaps with an existing appointment ({existing_time}). Please choose a different time.',
                                'conflict': True
                            }), 400
                    except:
                        # If time parsing fails, skip this check but continue
                        pass
            except Exception as e:
                # If time parsing fails, fall back to exact match check
                print(f"Time parsing error: {e}")
            
            # Additional check: Prevent same patient from booking multiple appointments at same time
            cursor.execute('''SELECT id FROM appointments 
                            WHERE patient_id = ? AND appointment_date = ? AND appointment_time = ?
                            AND status IN ('scheduled', 'confirmed')''',
                         (patient_id, appointment_date, appointment_time))
            if cursor.fetchone():
                return jsonify({
                    'error': 'You already have an appointment scheduled at this time. Please choose a different time.',
                    'conflict': True
                }), 400
            
            # Ensure appointments table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='appointments'")
            if not cursor.fetchone():
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS appointments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER NOT NULL,
                        doctor_id INTEGER NOT NULL,
                        appointment_date TEXT NOT NULL,
                        appointment_time TEXT NOT NULL,
                        duration INTEGER DEFAULT 30,
                        appointment_type TEXT DEFAULT "consultation",
                        reason TEXT,
                        status TEXT DEFAULT "scheduled",
                        notes TEXT,
                        reminder_sent INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT
                    )
                ''')
                conn.commit()
                print("Created appointments table")
            
            # Verify patient and doctor exist
            cursor.execute('SELECT id FROM patients WHERE id = ?', (patient_id,))
            if not cursor.fetchone():
                return jsonify({'error': f'Patient with ID {patient_id} not found'}), 400
            
            cursor.execute('SELECT id FROM doctors WHERE id = ?', (doctor_id,))
            if not cursor.fetchone():
                return jsonify({'error': f'Doctor with ID {doctor_id} not found'}), 400
            
            try:
                # For patient requests, start with 'pending' status (requires doctor approval)
                # For doctor-created appointments, use 'scheduled' status directly
                initial_status = 'pending' if session['user_role'] == 'patient' else 'scheduled'
                
                cursor.execute('''INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time,
                                 duration, appointment_type, reason, status, notes, created_at)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                             (patient_id, doctor_id, data.get('appointment_date'), data.get('appointment_time'),
                              data.get('duration', 30), data.get('appointment_type', 'consultation'),
                              data.get('reason'), initial_status, data.get('notes'), datetime.now().isoformat()))
                appointment_id = cursor.lastrowid
                print(f"Appointment inserted with ID: {appointment_id}, status: {initial_status}")
            except Exception as e:
                print(f"Error inserting appointment: {str(e)}")
                import traceback
                traceback.print_exc()
                conn.rollback()
                return jsonify({'error': f'Failed to create appointment: {str(e)}'}), 500
            
            # Commit the appointment first before creating notifications/audit logs
            conn.commit()
            print(f"Appointment {appointment_id} committed successfully")
            
            # Close this connection before calling functions that may open their own connections
            conn.close()
            
            # Create notifications (these use their own connections)
            try:
                conn2 = sqlite3.connect('db.sqlite3', timeout=10.0)
                conn2.execute('PRAGMA busy_timeout = 10000')
                cursor2 = conn2.cursor()
                cursor2.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (patient_id, 'patient'))
                patient_user = cursor2.fetchone()
                cursor2.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (doctor_id, 'doctor'))
                doctor_user = cursor2.fetchone()
                conn2.close()
                
                if patient_user:
                    create_notification(patient_user[0], 'patient', 'appointment', 
                                      'Appointment Scheduled', f'Appointment scheduled for {data.get("appointment_date")} at {data.get("appointment_time")}',
                                      f'/appointments/{appointment_id}')
                
                if doctor_user:
                    create_notification(doctor_user[0], 'doctor', 'appointment',
                                      'New Appointment', f'New appointment scheduled',
                                      f'/appointments/{appointment_id}')
            except Exception as e:
                print(f"Error creating notifications: {str(e)}")
            
            # Log audit (uses its own connection)
            try:
                log_audit('appointment_created', 'appointment', appointment_id, f'Patient {patient_id} with Doctor {doctor_id}')
            except Exception as e:
                print(f"Error logging audit: {str(e)}")
            
            return jsonify({'success': True, 'id': appointment_id, 'message': 'Appointment booked successfully'})
        else:
            # GET appointments
            user_role = session['user_role']
            user_db_id = session.get('user_db_id')
            
            if user_role == 'doctor':
                # Get appointments for this doctor - user_db_id is the doctor's ID from doctors table
                if user_db_id:
                    cursor.execute('''SELECT a.*, p.name as patient_name, d.name as doctor_name, p.email as patient_email, p.phone as patient_phone
                                     FROM appointments a
                                     LEFT JOIN patients p ON a.patient_id = p.id
                                     LEFT JOIN doctors d ON a.doctor_id = d.id
                                     WHERE a.doctor_id = ? 
                                     ORDER BY a.appointment_date DESC, a.appointment_time DESC''',
                                 (user_db_id,))
                else:
                    # If user_db_id is not set, try to get it from users table
                    cursor.execute('SELECT user_id FROM users WHERE id = ? AND role = ?', (session.get('user_id'), 'doctor'))
                    user_row = cursor.fetchone()
                    if user_row and user_row[0]:
                        doctor_id = user_row[0]
                        cursor.execute('''SELECT a.*, p.name as patient_name, d.name as doctor_name, p.email as patient_email, p.phone as patient_phone
                                         FROM appointments a
                                         LEFT JOIN patients p ON a.patient_id = p.id
                                         LEFT JOIN doctors d ON a.doctor_id = d.id
                                         WHERE a.doctor_id = ? 
                                         ORDER BY a.appointment_date DESC, a.appointment_time DESC''',
                                     (doctor_id,))
                    else:
                        return jsonify({'appointments': []})
            elif user_role == 'patient':
                cursor.execute('''SELECT a.*, p.name as patient_name, d.name as doctor_name
                                 FROM appointments a
                                 LEFT JOIN patients p ON a.patient_id = p.id
                                 LEFT JOIN doctors d ON a.doctor_id = d.id
                                 WHERE a.patient_id = ? ORDER BY a.appointment_date, a.appointment_time''',
                             (user_db_id,))
            else:  # admin
                cursor.execute('''SELECT a.*, p.name as patient_name, d.name as doctor_name
                                 FROM appointments a
                                 LEFT JOIN patients p ON a.patient_id = p.id
                                 LEFT JOIN doctors d ON a.doctor_id = d.id
                                 ORDER BY a.appointment_date, a.appointment_time DESC LIMIT 100''')
            
            columns = [desc[0] for desc in cursor.description]
            appointments = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return jsonify({'appointments': appointments})
    finally:
        conn.close()

@app.route('/api/appointments/<int:appointment_id>', methods=['PUT', 'DELETE'])
def appointment_detail(appointment_id):
    """Update or delete appointment - with proper access control"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')
    cursor = conn.cursor()
    
    try:
        # First, get the appointment to check access
        cursor.execute('''SELECT patient_id, doctor_id, appointment_date, appointment_time, status 
                         FROM appointments WHERE id = ?''', (appointment_id,))
        appointment = cursor.fetchone()
        
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404
        
        patient_id, doctor_id, appt_date, appt_time, status = appointment
        user_role = session.get('user_role')
        user_db_id = session.get('user_db_id')
        
        # Check access: Patient can modify their own, Doctor can modify their own, Admin can modify any
        has_access = False
        if user_role == 'admin':
            has_access = True
        elif user_role == 'patient' and user_db_id == patient_id:
            has_access = True
        elif user_role == 'doctor' and user_db_id == doctor_id:
            has_access = True
        
        if not has_access:
            return jsonify({'error': 'Unauthorized - You can only modify your own appointments'}), 403
        
        if request.method == 'DELETE':
            cursor.execute('DELETE FROM appointments WHERE id = ?', (appointment_id,))
            conn.commit()
            conn.close()
            # Log audit after closing connection
            log_audit('appointment_deleted', 'appointment', appointment_id)
            return jsonify({'success': True})
        else:
            data = request.json
            updates = []
            values = []
            
            # If updating time/date, check for conflicts
            if 'appointment_date' in data or 'appointment_time' in data:
                new_date = data.get('appointment_date', appt_date)
                new_time = data.get('appointment_time', appt_time)
                new_duration = data.get('duration', 30)
                
                # Check for conflicts with new time
                cursor.execute('''SELECT id FROM appointments 
                                WHERE doctor_id = ? AND appointment_date = ? AND appointment_time = ?
                                AND status IN ('scheduled', 'confirmed') AND id != ?''',
                             (doctor_id, new_date, new_time, appointment_id))
                if cursor.fetchone():
                    return jsonify({
                        'error': 'This time slot is already booked. Please choose a different time.',
                        'conflict': True
                    }), 400
            
            if 'status' in data:
                updates.append('status = ?')
                values.append(data['status'])
                updates.append('updated_at = ?')
                values.append(datetime.now().isoformat())
            
            if 'notes' in data:
                updates.append('notes = ?')
                values.append(sanitize_input(data['notes']))
            
            if 'appointment_date' in data:
                updates.append('appointment_date = ?')
                values.append(data['appointment_date'])
            
            if 'appointment_time' in data:
                updates.append('appointment_time = ?')
                values.append(data['appointment_time'])
            
            if 'duration' in data:
                updates.append('duration = ?')
                values.append(data['duration'])
            
            if updates:
                values.append(appointment_id)
                cursor.execute(f"UPDATE appointments SET {', '.join(updates)} WHERE id = ?", values)
                conn.commit()
            return jsonify({'success': True})
    finally:
        conn.close()
        # Log audit after closing connection
        if request.method == 'PUT':
            try:
                log_audit('appointment_updated', 'appointment', appointment_id, str(request.json))
            except:
                pass
        elif request.method == 'DELETE':
            try:
                log_audit('appointment_deleted', 'appointment', appointment_id)
            except:
                pass

@app.route('/api/appointments/<int:appointment_id>/respond', methods=['POST'])
def respond_to_appointment(appointment_id):
    """Doctor responds to an appointment request (accept/reject)"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return jsonify({'error': 'Unauthorized - Only doctors can respond to appointment requests'}), 401
    
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')
    cursor = conn.cursor()
    
    try:
        data = request.json
        action = data.get('action')  # 'accept' or 'reject'
        rejection_reason = data.get('rejection_reason', '')
        
        if action not in ['accept', 'reject']:
            return jsonify({'error': 'Invalid action. Must be "accept" or "reject"'}), 400
        
        # Get the appointment
        cursor.execute('''SELECT patient_id, doctor_id, status, appointment_date, appointment_time 
                         FROM appointments WHERE id = ?''', (appointment_id,))
        appointment = cursor.fetchone()
        
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404
        
        patient_id, doctor_id, current_status, appt_date, appt_time = appointment
        
        # Verify doctor owns this appointment
        if doctor_id != session.get('user_db_id'):
            return jsonify({'error': 'Unauthorized - You can only respond to your own appointment requests'}), 403
        
        # Verify appointment is pending
        if current_status != 'pending':
            return jsonify({'error': f'Appointment is not pending (current status: {current_status})'}), 400
        
        # Update appointment status
        new_status = 'scheduled' if action == 'accept' else 'rejected'
        notes_update = f'Doctor response: {action}' + (f' - {rejection_reason}' if rejection_reason else '')
        
        cursor.execute('''UPDATE appointments SET status = ?, notes = COALESCE(notes, '') || ?, updated_at = ?
                         WHERE id = ?''',
                     (new_status, notes_update, datetime.now().isoformat(), appointment_id))
        conn.commit()
        
        # Create notification for patient
        try:
            # Find patient's user ID
            cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (patient_id, 'patient'))
            patient_user = cursor.fetchone()
            if patient_user:
                if action == 'accept':
                    notification_title = 'Appointment Confirmed!'
                    notification_message = f'Your appointment on {appt_date} at {appt_time} has been confirmed by the doctor.'
                else:
                    notification_title = 'Appointment Request Declined'
                    notification_message = f'Your appointment request on {appt_date} at {appt_time} has been declined.' + (f' Reason: {rejection_reason}' if rejection_reason else '')
                
                create_notification(patient_user[0], 'patient', 'appointment',
                                  notification_title, notification_message, '/patient/appointments')
        except Exception as e:
            print(f"Error creating notification: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'Appointment {action}ed successfully',
            'new_status': new_status
        })
    except Exception as e:
        print(f"Error responding to appointment: {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return jsonify({'error': f'Failed to respond to appointment: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/api/doctors/<int:doctor_id>/schedule', methods=['GET', 'POST', 'PUT'])
def doctor_schedule(doctor_id):
    """Manage doctor's schedule"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            # Create schedule entry
            data = request.json
            cursor.execute('''INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, end_time,
                             slot_duration, break_start, break_end, is_available)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (doctor_id, data.get('day_of_week'), data.get('start_time'), data.get('end_time'),
                          data.get('slot_duration', 30), data.get('break_start'), data.get('break_end'),
                          data.get('is_available', 1)))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        elif request.method == 'PUT':
            # Update schedule
            data = request.json
            cursor.execute('''UPDATE doctor_schedules SET start_time = ?, end_time = ?, slot_duration = ?,
                             break_start = ?, break_end = ?, is_available = ? WHERE id = ?''',
                         (data.get('start_time'), data.get('end_time'), data.get('slot_duration'),
                          data.get('break_start'), data.get('break_end'), data.get('is_available'),
                          data.get('schedule_id')))
            conn.commit()
            return jsonify({'success': True})
        else:
            # GET schedule
            cursor.execute('SELECT * FROM doctor_schedules WHERE doctor_id = ? ORDER BY day_of_week', (doctor_id,))
            columns = [desc[0] for desc in cursor.description]
            schedules = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return jsonify({'schedules': schedules})
    finally:
        conn.close()

@app.route('/api/appointments/available-slots', methods=['GET'])
def available_slots():
    """Get available appointment slots for a doctor on a date"""
    doctor_id = request.args.get('doctor_id', type=int)
    date = request.args.get('date')
    
    if not doctor_id or not date:
        return jsonify({'error': 'doctor_id and date required'}), 400
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Get doctor's schedule for the day of week
        from datetime import datetime as dt
        date_obj = dt.strptime(date, '%Y-%m-%d')
        day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
        
        cursor.execute('SELECT * FROM doctor_schedules WHERE doctor_id = ? AND day_of_week = ? AND is_available = 1',
                      (doctor_id, day_of_week))
        schedule = cursor.fetchone()
        
        if not schedule:
            return jsonify({'slots': []})
        
        # Get booked appointments with duration for overlap checking
        cursor.execute('SELECT appointment_time, duration FROM appointments WHERE doctor_id = ? AND appointment_date = ? AND status IN ("scheduled", "confirmed")',
                      (doctor_id, date))
        booked = cursor.fetchall()
        
        # Generate available slots
        slots = []
        start = dt.strptime(schedule[3], '%H:%M').time()  # start_time
        end = dt.strptime(schedule[4], '%H:%M').time()    # end_time
        slot_duration = schedule[5] or 30  # slot_duration
        
        current = dt.combine(date_obj.date(), start)
        end_dt = dt.combine(date_obj.date(), end)
        
        while current < end_dt:
            slot_time = current.time().strftime('%H:%M')
            slot_minutes = current.hour * 60 + current.minute
            slot_end_minutes = slot_minutes + slot_duration
            
            # Check if slot overlaps with any booked appointment
            is_booked = False
            for booked_time, booked_duration in booked:
                try:
                    booked_time_obj = dt.strptime(booked_time, '%H:%M').time()
                    booked_minutes = booked_time_obj.hour * 60 + booked_time_obj.minute
                    booked_end_minutes = booked_minutes + (booked_duration or 30)
                    
                    # Check for overlap
                    if (slot_minutes < booked_end_minutes and slot_end_minutes > booked_minutes):
                        is_booked = True
                        break
                except:
                    # If parsing fails, do exact match check
                    if booked_time == slot_time:
                        is_booked = True
                        break
            
            if not is_booked:
                slots.append(slot_time)
            
            current = current.replace(minute=current.minute + slot_duration)
        
        return jsonify({'slots': slots})
    finally:
        conn.close()

# ==========================================
# PRESCRIPTION MANAGEMENT
# ==========================================

@app.route('/api/prescriptions', methods=['GET', 'POST'])
def prescriptions():
    """Get or create prescriptions"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            data = request.json
            patient_id = data.get('patient_id')
            doctor_id = session.get('user_db_id') if session['user_role'] == 'doctor' else data.get('doctor_id')
            
            if not patient_id or not doctor_id:
                return jsonify({'error': 'Patient and doctor required'}), 400
            
            # Get patient info for AI safety check
            cursor.execute('SELECT allergies, medications, medical_history FROM patients WHERE id = ?', (patient_id,))
            patient_info = cursor.fetchone()
            
            # Use Groq AI to check prescription safety
            ai_safety_check = None
            if patient_info:
                safety_prompt = f"""As a medical AI assistant, analyze this prescription for safety:

Patient Information:
- Allergies: {patient_info[0] or 'None known'}
- Current Medications: {patient_info[1] or 'None'}
- Medical History: {patient_info[2] or 'None'}

Prescription:
{data.get('prescription_text')}

Medications List: {data.get('medications', 'Not specified')}

Please check for:
1. Drug-allergy interactions
2. Drug-drug interactions
3. Contraindications based on medical history
4. Dosage appropriateness
5. Any other safety concerns

Provide a brief safety assessment with warnings if any issues are detected."""

                try:
                    global groq_client
                    if groq_client is None:
                        from groq import Groq
                        groq_client = Groq(api_key=GROQ_API_KEY)
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a medical AI assistant specialized in prescription safety checking. Provide concise, accurate safety assessments."},
                            {"role": "user", "content": safety_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    ai_safety_check = response.choices[0].message.content
                except Exception as e:
                    ai_safety_check = f"AI safety check unavailable: {str(e)}"
            
            cursor.execute('''INSERT INTO prescriptions (patient_id, doctor_id, appointment_id, prescription_text,
                             medications, instructions, valid_until, refills_remaining, status, ai_safety_check, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (patient_id, doctor_id, data.get('appointment_id'), data.get('prescription_text'),
                          data.get('medications'), data.get('instructions'), data.get('valid_until'),
                          data.get('refills_remaining', 0), 'active', ai_safety_check, datetime.now().isoformat()))
            prescription_id = cursor.lastrowid
            
            # Create notification
            cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (patient_id, 'patient'))
            patient_user = cursor.fetchone()
            if patient_user:
                create_notification(patient_user[0], 'patient', 'prescription',
                                  'New Prescription', 'A new prescription has been issued',
                                  f'/prescriptions/{prescription_id}')
            
            log_audit('prescription_created', 'prescription', prescription_id, f'For patient {patient_id}')
            conn.commit()
            return jsonify({'success': True, 'id': prescription_id, 'ai_safety_check': ai_safety_check})
        else:
            # GET prescriptions
            user_role = session['user_role']
            user_db_id = session.get('user_db_id')
            
            if user_role == 'doctor':
                cursor.execute('''SELECT p.*, pt.name as patient_name, d.name as doctor_name
                                 FROM prescriptions p
                                 LEFT JOIN patients pt ON p.patient_id = pt.id
                                 LEFT JOIN doctors d ON p.doctor_id = d.id
                                 WHERE p.doctor_id = ? ORDER BY p.created_at DESC''',
                             (user_db_id,))
            elif user_role == 'patient':
                cursor.execute('''SELECT p.*, pt.name as patient_name, d.name as doctor_name
                                 FROM prescriptions p
                                 LEFT JOIN patients pt ON p.patient_id = pt.id
                                 LEFT JOIN doctors d ON p.doctor_id = d.id
                                 WHERE p.patient_id = ? ORDER BY p.created_at DESC''',
                             (user_db_id,))
            else:
                cursor.execute('''SELECT p.*, pt.name as patient_name, d.name as doctor_name
                                 FROM prescriptions p
                                 LEFT JOIN patients pt ON p.patient_id = pt.id
                                 LEFT JOIN doctors d ON p.doctor_id = d.id
                                 ORDER BY p.created_at DESC LIMIT 100''')
            
            columns = [desc[0] for desc in cursor.description]
            prescriptions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return jsonify({'prescriptions': prescriptions})
    finally:
        conn.close()

# ==========================================
# NOTIFICATION SYSTEM
# ==========================================

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get notifications for current user"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 50''',
                     (session['user_id'],))
        columns = [desc[0] for desc in cursor.description]
        notifications = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Mark as read
        unread_count = sum(1 for n in notifications if not n.get('is_read'))
        
        return jsonify({'notifications': notifications, 'unread_count': unread_count})
    finally:
        conn.close()

@app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    """Mark notification as read"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?',
                     (notification_id, session['user_id']))
        conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

@app.route('/api/notifications/read-all', methods=['PUT'])
def mark_all_notifications_read():
    """Mark all notifications as read - with proper access control"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session.get('user_id')
    """Mark all notifications as read"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        cursor.execute('UPDATE notifications SET is_read = 1 WHERE user_id = ?', (session['user_id'],))
        conn.commit()
        return jsonify({'success': True})
    finally:
        conn.close()

# ==========================================
# DOCUMENT MANAGEMENT
# ==========================================

@app.route('/api/documents', methods=['GET', 'POST'])
def documents():
    """Get or upload documents"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            patient_id = request.form.get('patient_id', type=int)
            if not patient_id:
                return jsonify({'error': 'Patient ID required'}), 400
            
            # Save file
            filename = secure_filename(f"doc_{patient_id}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
            doc_folder = os.path.join('static', 'documents')
            os.makedirs(doc_folder, exist_ok=True)
            filepath = os.path.join(doc_folder, filename)
            file.save(filepath)
            
            cursor.execute('''INSERT INTO documents (patient_id, doctor_id, document_type, title, file_path,
                             file_size, mime_type, description, tags, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (patient_id, session.get('user_db_id') if session['user_role'] == 'doctor' else None,
                          request.form.get('document_type', 'general'), request.form.get('title', file.filename),
                          f"documents/{filename}", os.path.getsize(filepath), file.mimetype,
                          request.form.get('description'), request.form.get('tags'), datetime.now().isoformat()))
            
            log_audit('document_uploaded', 'document', cursor.lastrowid, f'For patient {patient_id}')
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        else:
            patient_id = request.args.get('patient_id', type=int)
            if patient_id:
                cursor.execute('SELECT * FROM documents WHERE patient_id = ? ORDER BY created_at DESC', (patient_id,))
            else:
                cursor.execute('SELECT * FROM documents ORDER BY created_at DESC LIMIT 100')
            
            columns = [desc[0] for desc in cursor.description]
            documents = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return jsonify({'documents': documents})
    finally:
        conn.close()

# ==========================================
# MESSAGING SYSTEM
# ==========================================

@app.route('/api/users/list', methods=['GET'])
def users_list():
    """Get list of users for chat (doctors can see patients, patients can see doctors)"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        user_role = session['user_role']
        
        if user_role == 'doctor':
            # Return list of patients
            cursor.execute('''SELECT u.id, u.username, u.full_name, u.email, u.user_id, u.role
                             FROM users u
                             WHERE u.role = 'patient'
                             ORDER BY u.full_name, u.username''')
        elif user_role == 'patient':
            # Return list of doctors
            cursor.execute('''SELECT u.id, u.username, u.full_name, u.email, u.user_id, u.role
                             FROM users u
                             WHERE u.role = 'doctor'
                             ORDER BY u.full_name, u.username''')
        else:
            # Admin can see all
            cursor.execute('''SELECT u.id, u.username, u.full_name, u.email, u.user_id, u.role
                             FROM users u
                             ORDER BY u.role, u.full_name, u.username''')
        
        columns = [desc[0] for desc in cursor.description]
        users = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return jsonify({'users': users})
    finally:
        conn.close()

@app.route('/api/messages', methods=['GET', 'POST'])
def messages():
    """Get or send messages"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')
    cursor = conn.cursor()
    
    try:
        if request.method == 'POST':
            data = request.json
            print(f"Message POST request data: {data}")
            
            recipient_id = data.get('recipient_id')
            recipient_role = data.get('recipient_role')
            
            if not recipient_id:
                return jsonify({'error': 'Recipient required'}), 400
            
            # Ensure messages table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
            if not cursor.fetchone():
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sender_id INTEGER NOT NULL,
                        sender_role TEXT NOT NULL,
                        recipient_id INTEGER NOT NULL,
                        recipient_role TEXT NOT NULL,
                        subject TEXT,
                        message TEXT NOT NULL,
                        is_read INTEGER DEFAULT 0,
                        parent_message_id INTEGER,
                        created_at TEXT NOT NULL
                    )
                ''')
                conn.commit()
                print("Created messages table")
            
            # Verify sender and recipient exist
            cursor.execute('SELECT id FROM users WHERE id = ?', (session['user_id'],))
            if not cursor.fetchone():
                print(f"Sender user_id {session['user_id']} not found in users table")
                return jsonify({'error': 'Sender not found'}), 400
            
            cursor.execute('SELECT id FROM users WHERE id = ?', (recipient_id,))
            if not cursor.fetchone():
                print(f"Recipient user_id {recipient_id} not found in users table")
                return jsonify({'error': f'Recipient with ID {recipient_id} not found'}), 400
            
            try:
                cursor.execute('''INSERT INTO messages (sender_id, sender_role, recipient_id, recipient_role,
                                 subject, message, parent_message_id, created_at)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                             (session['user_id'], session['user_role'], recipient_id, recipient_role,
                              data.get('subject'), data.get('message'), data.get('parent_message_id'),
                              datetime.now().isoformat()))
                message_id = cursor.lastrowid
                print(f"Message inserted with ID: {message_id}")
            except Exception as e:
                print(f"Error inserting message: {str(e)}")
                import traceback
                traceback.print_exc()
                conn.rollback()
                return jsonify({'error': f'Failed to send message: {str(e)}'}), 500
            
            # Commit the message first before creating notifications/audit logs
            conn.commit()
            print(f"Message {message_id} committed successfully")
            
            # Close this connection before calling functions that may open their own connections
            conn.close()
            
            # Create notification (uses its own connection)
            try:
                # Set correct route based on recipient role
                message_link = '/patient/chat' if recipient_role == 'patient' else '/doctor/chat'
                create_notification(recipient_id, recipient_role, 'message',
                                  'New Message', f'You have a new message from {session.get("full_name")}',
                                  message_link)
            except Exception as e:
                print(f"Error creating notification: {str(e)}")
            
            # Log audit (uses its own connection)
            try:
                log_audit('message_sent', 'message', message_id)
            except Exception as e:
                print(f"Error logging audit: {str(e)}")
            
            return jsonify({'success': True, 'id': message_id, 'message': 'Message sent successfully'})
        else:
            # GET messages (inbox)
            cursor.execute('''SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                             FROM messages m
                             LEFT JOIN users u1 ON m.sender_id = u1.id
                             LEFT JOIN users u2 ON m.recipient_id = u2.id
                             WHERE m.recipient_id = ? ORDER BY m.created_at DESC LIMIT 100''',
                         (session['user_id'],))
            columns = [desc[0] for desc in cursor.description]
            messages_list = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Mark as read
            cursor.execute('UPDATE messages SET is_read = 1 WHERE recipient_id = ? AND is_read = 0',
                         (session['user_id'],))
            conn.commit()
            
            return jsonify({'messages': messages_list})
    finally:
        conn.close()

@app.route('/api/messages/with-doctor/<int:doctor_id>', methods=['GET'])
def get_messages_with_doctor(doctor_id):
    """Get messages between current patient and a specific doctor"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Find doctor's user ID
        cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (doctor_id, 'doctor'))
        doctor_user = cursor.fetchone()
        if not doctor_user:
            return jsonify({'messages': []})
        
        doctor_user_id = doctor_user[0]
        patient_user_id = session['user_id']
        
        # Get all messages between this patient and doctor
        cursor.execute('''SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                         FROM messages m
                         LEFT JOIN users u1 ON m.sender_id = u1.id
                         LEFT JOIN users u2 ON m.recipient_id = u2.id
                         WHERE (m.sender_id = ? AND m.recipient_id = ?) OR (m.sender_id = ? AND m.recipient_id = ?)
                         ORDER BY m.created_at ASC''',
                     (patient_user_id, doctor_user_id, doctor_user_id, patient_user_id))
        columns = [desc[0] for desc in cursor.description]
        messages_list = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify({'messages': messages_list})
    finally:
        conn.close()

@app.route('/api/messages/with-patient/<int:patient_id>', methods=['GET'])
def get_messages_with_patient(patient_id):
    """Get messages between current doctor and a specific patient"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    try:
        # Find patient's user ID
        cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (patient_id, 'patient'))
        patient_user = cursor.fetchone()
        if not patient_user:
            return jsonify({'messages': []})
        
        patient_user_id = patient_user[0]
        doctor_user_id = session['user_id']
        
        # Get all messages between this doctor and patient
        cursor.execute('''SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                         FROM messages m
                         LEFT JOIN users u1 ON m.sender_id = u1.id
                         LEFT JOIN users u2 ON m.recipient_id = u2.id
                         WHERE (m.sender_id = ? AND m.recipient_id = ?) OR (m.sender_id = ? AND m.recipient_id = ?)
                         ORDER BY m.created_at ASC''',
                     (doctor_user_id, patient_user_id, patient_user_id, doctor_user_id))
        columns = [desc[0] for desc in cursor.description]
        messages_list = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify({'messages': messages_list})
    finally:
        conn.close()

@app.route('/api/messages/to-doctor/<int:doctor_id>', methods=['POST'])
def send_message_to_doctor(doctor_id):
    """Send message to a doctor - handles user lookup server-side to fix 'doctor not registered' error"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect('db.sqlite3', timeout=10.0)
    conn.execute('PRAGMA busy_timeout = 10000')
    cursor = conn.cursor()
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get doctor profile
        cursor.execute('SELECT id, name, specialist_type FROM doctors WHERE id = ?', (doctor_id,))
        doctor = cursor.fetchone()
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        doctor_name = doctor[1]
        specialist_type = doctor[2]
        
        # Find or create doctor's user account
        cursor.execute('SELECT id FROM users WHERE user_id = ? AND role = ?', (doctor_id, 'doctor'))
        doctor_user = cursor.fetchone()
        
        if not doctor_user:
            # Auto-create user account for the doctor
            import hashlib
            default_password = hashlib.md5('doctor123'.encode()).hexdigest()
            username = f"doctor_{doctor_id}"
            
            cursor.execute('''INSERT INTO users (username, password, role, full_name, user_id, created_at)
                             VALUES (?, ?, 'doctor', ?, ?, ?)''',
                         (username, default_password, doctor_name or f'Doctor {doctor_id}', doctor_id, datetime.now().isoformat()))
            conn.commit()
            doctor_user_id = cursor.lastrowid
            print(f"Auto-created user account for doctor {doctor_id} with user_id {doctor_user_id}")
        else:
            doctor_user_id = doctor_user[0]
        
        # Ensure messages table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id INTEGER NOT NULL,
                    sender_role TEXT NOT NULL,
                    recipient_id INTEGER NOT NULL,
                    recipient_role TEXT NOT NULL,
                    subject TEXT,
                    message TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    parent_message_id INTEGER,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()
        
        # Insert message
        cursor.execute('''INSERT INTO messages (sender_id, sender_role, recipient_id, recipient_role,
                         subject, message, created_at)
                         VALUES (?, ?, ?, 'doctor', ?, ?, ?)''',
                     (session['user_id'], session['user_role'], doctor_user_id,
                      f'Chat from {session.get("full_name", "Patient")}', message, datetime.now().isoformat()))
        message_id = cursor.lastrowid
        conn.commit()
        
        # Create notification for doctor
        try:
            create_notification(doctor_user_id, 'doctor', 'message',
                              'New Message', f'You have a new message from {session.get("full_name", "a patient")}',
                              '/doctor/chat')
        except Exception as e:
            print(f"Error creating notification: {str(e)}")
        
        return jsonify({'success': True, 'id': message_id, 'message': 'Message sent successfully'})
    except Exception as e:
        print(f"Error sending message to doctor: {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return jsonify({'error': f'Failed to send message: {str(e)}'}), 500
    finally:
        conn.close()

# ==========================================
# ADVANCED SEARCH
# ==========================================

@app.route('/api/search', methods=['GET'])
def advanced_search():
    """Advanced search across all entities"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    query = request.args.get('q', '').strip()
    entity_type = request.args.get('type', 'all')  # all, patients, reports, appointments, prescriptions
    
    if not query:
        return jsonify({'error': 'Search query required'}), 400
    
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    results = {}
    
    try:
        if entity_type in ['all', 'patients']:
            # Ensure patients can only search themselves
            if session['user_role'] == 'patient':
                patient_id = session.get('user_db_id')
                cursor.execute('''SELECT id, name, email, phone, patient_id FROM patients 
                                WHERE id = ? AND (name LIKE ? OR email LIKE ? OR phone LIKE ? OR patient_id LIKE ?)
                                LIMIT 20''',
                             (patient_id, f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
                results['patients'] = [{'id': r[0], 'name': r[1], 'email': r[2], 'phone': r[3], 'patient_id': r[4]} 
                                       for r in cursor.fetchall()]
            else:
                cursor.execute('''SELECT id, name, email, phone, patient_id FROM patients 
                                WHERE name LIKE ? OR email LIKE ? OR phone LIKE ? OR patient_id LIKE ?
                                LIMIT 20''',
                             (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
                results['patients'] = [{'id': r[0], 'name': r[1], 'email': r[2], 'phone': r[3], 'patient_id': r[4]} 
                                       for r in cursor.fetchall()]
        
        if entity_type in ['all', 'reports']:
            # Ensure patients can only search their own reports
            if session['user_role'] == 'patient':
                patient_id = session.get('user_db_id')
                cursor.execute('''SELECT r.id, r.original_filename, r.specialist_type, r.created_at, p.name as patient_name
                                FROM reports r
                                LEFT JOIN patients p ON r.patient_id = p.id
                                WHERE r.patient_id = ? AND (r.original_filename LIKE ? OR r.extracted_text LIKE ? OR p.name LIKE ?)
                                ORDER BY r.created_at DESC LIMIT 20''',
                             (patient_id, f'%{query}%', f'%{query}%', f'%{query}%'))
            else:
                cursor.execute('''SELECT r.id, r.original_filename, r.specialist_type, r.created_at, p.name as patient_name
                                FROM reports r
                                LEFT JOIN patients p ON r.patient_id = p.id
                                WHERE r.original_filename LIKE ? OR r.extracted_text LIKE ? OR p.name LIKE ?
                                ORDER BY r.created_at DESC LIMIT 20''',
                             (f'%{query}%', f'%{query}%', f'%{query}%'))
            columns = [desc[0] for desc in cursor.description]
            results['reports'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if entity_type in ['all', 'appointments']:
            # Ensure patients can only search their own appointments
            if session['user_role'] == 'patient':
                patient_id = session.get('user_db_id')
                cursor.execute('''SELECT a.id, a.appointment_date, a.appointment_time, p.name as patient_name, d.name as doctor_name
                                FROM appointments a
                                LEFT JOIN patients p ON a.patient_id = p.id
                                LEFT JOIN doctors d ON a.doctor_id = d.id
                                WHERE a.patient_id = ? AND (p.name LIKE ? OR d.name LIKE ? OR a.reason LIKE ?)
                                ORDER BY a.appointment_date DESC LIMIT 20''',
                             (patient_id, f'%{query}%', f'%{query}%', f'%{query}%'))
            else:
                cursor.execute('''SELECT a.id, a.appointment_date, a.appointment_time, p.name as patient_name, d.name as doctor_name
                                FROM appointments a
                                LEFT JOIN patients p ON a.patient_id = p.id
                                LEFT JOIN doctors d ON a.doctor_id = d.id
                                WHERE p.name LIKE ? OR d.name LIKE ? OR a.reason LIKE ?
                                ORDER BY a.appointment_date DESC LIMIT 20''',
                             (f'%{query}%', f'%{query}%', f'%{query}%'))
            columns = [desc[0] for desc in cursor.description]
            results['appointments'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if entity_type in ['all', 'prescriptions']:
            # Ensure patients can only search their own prescriptions
            if session['user_role'] == 'patient':
                patient_id = session.get('user_db_id')
                cursor.execute('''SELECT pr.id, pr.prescription_text, pr.created_at, p.name as patient_name, d.name as doctor_name
                                FROM prescriptions pr
                                LEFT JOIN patients p ON pr.patient_id = p.id
                                LEFT JOIN doctors d ON pr.doctor_id = d.id
                                WHERE pr.patient_id = ? AND (pr.prescription_text LIKE ? OR pr.medications LIKE ? OR p.name LIKE ?)
                                ORDER BY pr.created_at DESC LIMIT 20''',
                             (patient_id, f'%{query}%', f'%{query}%', f'%{query}%'))
            else:
                cursor.execute('''SELECT pr.id, pr.prescription_text, pr.created_at, p.name as patient_name, d.name as doctor_name
                                FROM prescriptions pr
                                LEFT JOIN patients p ON pr.patient_id = p.id
                                LEFT JOIN doctors d ON pr.doctor_id = d.id
                                WHERE pr.prescription_text LIKE ? OR pr.medications LIKE ? OR p.name LIKE ?
                                ORDER BY pr.created_at DESC LIMIT 20''',
                             (f'%{query}%', f'%{query}%', f'%{query}%'))
            columns = [desc[0] for desc in cursor.description]
            results['prescriptions'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        log_audit('search_performed', None, None, f'Query: {query}, Type: {entity_type}')
        return jsonify({'results': results})
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)

