import os
import json
import re
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import joblib
import numpy as np
import nltk

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'truthlens_secret_key_2026_super_secure')

# Database Configuration (MySQL default with automatic SQLite fallback)
DB_TYPE = os.getenv('DB_TYPE', 'mysql').lower()
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'root')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'truthlens_db')
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'truthlens.db')

mysql_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
sqlite_uri = f"sqlite:///{os.path.join(app.root_path, SQLITE_DB_PATH)}"

# Check database availability
use_mysql = False
if DB_TYPE == 'mysql':
    try:
        import pymysql
        conn = pymysql.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=2
        )
        conn.close()
        use_mysql = True
        print("[+] MySQL server is reachable.")
    except Exception as err:
        print(f"[!] MySQL server unreachable ({err}). Automatically using SQLite fallback database.")

if use_mysql:
    app.config['SQLALCHEMY_DATABASE_URI'] = mysql_uri
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = sqlite_uri

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Database ORM Models
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user') # 'user' or 'admin'
    full_name = db.Column(db.String(100), default='')
    bio = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    news_title = db.Column(db.String(255), default='Untitled Article')
    news_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False) # 'REAL' or 'FAKE'
    confidence = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------------------------------------------------------------------
# Machine Learning Model Manager
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(app.root_path, 'model', 'news_classifier.joblib')
VECTORIZER_PATH = os.path.join(app.root_path, 'model', 'tfidf_vectorizer.joblib')

ml_model = None
ml_vectorizer = None

def load_ml_artifacts():
    global ml_model, ml_vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[!] Model files not found. Auto-training ML model...")
        try:
            from train_model import train_fake_news_model
            train_fake_news_model()
        except Exception as err:
            print(f"[X] Model training error: {err}")
            return False
            
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        ml_model = joblib.load(MODEL_PATH)
        ml_vectorizer = joblib.load(VECTORIZER_PATH)
        print("[+] ML Model & Vectorizer loaded successfully into memory.")
        return True
    return False

# Clean text function consistent with train_model
def clean_news_input(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def get_explanation_keywords(raw_text, prediction_label, top_n=5):
    """
    Extract key influential words from the input text that contributed most to the prediction.
    """
    if ml_model is None or ml_vectorizer is None:
        return []
    
    cleaned = clean_news_input(raw_text)
    feature_names = ml_vectorizer.get_feature_names_out()
    tfidf_vector = ml_vectorizer.transform([cleaned])
    
    # Get non-zero indices for this document
    non_zero_indices = tfidf_vector.nonzero()[1]
    if len(non_zero_indices) == 0:
        return ["general context", "vocabulary structure"]
    
    coefficients = ml_model.coef_[0]
    
    # Positive weights correlate with FAKE (or REAL depending on classes array)
    classes = ml_model.classes_ # e.g. ['FAKE', 'REAL']
    fake_idx = np.where(classes == 'FAKE')[0][0] if 'FAKE' in classes else 0
    
    word_scores = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        val = tfidf_vector[0, idx]
        weight = coefficients[idx]
        # Score direction based on predicted label
        score = weight * val if prediction_label == 'FAKE' else -weight * val
        word_scores.append((word, score))
        
    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [w[0] for w in word_scores[:top_n]]
    return top_words if top_words else ["news tone", "unverified phrasing"]

# ---------------------------------------------------------------------------
# Auth Decorators
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Admin authorization required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------------------------------------------------------------------
# Web Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        
        if not username or not email or not password:
            flash('Please fill in all required fields.', 'danger')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
            
        if User.query.filter_by(username=username).first():
            flash('Username is already taken.', 'danger')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email address is already registered.', 'danger')
            return render_template('register.html')
            
        new_user = User(
            username=username,
            email=email,
            full_name=full_name or username,
            role='user'
        )
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account registered successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['full_name'] = user.full_name or user.username
            
            flash(f'Welcome back, {user.full_name or user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username/email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    
    # Calculate Statistics
    user_predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    total_scans = len(user_predictions)
    real_count = sum(1 for p in user_predictions if p.prediction == 'REAL')
    fake_count = sum(1 for p in user_predictions if p.prediction == 'FAKE')
    avg_confidence = round(sum(p.confidence for p in user_predictions) / total_scans, 1) if total_scans > 0 else 0.0
    
    recent_predictions = user_predictions[:5]
    
    return render_template(
        'dashboard.html',
        user=user,
        total_scans=total_scans,
        real_count=real_count,
        fake_count=fake_count,
        avg_confidence=avg_confidence,
        recent_predictions=recent_predictions
    )

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json() or {}
    news_title = data.get('news_title', '').strip() or 'Untitled News Article'
    news_text = data.get('news_text', '').strip()
    
    # 1. Run Machine Learning Pipeline & Strict Validation
    from backend.model_manager import predict_news
    result = predict_news(news_title, news_text)
    
    if result.get('status') == 'error':
        return jsonify({
            'status': 'error',
            'message': result.get('message', 'Please enter a meaningful news article.')
        }), 400
        
    prediction_class = result['prediction']
    confidence_score = result['confidence']
    explanation_str = result['explanation']
    
    # Combined explanation keywords summary
    all_keywords = result['real_indicators'] + result['fake_indicators']
    explanation_full = f"{explanation_str} Keywords: {', '.join(all_keywords)}"
    
    # 2. Save Prediction to Database
    new_pred = Prediction(
        user_id=session['user_id'],
        news_title=news_title[:250],
        news_text=news_text,
        prediction=prediction_class,
        confidence=confidence_score,
        explanation=explanation_full
    )
    db.session.add(new_pred)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'prediction_id': new_pred.id,
        'prediction': prediction_class,
        'confidence': confidence_score,
        'explanation': explanation_str,
        'real_indicators': result['real_indicators'],
        'fake_indicators': result['fake_indicators'],
        'word_count': result['word_count'],
        'char_count': result['char_count'],
        'reading_time_mins': result['reading_time_mins'],
        'model_used': result['model_used'],
        'timestamp': new_pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/history')
@login_required
def history():
    user_id = session['user_id']
    query_str = request.args.get('search', '').strip()
    filter_label = request.args.get('filter', '').strip().upper()
    
    pred_query = Prediction.query.filter_by(user_id=user_id)
    
    if filter_label in ['REAL', 'FAKE']:
        pred_query = pred_query.filter_by(prediction=filter_label)
        
    if query_str:
        pred_query = pred_query.filter(
            (Prediction.news_title.ilike(f'%{query_str}%')) |
            (Prediction.news_text.ilike(f'%{query_str}%'))
        )
        
    predictions = pred_query.order_by(Prediction.created_at.desc()).all()
    
    return render_template(
        'history.html',
        predictions=predictions,
        search_query=query_str,
        filter_label=filter_label
    )

@app.route('/api/history/<int:pred_id>/delete', methods=['DELETE', 'POST'])
@login_required
def delete_history_item(pred_id):
    pred = Prediction.query.filter_by(id=pred_id, user_id=session['user_id']).first()
    if not pred:
        return jsonify({'status': 'error', 'message': 'Prediction item not found.'}), 404
        
    db.session.delete(pred)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Prediction entry deleted successfully.'})

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = db.session.get(User, session['user_id'])
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update_profile':
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip().lower()
            bio = request.form.get('bio', '').strip()
            
            existing_email = User.query.filter(User.email == email, User.id != user.id).first()
            if existing_email:
                flash('Email is already taken by another account.', 'danger')
            else:
                user.full_name = full_name
                user.email = email
                user.bio = bio
                session['full_name'] = full_name or user.username
                db.session.commit()
                flash('Profile updated successfully!', 'success')
                
        elif action == 'change_password':
            current_password = request.form.get('current_password', '')
            new_password = request.form.get('new_password', '')
            confirm_new_password = request.form.get('confirm_new_password', '')
            
            if not user.check_password(current_password):
                flash('Incorrect current password.', 'danger')
            elif new_password != confirm_new_password:
                flash('New passwords do not match.', 'danger')
            elif len(new_password) < 6:
                flash('New password must be at least 6 characters.', 'danger')
            else:
                user.set_password(new_password)
                db.session.commit()
                flash('Password changed successfully!', 'success')
                
    return render_template('profile.html', user=user)

@app.route('/admin')
@login_required
@admin_required
def admin():
    users = User.query.order_by(User.created_at.desc()).all()
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    
    total_users = len(users)
    total_scans = len(predictions)
    real_count = sum(1 for p in predictions if p.prediction == 'REAL')
    fake_count = sum(1 for p in predictions if p.prediction == 'FAKE')
    
    return render_template(
        'admin.html',
        users=users,
        predictions=predictions[:10],
        total_users=total_users,
        total_scans=total_scans,
        real_count=real_count,
        fake_count=fake_count
    )

@app.route('/api/admin/user/<int:user_id>/toggle_role', methods=['POST'])
@login_required
@admin_required
def toggle_user_role(user_id):
    if user_id == session['user_id']:
        return jsonify({'status': 'error', 'message': 'You cannot change your own admin role.'}), 400
        
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found.'}), 404
        
    user.role = 'admin' if user.role == 'user' else 'user'
    db.session.commit()
    return jsonify({'status': 'success', 'new_role': user.role})

@app.route('/api/admin/user/<int:user_id>/delete', methods=['DELETE', 'POST'])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'status': 'error', 'message': 'You cannot delete your own admin account.'}), 400
        
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found.'}), 404
        
    db.session.delete(user)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'User account deleted.'})

@app.route('/api/stats')
@login_required
def api_stats():
    predictions = Prediction.query.all()
    real_count = sum(1 for p in predictions if p.prediction == 'REAL')
    fake_count = sum(1 for p in predictions if p.prediction == 'FAKE')
    
    # Calculate user prediction counts
    user_id = session['user_id']
    user_preds = [p for p in predictions if p.user_id == user_id]
    u_real = sum(1 for p in user_preds if p.prediction == 'REAL')
    u_fake = sum(1 for p in user_preds if p.prediction == 'FAKE')
    
    return jsonify({
        'global': {'real': real_count, 'fake': fake_count},
        'user': {'real': u_real, 'fake': u_fake}
    })

# ---------------------------------------------------------------------------
# Global Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'API endpoint not found.'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500
    flash('An internal server error occurred. Please try again.', 'danger')
    return redirect(url_for('dashboard'))

@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    db.session.rollback()
    print(f"[!] Unhandled Exception: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500
    flash('An error occurred while processing your request.', 'danger')
    return redirect(url_for('dashboard'))
def init_app_database():
    with app.app_context():
        db.create_all()
        # Seed admin user if no users exist
        if not User.query.filter_by(username='admin').first():
            admin_user = User(
                username='admin',
                email='admin@truthlens.ai',
                full_name='TruthLens Administrator',
                role='admin'
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            
        if not User.query.filter_by(username='demouser').first():
            demo_user = User(
                username='demouser',
                email='user@truthlens.ai',
                full_name='Demo Analyst',
                role='user'
            )
            demo_user.set_password('user123')
            db.session.add(demo_user)
            
        db.session.commit()

init_app_database()
load_ml_artifacts()

if __name__ == '__main__':
    print("\n========================================================")
    print(" [+] TruthLens AI Platform Running on http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
