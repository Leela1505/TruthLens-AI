# TruthLens AI – Intelligent Fake News Detection Platform

![TruthLens AI Platform](static/images/logo.svg)

TruthLens AI is an end-to-end full-stack web platform engineered to detect fake news, online disinformation, and unverified news articles using Natural Language Processing (NLP) and Machine Learning (ML).

---

## 🌟 Key Features

### 🔐 1. Authentication & Security
- User registration and login with session management.
- Password hashing using `werkzeug.security` (PBKDF2/scrypt).
- Role-based authorization (`user` vs. `admin`).

### 🌐 2. Modern Landing Page & Responsive UI
- Dark slate-blue aesthetic with glassmorphism design, Bootstrap 5, and smooth micro-animations.
- Hero, Features, Live Interactive Preview Demo, About, and Support Contact sections.

### 📊 3. User Dashboard
- Real-time statistics cards: Total Scans, Real News Count, Fake News Count, and Average Confidence.
- Interactive Chart.js visualizations (Doughnut charts showing verdict breakdown).
- Recent scans summary table with quick navigation.

### ⚡ 4. AI Fake News Detector
- Text analysis with Real vs. FAKE prediction classification.
- Confidence score percentage (0% to 100%).
- **Explainable AI**: Highlights top trigger keywords and linguistic patterns that drove the model's prediction.
- Print/Save PDF report option for journalists and researchers.

### 📜 5. Prediction History Log
- Automatic persistent storage of all verifications.
- Live keyword search and category filtering (`REAL` / `FAKE` / `ALL`).
- AJAX deletion with instant table updates.

### 👤 6. Profile Management
- Update full name, email, and bio.
- Secure password change interface with confirmation validation.

### 🛡️ 7. Admin Control Panel
- Global system statistics (Total registered users, total news scans, verdict breakdown).
- Global user management table with role toggling (`User` ↔ `Admin`) and user deletion.
- Audit prediction logs for full administrative visibility.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 (Custom Glassmorphism), JavaScript (ES6+ AJAX), Bootstrap 5, Chart.js, FontAwesome 6 |
| **Backend** | Python 3, Flask, Flask-SQLAlchemy, Flask-CORS, Werkzeug, Python-Dotenv |
| **Database** | MySQL (with dynamic SQLite automatic fallback) |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib, NLTK (TF-IDF Vectorizer & Logistic Regression Classifier) |

---

## 📂 Project Structure

```
TruthLens-AI/
├── app.py                     # Main Flask Application & Web API Routes
├── train_model.py             # NLP Preprocessing & Machine Learning Training Script
├── requirements.txt           # Python Project Dependencies
├── .env.example               # Environment Configuration Example
├── README.md                  # Comprehensive Documentation
├── backend/
│   ├── __init__.py            # Package Init
│   └── ml_engine.py           # NLP Text Cleaning & Model Utilities
├── frontend/
│   └── README.md              # Frontend Architecture Guide
├── database/
│   └── schema.sql             # MySQL Schema Script & Seed Data
├── dataset/
│   └── fake_or_real_news.csv  # Training Dataset Corpus
├── model/
│   ├── news_classifier.joblib # Saved ML Classifier Artifact
│   └── tfidf_vectorizer.joblib# Saved TF-IDF Vectorizer Artifact
├── static/
│   ├── css/
│   │   └── style.css          # Custom Styling Tokens & Glassmorphism
│   ├── js/
│   │   ├── main.js            # AJAX Detection Handler & UI Logic
│   │   └── dashboard_charts.js# Chart.js Visualizations
│   └── images/
│       └── logo.svg           # Brand Vector Graphics
└── templates/
    ├── base.html              # Base Navigation Layout & Toast Container
    ├── index.html             # Landing Page
    ├── login.html             # User Login Page
    ├── register.html          # User Registration Page
    ├── dashboard.html         # User Dashboard
    ├── detect.html            # Fake News Detection Tool Page
    ├── history.html           # Prediction History Log Page
    ├── profile.html           # Profile Management & Password Reset
    └── admin.html             # Admin Panel & User Audit Management
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open Workspace
Navigate to the root directory of the project:
```bash
cd TruthLens-AI
```

### 2. Install Dependencies
Install all required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Database Setup
TruthLens AI supports both **MySQL** and **SQLite**:
- **MySQL**: Import `database/schema.sql` into your MySQL instance:
  ```bash
  mysql -u root -p < database/schema.sql
  ```
- **SQLite (Default Out-of-the-Box)**: If MySQL is not running locally, TruthLens AI automatically creates and seeds `truthlens.db` via SQLite!

### 4. Train the Machine Learning Model
Train the NLP TF-IDF classifier on the dataset:
```bash
python train_model.py
```
*Outputs `news_classifier.joblib` and `tfidf_vectorizer.joblib` into the `model/` folder.*

### 5. Run the Platform
Start the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 Default Login Credentials

| Account Role | Username / Email | Password |
| :--- | :--- | :--- |
| **Administrator** | `admin` / `admin@truthlens.ai` | `admin123` |
| **Demo Analyst** | `demouser` / `user@truthlens.ai` | `user123` |

---

## 🔌 API Endpoints Reference

- `POST /api/predict`: Analyzes news text input and returns prediction, confidence score, and explainable keyword tags.
- `GET /api/stats`: Returns JSON summary of real vs fake news counts for charts.
- `DELETE /api/history/<id>/delete`: Deletes a specific prediction entry.
- `POST /api/admin/user/<id>/toggle_role`: Toggles admin privileges for a user.
- `DELETE /api/admin/user/<id>/delete`: Permanently removes a user account.

---

## 🛡️ License & Acknowledgements

Developed as an intelligent fake news detection solution using open-source Machine Learning and Natural Language Processing frameworks.
