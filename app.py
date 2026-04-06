from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import jwt
import datetime
import bcrypt
import os
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'eduguide_v2.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'super-secret-key-eduguide'

db = SQLAlchemy(app)

# Gemini AI Configuration
GOOGLE_API_KEY = "AIzaSyBdrqSOsaCgsCFxVIabPcvvZepoHP3buNc"
genai.configure(api_key=GOOGLE_API_KEY)
ai_model = genai.GenerativeModel('gemini-pro')

# ML Model Loading
model_path = os.path.join(basedir, 'career_model.pkl')
try:
    with open(model_path, 'rb') as f:
        ml_model = pickle.load(f)
except Exception as e:
    ml_model = None
    print(f"Warning: Model not found. Did you run ml_model.py? {e}")

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='student')
    
    # New Profile Fields
    age = db.Column(db.String(10), default='')
    school = db.Column(db.String(150), default='')
    grade = db.Column(db.String(50), default='')
    bio = db.Column(db.Text, default='')
    
    # Personality Analysis
    personality_type = db.Column(db.String(50), default='Not Assessed')
    personality_scores = db.Column(db.Text, default='{}') # JSON: analytical, creative, leader, etc.
    xp = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Career(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=False)
    skills_required = db.Column(db.Text)
    roadmap = db.Column(db.Text)
    colleges = db.Column(db.Text)
    exams = db.Column(db.Text)
    
    # Market Insights
    avg_salary_in = db.Column(db.String(100)) # e.g. "₹8L - ₹25L"
    avg_salary_gl = db.Column(db.String(100)) # e.g. "$60k - $140k"
    demand_level = db.Column(db.String(50)) # High/Med/Low
    growth_rate = db.Column(db.String(50)) # e.g. "+15% YoY"
    
    # New Extra Fields
    resume_keywords = db.Column(db.Text)
    suggested_projects = db.Column(db.Text)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    career_id = db.Column(db.Integer, db.ForeignKey('career.id'), nullable=False)
    confidence_score = db.Column(db.Float)
    scores_data = db.Column(db.String(200)) # Store original scores as JSON string to compare later
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('recommendations', lazy=True))
    career = db.relationship('Career', backref=db.backref('recommendations', lazy=True))

class ForumPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), default='General')
    replies_data = db.Column(db.Text, default='[]') # JSON sorted log of replies for simplicity instead of full table overhead
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('posts', lazy=True))

class JobSuggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    career_id = db.Column(db.Integer, db.ForeignKey('career.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text)
    job_type = db.Column(db.String(50)) # Internship/Job
    skills = db.Column(db.Text)
    
    career = db.relationship('Career', backref=db.backref('jobs', lazy=True))

class ResumeAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    career_id = db.Column(db.Integer, db.ForeignKey('career.id'), nullable=False)
    resume_text = db.Column(db.Text)
    analysis_json = db.Column(db.Text) # Match %, missing skills, etc.
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('resumes', lazy=True))

class PersonalityTest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    trait_scores = db.Column(db.Text) # JSON: analytical: 80, creative: 40...
    personality_type = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('personality_tests', lazy=True))

class Mentor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    expertise = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text, nullable=False)
    email = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(255), default="https://via.placeholder.com/150")

# Helper Decorator for Token
def token_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            token = token.split(" ")[1] # Bearer <token>
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        if not current_user:
            return jsonify({'message': 'User not found!'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

# API Routes
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('name') or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing data'}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'User already exists'}), 400

    hashed_pwd = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Admin role is no longer available via public signup
    role = data.get('role', 'student').lower()
    if role not in ['student', 'parent']:
        role = 'student'

    new_user = User(
        name=data['name'], 
        email=data['email'], 
        password_hash=hashed_pwd,
        role=role
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if not user:
        return jsonify({'message': 'Invalid credentials'}), 401

    if bcrypt.checkpw(data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'token': token, 
            'user': {'id': user.id, 'name': user.name, 'email': user.email, 'xp': user.xp, 'role': user.role}
        }), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/auth/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if not user or user.role != 'admin':
        return jsonify({'message': 'Invalid admin credentials'}), 401

    if bcrypt.checkpw(data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'token': token, 
            'user': {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role}
        }), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/profile', methods=['GET', 'PUT'])
@token_required
def profile(current_user):
    if request.method == 'GET':
        return jsonify({
            'name': current_user.name,
            'email': current_user.email,
            'age': current_user.age,
            'school': current_user.school,
            'grade': current_user.grade,
            'bio': current_user.bio,
            'xp': current_user.xp
        }), 200
        
    if request.method == 'PUT':
        data = request.get_json()
        current_user.name = data.get('name', current_user.name)
        current_user.age = data.get('age', current_user.age)
        current_user.school = data.get('school', current_user.school)
        current_user.grade = data.get('grade', current_user.grade)
        current_user.bio = data.get('bio', current_user.bio)
        db.session.commit()
        return jsonify({'message': 'Profile updated successfully'})

@app.route('/api/career/<int:career_id>', methods=['GET'])
def get_career(career_id):
    career = Career.query.get_or_404(career_id)
    return jsonify({
        'id': career.id,
        'title': career.title,
        'description': career.description,
        'skills_required': career.skills_required,
        'roadmap': career.roadmap,
        'colleges': career.colleges,
        'exams': career.exams,
        'resume_keywords': career.resume_keywords,
        'suggested_projects': career.suggested_projects
    })

@app.route('/api/dashboard', methods=['GET'])
@token_required
def dashboard(current_user):
    recs = Recommendation.query.filter_by(user_id=current_user.id).order_by(Recommendation.created_at.desc()).all()
    history = []
    for r in recs:
        # Load the scores associated with this recommendation to do skill gap analysis
        import json
        scores = []
        if r.scores_data:
            scores = json.loads(r.scores_data)
        
        history.append({
            'id': r.id,
            'career_id': r.career.id,
            'career_title': r.career.title,
            'confidence_score': r.confidence_score,
            'date': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'user_scores': scores,
            'resume_keywords': r.career.resume_keywords,
            'suggested_projects': r.career.suggested_projects,
            'roadmap': r.career.roadmap,
            'colleges': r.career.colleges,
            'exams': r.career.exams
        })
    return jsonify({
        'user': {
            'name': current_user.name, 
            'email': current_user.email, 
            'role': current_user.role,
            'xp': current_user.xp,
            'personality_type': current_user.personality_type
        },
        'recommendations': history
    }), 200

@app.route('/api/questions', methods=['GET'])
def get_questions():
    questions = [
        {"id": 1, "text": "How good are you at Mathematics and analytical solving?", "type": "math"},
        {"id": 2, "text": "Are you interested in Science, research, and biology?", "type": "science"},
        {"id": 3, "text": "How confident are you in public speaking and management?", "type": "communication"},
        {"id": 4, "text": "Do you enjoy art, design, writing or content creation?", "type": "creativity"},
        {"id": 5, "text": "How interested are you in computers, coding, and algorithms?", "type": "programming"}
    ]
    return jsonify(questions), 200

@app.route('/api/recommend', methods=['POST'])
@token_required
def recommend(current_user):
    data = request.get_json()
    scores = data.get('scores', [])
    if len(scores) != 5:
         return jsonify({'message': 'Please provide scores for all 5 categories.'}), 400
    
    if not ml_model:
        return jsonify({'message': 'ML Model not trained yet.'}), 500

    features = np.array([scores])
    probabilities = ml_model.predict_proba(features)[0]
    
    
    # 3. New: Weighted Personality Adjustment
    if current_user.personality_type != 'Not Assessed':
        import json
        p_scores = json.loads(current_user.personality_scores)
        
        # Mapping: Personality affects certain careers
        # Engineering -> Analytical+, Creative-
        # Medicine -> Analytical+, Patient/Communication+ (represented by Science/Comm in skill quiz)
        # Arts -> Creative+, Analytical-
        # Business -> Leader+, Comm+
        
        # career_mapping: 0: Engineering, 1: Medicine, 2: Arts, 3: Commerce
        if p_scores.get('analytical', 0) > 4: probabilities[0] *= 1.2
        if p_scores.get('creative', 0) > 4: probabilities[2] *= 1.2
        if p_scores.get('leader', 0) > 4: probabilities[3] *= 1.2
        
        # Normalize probabilities again
        probabilities = probabilities / np.sum(probabilities)
        top_indices = np.argsort(probabilities)[::-1][:3]
    
    import json
    results = []
    
    # Insert Top 3 into DB and Results Array
    for idx_class in top_indices:
        title = career_mapping.get(idx_class)
        career = Career.query.filter_by(title=title).first()
        if career:
            conf = float(probabilities[idx_class])
            results.append({
                'career_id': career.id,
                'recommended_career': career.title,
                'confidence': conf,
                'description': career.description
            })
            
            # Save historical recommendation
            rec = Recommendation(
                user_id=current_user.id,
                career_id=career.id,
                confidence_score=conf,
                scores_data=json.dumps(scores)
            )
            db.session.add(rec)
            
    # Award XP for taking assessment
    current_user.xp += 50

    db.session.commit()
    
    if results:
        return jsonify(results), 200
    else:
        return jsonify({'message': 'Career not found in DB.'}), 500

@app.route('/api/forums', methods=['GET', 'POST'])
@token_required
def forums(current_user):
    import json
    if request.method == 'GET':
        posts = ForumPost.query.order_by(ForumPost.created_at.desc()).all()
        result = []
        for p in posts:
            result.append({
                'id': p.id,
                'user_name': p.user.name,
                'title': p.title,
                'content': p.content,
                'category': p.category,
                'replies': json.loads(p.replies_data),
                'date': p.created_at.strftime('%b %d, %Y')
            })
        return jsonify(result), 200
        
    if request.method == 'POST':
        data = request.get_json()
        if not data.get('title') or not data.get('content'):
            return jsonify({'message': 'Missing data'}), 400
            
        post = ForumPost(
            user_id=current_user.id,
            title=data['title'],
            content=data['content'],
            category=data.get('category', 'General')
        )
        db.session.add(post)
        
        # Award xp for posting
        current_user.xp += 10
        db.session.commit()
        return jsonify({'message': 'Post created successfully'}), 201

@app.route('/api/forums/<int:post_id>/reply', methods=['POST'])
@token_required
def forum_reply(current_user, post_id):
    import json
    post = ForumPost.query.get_or_404(post_id)
    data = request.get_json()
    
    if not data.get('content'):
        return jsonify({'message': 'Content missing'}), 400
        
    replies = json.loads(post.replies_data)
    replies.append({
        'user_name': current_user.name,
        'content': data['content'],
        'date': datetime.datetime.utcnow().strftime('%b %d, %Y')
    })
    post.replies_data = json.dumps(replies)
    
    # Award XP for replying
    current_user.xp += 5
    db.session.commit()
    
    return jsonify({'message': 'Replied successfully', 'replies': replies}), 200


@app.route('/api/news', methods=['GET'])
def get_news():
    import random
    import datetime
    # Mock database of career news
    news_items = [
        {"id": 1, "title": "The Rise of AI in Healthcare", "category": "Medicine & Tech", "summary": "How artificial intelligence is revolutionizing diagnostic medicine and creating new hybrid career paths for biology students.", "date": "2 Days Ago", "url": "#"},
        {"id": 2, "title": "Top Tech Skills Demanded in 2026", "category": "Engineering", "summary": "Cloud architecture, Rust programming, and Machine Learning engineering top the highest-paying entry-level jobs this year.", "date": "5 Days Ago", "url": "#"},
        {"id": 3, "title": "Why Creative Arts are Surviving Automation", "category": "Arts & Design", "summary": "While generative AI improves, the demand for human-driven UI/UX design and physical artistry is experiencing a renaissance.", "date": "1 Week Ago", "url": "#"},
        {"id": 4, "title": "New MBA Trends: Green Finance", "category": "Commerce / Business", "summary": "Top Ivy League business schools are overhauling their curriculum to focus heavily on sustainable energy markets and green finance.", "date": "2 Weeks Ago", "url": "#"},
    ]
    random.shuffle(news_items)
    return jsonify(news_items), 200

@app.route('/api/mentors', methods=['GET'])
@token_required
def get_mentors(current_user):
    mentors = Mentor.query.all()
    result = []
    for m in mentors:
        result.append({
            'id': m.id,
            'name': m.name,
            'expertise': m.expertise,
            'company': m.company,
            'bio': m.bio,
            'email': m.email,
            'image_url': m.image_url
        })
    return jsonify(result), 200

# AI Chat API (Gemini Powered with Rule-based Fallback)
@app.route('/api/chat', methods=['POST'])
def ai_chat():
    import re
    data = request.get_json()
    message = data.get('message', '')
    msg_lower = message.lower()
    
    if not message:
        return jsonify({'message': 'Query required'}), 400
        
    # Rule-based Expert System (Fallback & Keywords)
    intents = {
        'greeting': r'\b(hello|hi|hey|greetings|what\'s up)\b',
        'engineering': r'\b(engineer|tech|software|hardware|computing|coding|programming|btech)\b',
        'medicine': r'\b(medicine|doctor|medical|biology|health|hospital|mbbs|nurse)\b',
        'arts': r'\b(art|design|creative|writing|music|painting|bfa)\b',
        'commerce': r'\b(business|commerce|finance|accounting|management|mba|bba)\b',
        'quiz': r'\b(quiz|test|assessment|recommend|match)\b'
    }
    
    responses = {
        'greeting': "Hello! I am EduGuide's AI. I can advise you on career roadmaps, or overcoming skill gaps. What's on your mind?",
        'engineering': "Engineering is a fantastic path with high growth! It requires robust analytical and core programming skills.",
        'medicine': "Medicine is a highly rewarding, empathy-driven field. The roadmap involves Pre-Med, Medical School, and Residency.",
        'arts': "A career in the Arts is all about creative expression and design. Building a strong online portfolio is crucial.",
        'commerce': "Commerce and Business manage the financial world. Strong leadership and networking are key.",
        'quiz': "Take our AI Assessment Quiz to find your perfect career match based on 5 core metrics!"
    }

    try:
        # 1. Primary: Google Gemini AI
        prompt = f"You are EduGuide Assistant, an AI career counselor for students (9th-college). Answer this query briefly and professionally, providing helpful career advice: {message}"
        response = ai_model.generate_content(prompt)
        return jsonify({'reply': response.text}), 200
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # 2. Secondary: Rule-based Fallback
        matched_reply = "I'm currently in basic mode, but I can still help! Try asking about Engineering, Medicine, or our Career Quiz."
        for intent, pattern in intents.items():
            if re.search(pattern, msg_lower):
                matched_reply = responses[intent]
                break
        return jsonify({'reply': matched_reply}), 200

# ADVANCED FEATURES ENDPOINTS

@app.route('/api/personality/submit', methods=['POST'])
@token_required
def submit_personality(current_user):
    data = request.get_json()
    # Expecting scores for analytical, creative, leader (scale 1-5 or 1-10)
    scores = {
        'analytical': data.get('analytical', 0),
        'creative': data.get('creative', 0),
        'leader': data.get('leader', 0)
    }
    
    # Determine type
    p_type = "The Strategist" # Default
    if scores['leader'] > 7: p_type = "The Natural Leader"
    elif scores['creative'] > 7: p_type = "The Visionary Creator"
    elif scores['analytical'] > 7: p_type = "The Analytical Thinker"
    
    import json
    current_user.personality_type = p_type
    current_user.personality_scores = json.dumps(scores)
    
    test = PersonalityTest(user_id=current_user.id, trait_scores=json.dumps(scores), personality_type=p_type)
    db.session.add(test)
    current_user.xp += 100 # Big bonus for personality testing
    db.session.commit()
    
    return jsonify({'type': p_type, 'scores': scores}), 200

@app.route('/api/career/skill-gap/<int:career_id>', methods=['GET'])
@token_required
def skill_gap(current_user, career_id):
    career = Career.query.get_or_404(career_id)
    # Get latest recommendation to get user's skill scores
    latest_rec = Recommendation.query.filter_by(user_id=current_user.id).order_by(Recommendation.created_at.desc()).first()
    
    import json
    user_scores = []
    if latest_rec and latest_rec.scores_data:
        user_scores = json.loads(latest_rec.scores_data)
    
    # Map user scores [math, science, comm, creative, prog]
    # Career requirements (mock mapping)
    requirements = {
        'Engineering': [9, 8, 5, 4, 9],
        'Medicine': [7, 10, 8, 3, 4],
        'Arts': [3, 4, 7, 10, 5],
        'Commerce/Business': [7, 5, 10, 6, 4]
    }
    
    req = requirements.get(career.title, [5, 5, 5, 5, 5])
    labels = ["Math", "Science", "Communication", "Creativity", "Programming"]
    
    analysis = []
    for i in range(5):
        analysis.append({
            'label': labels[i],
            'user': user_scores[i] if len(user_scores) > i else 0,
            'required': req[i],
            'status': 'Strong' if (user_scores[i] if len(user_scores) > i else 0) >= req[i] else 'Needs Work'
        })
        
    return jsonify({
        'career': career.title,
        'analysis': analysis
    }), 200

@app.route('/api/resume/analyze', methods=['POST'])
@token_required
def analyze_resume(current_user):
    data = request.get_json()
    text = data.get('text', '')
    career_id = data.get('career_id')
    
    if not text or not career_id:
        return jsonify({'message': 'Text and career_id required'}), 400
        
    career = Career.query.get_or_404(career_id)
    
    try:
        # Use Gemini for deep analysis
        prompt = f"""
        Analyze this student resume text for a {career.title} role.
        Resume Text: {text}
        
        Compare it with these target keywords: {career.resume_keywords}
        
        Return a JSON with:
        1. match_percentage (0-100)
        2. found_skills (list)
        3. missing_skills (list)
        4. improvement_tips (list)
        """
        response = ai_model.generate_content(prompt)
        # Handle cases where response.text might not be clean JSON
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            analysis = json_match.group()
        else:
            analysis = "{}" # Fallback
            
        new_analysis = ResumeAnalysis(user_id=current_user.id, career_id=career.id, resume_text=text, analysis_json=analysis)
        db.session.add(new_analysis)
        current_user.xp += 30
        db.session.commit()
        
        import json
        return jsonify(json.loads(analysis)), 200
    except Exception as e:
        print(f"Resume Analysis Error: {e}")
        return jsonify({'message': 'AI analysis failed, please try again.'}), 500

@app.route('/api/jobs/<int:career_id>', methods=['GET'])
def get_jobs(career_id):
    jobs = JobSuggestion.query.filter_by(career_id=career_id).all()
    result = []
    for j in jobs:
        result.append({
            'title': j.title,
            'description': j.description,
            'type': j.job_type,
            'skills': j.skills
        })
    return jsonify(result), 200

# Parent Portal Endpoints
@app.route('/api/parent/link', methods=['POST'])
@token_required
def link_student(current_user):
    if current_user.role != 'parent':
        return jsonify({'message': 'Access denied'}), 403
        
    data = request.get_json()
    student_email = data.get('student_email')
    
    if not student_email:
        return jsonify({'message': 'Student email required'}), 400
        
    student = User.query.filter_by(email=student_email, role='student').first()
    if not student:
        return jsonify({'message': 'Student not found with this email'}), 404
        
    current_user.linked_student_email = student_email
    db.session.commit()
    
    return jsonify({'message': f'Linked to {student.name} successfully!'}), 200

@app.route('/api/parent/student-progress', methods=['GET'])
@token_required
def get_student_progress(current_user):
    if current_user.role != 'parent':
        return jsonify({'message': 'Access denied'}), 403
        
    if not current_user.linked_student_email:
        return jsonify({'message': 'No student linked'}), 400
        
    student = User.query.filter_by(email=current_user.linked_student_email, role='student').first()
    if not student:
        return jsonify({'message': 'Linked student not found'}), 404
        
    recs = Recommendation.query.filter_by(user_id=student.id).order_by(Recommendation.created_at.desc()).all()
    history = []
    for r in recs:
        history.append({
            'career_title': r.career.title,
            'confidence_score': r.confidence_score,
            'date': r.created_at.strftime('%Y-%m-%d')
        })
        
    return jsonify({
        'student': {
            'name': student.name,
            'email': student.email,
            'xp': student.xp,
            'grade': student.grade,
            'school': student.school
        },
        'recommendations': history
    }), 200

@app.route('/api/admin/stats', methods=['GET'])
@token_required
def admin_stats(current_user):
    if current_user.role != 'admin':
        return jsonify({'message': 'Access denied'}), 403
        
    total_users = User.query.count()
    total_posts = ForumPost.query.count()
    total_recs = Recommendation.query.count()
    top_careers = db.session.query(Career.title, db.func.count(Recommendation.id)).join(Recommendation).group_by(Career.title).all()
    
    users = User.query.order_by(User.created_at.desc()).limit(10).all()
    recent_users = [{'id': u.id, 'name': u.name, 'email': u.email, 'role': u.role, 'xp': u.xp} for u in users]

    return jsonify({
        'totals': {
            'users': total_users,
            'forum_posts': total_posts,
            'assessments': total_recs
        },
        'top_careers': [{'title': c[0], 'count': c[1]} for c in top_careers],
        'recent_users': recent_users
    }), 200

# Initialize DB and Seed Data
with app.app_context():
    db.create_all()
    # Seed careers if empty (using first() to avoid count() subquery issues in some SQLite versions)
    if not Career.query.first():
        careers = [
            Career(title="Engineering", description="Applying math and science to build technology.", 
                   skills_required="Problem Solving, Math, Programming", roadmap="B.Tech -> Internship -> Junior Engineer", colleges="MIT, Stanford, IITs", exams="JEE, SAT",
                   avg_salary_in="₹8L - ₹25L", avg_salary_gl="$80k - $160k", demand_level="High", growth_rate="+18% YoY",
                   resume_keywords="Python, Java, Data Structures, Algorithms, Problem Solving, Embedded Systems, Circuit Design",
                   suggested_projects="1. Build a full-stack web application. 2. Create a machine learning model. 3. Design a simple IoT device."),
            Career(title="Medicine", description="Diagnosing and treating diseases.", 
                   skills_required="Biology, Empathy, Patience", roadmap="Pre-Med -> Medical School -> Residency", colleges="Harvard Med, AIIMS", exams="NEET, MCAT",
                   avg_salary_in="₹12L - ₹40L", avg_salary_gl="$150k - $400k", demand_level="Critical", growth_rate="+10% YoY",
                   resume_keywords="Clinical Research, Anatomy, Public Health, Patient Care, Diagnostics, Data Analysis in Biology",
                   suggested_projects="1. Volunteer at local clinics. 2. Publish a high-school biology paper. 3. Shadow a doctor."),
            Career(title="Arts", description="Creative expression through visual, literary, or performing arts.", 
                   skills_required="Creativity, Communication, Design", roadmap="BFA -> Portfolio creation -> Freelance/Agency", colleges="NID, RISD", exams="NID DAT, UCEED",
                   avg_salary_in="₹5L - ₹15L", avg_salary_gl="$45k - $90k", demand_level="Medium", growth_rate="+8% YoY",
                   resume_keywords="Adobe Creative Suite, UI/UX Design, Figma, Typography, Content Creation, Digital Painting",
                   suggested_projects="1. Build an online art portfolio. 2. Design a brand identity package. 3. Write and illustrate a short comic."),
            Career(title="Commerce/Business", description="Managing finance, economics, and business operations.", 
                   skills_required="Leadership, Analytics, Finance", roadmap="BBA/B.Com -> MBA -> Management", colleges="Wharton, IIMs", exams="CAT, GMAT",
                   avg_salary_in="₹7L - ₹35L", avg_salary_gl="$70k - $180k", demand_level="High", growth_rate="+12% YoY",
                   resume_keywords="Financial Modeling, Excel, Marketing Analytics, Leadership, Strategic Management, Budgeting",
                   suggested_projects="1. Start a small online e-commerce shop. 2. Analyze a public company's stock patterns. 3. Create a marketing campaign for a local business.")
        ]
        db.session.add_all(careers)
        db.session.commit()
        
        # Seed Jobs
        for c in careers:
            jobs = [
                JobSuggestion(career_id=c.id, title=f"Junior {c.title} Intern", description="Entry level exposure", job_type="Internship", skills="Basics"),
                JobSuggestion(career_id=c.id, title=f"Associate {c.title} Strategist", description="High growth role", job_type="Job", skills="Advanced")
            ]
            db.session.add_all(jobs)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
