# How to Run AetherFlix AI 🚀

> **Complete step-by-step terminal guide** to set up and run the AetherFlix AI project from scratch.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.12+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **Git** ([Download](https://git-scm.com/))
- **Supabase Account** ([Sign up](https://supabase.com/))

---

## ⚡ Quick Setup (3 Steps)

### **Step 1: Clone the Repository**

```powershell
# Open PowerShell terminal

# Navigate to your projects folder
cd D:\Projects

# Clone the repository
git clone <your-repo-url>

# Navigate into project directory
cd "AetherFlix AI"
```

---

### **Step 2: Backend Setup & ML Models Training**

```powershell
# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python --version
pip list
```

#### **Train ML Models (First Time Only)**

```powershell
# Navigate to ML pipeline directory
cd ..\ml_pipeline

# Install ML dependencies
pip install -r requirements.txt

# Train all ML models (takes 2-3 minutes)
python train_ml_models.py
```

**Output**:
```
📊 Loading Netflix dataset...
✅ Loaded 8807 titles
🤖 Training Type Classification Model...
✅ Logistic Regression Accuracy: 0.972
✅ Linear SVC Accuracy: 0.968
🎯 Building Recommendation Engine...
💾 Saving models...
✨ All models trained and saved successfully!
```

#### **Start Backend Server**

```powershell
# Navigate back to backend directory
cd ..\backend

# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Start FastAPI server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output**:
```
🚀 Starting AetherFlix AI Backend...
✅ ML models loaded successfully
✅ Supabase connection verified
✨ AetherFlix AI Backend is ready!

INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Verify Backend**:
- Open browser: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

---

### **Step 3: Frontend Setup**

**Open a NEW PowerShell terminal** (keep backend running)

```powershell
# Navigate to project directory
cd "D:\Projects\AetherFlix AI"

# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

**Expected Output**:
```
VITE v5.0.11  ready in 301 ms

➜  Local:   http://localhost:5173/
➜  Network: http://192.168.x.x:5173/
➜  press h + enter to show help
```

**Access Application**:
- Open browser: http://localhost:5173
- Login with demo user:
  - Email: `demo1@aetherflix.com`
  - Password: `demo123`

---

## 🔧 Detailed Setup Instructions

### **1️⃣ Backend Setup (Detailed)**

```powershell
# Step 1: Navigate to backend
cd "D:\Projects\AetherFlix AI\backend"

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate virtual environment
.\venv\Scripts\Activate.ps1

# Step 4: Upgrade pip
python -m pip install --upgrade pip

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Create logs directory (if not exists)
New-Item -ItemType Directory -Force -Path logs

# Step 7: Verify Supabase credentials (optional)
# Edit .env file with your Supabase URL and API key

# Step 8: Start server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### **2️⃣ ML Training Pipeline (Detailed)**

```powershell
# Step 1: Navigate to ML pipeline
cd "D:\Projects\AetherFlix AI\ml_pipeline"

# Step 2: Ensure backend venv is activated
cd ..\backend
.\venv\Scripts\Activate.ps1
cd ..\ml_pipeline

# Step 3: Install ML dependencies
pip install -r requirements.txt

# Step 4: Verify dataset exists
# Check if backend/netflix_titles.csv exists

# Step 5: Run training script
python train_ml_models.py

# Step 6: Verify model files created
# Check backend/models/trained/ directory
cd ..\backend\models\trained
dir

# Expected files:
# - logreg_classifier.pkl
# - svc_classifier.pkl
# - tfidf_vectorizer.pkl
# - tfidf_recommender.pkl
# - cosine_similarity.pkl
```

---

### **3️⃣ Frontend Setup (Detailed)**

```powershell
# Step 1: Navigate to frontend
cd "D:\Projects\AetherFlix AI\frontend"

# Step 2: Install dependencies
npm install

# Step 3: Verify Supabase configuration
# Check src/lib/supabase.js for correct credentials

# Step 4: Start development server
npm run dev

# Step 5: Build for production (optional)
npm run build

# Step 6: Preview production build (optional)
npm run preview
```

---

## 🛠 Terminal Commands Reference

### **Backend Commands**

```powershell
# Activate virtual environment
cd "D:\Projects\AetherFlix AI\backend"
.\venv\Scripts\Activate.ps1

# Install new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt

# Run server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run server on different port
python -m uvicorn main:app --reload --port 8080

# Deactivate virtual environment
deactivate
```

### **Frontend Commands**

```powershell
# Navigate to frontend
cd "D:\Projects\AetherFlix AI\frontend"

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Install new package
npm install package-name

# Update package
npm update package-name
```

### **ML Pipeline Commands**

```powershell
# Train all models
cd "D:\Projects\AetherFlix AI\ml_pipeline"
python train_ml_models.py

# Train only classification models
# (Edit train_ml_models.py to comment out recommendation section)

# Check model performance
# View charts in frontend/public/ml_charts/
```

---

## 🔍 Verification & Testing

### **1. Backend Health Check**

```powershell
# Use curl (if installed)
curl http://localhost:8000/api/health

# Or use PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/api/health" | Select-Object -Expand Content
```

### **2. Test Classification API**

```powershell
# Test classification endpoint
$body = @{
    title = "Stranger Things"
    description = "A group of kids face supernatural forces in a small town"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/classify" -Method POST -Body $body -ContentType "application/json" | Select-Object -Expand Content
```

### **3. Test Recommendation API**

```powershell
# Test recommendation endpoint
$body = @{
    title = "Stranger Things"
    top_n = 5
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/recommend" -Method POST -Body $body -ContentType "application/json" | Select-Object -Expand Content
```

---

## 🐛 Troubleshooting

### **Issue: Port Already in Use**

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
python -m uvicorn main:app --reload --port 8080
```

### **Issue: Virtual Environment Not Activating**

```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\venv\Scripts\Activate.ps1
```

### **Issue: Module Not Found**

```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### **Issue: Frontend Not Loading**

```powershell
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install

# Restart dev server
npm run dev
```

### **Issue: ML Models Not Found**

```powershell
# Retrain models
cd ml_pipeline
python train_ml_models.py

# Verify model files exist
cd ..\backend\models\trained
dir
```

---

## 📦 Project URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | React application |
| **Backend** | http://localhost:8000 | FastAPI server |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |
| **Health Check** | http://localhost:8000/api/health | Server status |

---

## 🔐 Demo Users

Use these credentials to test the application:

| Email | Password | Description |
|-------|----------|-------------|
| demo1@aetherflix.com | demo123 | Demo User 1 |
| demo2@aetherflix.com | demo123 | Demo User 2 |
| demo3@aetherflix.com | demo123 | Demo User 3 |

---

## 📝 Notes

1. **Always activate virtual environment** before running backend commands
2. **Keep both terminals running** (backend + frontend) while developing
3. **ML models must be trained first** before starting backend server
4. **Supabase credentials required** for authentication to work
5. **Check logs** in `backend/logs/` for debugging

---

## 🎉 Success Checklist

- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:5173
- [ ] ML models trained and saved in `backend/models/trained/`
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] Can login with demo user credentials
- [ ] Can browse content and view recommendations

---

## 💡 Quick Tips

```powershell
# Open both terminals side-by-side in Windows Terminal
# Terminal 1: Backend
cd "D:\Projects\AetherFlix AI\backend" ; .\venv\Scripts\Activate.ps1 ; python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd "D:\Projects\AetherFlix AI\frontend" ; npm run dev
```

---

**Happy Coding! 🚀**

For issues or questions, contact: **j06haniel@gmail.com**
