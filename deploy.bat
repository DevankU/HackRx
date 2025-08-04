@echo off
echo 🚀 Deploying Ultra-Fast RAG System to Railway...

REM Check if Railway CLI is installed
railway --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Railway CLI not found. Please install from: https://railway.app/cli
    echo Or run: npm install -g @railway/cli
    pause
    exit /b 1
)

REM Login to Railway (if not already logged in)
echo 🔐 Checking Railway authentication...
railway whoami
if %errorlevel% neq 0 (
    railway login
)

REM Initialize project (if not already initialized)
if not exist "railway.toml" (
    echo 📦 Initializing Railway project...
    railway init
)

REM Set environment variables reminder
echo 🔧 Environment Variables Setup Required:
echo Please set these in Railway dashboard after deployment:
echo - PINECONE_API_KEY=your_pinecone_api_key
echo - GROQ_API_KEY=your_groq_api_key
echo - LANGSMITH_API_KEY=your_langsmith_api_key
echo.

REM Deploy
echo 🚀 Deploying to Railway...
railway up

echo ✅ Deployment complete!
echo 🌐 Your API will be available at the Railway-provided URL
echo 📊 Test with: GET https://your-app.railway.app/health
pause
