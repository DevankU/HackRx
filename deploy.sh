#!/bin/bash

# Deploy to Railway Script
echo "🚀 Deploying Ultra-Fast RAG System to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    curl -fsSL https://railway.app/install.sh | sh
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway whoami || railway login

# Initialize project (if not already initialized)
if [ ! -f "railway.toml" ]; then
    echo "📦 Initializing Railway project..."
    railway init
fi

# Set environment variables
echo "🔧 Setting environment variables..."
echo "Please set these environment variables in Railway dashboard:"
echo "PINECONE_API_KEY=your_pinecone_api_key"
echo "GROQ_API_KEY=your_groq_api_key"  
echo "LANGSMITH_API_KEY=your_langsmith_api_key"

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your API will be available at the Railway-provided URL"
echo "📊 Test with: GET https://your-app.railway.app/health"
