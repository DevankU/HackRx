#!/usr/bin/env python3
"""
Test script for the Ultra-Fast RAG System API
Usage: python test_api.py <base_url>
Example: python test_api.py https://your-app.railway.app
"""

import requests
import json
import time
import sys

def test_api(base_url):
    """Test the deployed API endpoints"""
    
    print(f"🧪 Testing API at: {base_url}")
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Sample RAG Query
    print("\n2️⃣ Testing RAG Endpoint...")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer dbbdb701cfc45d4041e22a03edbfc65753fe9d7b4b9ba1df4884e864f3bb934d"
    }
    
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    try:
        print("   Sending request... (this may take 15-30 seconds)")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/hackrx/run", 
            json=test_payload,
            headers=headers,
            timeout=120
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ RAG query successful! ({response_time:.2f} seconds)")
            print(f"   Questions: {len(test_payload['questions'])}")
            print(f"   Answers: {len(result['answers'])}")
            print("\n   Sample answers:")
            for i, answer in enumerate(result['answers'][:2]):
                print(f"   Q{i+1}: {answer[:100]}...")
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return False
    
    # Test 3: Metrics (optional)
    print("\n3️⃣ Testing Metrics Endpoint...")
    try:
        response = requests.get(
            f"{base_url}/metrics", 
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Metrics endpoint working!")
            metrics = response.json()
            print(f"   Total queries: {metrics.get('total_queries_24h', 0)}")
        else:
            print(f"⚠️ Metrics endpoint issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Metrics endpoint error: {e}")
    
    print(f"\n🎉 API testing complete! System is ready for hackathon use.")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <base_url>")
        print("Example: python test_api.py https://your-app.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print("🚀 Ultra-Fast RAG System API Tester")
    print("=" * 50)
    
    success = test_api(base_url)
    
    if success:
        print("\n✅ All tests passed! Your API is ready for the hackathon! 🏆")
        print(f"\n📋 API Usage Summary:")
        print(f"   Endpoint: POST {base_url}/hackrx/run")
        print(f"   Auth: Bearer dbbdb701cfc45d4041e22a03edbfc65753fe9d7b4b9ba1df4884e864f3bb934d")
        print(f"   Health: GET {base_url}/health")
    else:
        print("\n❌ Some tests failed. Please check your deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
