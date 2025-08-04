import requests
import json
import time
import asyncio
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemTester:
    def __init__(self, base_url: str = "http://localhost:8000", bearer_token: str = None):
        self.base_url = base_url
        self.bearer_token = bearer_token or "dbbdb701cfc45d4041e22a03edbfc65753fe9d7b4b9ba1df4884e864f3bb934d"
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        self.executor = ThreadPoolExecutor(max_workers=3)

    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: Status {response.status_code}")
                logger.error(f"Health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            logger.error(f"Health check error: {str(e)}")
            return False

    def test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Metrics endpoint passed: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"❌ Metrics endpoint failed: Status {response.status_code}")
                logger.error(f"Metrics endpoint failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics endpoint error: {str(e)}")
            logger.error(f"Metrics endpoint error: {str(e)}")
            return False

    def test_sample_query(self) -> bool:
        """Test with the provided sample data"""
        sample_data = {
            "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?",
                "What is the No Claim Discount (NCD) offered in this policy?",
                "Is there a benefit for preventive health check-ups?",
                "How does the policy define a 'Hospital'?",
                "What is the extent of coverage for AYUSH treatments?",
                "Are there any sub-limits on room rent and ICU charges for Plan A?"
            ]
        }

        try:
            print("🔄 Testing sample query...")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/hackrx/run",
                headers=self.headers,
                json=sample_data,
                timeout=120
            )

            end_time = time.time()
            latency = end_time - start_time

            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])

                print(f"✅ Sample query successful (Latency: {latency:.2f}s)")
                print(f"📊 Received {len(answers)} answers")

                # Print all answers for validation
                for i, (question, answer) in enumerate(zip(sample_data['questions'], answers)):
                    print(f"Q{i+1}: {question}")
                    print(f"A{i+1}: {answer[:200]}..." if len(answer) > 200 else f"A{i+1}: {answer}")
                    print("-" * 50)

                # Validate that we received answers for all questions
                if len(answers) == len(sample_data['questions']):
                    print("✅ All questions answered")
                    return True
                else:
                    print(f"❌ Incomplete response: Expected {len(sample_data['questions'])} answers, got {len(answers)}")
                    logger.warning(f"Incomplete response: Expected {len(sample_data['questions'])} answers, got {len(answers)}")
                    return False
            else:
                print(f"❌ Sample query failed: Status {response.status_code}")
                print(f"Response: {response.text}")
                logger.error(f"Sample query failed: Status {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Sample query error: {str(e)}")
            logger.error(f"Sample query error: {str(e)}")
            return False

    async def test_concurrent_queries(self, num_requests: int = 3) -> bool:
        """Test system under concurrent load"""
        async def make_request():
            try:
                response = requests.post(
                    f"{self.base_url}/hackrx/run",
                    headers=self.headers,
                    json={
                        "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
                        "questions": ["What is the grace period for premium payment?"]
                    },
                    timeout=60
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Concurrent query error: {str(e)}")
                return False

        print(f"🔄 Testing {num_requests} concurrent queries...")
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        print(f"✅ Concurrent test completed: {success_count}/{num_requests} successful")

        return success_count == num_requests

    def test_invalid_token(self) -> bool:
        """Test authentication with invalid token"""
        try:
            invalid_headers = {
                "Authorization": "Bearer invalid_token",
                "Content-Type": "application/json"
            }
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                headers=invalid_headers,
                json={
                    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
                    "questions": ["Test question"]
                },
                timeout=10
            )
            if response.status_code == 401:
                print("✅ Invalid token test passed: Correctly rejected")
                return True
            else:
                print(f"❌ Invalid token test failed: Expected 401, got {response.status_code}")
                logger.warning(f"Invalid token test failed: Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid token test error: {str(e)}")
            logger.error(f"Invalid token test error: {str(e)}")
            return False

    def test_invalid_url(self) -> bool:
        """Test with invalid document URL"""
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                headers=self.headers,
                json={
                    "documents": "https://invalid-url-that-does-not-exist.com/fake.pdf",  # Actually invalid URL
                    "questions": ["Test question"]
                },
                timeout=30
            )
            # Accept either 400 or 500 as valid error responses for invalid URLs
            if response.status_code in [400, 500]:
                print("✅ Invalid URL test passed: Correctly handled")
                return True
            else:
                print(f"❌ Invalid URL test failed: Expected 400/500, got {response.status_code}")
                logger.warning(f"Invalid URL test failed: Expected 400/500, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid URL test error: {str(e)}")
            logger.error(f"Invalid URL test error: {str(e)}")
            return False

    async def run_all_tests(self):
        """Run all test cases"""
        print("🚀 Starting RAG System Tests")
        print("=" * 50)

        results = {
            "health_check": self.test_health_check(),
            "metrics_endpoint": self.test_metrics_endpoint(),
            "sample_query": self.test_sample_query(),
            "concurrent_queries": await self.test_concurrent_queries(),
            "invalid_token": self.test_invalid_token(),
            "invalid_url": self.test_invalid_url()
        }

        print("\n📊 Test Summary")
        print("=" * 50)
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")

        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        return passed == total

def main():
    tester = RAGSystemTester()
    asyncio.run(tester.run_all_tests())

if __name__ == "__main__":
    main()