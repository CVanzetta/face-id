#!/usr/bin/env python3
"""
API Testing Script for Face Recognition System
"""
import requests
import json
import time
from pathlib import Path
import base64

class FaceRecognitionAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health(self):
        """Test health endpoint"""
        print("🩺 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
            print(f"   Database connected: {health['database_connected']}")
            print(f"   Total persons: {health['total_persons']}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_enroll(self, name="Test Person", image_path=None):
        """Test enrollment endpoint"""
        print(f"👤 Testing enrollment for {name}...")
        
        if not image_path or not Path(image_path).exists():
            print("⚠️ No test image provided, skipping enrollment test")
            return None
            
        try:
            with open(image_path, 'rb') as f:
                files = {'files': f}
                data = {
                    'name': name,
                    'source': 'api_test',
                    'consent_type': 'explicit',
                    'tags': 'test,api',
                    'notes': 'Test enrollment via API'
                }
                
                response = self.session.post(
                    f"{self.base_url}/enroll",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                
                if result['success']:
                    print(f"✅ Enrolled {result['faces_enrolled']} faces")
                    print(f"   Person ID: {result['person_id']}")
                    return result['person_id']
                else:
                    print(f"❌ Enrollment failed: {result['message']}")
                    return None
                    
        except Exception as e:
            print(f"❌ Enrollment test failed: {e}")
            return None
    
    def test_search(self, image_path=None):
        """Test search endpoint"""
        print("🔍 Testing search endpoint...")
        
        if not image_path or not Path(image_path).exists():
            print("⚠️ No test image provided, skipping search test")
            return
            
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'top_k': 5,
                    'threshold': 0.3
                }
                
                response = self.session.post(
                    f"{self.base_url}/search",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"✅ Search completed")
                print(f"   Faces detected: {result['faces_detected']}")
                print(f"   Matches found: {len(result['matches'])}")
                
                for i, match in enumerate(result['matches'][:3]):
                    similarity = int(match['similarity_score'] * 100)
                    print(f"   Match {i+1}: {match['name']} ({similarity}%)")
                    
        except Exception as e:
            print(f"❌ Search test failed: {e}")
    
    def test_verify(self, image1_path=None, image2_path=None):
        """Test verification endpoint"""
        print("🔄 Testing verification endpoint...")
        
        if not image1_path or not image2_path:
            print("⚠️ Two test images required, skipping verification test")
            return
            
        if not Path(image1_path).exists() or not Path(image2_path).exists():
            print("⚠️ Test images not found, skipping verification test")
            return
            
        try:
            with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
                files = {
                    'file1': f1,
                    'file2': f2
                }
                
                response = self.session.post(
                    f"{self.base_url}/verify",
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                
                similarity = int(result['similarity_score'] * 100)
                match_status = "MATCH" if result['is_match'] else "NO MATCH"
                
                print(f"✅ Verification completed")
                print(f"   Result: {match_status}")
                print(f"   Similarity: {similarity}%")
                print(f"   Confidence: {int(result['confidence'] * 100)}%")
                
        except Exception as e:
            print(f"❌ Verification test failed: {e}")
    
    def test_list_persons(self):
        """Test persons listing"""
        print("📋 Testing persons listing...")
        
        try:
            response = self.session.get(f"{self.base_url}/persons")
            response.raise_for_status()
            persons = response.json()
            
            print(f"✅ Found {len(persons)} persons in database")
            for person in persons[:5]:  # Show first 5
                print(f"   {person['name']} ({person['faces_count']} faces)")
                
            return persons
            
        except Exception as e:
            print(f"❌ Persons listing failed: {e}")
            return []
    
    def test_delete_person(self, person_id):
        """Test person deletion"""
        print(f"🗑️ Testing deletion of person {person_id}...")
        
        try:
            response = self.session.delete(f"{self.base_url}/person/{person_id}")
            response.raise_for_status()
            result = response.json()
            
            if result['success']:
                print(f"✅ Successfully deleted person")
                print(f"   Embeddings deleted: {result['embeddings_deleted']}")
            else:
                print(f"❌ Deletion failed: {result['message']}")
                
        except Exception as e:
            print(f"❌ Deletion test failed: {e}")
    
    def test_stats(self):
        """Test statistics endpoint"""
        print("📊 Testing statistics endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            stats = response.json()
            
            print("✅ Statistics retrieved:")
            print(f"   Total persons: {stats['database'].get('total_persons', 0)}")
            print(f"   Total embeddings: {stats['database'].get('total_embeddings', 0)}")
            print(f"   Uptime: {int(stats['uptime_seconds'])}s")
            
        except Exception as e:
            print(f"❌ Statistics test failed: {e}")

def main():
    print("🧪 Face Recognition API Test Suite")
    print("=" * 50)
    
    tester = FaceRecognitionAPITester()
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    for i in range(30):
        if tester.test_health():
            break
        time.sleep(1)
    else:
        print("❌ API not responding after 30 seconds")
        return
    
    print("\n" + "=" * 50)
    
    # Test endpoints
    tester.test_stats()
    print()
    
    tester.test_list_persons()
    print()
    
    # These tests require actual images
    # You can provide paths to test images
    test_image = "path/to/test/image.jpg"  # Update this path
    
    if Path(test_image).exists():
        # Test enrollment
        person_id = tester.test_enroll("API Test Person", test_image)
        print()
        
        # Test search
        tester.test_search(test_image)
        print()
        
        # Test verification (using same image twice for simplicity)
        tester.test_verify(test_image, test_image)
        print()
        
        # Clean up - delete test person
        if person_id:
            tester.test_delete_person(person_id)
    else:
        print(f"⚠️ Test image not found at {test_image}")
        print("   Update the test_image variable with a valid image path")
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")

if __name__ == "__main__":
    main()
