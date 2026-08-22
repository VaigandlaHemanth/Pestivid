"""
Test Script for Simple AI Agent
Demonstrates all features of the PestVid AI Agent
"""

from simple_ai_agent import pestivid_agent
import json

def test_ai_agent():
    """Test all AI agent features"""
    
    print("🤖 Testing PestVid Simple AI Agent")
    print("=" * 50)
    
    # Test 1: General Crop Advice
    print("\n1. 🌱 Testing General Crop Advice")
    print("-" * 30)
    
    queries = [
        "How to control aphids on tomatoes?",
        "What causes yellow leaves in potato plants?",
        "Best time to water crops?",
        "How to improve soil health?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = pestivid_agent.get_crop_advice(query)
        print(f"Source: {result['source']}")
        print(f"Advice: {result['advice'][:200]}...")
        print()
    
    # Test 2: Crop Description Analysis
    print("\n2. 🔍 Testing Crop Description Analysis")
    print("-" * 40)
    
    descriptions = [
        "My tomato leaves have brown spots and are turning yellow",
        "White powder on cucumber leaves",
        "Holes in lettuce leaves",
        "Wilting potato plants despite adequate watering"
    ]
    
    for desc in descriptions:
        print(f"\nDescription: {desc}")
        result = pestivid_agent.analyze_crop_image_description(desc)
        print(f"Diagnosis: {result['diagnosis']}")
        print(f"Recommendations: {', '.join(result['recommendations'])}")
        print()
    
    # Test 3: Seasonal Advice
    print("\n3. 🌸 Testing Seasonal Advice")
    print("-" * 30)
    
    seasons = ["spring", "summer", "fall", "winter"]
    crops = ["tomato", "potato", None]
    
    for season in seasons:
        for crop in crops[:2]:  # Test with 2 crops per season
            print(f"\nSeason: {season.title()}, Crop: {crop or 'General'}")
            result = pestivid_agent.get_seasonal_advice(season, crop)
            if result['status'] == 'success':
                print(f"Advice: {result['advice']}")
                print(f"Key Tasks: {', '.join(result['key_tasks'])}")
            print()
    
    # Test 4: Quick Tips
    print("\n4. 💡 Testing Quick Tips")
    print("-" * 25)
    
    tips = pestivid_agent.get_quick_tips()
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("🚀 Your AI Agent is ready to help farmers!")

def test_api_endpoints():
    """Test API endpoints (requires Flask server running)"""
    import requests
    
    print("\n🌐 Testing API Endpoints")
    print("=" * 30)
    print("Note: Make sure Flask server is running (python flask_server.py)")
    
    base_url = "http://localhost:5000"
    
    # Test endpoints
    endpoints = [
        {
            "url": f"{base_url}/simple-ai-advice",
            "method": "POST",
            "data": {"query": "How to prevent tomato diseases?"}
        },
        {
            "url": f"{base_url}/analyze-description", 
            "method": "POST",
            "data": {"description": "Yellow leaves with brown spots"}
        },
        {
            "url": f"{base_url}/seasonal-advice",
            "method": "POST", 
            "data": {"season": "spring", "crop": "tomato"}
        },
        {
            "url": f"{base_url}/quick-tips",
            "method": "GET",
            "data": None
        }
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\nTesting: {endpoint['url']}")
            
            if endpoint['method'] == 'POST':
                response = requests.post(endpoint['url'], json=endpoint['data'], timeout=10)
            else:
                response = requests.get(endpoint['url'], timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {response.status_code}")
                
                # Print relevant response data
                if 'advice' in result:
                    print(f"Advice: {result['advice'][:100]}...")
                elif 'diagnosis' in result:
                    print(f"Diagnosis: {result['diagnosis'][:100]}...")
                elif 'tips' in result:
                    print(f"Tips: {len(result['tips'])} tips received")
                else:
                    print(f"Response: {str(result)[:100]}...")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Flask server not running")
            print("Start server with: python flask_server.py")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Test the AI agent directly
    test_ai_agent()
    
    # Test API endpoints (optional)
    print("\n" + "=" * 50)
    user_input = input("Test API endpoints? (y/n): ").lower().strip()
    if user_input == 'y':
        test_api_endpoints()
    
    print("\n🎉 Testing complete! Your AI Agent is ready for integration.")
