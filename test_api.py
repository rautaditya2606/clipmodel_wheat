import requests
import os
from dotenv import load_dotenv

load_dotenv()

# URLs for testing
TEST_IMAGES = {
    "wheat_field": "https://img.freepik.com/free-photo/golden-wheat-field_158595-3341.jpg?semt=ais_incoming&w=740&q=80",
    "crop_close_up": "https://images.picxy.com/cache/2020/9/16/696fab6720fa12a66519dc45a0ac54f3.jpg",
    "city_street": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=800&q=80",
    "cat_photo": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&w=800&q=80",
    "wheat_close_up": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRt_4zMMTKOSWDROkVkfBdm_2U2g4VKOh1_Gg&s",
}

# Deployed URL: https://wheat-detection-es09.onrender.com/
API_URL = "https://wheat-detection-es09.onrender.com/verify-crop/"

def download_and_upload(name, url):
    print(f"\n--- Testing: {name} ---")
    print(f"URL: {url}")
    
    # Download the image temporarily
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download image from {url}")
        return

    # Upload to our FastAPI server
    files = {"file": (f"{name}.jpg", response.content, "image/jpeg")}
    try:
        api_response = requests.post(API_URL, files=files)
        print(f"Status Code: {api_response.status_code}")
        print(f"Response: {api_response.json()}")
    except Exception as e:
        print(f"Error connecting to API: {e}. Is the server running?")

if __name__ == "__main__":
    for name, url in TEST_IMAGES.items():
        download_and_upload(name, url)
