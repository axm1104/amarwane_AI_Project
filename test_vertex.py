
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Setup Env
os.environ["GOOGLE_CLOUD_PROJECT"] = "tranquil-well-478523-b3"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-west1"
service_key_path = "tranquil-well-478523-b3-8c0fee1723fc.json"
if os.path.exists(service_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(service_key_path)

print(f"Project: {os.environ['GOOGLE_CLOUD_PROJECT']}")
print(f"Location: {os.environ['GOOGLE_CLOUD_LOCATION']}")

try:
    vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
    
    print("Trying gemini-1.5-flash-001...")
    model = GenerativeModel("gemini-1.5-flash-001")
    response = model.generate_content("Hello, can you hear me?")
    print(f"Success! Response: {response.text}")

except Exception as e:
    print(f"Error with gemini-1.5-flash-001: {e}")

    print("\nTrying gemini-1.5-flash-002...")
    try:
        model = GenerativeModel("gemini-1.5-flash-002")
        response = model.generate_content("Hello")
        print(f"Success with 002! Response: {response.text}")
    except Exception as e2:
        print(f"Error with gemini-1.5-flash-002: {e2}")

    print("\nTrying gemini-1.0-pro...")
    try:
        model = GenerativeModel("gemini-1.0-pro")
        response = model.generate_content("Hello")
        print(f"Success with 1.0! Response: {response.text}")
    except Exception as e3:
        print(f"Error with gemini-1.0-pro: {e3}")
