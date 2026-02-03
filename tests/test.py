import base64
import requests

with open("tests/audio.mp3", "rb") as f:
    audio = f.read()
audio_base64 = base64.b64encode(audio).decode('utf-8')

response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "Content-Type": "application/json",
        "x-api-key": "sk_test_123456789"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)
print(audio_base64)
print(response.json())