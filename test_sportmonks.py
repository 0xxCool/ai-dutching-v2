import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_token = os.getenv("SPORTMONKS_API_TOKEN")
url = f"https://api.sportmonks.com/v3/football/fixtures?api_token={api_token}&filters=leagueIds:8&include=participants"

response = requests.get(url)
print(f"Status: {response.status_code}")
print(f"Fixtures gefunden: {len(response.json().get('data', []))}")

if response.status_code == 200:
    print("✅ API funktioniert!")
else:
    print(f"❌ Fehler: {response.text}")