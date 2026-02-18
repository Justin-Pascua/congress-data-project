import httpx
from ..config import settings

API_KEY = settings.API_KEY.get_secret_value()
BASE_URL = 'https://api.congress.gov/v3'

base_params = {
    'api_key': API_KEY,
    'format': 'json',
    'limit': 250
}

client = httpx.AsyncClient(base_url = BASE_URL,
                           params = base_params, 
                           follow_redirects = True, 
                           timeout = 600)

