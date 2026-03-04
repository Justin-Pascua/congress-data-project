import httpx 
import dotenv
import os
import sys

dotenv.load_dotenv()

def get_remaining():
    response = httpx.get(
        'https://api.congress.gov/v3/congress/current', 
        params = {'api_key': os.getenv('API_KEY')})

    return response.headers['x-ratelimit-remaining']

args_list = sys.argv[1:]

if __name__ == '__main__':
    if args_list[0] == 'remaining':
        print(get_remaining())