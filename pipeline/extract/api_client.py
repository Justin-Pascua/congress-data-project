import httpx
import asyncio
from typing import Literal, Optional, List

from ..exceptions import *


BASE_URL = 'https://api.congress.gov/v3'
RATE_THRESHOLD = 100
NUM_RETRIES = 5

class CongressAPIClient:
    def __init__(self, api_key: str):
        base_params = {
            'api_key': api_key,
            'format': 'json',
            'limit': 250
        }
        
        self.client = httpx.AsyncClient(
            base_url = BASE_URL,
            params = base_params, 
            follow_redirects = True, 
            timeout = 30,
            limits = httpx.Limits(max_connections = 10,
                                  max_keepalive_connections = 5))
        
        response = httpx.get(BASE_URL + '/congress/current', params = base_params)
        self.remaining_calls = int(response.headers['x-ratelimit-remaining'])

    def _check_exceptions(self, response: httpx.Response) -> None:
        """
        Checks response for common 5xx or 4xx status codes, and raises exception if found.
        Args:
            response: An `httpx.Response` instance representing the Congress API's response to some call.
        """
        status_code = response.status_code
        if status_code >= 500:
            raise RuntimeError("Unexpected server error from Congress API")
        elif status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif status_code == 403:
            raise AuthorizationError("No API key, or invalid API key sent in request.")
        
    def _update_call_counter(self, response: httpx.Response) -> None:
        """
        Updates the internal counter for number of remaining calls using info in header of response sent by API
        Args:
            response: An `httpx.Response` instance representing the Congress API's response to some call.
        """
        self.remaining_calls = int(response.headers['x-ratelimit-remaining'])

    def _check_rate_limit(self) -> None:
        """
        Checks if number of remaining API calls is above `RATE_THRESHOLD`. 
        Raises `RateLimitError` if threshold has been met.
        """
        if self.remaining_calls <= RATE_THRESHOLD:
            raise RateLimitError("API rate limit threshold met")

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Wrapper for API calls. Calls `self._check_rate_limit` to remain within API rate limit,
        performs retries with exponential backoff, and calls `self._update_call_counter` to update internal counter.
        Args:
            method: A string specifying the HTTP method, usually "get"
            url: The url of the endpoint to be called
        """
        # exponential backoff retry
        base_delay = 0.5

        for attempt in range(NUM_RETRIES):
            self._check_rate_limit()
            try:
                response = await self.client.request(method, url, **kwargs)
                self._check_exceptions(response)
                self._update_call_counter(response)
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, RuntimeError) as e:
                if attempt == NUM_RETRIES - 1:
                    raise 

                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    async def get_current_congress(self) -> dict:
        """
        Returns the most recent congress and session number.
        """
        response = await self._request_with_retry('get', '/congress/current')
        data = response.json()

        congress_num = data['congress']['number']
        session_num = max([chamber['number'] for chamber in data['congress']['sessions']])

        return {'congress_num': congress_num,
                'session_num': session_num}
    
    async def get_all_members(self, congress_num: int) -> list:
        """
        Returns all the congress members from a specified congress.
        Args:
            congress_num: the number of the congress (e.g. 119)
        """
        members = []

        i = 0
        while True:
            response = await self._request_with_retry(
                'get', 
                f'/member/congress/{congress_num}', 
                params = {'offset': 250*i})
            data = response.json()
            if len(data['members']) == 0:
                break
            members.extend(data['members'])
            i += 1

        return members
    
    async def get_all_bills(self, congress_num: int, bill_type: str) -> list:
        """
        Returns a list of all bills in a specified congress of a specified type.
        Args:
            congress_num: the number of the congress (e.g. 119)
            bill_type: the type of bill. Acceptable values are "hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", or "sres"
        """
        bills = []

        i = 0
        while True:
            response = await self._request_with_retry('get', f'/bill/{congress_num}/{bill_type}',params = {'offset': 250*i})
            data = response.json()
                
            if len(data['bills']) == 0:
                break
            bills.extend(data['bills'])
            i += 1

        return bills
    
    async def get_member_history(self, bioguideId: str) -> dict:
        """
        Returns the bills sponsored and cosponsored by a given representative (across their congressional career)
        Args:
            bioguideId: The bioguide identifier for the congressional member (e.g. L000174)
        """
        sponsor_history = []
        i = 0
        while True:    
            response = await self._request_with_retry('get', f'/member/{bioguideId}/sponsored-legislation', params = {'offset': 250*i})
            data = response.json()

            if len(data['sponsoredLegislation']) == 0:
                break
            sponsor_history.extend(data['sponsoredLegislation'])
            i += 1

        cosponsor_history = []
        i = 0
        while True:    
            response = await self._request_with_retry('get', f'/member/{bioguideId}/cosponsored-legislation', params = {'offset': 250*i})
            data = response.json()
            if len(data['cosponsoredLegislation']) == 0:
                break
            cosponsor_history.extend(data['cosponsoredLegislation'])
            i += 1

        return {'sponsor_history': sponsor_history,
                'cosponsor_history': cosponsor_history}

    async def get_bill_summary(self, congress_num: int, bill_type: str, bill_num: int) -> dict:
        """
        Returns a summary of a specified bill. 
        This method first tries to find a summary from the "/bill/{congress_num}/{bill_type}/{bill_num}/summaries" endpoint. 
        If not found, then this method then tries to find a summary from the "/bill/{congress_num}/{bill_type}/{bill_num}/text" endpoint.
        If this fails, then the summary is left as `None`.
        Args:
            congress_num: the number of the congress (e.g. 119)
            bill_type: the type of bill. Acceptable values are "hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", or "sres"
            bill_num: the bill's assigned number (e.g. 3076)
        """
        result = {'summary': None}
        summary_response = await self._request_with_retry('get', f'/bill/{congress_num}/{bill_type}/{bill_num}/summaries')
        data = summary_response.json()

        try:
            summary = data['summaries'][-1]['text']
            result['summary'] = summary
        except:
            text_response = await self._request_with_retry('get', f'/bill/{congress_num}/{bill_type}/{bill_num}/text')
            text_data = text_response.json()
            
            formats = text_data['textVersions'][-1]['formats']
            for format in formats:
                if format['type'] == 'Formatted Text':
                    html_content = await self.client.get(format['url'])
                    result['summary'] = html_content.text
                    break
                
        return result
    
    async def get_bill_info(self, congress_num: int, bill_type: str, bill_num: int) -> dict:
        """
        Returns information on a specified bill, including its title, type, chamber of origin, sponsor, and policy area
        Args:
            congress_num: the number of the congress (e.g. 119)
            bill_type: the type of bill. Acceptable values are "hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", or "sres"
            bill_num: the bill's assigned number (e.g. 3076)
        """
        response = await self._request_with_retry('get', f'/bill/{congress_num}/{bill_type}/{bill_num}')
        data = response.json()

        return data['bill']
    
    async def get_bill_cosponsors(self, congress_num: int, bill_type: str, bill_num: int) -> list:
        """
        Returns the list of representatives that cosponsored a specified bill
        Args:
            congress_num: the number of the congress (e.g. 119)
            bill_type: the type of bill. Acceptable values are "hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", or "sres"
            bill_num: the bill's assigned number (e.g. 3076)
        """
        cosponsors = []
        
        i = 0
        while True:
            response = await self._request_with_retry('get', f'/bill/{congress_num}/{bill_type}/{bill_num}/cosponsors', params = {'offset': 250*i})
            data = response.json()
            
            cosponsors.extend(data['cosponsors'])
            if i == 0 and data['pagination']['count'] < 250:
                break
            elif len(data['cosponsors']) == 0:
                break
            
            i += 1

        return cosponsors
    
