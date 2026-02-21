from .api_client import CongressAPIClient
from ..config import settings
import time
from typing import List, Optional, Literal

# idea:
# - extract layer calls the api to get the bill types and bill numbers for the bills that the user wants to extract
# - extract layer writes these bill types/numbers into a temp file
# - extract layer extracts bill info until failure (i.e. until rate limit is exceeded) and updates the temp file to indicate which bills have been extracted
# - after an hour, continue running extraction until failure
# - continue until all bills have been extracted

# omitted "hjres", "sjres", "hconres", "sconres", "hres", "sres" from valid types because only hr and s can become laws
VALID_BILL_TYPES = ["hr", "s"]

client = CongressAPIClient(settings.API_KEY.get_secret_value())

async def extract_reps(congress_num: int):
    # get all reps
    representatives = await client.get_all_members(congress_num)

async def get_bill_ids(congress_num: int = None) -> List[tuple]:
    """
    Returns a list of bill identifiers, whose elements are tuples of the form (`bill_type`, `bill_num`)
    """
    if congress_num is None:
        current_details = await client.get_current_congress()
        congress_num = current_details['congress_num']

    bill_ids = []
    for bill_type in VALID_BILL_TYPES:
        bills_of_type = await client.get_all_bills(congress_num, bill_type)
        current_bill_ids = [(bill_type, bill['number']) for bill in bills_of_type]
        bill_ids.extend(current_bill_ids)

    return bill_ids

async def write_progress():
    # initializes progress file
    pass

async def update_progress():
    # updates progress file
    pass

async def read_progress():
    # reads progress file
    pass

async def batch_extract_bill_info(congress_num: int, bill_ids: list):
    # get detailed bill info and cosponsors
    bills_detailed = {type: dict() for type in VALID_BILL_TYPES}
    bill_cosponsors = {type: dict() for type in VALID_BILL_TYPES}

    for bill_identifier in bill_ids:
        bill_type, bill_num = bill_identifier

        bill_info = await client.get_bill_info(congress_num, bill_type, bill_num)
        bills_detailed[bill_type][bill_num] = bill_info
        
        cosponsors = await client.get_bill_cosponsors(congress_num, bill_type, bill_num)
        bill_cosponsors[bill_type][bill_num] = cosponsors

    return {'bills': bills_detailed,
            'cosponsors': bill_cosponsors}

