from .api_client import CongressAPIClient
from ..config import settings
import time
from typing import List, Optional, Literal

# omitted "hjres", "sjres", "hconres", "sconres", "hres", "sres" from valid types because only hr and s can become laws
VALID_BILL_TYPES = ["hr", "s"]

client = CongressAPIClient(settings.API_KEY.get_secret_value())

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

async def run_extract(congress_num: int = None):
    if congress_num is None:
        current_details = await client.get_current_congress()
        congress_num = current_details['congress_num']


    # get all reps
    representatives = await client.get_all_members(congress_num)
    
    # get detailed bill info and cosponsors
    bills_detailed = {type: dict() for type in VALID_BILL_TYPES}
    bill_cosponsors = {type: dict() for type in VALID_BILL_TYPES}

    bill_ids = await get_bill_ids(congress_num) # list of tuples of form (bill_type, bill_num) 
    for bill_identifier in bill_ids:
        bill_type, bill_num = bill_identifier

        bill_info = await client.get_bill_info(congress_num, bill_type, bill_num)
        bills_detailed[bill_type][bill_num] = bill_info
        
        cosponsors = await client.get_bill_cosponsors(congress_num, bill_type, bill_num)
        bill_cosponsors[bill_type][bill_num] = cosponsors

    return {'representatives': representatives,
            'bills': bills_detailed,
            'cosponsors': bill_cosponsors
            }
