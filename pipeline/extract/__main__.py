from .api_client import CongressAPIClient
from ..config import settings

client = CongressAPIClient(settings.API_KEY.get_secret_value())

async def run_extract():

    current_details = await client.get_current_congress()
    current_congress = current_details['congress_num']
    
    # get all reps
    representatives = await client.get_all_members(current_congress)
    
    # get all bills (comes with sponsors)
    valid_bill_types = ["hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"]
    bills = dict()
    
    for bill_type in valid_bill_types:
        bills_of_type = await client.get_all_bills(current_congress, bill_type)
        bills[bill_type] = bills_of_type    
    
    # get bill cosponsors
    cosponsorsip = dict()
    for bill_type, bills_of_type in bills.items():
        cosponsorsip[bill_type] = dict()
        for bill in bills_of_type:
            bill_num = bill['number']
            cosponsors_of_bill = await client.get_bill_cosponsors(current_congress, bill_type, bill_num)
            cosponsorsip[bill_type][bill_num] = cosponsors_of_bill

    return {'representatives': representatives,
            'bills': bills,
            'cosponsorship': cosponsorsip}

async def run_incremental_extract(congress_num: int):
    pass
