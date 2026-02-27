from typing import List

from .schemas import MemberClean, BillClean, BillSponsorshipClean
from ..load import models

def nested_get(input_dict: dict, *args):
    """
    Custom version of `dict.get` for nested dictionaries
    """
    try:
        result = input_dict
        for arg in args:
            result = result[arg] 
        return result
    except:
        return None


def transform_members(raw_members: List[dict]) -> List[MemberClean]:
    """
    Applies transformations to list of dicts representing members.
    Args:
        raw_members: a list of dicts representing members as returned by `pipeline.extract.extract_members`
    """
    result = []
    for raw_members in raw_members:
        clean_rep = MemberClean(
            bio_guide_id = raw_members['bioguideId'],
            name = raw_members['name'],
            party = raw_members['partyName'],
            state = raw_members['state'],
            district = raw_members['district'],
            chamber = raw_members['terms']['item'][0]['chamber']
        )
        result.append(clean_rep)

    return result

def transform_bills(raw_bills: List[dict]) -> List[BillClean]:
    """
    Applies transformations to list of dicts representing bills to extract bill details.
    Args:
        raw_bills: a list of bills as returned by `pipeline.extract.batch_extract_bill_info`
    """
    result = []
    for raw_bill in raw_bills:

        clean_bill = BillClean(
            congress_num = raw_bill['bill']['congress'],
            bill_type = raw_bill['bill']['type'],
            bill_num = raw_bill['bill']['number'],
            
            title = raw_bill['bill']['title'],
            chamber = raw_bill['bill']['originChamber'],
            # only using nested get here because only these two fields are nullable
            policy_area = nested_get(raw_bill, 'bill', 'policyArea', 'name'),   
            summary = nested_get(raw_bill, 'summary', 'summary')
        )
        result.append(clean_bill)
    
    return result

def transform_bill_sponsorship(raw_bills: List[dict]) -> List[BillSponsorshipClean]:
    """
    Applies transformations to list of dicts representing bills to extract sponsorship details.
    Args:
        raw_bills: a list of bills as returned by `pipeline.extract.batch_extract_bill_info`
    """
    result = []
    
    for raw_bill in raw_bills:
        for member in raw_bill['bill']['sponsors']:
            membership = BillSponsorshipClean(
                bio_guide_id = member['bioguideId'],
                congress_num = raw_bill['bill']['congress'],
                bill_type = raw_bill['bill']['type'],
                bill_num = raw_bill['bill']['number'],
                sponsorship_type = 'sponsor'
            )
            result.append(membership)

        for member in raw_bill['cosponsors']:
            membership = BillSponsorshipClean(
                bio_guide_id = member['bioguideId'],
                congress_num = raw_bill['bill']['congress'],
                bill_type = raw_bill['bill']['type'],
                bill_num = raw_bill['bill']['number'],
                sponsorship_type = 'cosponsor'
            )
            result.append(membership)
    
    return result