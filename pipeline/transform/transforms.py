from typing import List

from .schemas import MemberClean, BillClean, BillSponsorshipClean
from ..load import models

def transform_members(raw_members: List[dict]) -> List[MemberClean]:
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
    result = []
    for raw_bill in raw_bills:
        clean_bill = BillClean(
            congress_num = raw_bill['bill']['congress'],
            bill_type = raw_bill['bill']['type'],
            bill_num = raw_bill['bill']['number'],
            
            title = raw_bill['bill']['title'],
            chamber = raw_bill['bill']['originChamber'],
            policy_area = raw_bill['bill']['policyArea']['name'],
            summary = raw_bill['summary']['summary']
        )
        result.append(clean_bill)
    
    return result

def transform_bill_membership(raw_bills: List[dict]) -> List[BillSponsorshipClean]:
    result = []

    for raw_bill in raw_bills:
        bill_members = raw_bill['bill']['sponsors'] + raw_bill['cosponsors']
        for member in bill_members:
            membership = BillSponsorshipClean(
                bio_guide_id = member['bioguideId'],
                congress_num = raw_bill['bill']['congress'],
                bill_type = raw_bill['bill']['type'],
                bill_num = raw_bill['bill']['number']
            )
            result.append(membership)
    
    return result