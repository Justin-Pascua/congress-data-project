from typing import List

from .schemas import RepresentativeClean, BillClean, BillMembershipClean
from ..load import models

def transform_representatives(raw_representatives: List[dict]) -> List[RepresentativeClean]:
    result = []
    for raw_rep in raw_representatives:
        clean_rep = RepresentativeClean(
            bio_guide_id = raw_rep['bioguideId'],
            name = raw_rep['name'],
            party = raw_rep['partyName'],
            state = raw_rep['state'],
            district = raw_rep['district'],
            chamber = raw_rep['terms']['item'][0]['chamber']
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

def transform_bill_membership(raw_bills: List[dict]) -> List[BillMembershipClean]:
    result = []

    for raw_bill in raw_bills:
        bill_members = raw_bill['bill']['sponsors'] + raw_bill['cosponsors']
        for member in bill_members:
            membership = BillMembershipClean(
                bio_guide_id = member['bioguideId'],
                congress_num = raw_bill['bill']['congress'],
                bill_type = raw_bill['bill']['type'],
                bill_num = raw_bill['bill']['number']
            )
            result.append(membership)
    
    return result