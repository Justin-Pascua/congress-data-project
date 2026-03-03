from typing import List
import logging
from datetime import datetime

from .schemas import MemberClean, BillClean, BillSponsorshipClean

logger = logging.getLogger('pipeline.transform')

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

def transform_members(congress_num: int, raw_members: List[dict]) -> List[MemberClean]:
    """
    Applies transformations to list of dicts representing members.
    Args:
        raw_members: a list of dicts representing members as returned by `pipeline.extract.extract_members`
    """
    
    logger.info(f"Starting member transformation: {len(raw_members)} raw records")
    
    result = []
    failures = 0

    for raw_member in raw_members:
        try:
            clean_rep = MemberClean(
                congress_num = congress_num,
                bio_guide_id = raw_member['bioguideId'],
                name = raw_member['name'],
                party = raw_member['partyName'],
                state = raw_member['state'],
                district = raw_member['district'],
                chamber = raw_member['terms']['item'][0]['chamber']
            )
            result.append(clean_rep)
        except Exception as e:
            failures += 1
            logger.warning(f"Member transformation failed. ID: {raw_member['bioguideId']} | Error: {e}")

    logger.info(f"Completed member transformation: {len(result)} cleaned, {failures} failures")

    return result

def transform_bills(raw_bills: List[dict]) -> List[BillClean]:
    """
    Applies transformations to list of dicts representing bills to extract bill details.
    Args:
        raw_bills: a list of bills as returned by `pipeline.extract.batch_extract_bill_info`
    """
    
    logger.info(f"Starting bill transformation: {len(raw_bills)} raw records")
    
    result = []
    failures = 0

    for raw_bill in raw_bills:
        try:
            clean_bill = BillClean(
                congress_num = raw_bill['bill']['congress'],
                bill_type = raw_bill['bill']['type'],
                bill_num = raw_bill['bill']['number'],

                introduced_date = datetime.strptime(raw_bill['bill']['introducedDate'],"%Y-%m-%d"),
                title = raw_bill['bill']['title'],
                chamber = raw_bill['bill']['originChamber'],
                # only using nested get here because only these two fields are nullable
                policy_area = nested_get(raw_bill, 'bill', 'policyArea', 'name'),   
                summary = nested_get(raw_bill, 'summary', 'summary')
            )
            result.append(clean_bill)
        except Exception as e:
            failures += 1
            logger.warning(f"Bill transformation failed. ID: {raw_bill['bill']['congress'], raw_bill['bill']['type'], raw_bill['bill']['number']}"
                           f" | Error: {e}")

    
    logger.info(f"Completed bill transformation: {len(result)} cleaned, {failures} failures")
    
    return result

def transform_bill_sponsorships(raw_bills: List[dict]) -> List[BillSponsorshipClean]:
    """
    Applies transformations to list of dicts representing bills to extract sponsorship details.
    Args:
        raw_bills: a list of bills as returned by `pipeline.extract.batch_extract_bill_info`
    """

    logger.info(f"Starting sponsorship transformation: {len(raw_bills)} raw records")
    
    seen = set()
    result = []
    duplicates = 0
    
    for raw_bill in raw_bills:
        congress_num = raw_bill['bill']['congress']
        bill_type = raw_bill['bill']['type']
        bill_num = raw_bill['bill']['number']

        for member in raw_bill['bill'].get('sponsors', []):
            key = (member['bioguideId'], congress_num, bill_type, bill_num, 'sponsor')
            if key not in seen:
                seen.add(key)
                result.append(BillSponsorshipClean(
                    bio_guide_id = member['bioguideId'],
                    congress_num = congress_num,
                    bill_type = bill_type,
                    bill_num = bill_num,
                    sponsorship_type = 'sponsor'
                ))
            else:
                duplicates += 1
                logger.warning(f"Duplicate sponsor: {key}")

        for member in raw_bill['cosponsors']:
            # members who've withdrawn cosponsorship have a "sponsorshipWithdrawnDate" field
            if 'sponsorshipWithdrawnDate' in member:
                continue
            
            key = (member['bioguideId'], congress_num, bill_type, bill_num, 'cosponsor')
            if key not in seen:
                seen.add(key)
                result.append(BillSponsorshipClean(
                    bio_guide_id = member['bioguideId'],
                    congress_num = congress_num,
                    bill_type = bill_type,
                    bill_num = bill_num,
                    sponsorship_type = 'cosponsor'
                ))
            else:
                duplicates += 1
                logger.warning(f"Duplicate cosponsor: {key}")
    
    logger.info(f"Completed sponsorship transformation: {len(result)} distinct, {duplicates} duplicates")

    return result