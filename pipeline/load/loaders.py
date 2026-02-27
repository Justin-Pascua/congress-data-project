from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from typing import List

from .models import Member, Bill, BillSponsorship
from .database import Session
from ..transform.schemas import MemberClean, BillClean, BillSponsorshipClean

def upsert_members(clean_members: List[MemberClean]) -> None:
    """
    Upsert member info into db.
    Args:
        clean_members: a list of members as returned by `pipeline.transform.transform_members`
    """
    with Session() as db:
        member_dicts = [member.model_dump() for member in clean_members]
        stmt = insert(Member).values(member_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['bio_guide_id'],
            set_ = {
                'name': stmt.excluded.name,
                'chamber': stmt.excluded.chamber,
                'state': stmt.excluded.state,
                'district': stmt.excluded.district,
                'party': stmt.excluded.party
            }
        )
        db.execute(stmt)
        db.commit()

def upsert_bills(clean_bills: List[BillClean]) -> None:
    """
    Upsert bill info into db.
    Args:
        clean_bills: a list of bills as returned by `pipeline.transform.transform_bills`
    """
    with Session() as db:
        bill_dicts = [bill.model_dump() for bill in clean_bills]
        stmt = insert(Bill).values(bill_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['congress_num', 'bill_type', 'bill_num'],
            set_ = {
                'title': stmt.excluded.title,
                'policy_area': stmt.excluded.policy_area,
                'summary': stmt.excluded.summary
            }
        )
        db.execute(stmt)
        db.commit()

def upsert_sponsorship(clean_sponsorships: List[BillSponsorshipClean], refresh_time: datetime = datetime.now()) -> None:
    """
    Upsert bill info into db.
    Args:
        clean_sponsorship: a list of bill sponsorship details as returned by `pipeline.transform.transform_bill_sponsorship`
        refresh_time: a `datetime` object indicating the time of extraction. This value gets written into the "last_refresh" column of the "bill_sponsorship" table. 
    """
    with Session() as db:
        sponsorship_dicts = [sponsorship.model_dump() | {'is_active': True, 'last_refresh': refresh_time}
                            for sponsorship in clean_sponsorships]
        stmt = insert(BillSponsorship).values(sponsorship_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['bio_guide_id', 'congress_num', 'bill_type', 'bill_num'],
            set_ = {
                'sponsorship_type': stmt.excluded.sponsorship_type,
                'is_active': stmt.excluded.is_active,
                'last_refresh': stmt.excluded.last_refresh,
            }
        )
        db.execute(stmt)
        db.commit()