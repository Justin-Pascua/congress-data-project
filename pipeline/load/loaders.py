from sqlalchemy import column
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from typing import List
import logging

from .models import Member, Bill, BillSponsorship
from .database import Session
from ..transform.schemas import MemberClean, BillClean, BillSponsorshipClean

logger = logging.getLogger('pipeline.load')

def upsert_members(clean_members: List[MemberClean]) -> None:
    """
    Upsert member info into db.
    Args:
        clean_members: a list of members as returned by `pipeline.transform.transform_members`
    """
    logger.info(f"Starting member upsert: {len(clean_members)} records")

    with Session() as db:
        member_dicts = [member.model_dump() for member in clean_members]
        stmt = insert(Member).values(member_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['congress_num', 'bio_guide_id'],
            set_ = {
                'name': stmt.excluded.name,
                'chamber': stmt.excluded.chamber,
                'state': stmt.excluded.state,
                'district': stmt.excluded.district,
                'party': stmt.excluded.party
            }
        ).returning(
            Member.bio_guide_id,
            (column('xmax') == 0).label('was_inserted') # used to determine if row was inserted or updated
        )
        
        result = db.execute(stmt)
        db.commit()

        rows = result.all()
        inserted = sum(1 for row in rows if row.was_inserted)
        updated = len(rows) - inserted

    logger.info(f"Completed member upsert: {inserted} inserted, {updated} updated")
    
def upsert_bills(clean_bills: List[BillClean]) -> None:
    """
    Upsert bill info into db.
    Args:
        clean_bills: a list of bills as returned by `pipeline.transform.transform_bills`
    """
    logger.info(f"Starting bill upsert: {len(clean_bills)} records")
    
    with Session() as db:
        bill_dicts = [bill.model_dump() for bill in clean_bills]
        stmt = insert(Bill).values(bill_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['congress_num', 'bill_type', 'bill_num'],
            set_ = {
                'introduced_date': stmt.excluded.introduced_date,
                'title': stmt.excluded.title,
                'policy_area': stmt.excluded.policy_area,
                'summary': stmt.excluded.summary
            }
        ).returning(
            Bill.congress_num, Bill.bill_type, Bill.bill_num,
            (column('xmax') == 0).label('was_inserted')
        )

        result = db.execute(stmt)
        db.commit()

        rows = result.all()
        inserted = sum(1 for row in rows if row.was_inserted)
        updated = len(rows) - inserted

    logger.info(f"Completed bill upsert: {inserted} inserted, {updated} updated")

def upsert_sponsorships(clean_sponsorships: List[BillSponsorshipClean], refresh_time: datetime = datetime.now()) -> None:
    """
    Upsert bill info into db.
    Args:
        clean_sponsorship: a list of bill sponsorship details as returned by `pipeline.transform.transform_bill_sponsorship`
        refresh_time: a `datetime` object indicating the time of extraction. This value gets written into the "last_refresh" column of the "bill_sponsorship" table. 
    """
    logger.info(f"Starting sponsorship upsert: {len(clean_sponsorships)} records")

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
        ).returning(
            BillSponsorship.bio_guide_id,
            BillSponsorship.congress_num,
            BillSponsorship.bill_type,
            BillSponsorship.bill_num,
            (column('xmax') == 0).label('was_inserted')
        )
        result = db.execute(stmt)
        db.commit()

        rows = result.all()
        inserted = sum(1 for row in rows if row.was_inserted)
        updated = len(rows) - inserted

    logger.info(f"Completed sponsorship upsert: {inserted} inserted, {updated} updated")
