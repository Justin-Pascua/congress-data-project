from sqlalchemy import column
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from typing import List
import logging

from database.models import Member, Bill, BillSponsorship
from database.conn import Session
from ..transform.schemas import MemberClean, BillClean, BillSponsorshipClean
from ..tracking import utils
from ..tracking.status import LoadStatus

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
    
def upsert_bills(congress_num: int, clean_bills: List[BillClean]) -> None:
    """
    Upsert bill info into db.
    Args:
        congress_num: the number of the congress (e.g. 119). Used to fetch and update queue.
        clean_bills: a list of bills as returned by `pipeline.transform.transform_bills`
    """
    logger.info(f"Starting bill upsert: {len(clean_bills)} records")
    queue_df = utils.read_queue(congress_num)
    queue_indices = list(set((item.congress_num, item.bill_type.value, item.bill_num) for item in clean_bills))
    
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
        try:
            result = db.execute(stmt)
            db.commit()

            rows = result.all()
            inserted = sum(1 for row in rows if row.was_inserted)
            updated = len(rows) - inserted

            queue_df.loc[queue_indices, 'Load Status'] = LoadStatus.SUCCESSFUL.value
            logger.info(f"Completed bill upsert: {inserted} inserted, {updated} updated")
        except Exception as e:
            # sqlalchemy error messages usually very long, so take only first 100 chars
            error_str = f"({type(e)}) {str(e)[:100]}"
            queue_df.loc[queue_indices, 'Load Status'] = LoadStatus.FAILED.value
            queue_df.loc[queue_indices, 'Error'] = error_str
            logger.info(f"Failed bill upsert: {error_str}")

        utils.commit_queue(congress_num, queue_df)

def upsert_sponsorships(congress_num: int, clean_sponsorships: List[BillSponsorshipClean]) -> None:
    """
    Upsert bill info into db.
    Args:
        congress_num: the number of the congress (e.g. 119). Used to fetch and update queue.
        clean_sponsorship: a list of bill sponsorship details as returned by `pipeline.transform.transform_bill_sponsorship`
        """
    logger.info(f"Starting sponsorship upsert: {len(clean_sponsorships)} records")
    queue_df = utils.read_queue(congress_num)
    queue_indices = list(set((item.congress_num, item.bill_type.value, item.bill_num) for item in clean_sponsorships))

    with Session() as db:
        sponsorship_dicts = [sponsorship.model_dump() for sponsorship in clean_sponsorships]
        stmt = insert(BillSponsorship).values(sponsorship_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements = ['bio_guide_id', 'congress_num', 'bill_type', 'bill_num'],
            set_ = {
                'sponsorship_type': stmt.excluded.sponsorship_type
            }
        ).returning(
            BillSponsorship.bio_guide_id,
            BillSponsorship.congress_num,
            BillSponsorship.bill_type,
            BillSponsorship.bill_num,
            (column('xmax') == 0).label('was_inserted')
        )
        try:
            result = db.execute(stmt)
            db.commit()

            rows = result.all()
            inserted = sum(1 for row in rows if row.was_inserted)
            updated = len(rows) - inserted

            # leave success status up to upsert_bills
            logger.info(f"Completed sponsorship upsert: {inserted} inserted, {updated} updated")
        except Exception as e:
            # sqlalchemy error messages usually very long, so take only first 100 chars
            error_str = f"({type(e)}) {str(e)[:100]}"
            queue_df.loc[queue_indices, 'Load Status'] = LoadStatus.FAILED.value
            queue_df.loc[queue_indices, 'Error'] = error_str
            logger.info(f"Failed sponsorship upsert: {error_str}")

    utils.commit_queue(congress_num, queue_df)