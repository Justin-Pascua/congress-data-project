from sqlalchemy import select

from typing import List, Literal
from datetime import datetime

from .conn import Session
from .models import Member, Bill, BillSponsorship

def read_members(congress_num: int, 
                 parties: List[Literal['Democratic', 'Republican', 'Independent']] = None, 
                 chambers: List[Literal['HR', 'S']] = None, 
                 states: List[str] = None):
    """
    Pulls members from members table in local db.
    Args:
        congress_num: the number of the congress (e.g. 119) to filter by
        parties: the political parties to filter by
        chambers: the chambers of congress to filter by
        states: the states to filter by
    """
    with Session() as db:
        stmt = select(Member).where(Member.congress_num == congress_num)
        if parties is not None:
            stmt = stmt.where(Member.party.in_(parties))
        if chambers is not None:
            stmt = stmt.where(Member.chamber.in_(chambers))
        if states is not None:
            stmt = stmt.where(Member.state.in_(states))

        result = db.scalars(stmt).all()
    return result

def read_bills(congress_num: int = None,
               chambers: List[Literal['HR', 'S']] = None,
               start_date: datetime = None,
               end_date: datetime = None):
    """
    Pulls bills from bills table in local db.
    Args:
        congress_num: the number of the congress (e.g. 119) to filter by
        chambers: the chambers of congress to filter by
        start_date: the starting timestamp to filter by bill introduction date
        end_date: the ending timestamp to filter by bill introduction date
    """
    with Session() as db:
        stmt = select(Bill)
        if congress_num is not None:
            stmt = stmt.where(Bill.congress_num == congress_num)
        if chambers is not None:
            bill_types = []
            if 'HR' in chambers:
                bill_types.extend(['HR', 'HJRES', 'HCONRES', 'HRES'])
            if 'S' in chambers:
                bill_types.extend(['S', 'SJRES', 'SCONRES', 'SRES'])
            stmt = stmt.where(Bill.bill_type.in_(bill_types))
        if start_date is not None:
            stmt = stmt.where(start_date <= Bill.introduced_date)
        if end_date is not None:
            stmt = stmt.where(Bill.introduced_date <= end_date)
        result = db.scalars(stmt).all()
    return result

def read_sponsorships(congress_num: int,
                      chambers: List[Literal['HR', 'S']] = None,
                      start_date: datetime = None,
                      end_date: datetime = None):
    """
    Pulls sponsorship items from bill_sponsorship table in local db.
    Args:
        congress_num: the number of the congress (e.g. 119) to filter by
        chambers: the chambers of congress to filter by
        start_date: the starting timestamp to filter by bill introduction date
        end_date: the ending timestamp to filter by bill introduction date
    """
    with Session() as db:
        stmt = (select(BillSponsorship)
                .where(BillSponsorship.congress_num == congress_num))
        if chambers is not None:
            bill_types = []
            if 'HR' in chambers:
                bill_types.extend(['HR', 'HJRES', 'HCONRES', 'HRES'])
            if 'S' in chambers:
                bill_types.extend(['S', 'SJRES', 'SCONRES', 'SRES'])
            stmt = stmt.where(BillSponsorship.bill_type.in_(bill_types))
        if start_date is not None:
            stmt = stmt.join(
                Bill,
                (Bill.congress_num == BillSponsorship.congress_num) &
                (Bill.bill_type == BillSponsorship.bill_type) &
                (Bill.bill_num == BillSponsorship.bill_num)
            ).where(start_date <= Bill.introduced_date).where(Bill.introduced_date <= end_date)
        result = db.scalars(stmt).all()
    return result
