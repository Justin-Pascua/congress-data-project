import pydantic
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from database.enums import Chamber, BillType, SponsorshipType

class MemberClean(BaseModel):
    congress_num: int
    bio_guide_id: str
    name: str
    party: str | None = None
    state: str
    district: int | None = None
    chamber: Chamber

class BillClean(BaseModel):
    congress_num: int
    bill_type: BillType
    bill_num: int
    introduced_date: datetime

    title: str
    chamber: Chamber
    policy_area: Optional[str] = None
    summary: Optional[str] = None

class BillSponsorshipClean(BaseModel):
    # identifier for representative
    bio_guide_id: str

    # identifiers for bill
    congress_num: int
    bill_type: BillType
    bill_num: int

    sponsorship_type: SponsorshipType