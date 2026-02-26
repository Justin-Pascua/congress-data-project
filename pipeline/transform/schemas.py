import pydantic
from pydantic import BaseModel
from typing import List, Optional
from .enums import Chamber, BillType

class RepresentativeClean(BaseModel):
    bio_guide_id: str
    name: str
    party: Optional[str]
    state: str
    district: int | str | None = None
    chamber: Chamber

class BillClean(BaseModel):
    congress_num: int
    bill_type: BillType
    bill_num: int

    title: str
    chamber: Chamber
    policy_area: Optional[str] = None
    summary: Optional[str] = None

class BillMembershipClean(BaseModel):
    # identifier for representative
    bio_guide_id: str

    # identifiers for bill
    congress_num: int
    bill_type: BillType
    bill_num: int