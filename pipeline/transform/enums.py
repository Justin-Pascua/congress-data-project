import enum

class Chamber(enum.Enum):
    HR = "House of Representatives"
    S = "Senate"

class SponsorshipType(enum.Enum):
    SPONSOR = "sponsor"
    COSPONSOR = "cosponsor"

class BillTypes(enum.Enum):
    HR = "HR"
    S = "S"
    HJRES = "HJRES"
    SJRES = "SJRES"
    HCONRES = "HCONRES"
    SCONRES = "SCONRES"
    HRES = "HRES"
    SRES = "SRES"