import enum

class Chamber(enum.Enum):
    HR = "House of Representatives"
    S = "Senate"

    @classmethod
    def _missing_(cls, value):
        missing_map = {
            "H": cls.HR,
            "HR": cls.HR,
            "House": cls.HR,
            "S": cls.S,
        }
        if value in missing_map:
            return missing_map[value]
        
        return super()._missing_(value)

class SponsorshipType(enum.Enum):
    SPONSOR = "sponsor"
    COSPONSOR = "cosponsor"

class BillType(enum.Enum):
    # need lower-case to allow values to be used in API calls
    HR = "hr"
    S = "s"
    HJRES = "hjres"
    SJRES = "sjres"
    HCONRES = "hconres"
    SCONRES = "sconres"
    HRES = "hres"
    SRES = "sres"

    # used for case-insensitive instantiation
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
    
        return super()._missing_(value)
