import enum

class ExtractStatus(enum.Enum):
    UNATTEMPTED = 'UNATTEMPTED'
    FAILED = 'FAILED'
    EXTRACTED = 'EXTRACTED'