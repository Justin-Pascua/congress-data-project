from enum import Enum, auto

class ExtractStatus(Enum):
    UNATTEMPTED = "E_UNATTEMPTED"
    FAILED = "E_FAILED"
    SUCCESSFUL = "E_SUCCESSFUL"

class TransformStatus(Enum):
    UNATTEMPTED = "T_UNATTEMPTED"
    FAILED = "T_FAILED"
    SUCCESSFUL = "T_SUCCESSFUL"

class LoadStatus(Enum):
    UNATTEMPTED = "L_UNATTEMPTED"
    BILL_FAILED = "L_B_FAILED"
    SPONSORSHIP_FAILED = "L_S_FAILED"
    SUCCESSFUL = "L_SUCCESSFUL"
