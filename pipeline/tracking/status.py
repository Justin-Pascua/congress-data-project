from enum import Enum, auto

class ExtractStatus(Enum):
    UNATTEMPTED = auto()
    FAILED = auto()
    SUCCESSFUL = auto()

class TransformStatus(Enum):
    UNATTEMPTED = auto()
    FAILED = auto()
    SUCCESSFUL = auto()

class LoadStatus(Enum):
    UNATTEMPTED = auto()
    FAILED = auto()
    SUCCESSFUL = auto()