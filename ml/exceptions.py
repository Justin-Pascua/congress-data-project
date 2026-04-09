class EmptyDatasetError(Exception):
    """
    Thrown by data utils and preprocessing pipeline when no data is left are preprocessing.
    Bill data is dropped when either the bill summary or policy area are null. 
    So, this error is typically raised when pulling only recent bills (because newer
    bills tend to be unclassified or lacking a summary).
    """