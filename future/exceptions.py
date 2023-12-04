class DimensionMismatch(Exception):
    """raised when arguments provided are not of compatible dimensions"""
    pass

class InvalidHeaderException(Exception):
    """raised when user provided header is not in the current `SQLDataModel`"""
    pass

class ModelIndexException(Exception):
    """raised when `SQLDataModel` index provided is out of bounds of current model"""
    pass

class InvalidArgumentException(Exception):
    """raise when an incompatible argument is provided to `SQLDataModel` that requires exiting the program"""