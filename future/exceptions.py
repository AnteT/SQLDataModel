class DimensionError(Exception):
    """raised when arguments provided to `SQLDataModel` are not of compatible dimensions, ie trying to join a (10, 4) dimensional model to a (7, 5) dimensional model"""

class SQLProgrammingError(Exception):
    """raised when invalid or malformed SQL prevents execution of method or returns unexpected behavior, ie trying to select a column that does not exist in the current model"""