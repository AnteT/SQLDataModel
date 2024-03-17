class DimensionError(Exception):
    """
    Raised when arguments provided to ``SQLDataModel`` are not of compatible dimensions,
    for example, trying to join a (10, 4) dimensional model to a (7, 5) dimensional model.

    Attributes:
        ``message`` (str): A detailed error message describing the dimension mismatch.

    Example::
    
        import SQLDataModel

        # Example headers and data
        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model with correct dimensions
        sqldm = SQLDataModel(data,headers)

        # This time with one less header, which raises `DimensionError` exception:
        try:
            sqldm = SQLDataModel(data,headers[:-1])
        except DimensionError as e:
            print(e)
        
        # Attempting to assign a row with incompatible shape which also raises `DimensionError` exception:
        try:
            sqldm[1] = ['sarah', 'west', 30, 'new york']
        except DimensionError as e:
            print(e)
    
    Note:
        - An argument could be made for using `ValueError` instead, but there's enough difference to justify a new error.
    """

class SQLProgrammingError(Exception):
    """
    Raised when invalid or malformed SQL prevents the execution of a method or returns unexpected behavior,
    for example, trying to select a column that does not exist in the current model.

    Attributes:
        ``message`` (str): A detailed error message describing the SQL programming error.

    Example::
    
        import SQLDataModel

        # Example headers and data
        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model
        sqldm = SQLDataModel(data,headers)

        # Query with invalid syntax to raise `SQLProgrammingError` exception:
        try:
            sqldm = SQLDataModel.fetch_query("selct first, last from sdmmodel where age > 30")
        except SQLProgrammingError as e:
            print(e)

        # Query for non-existing column to raise `SQLProgrammingError` exception:
        try:
            sqldm = SQLDataModel.fetch_query("select first, last, date_of_birth from sdmmodel")
        except SQLProgrammingError as e:
            print(e)    

    Note:
        - This exception is used to wrap any ``sqlite3.ProgrammingError`` encountered during SQL related operations.
        
    """