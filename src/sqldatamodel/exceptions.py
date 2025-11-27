class DimensionError(Exception):
    """
    Raised when arguments provided to ``SQLDataModel`` are not of compatible dimensions,
    for example, trying to join a ``(10, 4)`` dimensional model to a ``(7, 5)`` dimensional model.

    Attributes:
        ``message`` (str): A detailed error message describing the dimension mismatch.

    Example::
    
        from SQLDataModel import SQLDataModel

        # Example headers and data
        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model with correct dimensions
        sdm = SQLDataModel(data,headers)

        # This time with one less header, which raises `DimensionError` exception:
        try:
            sdm = SQLDataModel(data,headers[:-1])
        except DimensionError as e:
            print(e)
        
        # Attempting to assign a row with incompatible shape which also raises `DimensionError` exception:
        try:
            sdm[1] = ['sarah', 'west', 30, 'new york']
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
    
        from SQLDataModel import SQLDataModel

        # Example headers and data
        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model
        sdm = SQLDataModel(data,headers)

        # Query with invalid syntax to raise `SQLProgrammingError` exception:
        try:
            sdm = SQLDataModel.fetch_query("selct first, last from sdm where age > 30")
        except SQLProgrammingError as e:
            print(e)

        # Query for non-existing column to raise `SQLProgrammingError` exception:
        try:
            sdm = SQLDataModel.fetch_query("select first, last, date_of_birth from sdm")
        except SQLProgrammingError as e:
            print(e)    

    Note:
        - This exception is used to wrap any ``sqlite3.ProgrammingError`` encountered during SQL related operations.
    """

def WarnFormat(warn:str) -> str:
    """
    Formats a warning message with ANSI color coding.

    Parameters:
        ``warn``: The warning message to be formatted.

    Returns:
        ``str``: The modified string with ANSI color coding, highlighting the class name in bold yellow.

    Example::
    
        # Warning to format
        formatted_warning = WarnFormat("DeprecationWarning: This method is deprecated.")
        
        # Styled message to pass with error or exception
        print(formatted_warning)
    
    Changelog:
        - Version 0.12.1 (2024-07-07):
            - Moved into exceptions module to avoid redefinition in each module.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    warned_by, warning_description = warn.split(':',1)
    return f"""\r\033[1m\033[38;2;246;221;109m{warned_by}:\033[0m\033[39m\033[49m{warning_description}"""
    
def ErrorFormat(error:str) -> str:
    """
    Formats an error message with ANSI color coding.

    Parameters:
        ``error``: The error message to be formatted.

    Returns:
        ``str``: The modified string with ANSI color coding, highlighting the error type in bold red.

    Example::
        
        # Format the error message
        formatted_error = SQLDataModel.ErrorFormat("ValueError: Invalid value provided.")
        
        # Should display a colored string to display along with error or exception
        print(formatted_error)

    Changelog:        
        - Version 0.12.1 (2024-07-07):
            - Moved into exceptions module to avoid redefinition in each module.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    error_type, error_description = error.split(':',1)
    return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""