import sqlite3
import typing

from .sqldatamodel import SQLDataModel
from .converters import register_adapters_and_converters

register_adapters_and_converters()
del(register_adapters_and_converters)

def from_csv(csv_source:str, infer_types:bool=True, encoding:str = 'Latin1', delimiter:str = ',', quotechar:str = '"', headers:list[str] = None, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` generated from the provided CSV source, which can be either a file path or a raw delimited string.

    Parameters:
        ``csv_source`` (str): The path to the CSV file or a raw delimited string.
        ``infer_types`` (bool, optional): Infer column types based on random subset of data. Default is True, when False, all columns are str type.
        ``encoding`` (str, optional): The encoding used to decode the CSV source if it is a file. Default is 'Latin1'.
        ``delimiter`` (str, optional): The delimiter to use when parsing CSV source. Default is ``,``.
        ``quotechar`` (str, optional): The character used for quoting fields. Default is ``"``.
        ``headers`` (List[str], optional): List of column headers. If None, the first row of the CSV source is assumed to contain headers.
        ``**kwargs``: Additional keyword arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the provided CSV source.

    Raises:
        ``ValueError``: If no delimited data is found in ``csv_source`` or if parsing with delimiter does not yield valid tabular data.
        ``Exception``: If an error occurs while attempting to read from or process the provided CSV source.

    Examples:

    From CSV File
    -------------

    ```python
        from SQLDataModel import SQLDataModel

        # CSV file path or raw CSV string
        csv_source = "/path/to/data.csv"

        # Create the model using the CSV file, providing custom headers
        sdm = SQLDataModel.from_csv(csv_source, headers=['ID', 'Name', 'Value'])
    ```

    From CSV Literal
    ----------------

    ```python
        from SQLDataModel import SQLDataModel

        # CSV data
        data = '''
        A, B, C
        1a, 1b, 1c
        2a, 2b, 2c
        3a, 3b, 3c
        '''

        # Create the model
        sdm = SQLDataModel.from_csv(data)

        # View result
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────┬──────┬──────┐
        │ A    │ B    │ C    │
        ├──────┼──────┼──────┤
        │ 1a   │ 1b   │ 1c   │
        │ 2a   │ 2b   │ 2c   │
        │ 3a   │ 3b   │ 3c   │
        └──────┴──────┴──────┘
        [3 rows x 3 columns]
    ```

    Note:
        - If ``csv_source`` is delimited by characters other than those specified, use :meth:`SQLDataModel.from_delimited()` and provide delimiter to ``delimiters``.
        - If ``headers`` are provided, the first row parsed from source will be the first row in the table and not discarded.
        - The ``infer_types`` argument can be used to infer the appropriate data type for each column:

            - When ``infer_types = True``, a random subset of the data will be used to infer the correct type and cast values accordingly
            - When ``infer_types = False``, values from the first row only will be used to assign types, almost always 'str' when reading from CSV.

    Changelog:
        - Version 0.4.0 (2024-04-23):
            - Modifed to only parse CSV files and removed all delimiter sniffing with introduction of new method :meth:`SQLDataModel.from_delimited()` to handle other delimiters.
            - Renamed ``delimiters`` parameter to ``delimiter`` with ``,`` set as new default to reflect revised focus on CSV files only.
    """
    return SQLDataModel.from_csv(csv_source, infer_types, encoding, delimiter, quotechar, headers, **kwargs)

def from_data(data:typing.Any=None, **kwargs) -> SQLDataModel:
    """
    Convenience method to infer the source of ``data`` and return the appropriate constructor method to generate a new ``SQLDataModel`` instance.

    Parameters:
        ``data`` (Any, required): The input data from which to create the SQLDataModel object. 
        ``**kwargs``: Additional keyword arguments to be passed to the constructor method, see init method for arguments.
        
    Constructor methods are called according to the input type:
        - ``dict``: If all values are python datatypes, passed as ``dtypes`` to constructor, otherwise as ``data`` to :meth:`SQLDataModel.from_dict()`.
        - ``list``: If single dimension, passed as ``headers`` to constructor, otherwise as ``data`` containing list of lists.
        - ``tuple``: Same as with list, if single dimension passed as ``headers``, otherwise as ``data`` containing tuple of lists.
        - ``numpy.ndarray``: passed to :meth:`SQLDataModel.from_numpy()` as array data.
        - ``pandas.DataFrame``: passed to :meth:`SQLDataModel.from_pandas()` as dataframe data.
        - ``polars.DataFrame``: passed to :meth:`SQLDataModel.from_polars()` as dataframe data.
        - ``str``: If starts with 'http', passed to :meth:`SQLDataModel.from_html()` as url, otherwise:
    
            - ``'.csv'``: passed to :meth:`SQLDataModel.from_csv()` as csv source data.
            - ``'.html'``: passed to :meth:`SQLDataModel.from_html()` as html source data.
            - ``'.json'``: passed to :meth:`SQLDataModel.from_json()` as json source data.
            - ``'.md'``: passed to :meth:`SQLDataModel.from_markdown()` as markdown source data.
            - ``'.parquet'``: passed to :meth:`SQLDataModel.from_parquet()` as parquet source data.
            - ``'.pkl'``: passed to :meth:`SQLDataModel.from_pickle()` as pickle source data.
            - ``'.sdm'``: passed to :meth:`SQLDataModel.from_pickle()` as pickle source data.
            - ``'.tex'``: passed to :meth:`SQLDataModel.from_latex()` as latex source data.
            - ``'.tsv'``: passed to :meth:`SQLDataModel.from_csv()` as csv source data.
            - ``'.txt'``: passed to :meth:`SQLDataModel.from_text()` as text source data.
            - ``'.xlsx'``: passed to :meth:`SQLDataModel.from_excel()` as excel source data.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the provided data.

    Raises:
        ``TypeError``: If the type of ``data`` is not supported.
        ``ValueError``: If the file extension is not found, unsupported, or if the SQL extension is not supported.
        ``Exception``: If an OS related error occurs during file read operations if ``data`` is a filepath.

    Example::

        from SQLDataModel import SQLDataModel

        # Create SQLDataModel from a CSV file
        sdm_csv = SQLDataModel.from_data("data.csv", headers=['ID', 'Name', 'Value'])

        # Create SQLDataModel from a dictionary
        sdm_dict = SQLDataModel.from_data({"ID": int, "Name": str, "Value": float})

        # Create SQLDataModel from a list of tuples
        sdm_list = SQLDataModel.from_data([(1, 'Alice', 100.0), (2, 'Bob', 200.0)], headers=['ID', 'Name', 'Value'])

        # Create SQLDataModel from raw string literal
        delimited_literal = '''
        A, B, C
        1, 2, 3
        4, 5, 6
        7, 8, 9
        '''

        # Create the model by having correct constructor inferred
        sdm = SQLDataModel.from_data(delimited_literal)

        # View output
        print(sdm)
    
    This will output:
    
    ```shell            
        ┌────┬────┬────┐
        │ A  │ B  │ C  │
        ├────┼────┼────┤
        │ 1  │ 2  │ 3  │
        │ 4  │ 5  │ 6  │
        │ 7  │ 8  │ 9  │
        └────┴────┴────┘
        [3 rows x 3 columns]
    ```

    Note:
        - This method attempts to infer the correct method to call based on ``data`` argument, if one cannot be inferred an exception is raised.
        - For data type specific implementation or examples, see related method for appropriate data type.
    """
    return SQLDataModel.from_data(data, **kwargs)

def from_delimited(source:str, infer_types:bool=True, encoding:str='Latin1', delimiters:str=', \t;|:', quotechar:str='"', headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` generated from the provided delimited source, which can be either a file path or a raw delimited string.

    Parameters:
        ``source`` (str): The path to the delimited file or a raw delimited string.
        ``infer_types`` (bool, optional): Infer column types based on random subset of data. Default is True, when False, all columns are str type.
        ``encoding`` (str, optional): The encoding used to decode the source if it is a file. Default is ``'Latin1'``.
        ``delimiters`` (str, optional): Possible delimiters. Default is ``\\s``, ``\\t``, ``;``, ``|``, ``:`` or ``,`` (space, tab, semicolon, pipe, colon or comma).
        ``quotechar`` (str, optional): The character used for quoting fields. Default is ``"``.
        ``headers`` (list[str], optional): List of column headers. If None, the first row of the delimited source is assumed to be the header row.
        ``**kwargs``: Additional keyword arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the provided CSV source.

    Raises:
        ``ValueError``: If no delimiter is found in ``source`` or if parsing with delimiter does not yield valid tabular data.
        ``Exception``: If an error occurs while attempting to read from or process the provided CSV source.

    Example:

    From Delimited Literal
    ----------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Space delimited literal
        source_data = '''
        Name Age Height
        Beth 27 172.4
        Kate 28 162.0
        John 30 175.3
        Will 35 185.8'''

        # Create the model
        sdm = SQLDataModel.from_delimited(source_data)

        # View output
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────┬─────┬─────────┐
        │ Name │ Age │  Height │
        ├──────┼─────┼─────────┤
        │ Beth │  27 │  172.40 │
        │ Kate │  28 │  162.00 │
        │ John │  30 │  175.30 │
        │ Will │  35 │  185.80 │
        └──────┴─────┴─────────┘
        [4 rows x 3 columns]
    ```

    From Delimited File
    -------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Tab separated file
        tsv_file = 'persons.tsv'

        # Create the model
        sdm = SQLDataModel.from_delimited(tsv_file)
    ```

    Note:
        - Use :meth:`SQLDataModel.from_csv()` if delimiter in source is already known and available as this method requires more compute to determine a plausible delimiter.
        - Use :meth:`SQLDataModel.from_text()` if data is not delimited but is a string representation such as an ASCII table or the output from another ``SQLDataModel`` instance.
        - If file is delimited by delimiters other than the default targets ``\\s``, ``\\t``, ``;``, ``|``, ``:`` or ``,`` (space, tab, semicolon, pipe, colon or comma) make sure they are provided as single character values to ``delimiters``.

    Changelog:
        - Version 0.4.0 (2024-04-23):
            - New method.
    """
    return SQLDataModel.from_delimited(source, infer_types, encoding, delimiters, quotechar, headers, **kwargs)

def from_dict(data:dict|list, **kwargs) -> SQLDataModel:
    """
    Create a new ``SQLDataModel`` instance from the provided dictionary.

    Parameters:
        ``data`` (dict): The dictionary or list of dictionaries to convert to SQLDataModel. If keys are of type int, they will be used as row indexes; otherwise, keys will be used as headers.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the provided dictionary.

    Raises:
        ``TypeError``: If the provided dictionary values are not of type 'list', 'tuple', or 'dict'.
        ``ValueError``: If the provided data appears to be a list of dicts but is empty.

    Example::

        from SQLDataModel import SQLDataModel
        
        # Sample data with column orientation
        data = {
            'Name': ['Beth', 'John', 'Alice', 'Travis'], 
            'Height': [172.4, 175.3, 162.0, 185.8],
            'Age': [27, 30, 28, 35]
        }

        # Create the model
        sdm = SQLDataModel.from_dict(data)

        # View it
        print(sdm)
    
    This will output:

    ```shell
        ┌────────┬─────────┬─────┐
        │ Name   │  Height │ Age │
        ├────────┼─────────┼─────┤
        │ Beth   │  172.40 │  27 │
        │ John   │  175.30 │  30 │
        │ Alice  │  162.00 │  28 │
        │ Travis │  185.80 │  35 │
        └────────┴─────────┴─────┘
        [4 rows x 3 columns]
    ```

    We can also create a model using a dictionary with row orientation:
    
    ```python
        from SQLDataModel import SQLDataModel

        # Sample data with row orientation
        data = {
                0: ['Mercury', 0.38]
            ,1: ['Venus', 0.91]
            ,2: ['Earth', 1.00]
            ,3: ['Mars', 0.38]
        }

        # Create the model with custom headers
        sdm = SQLDataModel.from_dict(data, headers=['Planet', 'Gravity'])

        # View output
        print(sdm)
    ```

    This will output the model created using row-wise dictionary data:
    
    ```shell            
        ┌─────────┬─────────┐
        │ Planet  │ Gravity │
        ├─────────┼─────────┤
        │ Mercury │    0.38 │
        │ Venus   │    0.91 │
        │ Earth   │    1.00 │
        │ Mars    │    0.38 │
        └─────────┴─────────┘
        [4 rows x 2 columns]
    ```

    Note:
        - If data orientation suggests JSON like structure, then :meth:`SQLDataModel.from_json()` will attempt to construct the model.
        - Dictionaries in list like orientation can also be used with structures similar to JSON objects.
        - The method determines the structure of the SQLDataModel based on the format of the provided dictionary.
        - If the keys are integers, they are used as row indexes; otherwise, keys are used as headers.
        - See :meth:`SQLDataModel.to_dict()` for converting existing instances of ``SQLDataModel`` to dictionaries.

    Changelog:
        - Version 0.6.3 (2024-05-16):
            - Modified to try parsing input data as JSON if initial inspection does not signify row or column orientation.
        - Version 0.1.5 (2023-11-24):
            - New method.            
    """
    return SQLDataModel.from_dict(data, **kwargs)

def from_excel(filename:str, worksheet:int|str=0, min_row:int|None=None, max_row:int|None=None, min_col:int|None=None, max_col:int|None=None, headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` instance from the specified Excel file.

    Parameters:
        ``filename`` (str): The file path to the Excel file, e.g., ``filename = 'titanic.xlsx'``.
        ``worksheet`` (int | str, optional): The index or name of the worksheet to read from. Defaults to 0, indicating the first worksheet.
        ``min_row`` (int | None, optional): The minimum row number to start reading data from. Defaults to None, indicating the first row.
        ``max_row`` (int | None, optional): Maximum row index (1-based) to import. Defaults to None, indicating all rows are read.
        ``min_col`` (int | None, optional): Minimum column index (1-based) to import. Defaults to None, indicating the first column.
        ``max_col`` (int | None, optional): Maximum column index (1-based) to import. Defaults to None, indicating all the columns are read.
        ``headers`` (List[str], optional): The column headers for the data. Default is None, using the first row of the Excel sheet as headers.
        ``**kwargs``: Additional keyword arguments to pass to the ``SQLDataModel`` constructor.

    Raises:
        ``ModuleNotFoundError``: If the required package ``openpyxl`` is not installed as determined by ``_has_xl`` flag.
        ``TypeError``: If the ``filename`` argument is not of type 'str' representing a valid Excel file path.
        ``Exception``: If an error occurs during Excel read and write operations related to openpyxl processing.

    Returns:
        ``SQLDataModel``: A new instance of ``SQLDataModel`` created from the Excel file.

    Examples:

    We'll use this Excel file, ``data.xlsx``, as the source for the below examples:

    ```text
            ┌───────┬─────┬────────┬───────────┐
            │ A     │ B   │ C      │ D         │
        ┌───┼───────┼─────┼────────┼───────────┤
        │ 1 │ Name  │ Age │ Gender │ City      │
        │ 2 │ John  │ 25  │ Male   │ Boston    │
        │ 3 │ Alice │ 30  │ Female │ Milwaukee │
        │ 4 │ Bob   │ 22  │ Male   │ Chicago   │
        │ 5 │ Sarah │ 35  │ Female │ Houston   │
        │ 6 │ Mike  │ 28  │ Male   │ Atlanta   │
        └───┴───────┴─────┴────────┴───────────┘
        [ Sheet1 ]
    ```
    
    Example 1: Load Excel file with default parameters

    ```python
        from SQLDataModel import SQLDataModel

        # Create the model using the default parameters
        sdm = SQLDataModel.from_excel('data.xlsx')

        # View imported data
        print(sdm)
    ```

    This will output all of the data starting from 'A1':

    ```shell
        ┌───────┬──────┬────────┬───────────┐
        │ Name  │  Age │ Gender │ City      │
        ├───────┼──────┼────────┼───────────┤
        │ John  │   25 │ Male   │ Boston    │
        │ Alice │   30 │ Female │ Milwaukee │
        │ Bob   │   22 │ Male   │ Chicago   │
        │ Sarah │   35 │ Female │ Houston   │
        │ Mike  │   28 │ Male   │ Atlanta   │
        └───────┴──────┴────────┴───────────┘
        [5 rows x 4 columns]
    ```

    Example 2: Load Excel file from specific worksheet

    ```python
        from SQLDataModel import SQLDataModel

        # Create the model from 'Sheet2'
        sdm = SQLDataModel.from_excel('data.xlsx', worksheet='Sheet2')

        # View imported data
        print(sdm)
    ```
    
    This will output the contents of 'Sheet2':

    ```shell
        ┌────────┬───────┐
        │ Gender │ count │
        ├────────┼───────┤
        │ Male   │     3 │
        │ Female │     2 │
        └────────┴───────┘
        [2 rows x 2 columns]
    ```

    Example 3: Load Excel file with custom headers starting from different row

    ```python
        from SQLDataModel import SQLDataModel

        # Use our own headers instead of the Excel ones
        new_cols = ['Col A', 'Col B', 'Col C', 'Col D']

        # Create the model starting from the 2nd row to ignore the original headers
        sdm = SQLDataModel.from_excel('data.xlsx', min_row=2, headers=new_cols)

        # View the data
        print(sdm)
    ```

    This will output the data with our renamed headers:

    ```shell
        ┌───────┬───────┬────────┬───────────┐
        │ Col A │ Col B │ Col C  │ Col D     │
        ├───────┼───────┼────────┼───────────┤
        │ John  │    25 │ Male   │ Boston    │
        │ Alice │    30 │ Female │ Milwaukee │
        │ Bob   │    22 │ Male   │ Chicago   │
        │ Sarah │    35 │ Female │ Houston   │
        │ Mike  │    28 │ Male   │ Atlanta   │
        └───────┴───────┴────────┴───────────┘
        [5 rows x 4 columns]
    ```
    
    Example 4: Load Excel file with specific subset of columns

    ```python
        from SQLDataModel import SQLDataModel

        # Create the model using the middle two columns only
        sdm = SQLDataModel.from_excel('data.xlsx', min_col=2, max_col=3)

        # View the data
        print(sdm)
    ```        

    This will output only the middle two columns:

    ```shell
        ┌──────┬────────┐
        │  Age │ Gender │
        ├──────┼────────┤
        │   25 │ Male   │
        │   30 │ Female │
        │   22 │ Male   │
        │   35 │ Female │
        │   28 │ Male   │
        └──────┴────────┘
        [5 rows x 2 columns]
    ```

    Note:
        - This method entirely relies on ``openpyxl``, see their amazing documentation for further information on Excel file handling in python.
        - If custom ``headers`` are provided using the default ``min_row``, then the original headers, if present, will be duplicated.
        - All indicies for ``min_row``, ``max_row``, ``min_col`` and ``max_col`` are 1-based instead of 0-based, again see ``openpyxl`` for more details.
        - See related :meth:`SQLDataModel.to_excel()` for exporting an existing ``SQLDataModel`` to Excel.

    Changelog:
        - Version 0.2.2 (2024-03-26):
            - New method.
    """
    return SQLDataModel.from_excel(filename, worksheet, min_row, max_row, min_col, max_col, headers, **kwargs)

def from_json(json_source:str|list|dict, encoding:str='utf-8', **kwargs) -> SQLDataModel:
    """
    Creates a new ``SQLDataModel`` instance from JSON file path or JSON-like source, flattening if required.

    Parameters:
        ``json_source`` (str | list | dict): The JSON source. If a string, it can represent a file path or a JSON-like object.
        ``encoding`` (str): The encoding to use when reading from a file. Defaults to 'utf-8'.
        ``**kwargs``: Additional keyword arguments to pass to the ``SQLDataModel`` constructor.

    Returns:
        ``SQLDataModel``: A new SQLDataModel instance created from the JSON source.

    Raises:
        ``TypeError``: If the ``json_source`` argument is not of type 'str', 'list', or 'dict'.
        ``OSError``: If related exception occurs when trying to open and read from ``json_source`` as file path.

    Examples:

    From JSON String Literal
    ------------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Sample JSON string
        json_data = '''[{
            "id": 1,
            "color": "red",
            "value": "#f00"
            },
            {   
            "id": 2,
            "color": "green",
            "value": "#0f0"
            },
            {
            "id": 3,
            "color": "blue",
            "value": "#00f"
        }]'''   

        # Create the model
        sdm = SQLDataModel.from_json(json_data)

        # View result
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────┬───────┬───────┐
        │   id │ color │ value │
        ├──────┼───────┼───────┤
        │    1 │ red   │ #f00  │
        │    2 │ green │ #0f0  │
        │    3 │ blue  │ #00f  │
        └──────┴───────┴───────┘
        [3 rows x 3 columns]

    ```
    From JSON-like Object
    ---------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # JSON-like sample
        json_data = [{
            "alpha": "A",
            "value": "1"
        },  
        {
            "alpha": "B",
            "value": "2"
        },
        {
            "alpha": "C",
            "value": "3"
        }]

        # Create the model
        sdm = SQLDataModel.from_json(json_data)

        # Output
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌───────┬───────┐
        │ alpha │ value │
        ├───────┼───────┤
        │ A     │ 1     │
        │ B     │ 2     │
        │ C     │ 3     │
        └───────┴───────┘
        [3 rows x 2 columns]

    ```

    From JSON file
    --------------

    ```python
        from SQLDataModel import SQLDataModel

        # JSON file path
        json_data = 'data/json-sample.json'

        # Create the model
        sdm = SQLDataModel.from_json(json_data, encoding='latin-1')

        # View output
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌──────┬────────┬───────┬─────────┐
        │   id │ color  │ value │ notes   │
        ├──────┼────────┼───────┼─────────┤
        │    1 │ red    │ #f00  │ primary │
        │    2 │ green  │ #0f0  │         │
        │    3 │ blue   │ #00f  │ primary │
        │    4 │ cyan   │ #0ff  │         │
        │    5 │ yellow │ #ff0  │         │
        │    5 │ black  │ #000  │         │
        └──────┴────────┴───────┴─────────┘
        [6 rows x 4 columns]
    ```

    Note:
        - If ``json_source`` is deeply-nested it will be flattened according to the staticmethod :meth:`SQLDataModel.flatten_json()`
        - If ``json_source`` is a JSON-like string object that is not an array, it will be wrapped according as an array.
    
    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    return SQLDataModel.from_json(json_source, encoding, **kwargs)

def from_html(html_source:str, encoding:str='utf-8', table_identifier:int|str=1, infer_types:bool=True, **kwargs) -> SQLDataModel:
    """
    Parses HTML table element from one of three possible sources: web page at url, local file at path, raw HTML string literal.
    If ``table_identifier`` is not specified, the first <table> element successfully parsed is returned, otherwise if ``table_identifier`` is a ``str``, the parser will return the corresponding 'id' or 'name' HTML attribute that matches the identifier specified. 
    If ``table_identifier`` is an ``int``, the parser will return the table matched as a sequential index after parsing all <table> elements from the top of the page down, starting at '1', the first table found. 
    By default, the first <table> element found is returned if ``table_identifier`` is not specified.

    Parameters:
        ``html_source`` (str): The HTML source, which can be a URL, a valid path to an HTML file, or a raw HTML string.
            If starts with 'http', the argument is considered a url and the table will be parsed from returned the web request.
            If is a valid file path, the argument is considered a local file and the table will be parsed from its html.
            If is not a valid url or path, the argument is considered a raw HTML string and the table will be parsed directly from the input.
        ``encoding`` (str): The encoding to use for reading HTML when ``html_source`` is considered a valid url or file path (default is 'utf-8').
        ``table_identifier`` (int | str): An identifier to specify which table to parse if there are multiple tables in the HTML source. Default is 1, returning the first table element found.
        ``infer_types`` (bool, optional): If column data types should be inferred in the return model. Default is True, meaning column types will be inferred otherwise are returned as 'str' types.
            If is ``int``, identifier is treated as the indexed location of the <table> element on the page from top to bottom starting from zero and will return the corresponding position when encountered.
            If is ``str``, identifier is treated as a target HTML 'id' or 'name' attribute to search for and will return the first case-insensitive match when encountered.
        ``**kwargs``: Additional keyword arguments to pass when using ``urllib.request.urlopen`` to fetch HTML from a URL.

    Returns:
        ``SQLDataModel``: A new SQLDataModel instance containing the data from the parsed HTML table.

    Raises:
        ``TypeError``: If ``html_source`` is not of type ``str`` representing a possible url, filepath or raw HTML stream.
        ``HTTPError``: Raised from ``urllib`` when ``html_source`` is considered a url and an HTTP exception occurs.
        ``URLError``: Raised from ``urllib`` when ``html_source`` is considered a url and a URL exception occurs.
        ``ValueError``: If no <table> elements are found or if the targeted ``table_identifier`` is not found.
        ``OSError``: Related exceptions that may be raised when ``html_source`` is considered a file path.

    Examples:

    From Website URL
    ----------------

    ```python            
        from SQLDataModel import SQLDataModel

        # From URL
        url = 'https://en.wikipedia.org/wiki/1998_FIFA_World_Cup'
        
        # Lets get the 95th table from the 1998 World Cup
        sdm = SQLDataModel.from_html(url, table_identifier=95)

        # View result:
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌────┬─────────────┬────┬────┬────┬────┬────┬────┬────┬─────┬──────┐
        │  R │ Team        │ G  │  P │  W │  D │  L │ GF │ GA │ GD  │ Pts. │
        ├────┼─────────────┼────┼────┼────┼────┼────┼────┼────┼─────┼──────┤
        │  1 │ France      │ C  │  7 │  6 │  1 │  0 │ 15 │  2 │ +13 │   19 │
        │  2 │ Brazil      │ A  │  7 │  4 │  1 │  2 │ 14 │ 10 │ +4  │   13 │
        │  3 │ Croatia     │ H  │  7 │  5 │  0 │  2 │ 11 │  5 │ +6  │   15 │
        │  4 │ Netherlands │ E  │  7 │  3 │  3 │  1 │ 13 │  7 │ +6  │   12 │
        │  5 │ Italy       │ B  │  5 │  3 │  2 │  0 │  8 │  3 │ +5  │   11 │
        │  6 │ Argentina   │ H  │  5 │  3 │  1 │  1 │ 10 │  4 │ +6  │   10 │
        │  7 │ Germany     │ F  │  5 │  3 │  1 │  1 │  8 │  6 │ +2  │   10 │
        │  8 │ Denmark     │ C  │  5 │  2 │  1 │  2 │  9 │  7 │ +2  │    7 │
        └────┴─────────────┴────┴────┴────┴────┴────┴────┴────┴─────┴──────┘
        [8 rows x 11 columns]
    ```

    From Local File 
    ---------------

    ```python            
        from SQLDataModel import SQLDataModel

        # From HTML file
        sdm = SQLDataModel.from_html('path/to/file.html')

        # View output
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌─────────────┬────────┬──────┐
        │ Team        │ Points │ Rank │
        ├─────────────┼────────┼──────┤
        │ Brazil      │ 63.7   │ 1    │
        │ England     │ 50.7   │ 2    │
        │ Spain       │ 50.0   │ 3    │
        │ Germany [a] │ 49.3   │ 4    │
        │ Mexico      │ 47.3   │ 5    │
        │ France      │ 46.0   │ 6    │
        │ Italy       │ 44.3   │ 7    │
        │ Argentina   │ 44.0   │ 8    │
        └─────────────┴────────┴──────┘
        [8 rows x 3 columns]
    ```

    From Raw HTML
    -------------

    ```python        
        from SQLDataModel import SQLDataModel

        # Raw HTML
        raw_html = 
        '''<table id="find-me">
            <tr>
                <th>Col 1</th>
                <th>Col 2</th>
            </tr>    
            <tr>
                <td>A</td>
                <td>1</td>
            </tr>
            <tr>
                <td>B</td>
                <td>2</td>
            </tr>
            <tr>
                <td>C</td>
                <td>3</td>
            </tr>                
        </table>'''

        # Create the model and search for id attribute
        sdm = SQLDataModel.from_html(raw_html, table_identifier="find-me")

        # View output
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌───┬───────┬───────┐
        │   │ Col 1 │ Col 2 │
        ├───┼───────┼───────┤
        │ 1 │ B     │ 2     │
        │ 2 │ C     │ 3     │
        └───┴───────┴───────┘
        [3 rows x 2 columns]
    ```

    Note:
        - ``**kwargs`` passed to method are used in ``urllib.request.urlopen`` if ``html_source`` is being considered as a web url.
        - ``**kwargs`` passed to method are used in ``open`` if ``html_source`` is being considered as a filepath.
        - The largest row size encountered will be used as the ``column_count`` for the returned ``SQLDataModel``, rows will be padded with ``None`` if less.
        - See :meth:`SQLDataModel.generate_html_table_chunks()` for initial source chunking before content fed to :mod:`SQLDataModel.HTMLParser`.

    Changelog:
        - Version 0.9.0 (2024-06-26):
            - Modified ``table_identifier`` default value to 1, changing from zero-based to one-based indexing for referencing target table in source to align with similar extraction methods throughout package.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """        
    return SQLDataModel.from_html(html_source, encoding, table_identifier, infer_types, **kwargs)

def from_latex(latex_source:str, table_identifier:int=1, encoding:str='utf-8', **kwargs) -> SQLDataModel:
    """
    Creates a new ``SQLDataModel`` instance from the provided LaTeX file or raw literal.
    
    Parameters:
        ``latex_source`` (str): The LaTeX source containing one or more LaTeX tables.
            If ``latex_source`` is a valid system filepath, source will be treated as a ``.tex`` file and parsed.
            If ``latex_source`` is not a valid filepath, source will be parsed as raw LaTeX literal.
        ``table_identifier`` (int, optional): The index position of the LaTeX table to extract. Default is 1.
        ``encoding`` (str, optional): The file encoding to use if source is a LaTex filepath. Default is 'utf-8';.
        ``**kwargs``: Additional keyword arguments to be passed to the ``SQLDataModel`` constructor.

    Returns:
        ``SQLDataModel``: The ``SQLDataModel`` instance created from the parsed LaTeX table.            

    Raises:
        ``TypeError``: If the ``latex_source`` argument is not of type 'str', or if the ``table_identifier`` argument is not of type 'int'.
        ``ValueError``: If the ``table_identifier`` argument is less than 1, or if no tables are found in the LaTeX source.
        ``IndexError``: If the ``table_identifier`` is greater than the number of tables found in the LaTeX source.
    
    Table Indicies:    
        - In the last example, ``sdm`` will contain the data from the second table found in the LaTeX content.
        - Tables are indexed starting from index 1 at the top of the LaTeX content, incremented as they are found.
        - LaTeX parsing stops after the table specified at ``table_identifier`` is found without parsing the remaining content.    

    Examples:

    From LaTeX literal
    ------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Raw LaTeX literal
        latex_content = '''
        \\begin{tabular}{|l|r|r|}
        \\hline
            {Name} & {Age} & {Height} \\\\
        \\hline
            John    &   30 &  175.30 \\\\
            Alice   &   28 &  162.00 \\\\
            Michael &   35 &  185.80 \\\\
        \\hline
        \\end{tabular}
        '''

        # Create the model from the LaTeX
        sdm = SQLDataModel.from_latex(latex_content)

        # View result
        print(sdm)
    ```

    This will output:

    ```shell            
        ┌─────────┬──────┬─────────┐
        │ Name    │  Age │  Height │
        ├─────────┼──────┼─────────┤
        │ John    │   30 │  175.30 │
        │ Alice   │   28 │  162.00 │
        │ Michael │   35 │  185.80 │
        └─────────┴──────┴─────────┘
        [3 rows x 3 columns]
    ```

    From LaTeX file
    ---------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Load LaTeX content from file
        latex_file = 'path/to/latex/file.tex'

        # Create the model using the path
        sdm = SQLDataModel.from_latex(latex_file)
    ```

    Specifying table identifier
    ---------------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Raw LaTeX literal with multiple tables
        latex_content = '''
        %% LaTeX with a Table

        \\begin{tabular}{|l|l|}
        \\hline
            {Header A} & {Header B} \\\\
        \\hline
            Value A1 & Value B1 \\\\
            Value A2 & Value B2 \\\\
        \\hline
        \\end{tabular}

        %% Then another Table

        \\begin{tabular}{|l|l|}
        \\hline
            {Header X} & {Header Y} \\\\
        \\hline
            Value X1 & Value Y1 \\\\
            Value X2 & Value Y2 \\\\
        \\hline
        \\end{tabular}
        '''

        # Create the model from the 2nd table
        sdm = SQLDataModel.from_latex(latex_content, table_identifier=2)

        # View output
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────────┬──────────┐
        │ Header X │ Header Y │
        ├──────────┼──────────┤
        │ Value X1 │ Value Y1 │
        │ Value X2 │ Value Y2 │
        └──────────┴──────────┘
        [2 rows x 2 columns]
    ```

    Note:
        - LaTeX tables are identified based on the presence of tabular environments: ``\\begin{tabular}...\\end{tabular}``.
        - The ``table_identifier`` specifies which table to extract when multiple tables are present, beginning at position '1' from the top of the source.
        - The provided ``kwargs`` are passed to the ``SQLDataModel`` constructor for additional parameters to the instance returned.           

    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    return SQLDataModel.from_latex(latex_source, table_identifier, encoding, **kwargs)

def from_markdown(markdown_source: str, table_identifier:int=1, **kwargs) -> SQLDataModel:
    """
    Creates a new ``SQLDataModel`` instance from the provided Markdown source file or raw content.
    
    If ``markdown_source`` is a valid system path, the markdown file will be parsed. 
    Otherwise, the provided string will be parsed as raw markdown.

    Parameters:
        ``markdown_source`` (str): The Markdown source file path or raw content.
        ``table_identifier`` (int, optional): The index position of the markdown table to extract. Default is 1.
        ``**kwargs``: Additional keyword arguments to be passed to the ``SQLDataModel`` constructor.

    Raises:
        ``TypeError``: If the ``markdown_source`` argument is not of type 'str', or if the ``table_identifier`` argument is not of type 'int'.
        ``ValueError``: If the ``table_identifier`` argument is less than 1, or if no tables are found in the markdown source.
        ``IndexError``: If the ``table_identifier`` is greater than the number of tables found in the markdown source.
    
    Returns:
        ``SQLDataModel``: The SQLDataModel instance created from the parsed markdown table.

    Table indicies:    
        - In the last example, ``sdm`` will contain the data from the second table found in the markdown content.
        - Tables are indexed starting from index 1 at the top of the markdown content, incremented as they are found.
        - Markdown parsing stops after the table specified at ``table_identifier`` is found without parsing the remaining content.

    Examples:

    From Markdown Literal
    ---------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Raw markdown literal
        markdown_content = '''
        | Item          | Price | # In stock |
        |---------------|-------|------------|
        | Juicy Apples  | 1.99  | 37         |
        | Bananas       | 1.29  | 52         |
        | Pineapple     | 3.15  | 14         |
        '''

        # Create the model from the markdown
        sdm = SQLDataModel.from_markdown(markdown_content)

        # View result
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────────────┬───────┬────────────┐
        │ Item         │ Price │ # In stock │
        ├──────────────┼───────┼────────────┤
        │ Juicy Apples │ 1.99  │ 37         │
        │ Bananas      │ 1.29  │ 52         │
        │ Pineapple    │ 3.15  │ 14         │
        └──────────────┴───────┴────────────┘
        [3 rows x 3 columns]
    
    ```

    From Markdown File
    ------------------

    ```python
        from SQLDataModel import SQLDataModel

        # Load markdown content from file
        markdown_file_path = 'path/to/markdown_file.md'

        # Create the model using the path
        sdm = SQLDataModel.from_markdown(markdown_file_path)
    ```

    Specifying Table Identifier
    ---------------------------

    ```python            
        from SQLDataModel import SQLDataModel

        # Raw markdown literal with multiple tables
        markdown_content = '''
        ### Markdown with a Table

        | Header A | Header B |
        |----------|----------|
        | Value A1 | Value B1 |
        | Value A2 | Value B2 |

        ### Then another Table

        | Header X | Header Y |
        |----------|----------|
        | Value X1 | Value Y1 |
        | Value X2 | Value Y2 |

        '''
        # Create the model from the 2nd table
        sdm = SQLDataModel.from_markdown(markdown_content, table_identifier=2)

        # View output
        print(sdm)
    ```

    This will output:

    ```shell
        ┌──────────┬──────────┐
        │ Header X │ Header Y │
        ├──────────┼──────────┤
        │ Value X1 │ Value Y1 │
        │ Value X2 │ Value Y2 │
        └──────────┴──────────┘
        [2 rows x 2 columns]
    ```

    Note:
        - Markdown tables are identified based on the presence of pipe characters ``|`` defining table cells.
        - The ``table_identifier`` specifies which table to extract when multiple tables are present, beginning at position '1' from the top of the source.
        - Escaped pipe characters ``\\|`` within the markdown are replaced with the HTML entity reference ``&vert;`` for proper parsing.
        - The provided ``kwargs`` are passed to the ``SQLDataModel`` constructor for additional parameters to the instance returned.

    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    return SQLDataModel.from_markdown(markdown_source, table_identifier, **kwargs)

def from_numpy(array, headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a ``SQLDataModel`` object created from the provided numpy ``array``.

    Parameters:
        ``array`` (numpy.ndarray): The numpy array to convert to a SQLDataModel.
        ``headers`` (list of str, optional): The list of headers to use for the SQLDataModel. If None, no headers will be used, and the data will be treated as an n-dimensional array. Default is None.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the numpy array.

    Raises:
        ``ModuleNotFoundError``: If the required package ``numpy`` is not found.
        ``TypeError``: If ``array`` argument is not of type ``numpy.ndarray``.
        ``DimensionError``: If ``array.ndim != 2`` representing a `(row, column)` tabular array.

    Example::

        import numpy as np
        from SQLDataModel import SQLDataModel

        # Sample array
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Create the model with custom headers
        sdm = SQLDataModel.from_numpy(arr, headers=['Col A', 'Col B', 'Col C])

        # View output
        print(sdm)
    
    This will output:

    ```shell            
        ┌───────┬───────┬───────┐
        │ Col A │ Col B │ Col C │
        ├───────┼───────┼───────┤
        │     1 │     2 │     3 │
        │     4 │     5 │     6 │
        │     7 │     8 │     9 │
        └───────┴───────┴───────┘
        [3 rows x 3 columns]
    ```

    Note:
        - Numpy array must have '2' dimensions, the first representing the rows, and the second the columns.
        - If no headers are provided, default headers will be generated as 'col_N' where N represents the column integer index.

    Changelog:
        - Version 0.1.3 (2023-10-15):
            - New method.
    """
    return SQLDataModel.from_numpy(array, headers, **kwargs)

def from_pandas(df, headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a ``SQLDataModel`` object created from the provided ``df`` representing a Pandas ``DataFrame`` object. Note that ``pandas`` must be installed in order to use this method.

    Parameters:
        ``df`` (pandas.DataFrame): The pandas DataFrame to convert to a SQLDataModel.
        ``headers`` (list[str], optional): The list of headers to use for the SQLDataModel. Default is None, using the columns from the ``df`` object.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the pandas DataFrame.

    Raises:
        ``ModuleNotFoundError``: If the required package ``pandas`` is not found.
        ``TypeError``: If ``df`` argument is not of type ``pandas.DataFrame``.

    Example::

        import pandas as pd
        from SQLDataModel import SQLDataModel

        # Create a pandas DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        
        # Create the model
        sdm = SQLDataModel.from_pandas(df)

    Note:
        - If ``headers`` are not provided, the existing pandas columns will be used as the new ``SQLDataModel`` headers.

    Changelog:
        - Version 0.1.3 (2023-10-15):
            - New method.
    """
    return SQLDataModel.from_pandas(df, headers, **kwargs)

def from_parquet(filename:str, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` instance from the specified parquet file.

    Parameters:
        ``filename`` (str): The file path to the parquet file, e.g., ``filename = 'user/data/titanic.parquet'``.
        ``**kwargs``: Additional keyword arguments to pass to the pyarrow ``read_table`` function, e.g., ``filters = [('Name','=','Alice')]``.

    Returns:
        ``SQLDataModel``: A new instance of ``SQLDataModel`` created from the parquet file.

    Raises:
        ``ModuleNotFoundError``: If the required package ``pyarrow`` is not installed as determined by ``_has_pa`` flag.
        ``TypeError``: If the ``filename`` argument is not of type 'str' representing a valid parquet file path.
        ``FileNotFoundError``: If the specified parquet ``filename`` is not found.
        ``Exception``: If any unexpected exception occurs during the file or parquet reading process.

    Example::

        from SQLDataModel import SQLDataModel

        # Sample parquet file
        pq_file = "titanic.parquet"

        # Create the model
        sdm = SQLDataModel.from_parquet(pq_file)

        # View column counts
        print(sdm.count())

    This will output:

    ```shell            
        ┌────┬─────────────┬──────┬────────┬───────┬───────┐
        │    │ column      │   na │ unique │ count │ total │
        ├────┼─────────────┼──────┼────────┼───────┼───────┤
        │  0 │ PassengerId │    0 │    891 │   891 │   891 │
        │  1 │ Survived    │    0 │      2 │   891 │   891 │
        │  2 │ Pclass      │    0 │      3 │   891 │   891 │
        │  3 │ Name        │    0 │    891 │   891 │   891 │
        │  4 │ Sex         │    0 │      2 │   891 │   891 │
        │  5 │ Age         │  177 │     88 │   714 │   891 │
        │  6 │ SibSp       │    0 │      7 │   891 │   891 │
        │  7 │ Parch       │    0 │      7 │   891 │   891 │
        │  8 │ Ticket      │    0 │    681 │   891 │   891 │
        │  9 │ Fare        │    0 │    248 │   891 │   891 │
        │ 10 │ Cabin       │  687 │    147 │   204 │   891 │
        │ 11 │ Embarked    │    2 │      3 │   889 │   891 │
        └────┴─────────────┴──────┴────────┴───────┴───────┘
        [12 rows x 5 columns]
    ```

    Note:
        - The pyarrow package is required to use this method as well as the :meth:`SQLDataModel.to_parquet()` method.
        - Once the file is read into pyarrow.parquet, the ``to_pydict()`` method is used to pass the data to this package's :meth:`SQLDataModel.from_dict()` method.
        - Titanic parquet data used in example available at https://www.kaggle.com/code/taruntiwarihp/titanic-dataset
    """
    return SQLDataModel.from_parquet(filename, **kwargs)

def from_pickle(filename:str=None, **kwargs) -> SQLDataModel:
    """
    Returns the ``SQLDataModel`` object from the provided ``filename``. If ``None``, the current directory will be scanned for the default :meth:`SQLDataModel.to_pickle()` format.

    Parameters:
        ``filename`` (str, optional): The name of the pickle file to load. If None, the current directory will be scanned for the default filename. Default is None.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor, these will override the properties loaded from ``filename``.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the loaded pickle file.

    Raises:
        ``TypeError``: If filename is provided but is not of type 'str' representing a valid pickle filepath.
        ``FileNotFoundError``: If the provided filename could not be found or does not exist.
    
    Example::

        from SQLDataModel import SQLDataModel

        headers = ['Name','Age','Sex']
        data = [('Alice', 20, 'F'), ('Bob', 25, 'M'), ('Gerald', 30, 'M')]

        # Create the model with sample data
        sdm = SQLDataModel(data=data, headers=headers)

        # Filepath
        pkl_file = 'people.sdm'

        # Save the model
        sdm.to_pickle(filename=pkl_file)

        # Load it back from file
        sdm = SQLDataModel.from_pickle(filename=pkl_file)

    Note:
        - All data, headers, data types and display properties will be saved when pickling.
        - Any additional ``kwargs`` provided will override those saved in the pickled model.
    """
    return SQLDataModel.from_pickle(filename, **kwargs)

def from_polars(df, headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a ``SQLDataModel`` object created from the provided ``df`` representing a Polars ``DataFrame`` object. Note that ``polars`` must be installed in order to use this method.

    Parameters:
        ``df`` (polars.DataFrame): The Polars DataFrame to convert to a SQLDataModel.
        ``headers`` (list[str], optional): The list of headers to use for the SQLDataModel. Default is None, using the columns from the ``df`` object.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the Polars DataFrame.

    Raises:
        ``ModuleNotFoundError``: If the required package ``polars`` is not found.
        ``TypeError``: If ``df`` argument is not of type ``polars.DataFrame``.

    Example::

        import polars as pl
        from SQLDataModel import SQLDataModel

        # Sample data
        data = {
            'Name': ['Beth', 'John', 'Alice', 'Travis'], 
            'Age': [27, 30, 28, 35], 
            'Height': [172.4, 175.3, 162.0, 185.8]
        }

        # Create the polars DataFrame
        df = pl.DataFrame(data)
        
        # Create a SQLDataModel object
        sdm = SQLDataModel.from_polars(df)
    
        # View result
        print(sdm)

    This will output a ``SQLDataModel`` constructed from the Polars ``df``:

    ```shell
        ┌────────┬─────┬─────────┐
        │ Name   │ Age │  Height │
        ├────────┼─────┼─────────┤
        │ Beth   │  27 │  172.40 │
        │ John   │  30 │  175.30 │
        │ Alice  │  28 │  162.00 │
        │ Travis │  35 │  185.80 │
        └────────┴─────┴─────────┘
        [4 rows x 3 columns]
    ```

    Note:
        - If ``headers`` are not provided, the columns from the provided DataFrame's columns will be used as the new ``SQLDataModel`` headers.
        - Polars uses different data types than those used by ``SQLDataModel``, see :meth:`SQLDataModel.set_column_dtypes()` for specific casting rules.
        - See related :meth:`SQLDataModel.to_polars()` for the inverse method of converting a ``SQLDataModel`` into a Polars ``DataFrame`` object.

    Changelog:
        - Version 0.3.8 (2024-04-12):
            - New method.
    """
    return SQLDataModel.from_polars(df, headers, **kwargs)

def from_pyarrow(table, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` instance from the provided Apache Arrow object.

    Parameters:
        ``table`` (pyarrow.lib.Table): Apache Arrow object from which to construct a new ``SQLDataModel`` object.
        ``**kwargs``: Additional keyword arguments to pass to the SQLDataModel constructor.

    Raises:
        ``ModuleNotFoundError``: If the required package ``pyarrow`` is not installed.
        ``TypeError``: If the provided `table` argument is not of type 'pyarrow.lib.Table'.

    Returns:
        ``SQLDataModel``: A new SQLDataModel instance representing the data in the provided Apache Arrow object.

    Example::

        import pyarrow as pa
        from SQLDataModel import SQLDataModel

        # Sample data
        data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Grade': [3.8, 3.9, 3.2],
        }

        # Create PyArrow table from data
        table = pa.Table.from_pydict(data)

        # Create model from PyArrow table
        sdm = SQLDataModel.from_pyarrow(table)

    This will output:

    ```shell
        ┌─────────┬──────┬───────┐
        │ Name    │  Age │ Grade │
        ├─────────┼──────┼───────┤
        │ Alice   │   25 │  3.80 │
        │ Bob     │   30 │  3.90 │
        │ Charlie │   35 │  3.20 │
        └─────────┴──────┴───────┘
        [3 rows x 3 columns]
    ```            

    Note:
        - To convert an existing ``SQLDataModel`` instance to Apache Arrow format, see :meth:`SQLDataModel.to_pyarrow()`.
        - This method is only for in-memory Apache Arrow table objects, for reading and writing parquet see :meth:`SQLDataModel.from_parquet()`.

    Changelog:
        - Version 0.3.0 (2024-03-31):
            - Renamed ``include_index`` parameter to ``index`` for package consistency.
        - Version 0.2.3 (2024-03-28):
            - New method.            
    """
    return SQLDataModel.from_pyarrow(table, **kwargs)

def from_shape(shape:tuple[int, int], fill:typing.Any=None, headers:list[str]=None, dtype:typing.Literal['bytes','date','datetime','float','int','str']=None, **kwargs) -> SQLDataModel:
    """
    Returns a SQLDataModel from shape ``(N rows, M columns)`` as a convenience method to quickly build a model through an iterative approach. 
    By default, no particular data type is assigned given the flexibility of ``sqlite3``, however one can be inferred by providing an initial ``fill`` value or explicitly by providing the ``dtype`` argument.

    Parameters:
        ``shape`` (tuple[int, int]): The shape to initialize the SQLDataModel with as ``(M, N)`` where ``M`` is the number of rows and ``N`` is the number of columns.
        ``fill`` (Any, optional): The scalar fill value to populate the new SQLDataModel with. Default is None, using SQL null values or deriving from ``dype`` if provided.
        ``headers`` (list[str], optional): The headers to use for the model. Default is None, incrementing headers ``0, 1, ..., N`` where ``N`` is the number of columns.
        ``dtype`` (str, optional): A valid python or SQL datatype to initialize the n-dimensional model with. Default is None, using the SQL text type.
        ``**kwargs``: Additional keyword arguments to pass to the ``SQLDataModel`` constructor.            

    Raises:
        ``TypeError``: If ``M`` or ``N`` are not of type 'int' representing a valid shape to initialize a SQLDataModel with.
        ``ValueError``: If ``M`` or ``N`` are not positive integer values representing valid nonzero row and column dimensions.
        ``ValueError``: If ``dtype`` is not a valid python or SQL convertible datatype to initialize the model with.

    Returns:
        ``SQLDataModel``: Instance with the specified number of rows and columns, initialized with by ``dtype`` fill values or with ``None`` values (default).

    Example::
    
        from SQLDataModel import SQLDataModel

        # Create a 3x3 model filled by 'X'
        sdm = SQLDataModel.from_shape((3,3), fill='X')            

        # View it
        print(sdm)
        
    This will output a 3x3 grid of 'X' characters:

    ```text
        ┌───┬─────┬─────┬─────┐
        │   │ 0   │ 1   │ 2   │
        ├───┼─────┼─────┼─────┤
        │ 0 │ X   │ X   │ X   │
        │ 1 │ X   │ X   │ X   │
        │ 2 │ X   │ X   │ X   │
        └───┴─────┴─────┴─────┘
        [3 rows x 3 columns]
    ```

    We can iteratively build the model from the shape dimensions:

    ```python
        from SQLDataModel import SQLDataModel

        # Define shape
        shape = (6,6)
        
        # Initialize the multiplcation table with integer dtypes
        mult_table = SQLDataModel.from_shape(shape=shape, dtype='int')

        # Construct the table values
        for x in range(shape[0]):
            for y in range(shape[1]):
                mult_table[x, y] = x * y
        
        # View the multiplcation table
        print(mult_table)
    ```
    
    This will output our 6x6 multiplication table:

    ```text
        ┌───┬─────┬─────┬─────┬─────┬─────┬─────┐
        │   │   0 │   1 │   2 │   3 │   4 │   5 │
        ├───┼─────┼─────┼─────┼─────┼─────┼─────┤
        │ 0 │   0 │   0 │   0 │   0 │   0 │   0 │
        │ 1 │   0 │   1 │   2 │   3 │   4 │   5 │
        │ 2 │   0 │   2 │   4 │   6 │   8 │  10 │
        │ 3 │   0 │   3 │   6 │   9 │  12 │  15 │
        │ 4 │   0 │   4 │   8 │  12 │  16 │  20 │
        │ 5 │   0 │   5 │  10 │  15 │  20 │  25 │
        └───┴─────┴─────┴─────┴─────┴─────┴─────┘
        [6 rows x 6 columns]
    ```

    Note:
        - If both ``fill`` and ``dtype`` are provided, the data type will be derived from ``type(fill)`` overriding or ignoring the specified ``dtype``.
        - If only ``dtype`` is provided, sensible default initialization fill values will be used to populate the model such as 0 or 0.0 for numeric and empty string or null for others.
        - For those data types not natively implemented by ``sqlite3`` such as ``date`` and ``datetime``, today's date and now's datetime will be used respectively for initialization values.

    Changelog:
        - Version 0.5.2 (2024-05-13):
            - Added ``shape`` parameter in lieu of separate ``n_rows`` and ``n_cols`` arguments.
            - Added ``fill`` parameter to populate resulting SQLDataModel with values to override type-specific initialization defaults.
            - Added ``headers`` parameter to explicitly set column names when creating the SQLDataModel.
            - Added ``**kwargs`` parameter to align more closely with usage patterns of other model initializing constructor methods.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    return SQLDataModel.from_shape(shape, fill, headers, dtype, **kwargs)

def from_sql(sql: str, con: sqlite3.Connection|typing.Any, dtypes:dict=None, **kwargs) -> SQLDataModel:
    """
    Create a ``SQLDataModel`` object by executing the provided SQL query using the specified SQL connection.
    If a single word is provided as the ``sql``, the method wraps it and executes a select all treating the text as the target table.
    
    Supported Connection APIs:
        - SQLite using ``sqlite3`` or url with format ``'file:///path/to/database.db'``
        - PostgreSQL using ``psycopg2`` or url with format ``'postgresql://user:pass@hostname:port/db'``
        - SQL Server ODBC using ``pyodbc`` or url with format ``'mssql://user:pass@hostname:port/db'``
        - Oracle using ``cx_Oracle`` or url with format ``'oracle://user:pass@hostname:port/db'``
        - Teradata using ``teradatasql`` or url with format ``'teradata://user:pass@hostname:port/db'``

    Parameters:
        ``sql`` (str): The SQL query to execute and use to create the SQLDataModel.
        ``con`` (sqlite3.Connection | Any): The database connection object or url, supported connection APIs are ``sqlite3``, ``psycopg2``, ``pyodbc``, ``cx_Oracle``, ``teradatasql``.
        ``dtypes`` (dict, optional): A dictionary of the format ``'column': 'python dtype'`` to assign to values. Default is None, mapping types from source connection.
        ``**kwargs``: Additional arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the executed SQL query.

    Raises:
        ``TypeError``: If dtypes argument is provided and is not of type ``dict`` representing python data types to assign to values.
        ``SQLProgrammingError``: If the provided SQL connection is not opened or valid, or the SQL query is invalid or malformed.
        ``ModuleNotFoundError``: If ``con`` is provided as a connection url and the specified scheme driver module is not found.
        ``DimensionError``: If the provided SQL query returns no data.

    Examples:

    From SQL Table
    --------------
    
    ```python
        from SQLDataModel import SQLDataModel

        # Single word parameter
        sdm = SQLDataModel.from_sql("table_name", sqlite3.Connection)
        
        # Equilavent query executed
        sdm = SQLDataModel.from_sql("select * from table_name", sqlite3.Connection)
    ```

    From SQLite Database
    --------------------

    ```python            
        import sqlite3
        from SQLDataModel import SQLDataModel

        # Create connection object
        sqlite_db_conn = sqlite3.connect('./database/users.db')

        # Basic usage with a select query
        sdm = SQLDataModel.from_sql("SELECT * FROM my_table", sqlite_db_conn)

        # When a single word is provided, it is treated as a table name for a select all query
        sdm_table = SQLDataModel.from_sql("my_table", sqlite_db_conn)
    ```
        
    From PostgreSQL Database
    ------------------------

    ```python
        import psycopg2
        from SQLDataModel import SQLDataModel

        # Create connection object
        pg_db_conn = psycopg2.connect('dbname=users user=postgres password=postgres')
        
        # Basic usage with a select query
        sdm = SQLDataModel.from_sql("SELECT * FROM my_table", pg_db_conn)

        # When a single word is provided, it is treated as a table name for a select all query
        sdm_table = SQLDataModel.from_sql("my_table", pg_db_conn)
    ```

    From SQL Server Databse
    -----------------------

    ```python
        import pyodbc
        from SQLDataModel import SQLDataModel

        # Create connection object
        con = pyodbc.connect("DRIVER={SQL Server};SERVER=host;DATABASE=db;UID=user;PWD=pw;")
        
        # Basic usage with a select query
        sdm = SQLDataModel.from_sql("SELECT * FROM my_table", con)

        # When a single word is provided, it is treated as a table name for a select all query
        sdm_table = SQLDataModel.from_sql("my_table", con)
    ```

    Note:
        - When ``con`` is provided as a string a connection will be attempted using :meth:`SQLDataModel._create_connection()` if the path does not exist, otherwise a ``sqlite3`` local connection will be attempted.
        - When ``con`` is provided as an object a connection is assumed to be open and valid, if a cursor cannot be created from the object an exception will be raised. 
        - Unsupported connection object will output a ``SQLDataModelWarning`` advising unstable or undefined behaviour.
        - The ``dtypes``, if provided, are only applied to ``sqlite3`` connection objects as remaining supported connections implement SQL to python adapters.
        - See related :meth:`SQLDataModel.to_sql()` for writing to SQL database connections.
        - See utility methods :meth:`SQLDataModel._parse_connection_url()` and :meth:`SQLDataModel._create_connection()` for implementation on creating database connections from urls.                      

    Changelog:
        - Version 0.9.1 (2024-06-27):
            - Modified handling of ``con`` parameter to allow database connection url to also be provided as ``'scheme://user:pass@host:port/db'``
        - Version 0.8.2 (2024-06-24):
            - Modified handling of ``con`` parameter to allow providing SQLite database filepath directly as string to instantiate connection.
        - Version 0.3.0 (2024-03-31):
            - Renamed ``sql_query`` parameter to ``sql`` for consistency with similar method arguments.  
    """
    return SQLDataModel.from_sql(sql, con, dtypes, **kwargs)

def from_text(text_source:str, table_identifier:int=1, encoding:str='utf-8', headers:list[str]=None, **kwargs) -> SQLDataModel:
    """
    Returns a new ``SQLDataModel`` generated from the provided ``text_source``, either as a file if the path exists, or from a raw string literal if the path does not exist.

    Parameters:
        ``text_source`` (str): The path to the tabular data file or a raw string literal containing tabular data.
        ``table_identifier`` (int, optional): The index position of the target table within the text source. Default is 1.
        ``encoding`` (str, optional): The encoding used to decode the text source if it is a file. Default is 'utf-8'.
        ``headers`` (list, optional): The headers to use for the provided data. Default is to use the first row.
        ``**kwargs``: Additional keyword arguments to be passed to the SQLDataModel constructor.

    Returns:
        ``SQLDataModel``: The SQLDataModel object created from the provided tabular data.

    Raises:
        ``TypeError``: If ``text_source`` is not a string or ``table_identifier`` is not an integer.
        ``ValueError``: If no tabular data is found in ``text_source``, if parsing fails to extract valid tabular data, or if the provided ``table_identifier`` is out of range.
        ``IndexError``: If the provided ``table_identifier`` exceeds the number of tables found in ``text_source``.
        ``Exception``: If an error occurs while attempting to read from or process the provided ``text_source``.

    Example::

        from SQLDataModel import SQLDataModel

        # Text source containing tabular data
        text_source = "/path/to/tabular_data.txt"

        # Create the model using the text source
        sdm = SQLDataModel.from_text(text_source, table_identifier=2)

    Note:
        - This method is made for parsing ``SQLDataModel`` formatted text, such as the kind generated with ``print(sdm)`` or the output created by the inverse method :meth:`SQLDataModel.to_text()`
        - For parsing other delimited tabular data, this method calls the related :meth:`SQLDataModel.from_csv()` method, which parses tabular data constructed with common delimiters.

    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    return SQLDataModel.from_text(text_source, table_identifier, encoding, headers, **kwargs)