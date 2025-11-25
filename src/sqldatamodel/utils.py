from __future__ import annotations

import re
import sqlite3
import datetime
from urllib.parse import urlparse
from collections.abc import Generator
from collections import namedtuple
from typing import Literal, Any, NamedTuple
from ast import literal_eval

from .exceptions import ErrorFormat
from .optionals import _has_dateutil

def generate_html_table_chunks(html_source:str) -> Generator[str, None, None]:
    """
    Generate chunks of HTML content for all ``<table>`` elements found in provided source as complete and unbroken chunks for parsing.

    Parameters:
        ``html_source`` (str): The raw HTML content from which to generate chunks.

    Raises:
        ``ValueError``: If zero ``<table>`` elements were found in ``html_source`` provided.

    Yields:
        ``str``: Chunks of HTML content containing complete ``<table>`` elements.

    Example::

        import sqldatamodel as sdm

        # HTML content to chunk
        html_source = '''
        <html> 
            <table><tr><td>Table 1</td></tr></table>
            ...
            <p>Non-table elements</p>
            ...
            <table><tr><td>Table 2</td></tr></table>
        </html>
        '''

        # Generate and view the returned table chunks
        for chunk in sdm.SQLDataModel.generate_html_table_chunks(html_source):
            print('Chunk:', chunk)
        
    This will output:

    ```text
        Chunk: <table><tr><td>Table 1</td></tr></table>
        Chunk: <table><tr><td>Table 2</td></tr></table>
    ```
    
    Note:
        - HTML content before the first ``<table>`` element and after the last ``</table>`` element is ignored and not yielded.
        - See :meth:`SQLDataModel.from_html()` for full implementation and how this function is used for HTML parsing.

    Changelog:
        - Version 0.2.1 (2024-03-24):
            - New method.
    """
    start_index, table_found = 0, False
    while True:
        start = html_source.find('<table', start_index)
        end = html_source.find('</table>', start)
        if start == -1 or end == -1:
            break
        yield html_source[start:end + 8] # len('</table>')
        start_index = end + 8
        table_found = True
    if not table_found:
        raise ValueError(
            ErrorFormat(f"ValueError: zero table elements found in provided source, confirm `html_source` is valid HTML or check integrity of data")
        )
        
def infer_str_type(obj:str, date_format:str='%Y-%m-%d', datetime_format:str='%Y-%m-%d %H:%M:%S') -> str:    
    """
    Infer the data type of the input object.

    Parameters:
        ``obj`` (str): The object for which the data type is to be inferred.
        ``date_format`` (str): The format string to use for parsing date values. Default is `'%Y-%m-%d'`.
        ``datetime_format`` (str): The format string to use for parsing datetime values. Default is `'%Y-%m-%d %H:%M:%S'`.

    Returns:
        ``str``: The inferred data type.
    
    Inference:
        - ``'str'``: If the input object is a string, or cannot be parsed as another data type.
        - ``'date'``: If the input object represents a date without time information.
        - ``'datetime'``: If the input object represents a datetime with both date and time information.
        - ``'int'``: If the input object represents an integer.
        - ``'float'``: If the input object represents a floating-point number.
        - ``'bool'``: If the input object represents a boolean value.
        - ``'bytes'``: If the input object represents a binary array.
        - ``'None'``: If the input object is None, empty, or not a string.

    Note:
        - This method attempts to infer the data type of the input object by evaluating its content.
        - If the input object is a string, it is parsed to determine whether it represents a date, datetime, integer, or float.
        - If the input object is not a string or cannot be parsed, its type is determined based on its Python type (bool, int, float, bytes, or None).
    
    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """        
    if obj is None or obj == '' or not isinstance(obj, str):
        return type(obj).__name__ if obj is not None else 'None'
    try:
        obj = literal_eval(obj)
    except (ValueError, SyntaxError):
        pass
    if isinstance(obj, (bool, bytes, int, type(None))):
        return type(obj).__name__ if obj is not None else 'None'
    if isinstance(obj, float):
        return 'int' if obj.is_integer() else 'float'
    try:
        if _has_dateutil:
            from dateutil.parser import parse as dateparser
            dt_obj = dateparser(obj, fuzzy=False, fuzzy_with_tokens=False, ignoretz=True)
        else:
            try:
                dt_obj = datetime.datetime.strptime(obj, datetime_format)
            except:
                dt_obj = datetime.datetime.strptime(obj, date_format)
        return 'datetime' if dt_obj.time() != datetime.time.min else 'date'
    except:
        pass
    return 'str'

def infer_types_from_data(input_data:list[list], date_format:str='%Y-%m-%d', datetime_format:str='%Y-%m-%d %H:%M:%S') -> list[str]:
    """
    Infer the best types of ``input_data`` by using a simple presence-based voting scheme. Sampling is assumed prior to function call, treating ``input_data`` as already a sampled subset from the original data.

    Parameters:
        ``input_data`` (list[list]): A list of lists containing the input data.
        ``date_format`` (str): The format string to use for parsing date values. Default is `'%Y-%m-%d'`.
        ``datetime_format`` (str): The format string to use for parsing datetime values. Default is `'%Y-%m-%d %H:%M:%S'`.

    Returns:
        ``list``: A list representing the best-matching inferred types for each column based on the sampled data.
        
    Note:
        - If multiple types are present in the samples, the most appropriate type is inferred based on certain rules.
        - If a column contains both ``date`` and ``datetime`` instances, the type is inferred as ``datetime``.
        - If a column contains both ``int`` and ``float`` instances, the type is inferred as ``float``.
        - If a column contains only ``str`` instances or multiple types with no clear choice, the type remains as ``str``.
        - See :meth:`SQLDataModel.infer_str_type()` for type determination process.
    
    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """        
    n_rows, n_cols = len(input_data), len(input_data[0])
    rand_dtypes = [list(set([infer_str_type(input_data[i][j], date_format=date_format, datetime_format=datetime_format) for i in range(n_rows)])) for j in range(n_cols)]
    parsed_dtypes = ['str' for _ in range(n_cols)] # default type
    for cid in range(n_cols):
        col_type_ocurx = rand_dtypes[cid]
        if ('None' in col_type_ocurx) and (len(col_type_ocurx) > 1):
            col_type_ocurx = [x for x in col_type_ocurx if x != 'None']
        num_types = len(col_type_ocurx)
        if 'str' in col_type_ocurx or num_types > 2 or num_types < 1:
            continue # leave as is, too many types
        if num_types == 1:
            parsed_dtypes[cid] = col_type_ocurx[0]
            continue
        if num_types == 2:
            if 'date' in col_type_ocurx and 'datetime' in col_type_ocurx:
                parsed_dtypes[cid] = 'datetime'
                continue
            if 'int' in col_type_ocurx and 'float' in col_type_ocurx:
                parsed_dtypes[cid] = 'float'
                continue
    return parsed_dtypes

def sqlite_cast_type_format(param:str='?', dtype:Literal['None','int','float','str','bytes','date','datetime','NoneType','bool']='str', as_binding:bool=True, as_alias:bool=False):
    """
    Formats the specified param to be cast consistently into the python type specified for insert params or as a named alias param.

    Parameters:
        ``param`` (str): The parameter to be formatted.
        ``dtype`` (Literal['None', 'int', 'float', 'str', 'bytes', 'date', 'datetime', 'NoneType', 'bool']): The python data type of the parameter as a string.
        ``as_binding`` (bool, optional): Whether to format as a binding parameter (default is True).
        ``as_alias`` (bool, optional): Whether to include an alias for the parameter (default is False).

    Returns:
        ``str``: The parameter formatted for SQL type casting.

    Note:
        - This function provides consistent formatting for casting parameters into specific data types for SQLite, changing it will lead to unexpected behaviors.
        - Used by :meth:`SQLDataModel.__init__()` with ``as_binding=True`` to allow parameterized inserts to cast to appropriate data type.

    Changelog:
        - Version 0.7.6 (2024-06-16):
            - Added support for additional date formats when ``dtype='date'`` including: ``'%m/%d/%Y'``, ``'%m-%d-%Y'``, ``'%m.%d.%Y'``, ``'%Y/%m/%d'``, ``'%Y-%m-%d'``, ``'%Y.%m.%d'``.
            - Modified behavior when ``dtype='bytes'`` to avoid the need for any additional checks after insert.
        - Version 0.3.3 (2024-04-03):
            - New method.
    """
    param_alias =  f'''as "{param}"''' if as_alias else ''''''
    if dtype in ('str','None','NoneType'):
        return '''CAST(NULLIF(NULLIF(?,'None'),'') as TEXT)''' if as_binding else f'''CAST(NULLIF(NULLIF("{param}",'None'),'') as TEXT) {param_alias}'''
    elif dtype == 'int':
        return '''CAST(NULLIF(NULLIF(?,'None'),'') as INTEGER)''' if as_binding else f'''CAST(NULLIF(NULLIF("{param}",'None'),'') as INTEGER) {param_alias}'''
    elif dtype == 'float':
        return '''CAST(NULLIF(NULLIF(?,'None'),'') as REAL)''' if as_binding else f'''CAST(NULLIF(NULLIF("{param}",'None'),'') as REAL) {param_alias}'''
    elif dtype == 'bytes':
        return """(SELECT CAST(CASE WHEN (SUBSTR(val,1,2) = 'b''' AND SUBSTR(val,-1,1) ='''') THEN SUBSTR(val,3,LENGTH(val)-3) ELSE NULLIF(NULLIF(val,'None'),'') END as BLOB) FROM (SELECT ? AS val))""" if as_binding else f"""CAST(CASE WHEN (SUBSTR("{param}",1,2) = 'b''' AND SUBSTR("{param}",-1,1) ='''') THEN SUBSTR("{param}",3,LENGTH("{param}")-3) ELSE NULLIF(NULLIF("{param}",'None'),'') END as BLOB) {param_alias} """
    elif dtype == 'date':
        return '''(SELECT CASE WHEN SUBSTR(val, 3, 1) = '-' THEN DATE(SUBSTR(val, 7, 4) || '-' ||SUBSTR(val, 1, 2) || '-' ||SUBSTR(val, 4, 2)) ELSE DATE(NULLIF(NULLIF(val, 'None'), '')) END FROM (SELECT REPLACE(REPLACE(?, '/', '-'), '.', '-') AS val))''' if as_binding else f'''CASE WHEN SUBSTR(REPLACE(REPLACE("{param}",'/','-'),'.','-'),3,1) = '-' THEN DATE(SUBSTR(REPLACE(REPLACE("{param}",'/','-'),'.','-'), 7, 4) || '-' || SUBSTR(REPLACE(REPLACE("{param}",'/','-'),'.','-'), 1, 2) || '-' || SUBSTR(REPLACE(REPLACE("{param}",'/','-'),'.','-'), 4, 2)) ELSE DATE(NULLIF(NULLIF(REPLACE(REPLACE("{param}",'/','-'),'.','-'),'None'),'')) END {param_alias} '''
    elif dtype == 'datetime':
        return '''DATETIME(NULLIF(NULLIF(?,'None'),''))''' if as_binding else f'''DATETIME(NULLIF(NULLIF("{param}",'None'),'')) {param_alias}'''
    elif dtype == 'bool':
        return '''CAST(CASE COALESCE(NULLIF(?,''),'None') WHEN 'None' THEN null WHEN 'False' THEN 0 WHEN '0' THEN 0 WHEN 0 THEN 0 ELSE 1 END as INTEGER)''' if as_binding else f'''CAST(CASE coalesce(NULLIF("{param}",''),'None') WHEN 'None' THEN null WHEN 'False' THEN 0 WHEN '0' THEN 0 WHEN 0 THEN 0 ELSE 1 END as INTEGER) {param_alias}'''
    else:
        return '''NULLIF(NULLIF(?,'None'),'')''' if as_binding else f'''NULLIF(NULLIF("{param}",'None'),'') {param_alias}'''
    
def sqlite_printf_format(column:str, dtype:str, max_pad_width:int, float_precision:int=4, alignment:str=None, escape_newline:bool=False, truncation_chars:str='⠤⠄') -> str:
    """
    Formats SQLite SELECT clauses based on column parameters to provide preformatted fetches, providing most of the formatting for ``repr`` output.

    Parameters:
        ``column`` (str): The name of the column.
        ``dtype`` (str): The data type of the column ('float', 'int', 'bytes', 'index', or 'custom').
        ``max_pad_width`` (int): The maximum width to pad the output.
        ``float_precision`` (int, optional): The precision for floating-point numbers (default is 4).
        ``alignment`` (str, optional): The alignment of the output ('<', '>', or None for no alignment).
        ``escape_newline`` (bool, optional): If newline characters should be escaped when ``dtype = 'str'``. Default is False.
        ``truncation_chars`` (str, optional): Truncation characters to use if column exceeds maximum width. Default is ``'⠤⠄'``.

    Returns:
        ``str``: The formatted SELECT clause for SQLite.

    Note:
        - This function generates SQLite SELECT clauses for single column only.
        - The output preformats SELECT result to fit ``repr`` method for tabular output.
        - The return ``str`` is not valid SQL by itself, representing only the single column select portion.

    Changelog:
        - Version 0.11.0 (2024-07-05):
            - Added ``truncation_chars`` keyword argument to allow custom truncation characters when column value exceeds maximum width.
        - Version 0.10.4 (2024-07-03):
            - Added ``escape_newline`` keyword argument to escape newline characters to prevent wrapping lines when called by :meth:`SQLDataModel.__repr__()`
        - Version 0.7.0 (2024-06-08):
            - Added preemptive check for custom flag to pass through string formatting directly to support horizontally centered repr changes.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    if dtype == 'custom':
        return f"""printf('%{max_pad_width}s', '{column}') """ # treats column as literal argument for string format substitution
    if alignment is None: # dynamic alignment
        if dtype == 'float':
            select_item_fmt = f"""(CASE WHEN "{column}" IS NULL THEN printf('%{max_pad_width}s', '') WHEN LENGTH(printf('% .{float_precision}f',"{column}")) <= {max_pad_width} THEN printf('%.{max_pad_width}s', printf('% {max_pad_width}.{float_precision}f',"{column}")) ELSE SUBSTR(printf('% .{float_precision}f',"{column}"),1,{max_pad_width}-{len(truncation_chars)}) || '{truncation_chars}' END)"""
        elif dtype == 'int':
            select_item_fmt = f"""printf('%{max_pad_width}s', CASE WHEN length("{column}") <= ({max_pad_width}) THEN "{column}" ELSE substr("{column}",1,({max_pad_width})-{len(truncation_chars)})||'{truncation_chars}' END) """
        elif dtype == 'bytes':
            select_item_fmt = f"""printf('%!-{max_pad_width}s', CASE WHEN (length("{column}")+3) <= ({max_pad_width}) THEN ('b'''||"{column}"||'''') ELSE substr('b'''||"{column}",1,({max_pad_width})-{len(truncation_chars)})||'{truncation_chars}' END) """
        elif dtype == 'index':
            select_item_fmt = f"""printf('%{max_pad_width}s', "{column}") """
        else:
            column = f'"{column}"' if not escape_newline else  "".join((f'REPLACE("{column}",',"'\n','\\n')"))
            select_item_fmt = f"""printf('%!-{max_pad_width}s', CASE WHEN length({column}) <= ({max_pad_width}) THEN {column} ELSE substr({column},1,({max_pad_width})-{len(truncation_chars)})||'{truncation_chars}' END) """
        return select_item_fmt
    else: # left, right aligned
        if alignment in ("<", ">"):
            dyn_left_right = '-' if alignment == '<' else ''
            if dtype == 'float':
                select_item_fmt = f"""(CASE WHEN "{column}" IS NULL THEN printf('%{dyn_left_right}{max_pad_width}s', '') WHEN LENGTH(printf('%{dyn_left_right}.{float_precision}f',"{column}")) <= {max_pad_width} THEN printf('%.{max_pad_width}s', printf('%{dyn_left_right}{max_pad_width}.{float_precision}f',"{column}")) ELSE SUBSTR(printf('%{dyn_left_right}.{float_precision}f',"{column}"),1,{max_pad_width}-{len(truncation_chars)}) || '{truncation_chars}' END)"""
            elif dtype == 'bytes':
                select_item_fmt = f"""printf('%!{dyn_left_right}{max_pad_width}s', CASE WHEN (length("{column}")+3) <= ({max_pad_width}) THEN ('b'''||"{column}"||'''') ELSE substr('b'''||"{column}"||'''',1,{max_pad_width}-{len(truncation_chars)})||'{truncation_chars}' END) """
            elif dtype == 'index':
                select_item_fmt = f"""printf('%{max_pad_width}s', "{column}") """
            else:
                column = f'"{column}"' if not escape_newline else  "".join((f'REPLACE("{column}",',"'\n','\\n')"))
                select_item_fmt = f"""printf('%!{dyn_left_right}{max_pad_width}s', CASE WHEN length({column}) <= ({max_pad_width}) THEN {column} ELSE substr({column},1,({max_pad_width})-{len(truncation_chars)})||'{truncation_chars}' END) """
            return select_item_fmt            
        else: # center aligned
            if dtype == 'index':
                select_item_fmt = f"""printf('%{max_pad_width}s', "{column}") """
            else:
                # Negative numbers favor right side, positive the left when no even split is possible, use ON_UNEVEN_SPLIT_{DTYPE} = +1 to replicate pythons uneven break behavior, which is used by the headers
                ON_UNEVEN_SPLIT_FLOAT = 1 # break left
                ON_UNEVEN_SPLIT_INT = 1 # break left
                ON_UNEVEN_SPLIT_BYTES = 0 # break arbitrarily
                ON_UNEVEN_SPLIT_REMAINING = 1 # break left
                if dtype == 'float':
                    col_discriminator = f"""(CASE WHEN LENGTH(printf('%.{float_precision}f',"{column}")) <= {max_pad_width} THEN (printf('%*.{float_precision}f',{max_pad_width}-(({max_pad_width}+{ON_UNEVEN_SPLIT_FLOAT} /* [Favor left (-) or right (+) on uneven split] */ - length(printf('%.{float_precision}f',"{column}")))/2),"{column}")) ELSE SUBSTR(printf('%.{float_precision}f',"{column}"),1,{max_pad_width}-{len(truncation_chars)}) || '{truncation_chars}' END)"""
                elif dtype == 'int':
                    col_discriminator = f"""(CASE WHEN LENGTH("{column}") <= {max_pad_width} THEN printf('%!*s',{max_pad_width}-(({max_pad_width}+{ON_UNEVEN_SPLIT_INT} /* [Favor left (-) or right (+) on uneven split] */ - length("{column}"))/2),"{column}") ELSE SUBSTR(printf('%!s',"{column}"),1,{max_pad_width}-{len(truncation_chars)})||'{truncation_chars}' END)"""                        
                elif dtype == 'bytes':
                    col_discriminator = f"""(CASE WHEN LENGTH("{column}")+3 <= {max_pad_width} THEN printf('%!*s',{max_pad_width}-(({max_pad_width}+{ON_UNEVEN_SPLIT_BYTES} /* [Favor left (-) or right (+) on uneven split] */ - (length("{column}")+3))/2),('b'''||"{column}"||'''')) ELSE SUBSTR('b'''||"{column}"||'''',1,{max_pad_width}-{len(truncation_chars)})||'{truncation_chars}' END)"""
                else:
                    column = f'"{column}"' if not escape_newline else  "".join((f'REPLACE("{column}",',"'\n','\\n')"))
                    col_discriminator = f"""(CASE WHEN LENGTH({column}) <= {max_pad_width} THEN printf('%!*s',{max_pad_width}-(({max_pad_width}+{ON_UNEVEN_SPLIT_REMAINING} /* [Favor left (-) or right (+) on uneven split] */ - length({column}))/2),{column}) ELSE SUBSTR(printf('%!s',{column}),1,{max_pad_width}-{len(truncation_chars)})||'{truncation_chars}' END)"""
                    string_only_select_item_fmt = f"""CASE WHEN {column} IS NULL THEN printf('%{max_pad_width}s',"") ELSE printf('%!-{max_pad_width}.{max_pad_width}s',{col_discriminator}) END"""
                    return string_only_select_item_fmt
                select_item_fmt = f"""CASE WHEN "{column}" IS NULL THEN printf('%{max_pad_width}s',"") ELSE printf('%!-{max_pad_width}.{max_pad_width}s',{col_discriminator}) END"""
    return select_item_fmt

def alias_duplicates(headers:list) -> Generator:
    """
    Rename duplicate column names in a given list by appending an underscore and a numerical suffix.

    Parameters:
        ``headers`` (list): A list of column names that require parsing for duplicates.

    Yields:
        ``Generator``: A generator object that yields the original or modified column names.

    Example::

        import sqldatamodel as sdm

        # Original list of column names with duplicates
        original_headers = ['ID', 'Name', 'Amount', 'Name', 'Date', 'Amount']

        # Use the static method to return a generator for the duplicates
        renamed_generator = sdm.SQLDataModel.alias_duplicates(original_headers)

        # Obtain the modified column names
        modified_headers = list(renamed_generator)

        # View modified column names
        print(modified_headers)

        # Output
        modified_headers = ['ID', 'Name', 'Amount', 'Name_2', 'Date', 'Amount_2']
    
    Example of implementation for SQLDataModel:

    ```python
        # Given a list of headers
        original_headers = ['ID', 'ID', 'Name', 'Name', 'Name', 'Unique']

        # Create a separate list for aliasing duplicates
        aliased_headers = list(SQLDataModel.alias_duplicates(original_headers))

        # View aliases
        for col, alias in zip(original_headers, aliased_headers):
            print(f"{col} as {alias}")
    ```

    This will output:

    ```shell
        ID as ID
        ID as ID_2
        Name as Name
        Name as Name_2
        Name as Name_3
        Unique as Unique
    ```

    Note:
        - Used by :meth:`SQLDataModel.execute_fetch()` when column selection is unknown and may require duplicate aliasing.

    Changelog:
        - Version 0.3.4 (2024-04-05):
            - Modified to re-alias partially aliased input to prevent runaway incrementation on suffixes.
        - Version 0.1.9 (2024-03-19):
            - New method.
    """        
    dupes = {}
    for col in headers:
        if col in dupes:
            dupes[col] += 1
            new_col = f"{col}_{dupes[col]}"
            while new_col in headers:
                dupes[col] += 1
                new_col = f"{col}_{dupes[col]}"
            yield new_col
        else:
            dupes[col] = 1
            yield col

def flatten_json(json_source:list|dict, flatten_rows:bool=True, level_sep:str='_', key_prefix:str=None) -> dict:
    """
    Parses raw JSON data and flattens it into a dictionary with optional normalization.

    Parameters:
        ``json_source`` (dict | list): The raw JSON data to be parsed.
        ``flatten_rows`` (bool): If True, the data will be normalized into columns and rows. If False,
            columns will be concatenated from each row using the specified `key_prefix`.
        ``level_sep`` (str): Separates nested levels from other levels and used to concatenate prefix to column.
        ``key_prefix`` (str): The prefix to prepend to the JSON keys. If None, an empty string is used.

    Returns:
        ``dict``: A flattened dictionary representing the parsed JSON data.

    Example::
    
        import sqldatamodel as sdm
    
        # Sample JSON
        json_source = [{
            "alpha": "A",
            "value": 1
        },  
        {
            "alpha": "B",
            "value": 2
        },
        {
            "alpha": "C",
            "value": 3
        }]

        # Flatten JSON with normalization
        flattened_data = sdm.SQLDataModel.flatten_json(json_data, flatten_rows=True)

        # Format of result
        flattened_data = {"alpha": ['A','B','C'], "value": [1, 2, 3]}

        # Alternatively, flatten columns without rows and adding a prefix
        flattened_data = sdm.SQLDataModel.flatten_json(raw_input,key_prefix='row_',flatten_rows=False)

        # Format of result
        flattened_data = {'row_0_alpha': 'A', 'row_0_value': 1, 'row_1_alpha': 'B', 'row_1_value': 2, 'row_2_alpha': 'C', 'row_2_value': 3}

    Note:
        - Used by :meth:`SQLDataModel.from_dict()` to flatten deeply nested JSON objects into 2 dimensions when encountered.

    Changelog:
        - Version 0.1.9 (2024-03-19):
            - New method.
    """
    if isinstance(json_source, dict):
        json_source = [json_source]
    key_prefix = key_prefix if key_prefix is not None else ''
    headers, rows, output = [], [], {}
    def flatten(x:list|dict, pref:str='', cols:list=[], rows:list=[]):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], pref + a + level_sep, cols=cols)
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, pref + f"{i}" + level_sep, cols=cols)
                if i not in rows:
                    rows.append(i)
                i += 1
        else:
            output[pref[:-1]] = x
            col_id = pref[len(key_prefix):-1].split(level_sep,1)[-1]
            if col_id not in cols:
                cols.append(col_id)
    flatten(json_source, pref=key_prefix, cols=headers, rows=rows)
    if not flatten_rows:
        return output
    flat_dict = {col:[] for col in headers}
    for col in headers:
        for rfound in rows:
            rowcol = f"{key_prefix}{rfound}_{col}"
            if rowcol in output:
                flat_dict[col].append(output[rowcol])
            else:
                flat_dict[col].append(None)
    return flat_dict

def _parse_connection_url(url:str) -> NamedTuple:
    """
    Parses database connection url into component parameters and returns the parsed components as a NamedTuple

    Parameters:
        ``url`` (str): The url connection string provided in the format of ``'scheme://user:pass@host:port/path'``
    
    Raises:
        ``AttributeError``: If ``url`` provided could not be parsed into expected component properties.
        ``ValueError``: If scheme is not provided or is not one of the currently supported driver formats or module aliases below
            SQLite: ``'file'`` or ``'sqlite3'``
            PostgreSQL: ``'postgresql'`` or ``'psycopg2'``
            SQL Server ODBC: ``'mssql'`` or ``'pyodbc'``
            Oracle: ``'oracle'`` or ``'cx_oracle'``
            Teradata: ``'teradata'`` or ``'teradatasql'``

    Returns:
        ``ConnectionDetails``: The parsed details as ``ConnectionDetails('scheme', 'user', 'cred', 'host', 'port', 'db')``
    
    Supported Formats:
        - SQLite using ``sqlite3`` with format ``'file:///path/to/database.db'``
        - PostgreSQL using ``psycopg2`` with format ``'postgresql://user:pass@hostname:port/db'``
        - SQL Server ODBC using ``pyodbc`` with format ``'mssql://user:pass@hostname:port/db'``
        - Oracle using ``cx_Oracle`` with format ``'oracle://user:pass@hostname:port/db'``
        - Teradata using ``teradatasql`` with format ``'teradata://user:pass@hostname:port/db'``
    
    Example::

        import sqldatamodel as sdm

        # SQLite connection url
        url = 'file:///home/database/users.db'

        # Parse the connection properties
        url_props = sdm.SQLDataModel._parse_connection_url(url)

        # View attributes
        print(url_props)
    
    This will output the connection details for a local SQLite database file:

    ```text
        ConnectionDetails(
            scheme='file', user=None, cred=None, host=None, port=None, db='/home/database/users.db'
        )
    ```

    PostgreSQL connections can be parsed from a valid format:

    ```python
        import sqldatamodel as sdm

        # PostgreSQL connection url
        url = 'postgresql://scott:tiger@12.34.56.78:5432/pgdb'

        # Parse the connection properties
        url_props = sdm.SQLDataModel._parse_connection_url(url)

        # View attributes
        print(url_props)
    ```

    This will output the connection details for a PostgreSQL connection:

    ```text
        ConnectionDetails(
            scheme='postgresql', user='scott', cred='tiger', host='12.34.56.78', port=5432, db='pgdb'
        )
    ```

    Note:
        - This method is used by :meth:`SQLDataModel._create_connection()` to parse details from url and create a connection object.
        - This method can be used by :meth:`SQLDataModel.from_sql()` and :meth:`SQLDataModel.to_sql()` to parsed connection details when connection parameter provided as string.

    Changelog:
        - Version 0.9.3 (2024-06-28):
            - Modified behavior when ``scheme`` is not provided, treating as file path when parsed in absence of auth related properties to retain prior version behavior of creating new sqlite3 database file when path is provided.
            - Added driver module names as valid aliases for relevant connection drivers, valid schemes now include 'file', 'sqlite3', 'postgresql', 'psycopg2', 'mssql', 'pyodbc', 'oracle', 'cx_oracle', 'teradata', 'teradatasql'
        - Version 0.9.2 (2024-06-27):
            - Modified to use ``urllib.parse.urlparse`` instead of added 3rd party package dependency.
        - Version 0.9.1 (2024-06-27):
            - New method.
    """
    ConnectionDetails = namedtuple('ConnectionDetails', ['scheme', 'user', 'cred', 'host', 'port', 'db'])
    # valid_connection_drivers: file|sqlite3, mssql|pyodbc, postgresql|psycopg2, oracle|cx_oracle or teradata|teradatasql
    url = url.replace("cx_oracle", "cxoracle", 1) # Handle cx_oracle being interpreted as relative filepath
    is_windows = re.match(r"^[a-zA-Z]:\\", url) # Handle Windows paths by detecting them using a regex
    if is_windows:
        url = "".join(("file:///", url.replace('\\', '/')))
    try:
        url_details = urlparse(url)
    except Exception as e:
        raise type(e)(
            ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to parse connection url: '{url}'")
        ).with_traceback(e.__traceback__) from None         
    user, cred, host = url_details.username, url_details.password, url_details.hostname
    port, db = url_details.port, url_details.path
    scheme = url_details.scheme.lower() if url_details.scheme else 'file'
    if scheme not in ('file','sqlite3','postgresql','psycopg2','mssql','pyodbc','oracle','cxoracle','teradata','teradatasql'):
        raise ValueError(
            ErrorFormat(f"ValueError: invalid scheme '{scheme}', scheme must be one of 'file', 'postgresql', 'mssql', 'oracle' or 'teradata'")
        )        
    db = db.lstrip('/') if ((db is not None and scheme not in ('file','sqlite3')) or (scheme in ('file','sqlite3') and is_windows)) else db
    return ConnectionDetails(scheme=scheme, user=user, cred=cred, host=host, port=port, db=db)

def _create_connection(url:str) -> sqlite3.Connection|Any:
    """
    Parses database connection url into component parameters and creates the specified connection.
    
    Parameters:
        ``url`` (str): The url connection string provided in the format of ``'scheme://user:pass@host:port/path'``
    
    Raises:
        ``ValueError``: If scheme is provided and not one of the currently supported driver formats.
        ``ModuleNotFoundError``: If required driver for specified scheme is not installed or not found.

    Returns:
        ``Connection`` (sqlite3.Connection | Any): The driver connection object for the scheme specified.
    
    Supported Formats:
        - SQLite using ``sqlite3`` with format ``'file:///path/to/database.db'``
        - PostgreSQL using ``psycopg2`` with format ``'postgresql://user:pass@hostname:port/db'``
        - SQL Server ODBC using ``pyodbc`` with format ``'mssql://user:pass@hostname:port/db'``
        - Oracle using ``cx_Oracle`` with format ``'oracle://user:pass@hostname:port/db'``
        - Teradata using ``teradatasql`` with format ``'teradata://user:pass@hostname:port/db'``

    Examples:

    SQLite
    ------

    ```python
        import sqldatamodel as sdm

        # SQLite connection url
        url = 'file:///home/database/users.db'

        # Parse and create sqlite3 connection
        conn = sdm.SQLDataModel._create_connection(url)
    ```
    
    PostgreSQL
    ----------

    ```python
        import sqldatamodel as sdm

        # Sample url
        url = 'postgresql://scott:tiger@12.34.56.78:5432/pgdb'

        # Parse and create psycopg2 connection
        conn = sdm.SQLDataModel._create_connection(url)
    ```

    Note:
        - Used by :meth:`SQLDataModel.from_sql()` and :meth:`SQLDataModel.to_sql()` to parse and create connection objects from url.
        - See :meth:`SQLDataModel._parse_connection_url()` for implementation on parsing url properties from connection string.

    Changelog:
        - Version 0.9.2 (2024-06-27):
            - New method.
    """
    url_props = _parse_connection_url(url)
    driver = url_props.scheme
    # Valid drivers: ('file','sqlite3') or ('postgresql','psycopg2') or ('mssql','pyodbc') or ('oracle','cxoracle') or ('teradata','teradatasql')
    if driver in ('file', 'sqlite3'):
        try:
            conn = sqlite3.connect(url_props.db)
        except Exception as e:
            raise type(e)(
                ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to open sqlite3 connection")
            ).with_traceback(e.__traceback__) from None                  
    elif driver in ('postgresql', 'psycopg2'):
        try:
            import psycopg2
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                ErrorFormat(f"ModuleNotFoundError: required package not found, 'psycopg2' must be installed in order to use a PostgreSQL connection driver")
            ) from None
        try:
            conn = psycopg2.connect(host=url_props.host,database=url_props.db,user=url_props.user,password=url_props.cred,port=url_props.port)
        except Exception as e:
            raise type(e)(
                ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to open psycopg2 connection")
            ).with_traceback(e.__traceback__) from None                  
    elif driver in ('mssql', 'pyodbc'):
        try:
            import pyodbc # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                ErrorFormat(f"ModuleNotFoundError: required package not found, 'pyodbc' must be installed in order to use a SQL Servier connection driver")
            ) from None
        try:
            conn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',server=f'{url_props.host},{url_props.port}' if url_props.port else f'{url_props.host}' ,database=url_props.db,uid=url_props.user,pwd=url_props.cred)
        except Exception as e:
            raise type(e)(
                ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to open pyodbc connection")
            ).with_traceback(e.__traceback__) from None                   
    elif driver in ('oracle', 'cxoracle'):
        try:
            import cx_Oracle # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                ErrorFormat(f"ModuleNotFoundError: required package not found, 'cx_Oracle' must be installed in order to use an Oracle connection driver")
            ) from None        
        try:    
            conn = cx_Oracle.connect(user=url_props.user, password=url_props.cred, dsn=f"{url_props.host}:{url_props.port}/{url_props.db}")            
        except Exception as e:
            raise type(e)(
                ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to open cx_Oracle connection")
            ).with_traceback(e.__traceback__) from None                   
    elif driver in ('teradata', 'teradatasql'):
        try:
            import teradatasql # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                ErrorFormat(f"ModuleNotFoundError: required package not found, 'teradatasql' must be installed in order to use an Oracle connection driver")
            ) from None        
        try:
            conn = teradatasql.connect(host=url_props.host, user=url_props.user, password=url_props.cred, encryptdata='true')
        except Exception as e:
            raise type(e)(
                ErrorFormat(f"{type(e).__name__}: {e} encountered when trying to open teradatasql connection")
            ).with_traceback(e.__traceback__) from None                
    return conn
