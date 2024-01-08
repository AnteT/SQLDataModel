from __future__ import annotations
import sqlite3, os, csv, sys, datetime, pickle, re, shutil, datetime
from typing import Generator, Callable, Literal, Iterator
from collections import namedtuple
from pathlib import Path
from ast import literal_eval

from SQLDataModel.exceptions import DimensionError, SQLProgrammingError
from SQLDataModel.ANSIColor import ANSIColor
from SQLDataModel.StandardDeviation import StandardDeviation

try:
    from dateutil.parser import parse as dateparser
    _has_dateutil = True
except ModuleNotFoundError:
    _has_dateutil = False
    
try:
    import numpy as _np
    _has_np = True
except ModuleNotFoundError:
    _has_np = False

try:
    import pandas as _pd
    _has_pd = True
except ModuleNotFoundError:
    _has_pd = False

class SQLDataModel:
    """
    ## SQLDataModel 
    Primary class for the package of the same name. Its meant to provide a fast & light-weight alternative to the common pandas, numpy and sqlalchemy setup for moving data in a source/destination agnostic manner. It is not an ORM, any modifications outside of basic joins, group bys and table alteration requires knowledge of SQL. The primary use-case envisaged by the package is one where a table needs to be ETL'd from location A to destination B with arbitrary modifications made if needed:
    
    ---
    ### Usage
    ```python
    from SQLDataModel import SQLDataModel as SQLDM
    
    source_db_conn = pyodbc.connect(...) # your source connection details
    destination_db_conn = sqlite3.connect(...) # your destination connection details

    # get the data from sql, csv, pandas, numpy, pickle, dictionary, python lists etc.
    sdm = SQLDM.from_sql("select * from source_table", source_db_conn)
    
    # execute arbitrary SQL to transform the model
    sdm = sdm.execute_fetch("select first_name, last_name, dob, job_title from sdm where job_title = "sales" ")
    sdm.set_headers(["first", "last", "dob", "position"])

    # iterate through the data
    for row in sdm.iter_rows():
        print(row)
    
    # group or aggregate the data:
    sdm.group_by(["first", "last", "position"])
    
    # load it to another sql database:
    sdm.to_sql("new_table", destination_db_conn)

    # or save it for later in any number of formats:
    sdm.to_csv("modified_data.csv")
    sdm.to_text("modified_data.txt")
    sdm.to_pickle("modified_data.sdm")
    sdm.to_local_db("modified_data.db")

    # reload it later from any number of formats:
    sdm = SQLDM.from_csv("modified_data.csv")
    sdm = SQLDM.from_numpy(x) # numpy.ndarray
    sdm = SQLDM.from_pandas(df) # pandas.DataFrame
    sdm = SQLDM.from_pickle("modified_data.sdm")
    sdm = SQLDM.from_sql("modified_data", sqlite3.connect('modified_data.db'))
    ```
    ---

    ### Pretty Printing
    SQLDataModel also pretty prints your table in any color you specify, use `SQLDataModel.set_display_color(t_color)` and provide either a hex value or a tuple of rgb and print the table, example output:
    ```shell
    ┌───┬─────────────────────┬────────────┬──────────────────────────────────┬─────────────┬────────┬─────────┬────────────┐
    │   │ full_name           │ date       │ email                            │ country     │    pin │ service │         id │
    ├───┼─────────────────────┼────────────┼──────────────────────────────────┼─────────────┼────────┼─────────┼────────────┤
    │ 0 │ Pamela Berg         │ 2024-09-15 │ turpis.nulla.aliquet@aol.com     │ New Zealand │   3010 │    3.02 │  951869685 │
    │ 1 │ Mason Hoover        │ 2024-01-23 │ ultrices@outlook.com             │ Australia   │   6816 │    5.01 │  351202632 │
    │ 2 │ Veda Suarez         │ 2023-09-04 │ ipsum.phasellus@yahoo.com        │ Ukraine     │   1175 │    4.65 │  523370578 │
    │ 3 │ Guinevere Cleveland │ 2024-03-22 │ amet.ornare@hotmail.net          │ New Zealand │   4962 │    3.81 │  253821048 │
    │ 4 │ Vincent Mccoy       │ 2023-09-16 │ nulla@yahoo.ca                   │ France      │   4446 │    2.95 │  521824160 │
    │ 5 │ Holmes Kemp         │ 2024-11-13 │ gravida.non@protonmail.couk      │ Germany     │   9396 │    4.61 │  700698503 │
    │ 6 │ Donna Mays          │ 2023-06-06 │ nulla@hotmail.com                │ Costa Rica  │   8153 │    5.34 │  423818468 │
    │ 7 │ Rama Galloway       │ 2023-09-22 │ et@protonmail.edu                │ Italy       │   3384 │    3.87 │  254573974 │
    │ 8 │ Lucas Rodriquez     │ 2024-03-16 │ aliquam.fringilla.curs@yahoo.com │ New Zealand │   3278 │    2.73 │  684654705 │
    │ 9 │ Hunter Donaldson    │ 2023-06-30 │ dolor.nulla@google.edu           │ Belgium     │   1593 │    4.58 │  171152418 │
    └───┴─────────────────────┴────────────┴──────────────────────────────────┴─────────────┴────────┴─────────┴────────────┘
    ```
    ---
    ### Notes
    use `SQLDM.get_supported_sql_connections()` to view supported databases, please reach out with any issues or questions, thanks!
    """
    __slots__ = ('sql_idx','sql_model','display_max_rows','min_column_width','max_column_width','column_alignment','display_color','display_index','row_count','headers','column_count','static_py_to_sql_map_dict','static_sql_to_py_map_dict','sql_c','sql_db_conn','display_float_precision','header_master')
    debug:bool=False
    def __init__(self, data:list[list], headers:list[str]=None, display_max_rows:int=None, min_column_width:int=4, max_column_width:int=32, column_alignment:str=None, display_color:str=None, display_index:bool=True, display_float_precision:int=4):
        if not isinstance(data, list|tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: type mismatch, '{type(data).__name__}' is not a valid type for data, which must be of type list or tuple...")
                )
        if len(data) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: data not found, data of length '{len(data)}' is insufficient to construct a valid model, additional rows of data required...")
                )
        try:
            _ = data[0]
        except Exception as e:
            raise IndexError(
                SQLDataModel.ErrorFormat(f"IndexError: data index error, data index provided does not exist for length '{len(data)}' due to '{e}'...")
                ) from None
        if not isinstance(data[0], list|tuple):
            if type(data[0]).__module__ != 'pyodbc': # check for pyodbc.Row which is acceptable
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: type mismatch, '{type(data[0]).__name__}' is not a valid type for data rows, which must be of type list or tuple...")
                    )
        if len(data[0]) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: data rows not found, data rows of length '{len(data[0])}' are insufficient to construct a valid model, at least one row is required...")
                )
        if headers is not None:
            if not isinstance(headers, list|tuple):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid header types, '{type(headers).__name__}' is not a valid type for headers, please provide a tuple or list type...")
                    )
            if len(headers) != len(data[0]):
                raise DimensionError(
                    SQLDataModel.ErrorFormat(f"DimensionError: invalid data dimensions, provided headers length '{len(headers)} != {len(data[0])}', the implied column count for data provided...")
                    )                
            if isinstance(headers,tuple):
                try:
                    headers = list(headers)
                except:
                    raise TypeError(
                        SQLDataModel.ErrorFormat(f"TypeError: failed header conversion, unable to convert provided headers tuple to list type, please provide headers as a list type...")
                        ) from None
            if not all(isinstance(x, str) for x in headers):
                try:
                    headers = [str(x) for x in headers]
                except:
                    raise TypeError(
                        SQLDataModel.ErrorFormat(f"TypeError: invalid header values, all headers provided must be of type string...")
                        ) from None
        else:
            headers = [f"col_{x}" for x in range(len(data[0]))]
        self.sql_idx = "idx"
        self.sql_model = "sdm"
        self.display_max_rows = display_max_rows
        self.min_column_width = min_column_width
        self.max_column_width = max_column_width
        self.column_alignment = column_alignment
        self.display_index = display_index
        self.display_float_precision = display_float_precision
        self.row_count = len(data)
        had_idx = True if headers[0] == self.sql_idx else False
        dyn_idx_offset,dyn_idx_bind,dyn_add_idx_insert = (1, "?,", f'"{self.sql_idx}",') if had_idx else (0, "", "")
        headers = headers[dyn_idx_offset:]
        self.headers = headers
        self.column_count = len(self.headers)
        self.display_color = ANSIColor(display_color) if isinstance(display_color,str|tuple) else display_color if isinstance(display_color,ANSIColor) else None
        self.static_py_to_sql_map_dict = {'None': 'NULL','int': 'INTEGER','float': 'REAL','str': 'TEXT','bytes': 'BLOB', 'date':'DATE', 'datetime': 'TIMESTAMP', 'NoneType':'TEXT', 'bool':'INTEGER'}
        self.static_sql_to_py_map_dict = {'NULL': 'None','INTEGER': 'int','REAL': 'float','TEXT': 'str','BLOB': 'bytes', 'DATE': 'date', 'TIMESTAMP': 'datetime','':'str'}
        headers_to_py_dtypes_dict = {self.headers[i]:type(data[0][i+dyn_idx_offset]).__name__ if type(data[0][i+dyn_idx_offset]).__name__ != 'NoneType' else 'str' for i in range(self.column_count)}
        headers_to_sql_dtypes_dict = {k:self.static_py_to_sql_map_dict[v] for (k,v) in headers_to_py_dtypes_dict.items()}
        headers_with_sql_dtypes_str = ",".join(f'"{col}" {type}' for col,type in headers_to_sql_dtypes_dict.items())
        sql_create_stmt = f"""create table if not exists "{self.sql_model}" ("{self.sql_idx}" INTEGER PRIMARY KEY,{headers_with_sql_dtypes_str})"""
        sql_insert_stmt = f"""insert into "{self.sql_model}" ({dyn_add_idx_insert}{','.join([f'"{col}"' for col in self.headers])}) values ({dyn_idx_bind}{','.join(['?' if headers_to_py_dtypes_dict[col] not in ('datetime','date') else "datetime(?)" if headers_to_py_dtypes_dict[col] == 'datetime' else "date(?)" for col in self.headers])})"""
        self.sql_db_conn = sqlite3.connect(":memory:", uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
        self.sql_db_conn.create_aggregate("stdev", 1, StandardDeviation)
        self.sql_db_conn.execute(sql_create_stmt)
        self._update_model_metadata()
        if not had_idx:
            first_row_insert_stmt = f"""insert into "{self.sql_model}" ("{self.sql_idx}",{','.join([f'"{col}"' for col in self.headers])}) values (?,{','.join(['?' if headers_to_py_dtypes_dict[col] not in ('datetime','date') else "datetime(?)" if headers_to_py_dtypes_dict[col] == 'datetime' else "date(?)" for col in self.headers])})"""
            try:
                self.sql_db_conn.execute(first_row_insert_stmt, (0,*data[0]))
            except sqlite3.ProgrammingError as e:
                raise SQLProgrammingError(
                    SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or inconsistent data, failed with "{e}"...')
                ) from None                
            data = data[1:] # remove first row from remaining data
        try:
            self.sql_db_conn.executemany(sql_insert_stmt,data)
        except sqlite3.ProgrammingError as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or inconsistent data, failed with "{e}"...')
            ) from None   
                     
################################################################################################################
################################################ static methods ################################################
################################################################################################################

    @staticmethod
    def ErrorFormat(error:str) -> str:
        """
        Formats an error message with ANSI color coding.

        Parameters:
            - `error`: The error message to be formatted.

        Returns:
            - `str`: The modified string with ANSI color coding, highlighting the error type in bold red.

        Example:
        ```python
        formatted_error = SQLDataModel.ErrorFormat("ValueError: Invalid value provided.")
        print(formatted_error)
        ```
        """
        error_type, error_description = error.split(':',1)
        return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""

    @staticmethod
    def WarnFormat(warn:str) -> str:
        """
        Formats a warning message with ANSI color coding.

        Parameters:
            - `warn`: The warning message to be formatted.

        Returns:
            - `str`: The modified string with ANSI color coding, highlighting the class name in bold yellow.

        Example:
        ```python
        formatted_warning = WarnFormat("DeprecationWarning: This method is deprecated.")
        print(formatted_warning)
        ```
        """
        warned_by, warning_description = warn.split(':',1)
        return f"""\r\033[1m\033[38;2;246;221;109m{warned_by}:\033[0m\033[39m\033[49m{warning_description}"""

    @staticmethod
    def SuccessFormat(success:str) -> str:
        """
        Formats a success message with ANSI color coding.

        Parameters:
            - `success`: The success message to be formatted.

        Returns:
            - `str`: The modified string with ANSI color coding, highlighting the success source in bold green.

        Example:
        ```python
        formatted_success = SuccessFormat("FileCreated: The operation was successful with new file created.")
        print(formatted_success)
        ```
        """ 
        success_by, success_description = success.split(':',1)
        return f"""\r\033[1m\033[38;2;108;211;118m{success_by}:\033[0m\033[39m\033[49m{success_description}"""

    @staticmethod
    def sqlite_printf_format(column:str, dtype:str, max_pad_width:int, float_precision:int=4, alignment:str=None) -> str:
        """
        Formats SQLite SELECT clauses based on column parameters to provide preformatted fetches, providing most of the formatting for `repr` output.

        Parameters:
            - `column` (str): The name of the column.
            - `dtype` (str): The data type of the column ('float', 'int', 'index', or other).
            - `max_pad_width` (int): The maximum width to pad the output.
            - `float_precision` (int, optional): The precision for floating-point numbers (default is 4).
            - `alignment` (str, optional): The alignment of the output ('<', '>', or None for no alignment).

        Returns:
            `str`: The formatted SELECT clause for SQLite.

        Notes:
            - This function generates SQLite SELECT clauses for single column only.
            - The output preformats SELECT result to fit `repr` method for tabular output.
            - The return `str` is not valid SQL by itself, representing only the single column select portion.
        """
        if alignment is None:
            if dtype == 'float':
                select_item_fmt = f"""CASE WHEN "{column}" IS NULL THEN printf('%{max_pad_width}s', '') ELSE printf('%.{max_pad_width}s', printf('% {max_pad_width}.{float_precision}f',"{column}")) END""" # does the most precision then treats as string to trim
            elif dtype == 'int':
                select_item_fmt = f"""printf('% {max_pad_width}s', CASE WHEN length("{column}") <= ({max_pad_width}) THEN "{column}" ELSE substr("{column}",1,({max_pad_width})-2)||'⠤⠄' END) """
            elif dtype == 'bytes':
                select_item_fmt = f"""printf('%!-{max_pad_width}s', CASE WHEN (length("{column}")+3) <= ({max_pad_width}) THEN ('b'''||"{column}"||'''') ELSE substr('b'''||"{column}",1,({max_pad_width})-3)||'''⠤⠄' END) """
            elif dtype == 'index':
                select_item_fmt = f"""printf('%{max_pad_width}s', "{column}") """
            else:
                select_item_fmt = f"""printf('%!-{max_pad_width}s', CASE WHEN length("{column}") <= ({max_pad_width}) THEN "{column}" ELSE substr("{column}",1,({max_pad_width})-2)||'⠤⠄' END) """
            return select_item_fmt
        else:
            if alignment in ("<", ">"):
                column = f'"{column}"' if dtype not in "bytes" else f"""'b'''||"{column}"||''''"""
                numeric_discriminator = """ """ if dtype in ("int","float") else """"""
                float_int_discriminator = f""".{float_precision}f""" if dtype == 'float' else """d""" if dtype == 'int' else """s"""
                printf_alignment = "-" if alignment == "<" else ""
                select_item_fmt = f"""CASE WHEN {column} IS NULL THEN printf('%{max_pad_width}s','') 
                WHEN length(printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',{column})) <= {max_pad_width} THEN
                printf('%.{max_pad_width}s', printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',{column})) 
                ELSE substr(printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',{column}),1,({max_pad_width})-2)||'⠤⠄'END """                
            else:
                float_int_discriminator = f"""printf('% .{float_precision}f',"{column}")""" if dtype == 'float' else f"""printf('% d',"{column}")""" if dtype == 'int' else f"""printf('%s',"{column}")""" if dtype != 'bytes' else f"""printf('b''%!s''',"{column}")"""
                select_item_fmt = f"""CASE WHEN "{column}" IS NULL THEN printf('%{max_pad_width}s','')
                WHEN length({float_int_discriminator}) <= {max_pad_width} THEN
                printf("%-{max_pad_width}.{max_pad_width}s", /* change minus operator before max_pad_width to favor right alignment when pad width is odd number, right now set to favor left side */
                printf("%*s%s%*s", (({max_pad_width}-length({float_int_discriminator}))/2), "", {float_int_discriminator}, (({max_pad_width}-length({float_int_discriminator}))/2), ""))
                ELSE substr({float_int_discriminator},1,({max_pad_width})-2)||'⠤⠄' END """                               
        return select_item_fmt 
    
    @staticmethod
    def alias_duplicates(headers:list) -> Generator:
        """
        Rename duplicate column names in a given list by appending an underscore and a numerical suffix.

        Parameters:
            - `headers` (list): A list of column names that require parsing for duplicates.

        Yields:
            - `Generator`: A generator object that yields the original or modified column names.

        Basic Usage:
            - `headers_str = ",".join([f'"{col}" as "{alias}"' for col, alias in zip(headers, alias_duplicates(headers))])`

        Example:

        ```python
        import SQLDataModel

        # Original list of column names with duplicates
        original_headers = ['ID', 'Name', 'Amount', 'Name', 'Date', 'Amount']

        # Use the static method to return a generator for the duplicates
        renamed_generator = SQLDataModel.alias_duplicates(original_headers)

        # Obtain the modified column names
        modified_headers = list(renamed_generator)

        # View modified column names
        print(modified_headers)

        # Output
        modified_headers = ['ID', 'Name', 'Amount', 'Name_2', 'Date', 'Amount_2']
        ```
        """        
        dupes = {}
        for col in headers:
            if col in dupes:
                dupes[col] += 1
                yield f"{col}_{dupes[col]}"
            else:
                dupes[col] = 1
                yield col

    def get_header_at_index(self, index:int) -> str:
        """
        Retrieves the name of the column in the `SQLDataModel` at the specified index.

        Parameters:
            - `index` (int): The index of the column for which to retrieve the name.

        Raises:
            - `IndexError`: If the provided column index is outside the current column range.

        Returns:
            - `str`: The name of the column at the specified index.

        Note:
            - The method allows retrieving the name of a column identified by its index in the SQLDataModel.
            - Handles negative indices by adjusting them relative to the end of the column range.

        Usage:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create model with sample data
        sqldm = SQLDataModel(data,headers)

        # Example: Get the name of the column at index 1
        column_name = sqldm.get_header_at_index(1)

        # Outputs 'first'
        print(column_name)
        ```
        """
        if index < 0:
            index = self.column_count + (index)
        if (index < 0 or index >= self.column_count):
            raise IndexError(
                SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{index}', provided index is outside of current column range '0:{self.column_count}', use `.get_headers()` to view current valid columns")
            )        
        return self.headers[index]
    
    def rename_column(self, column:int|str, new_column_name:str) -> None:
        """
        Renames a column in the `SQLDataModel` at the specified index or using the old column name with the provided value in `new_column_name`.

        Parameters:
            - `column` (int|str): The index or current str value of the column to be renamed.
            - `new_column_name` (str): The new name as a str value for the specified column.

        Raises:
            - `TypeError`: If the `column` or `new_column_name` parameters are invalid types.
            - `IndexError`: If the provided column index is outside the current column range.
            - `SQLProgrammingError`: If there is an issue with the SQL execution during the column renaming.

        Note:
            - The method allows renaming a column identified by its index in the SQLDataModel.
            - Handles negative indices by adjusting them relative to the end of the column range.
            - If an error occurs during SQL execution, it rolls back the changes and raises a SQLProgrammingError with an informative message.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the model with sample data
        sqldm = SQLDataModel(data,headers)

        # Example: Rename the column at index 1 to 'first_name'
        sqldm.rename_column(1, 'first_name')

        # Get current values
        new_headers = sqldm.get_headers()

        # Outputs ['first_name', 'last', 'age']
        print(new_headers)
        ```
        """
        if not isinstance(column,int|str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column type '{type(column).__name__}', `column` parameter must be of type 'int' or 'str'")
            )
        if isinstance(column,int):
            if column < 0:
                column = self.column_count + (column)
            if (column < 0 or column >= self.column_count):
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{column}', provided index is outside of current column range '0:{self.column_count}', use `.get_headers()` to view current valid columns")
                )
            column = self.headers[column]
        else:
            if column not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{column}', use `.get_headers()` to view current valid column arguments")
                )
        rename_stmts = f"""alter table "{self.sql_model}" rename column "{column}" to "{new_column_name}" """
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers[self.headers.index(column)] = new_column_name # replace old column with new column in same position as original column
        self._update_model_metadata()

    def replace(self, pattern:str, replacement:str, inplace:bool=False, **kwargs) -> SQLDataModel:
        """
        Replaces matching occurrences of a specified pattern with a replacement value in the `SQLDataModel` instance. 
        If inplace is True, the method updates the existing SQLDataModel; otherwise, it returns a new `SQLDataModel` with the replacements applied.

        Parameters:
            - `pattern` (str): The substring or regular expression pattern to search for in each column.
            - `replacement` (str): The string to replace the matched pattern with.
            - `inplace` (bool, optional): If True, modifies the current SQLDataModel instance in-place. Default is False.
            - **kwargs: Additional keyword arguments to be passed to the `execute_fetch` method when not in-place.
        
        Raises:
            - `TypeError`: If the `pattern` or `replacement` parameters are invalid types.

        Returns:
            - `SQLDataModel`: If inplace=True, modifies the current instance in-place and returns None. Otherwise, returns a new SQLDataModel
            with the specified replacements applied.

        Example:
        ```python
        import SQLDataModel

        headers = ['first', 'last', 'age', 'service']
        data = [
            ('John', 'Smith', 27, 1.22),
            ('Sarah', 'West', 39, 0.7),
            ('Mike', 'Harlin', 36, 3),
            ('Pat', 'Douglas', 42, 11.5)
        ]        

        # Create the model
        sdm = SQLDataModel(data, headers,display_float_precision=2, display_index=False)

        # Replace 'John' in the 'first' column
        sdm['first'] = sdm['first'].replace("John","Jane")
        
        # View model
        print(sdm)

        # Output    
        ```
        ```shell
        ┌───────┬─────────┬──────┬─────────┐
        │ first │ last    │  age │ service │
        ├───────┼─────────┼──────┼─────────┤
        │ Jane  │ Smith   │   27 │    1.22 │
        │ Sarah │ West    │   39 │    0.70 │
        │ Mike  │ Harlin  │   36 │    3.00 │
        │ Pat   │ Douglas │   42 │   11.50 │
        └───────┴─────────┴──────┴─────────┘   
        [4 rows x 4 columns] 
        ```
        """
        if not isinstance(pattern, str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid argument type '{type(pattern).__name__}', method argument `pattern` must be of type 'str' for `replace()` method")
            )
        if not isinstance(replacement, str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid argument type '{type(pattern).__name__}', method argument `replacement` must be of type 'str' for `replace()` method")
            )
        if inplace:
            replace_stmt = " ".join((f"""update "{self.sql_model}" set""",",".join([f""" "{col}"=replace("{col}",'{pattern}','{replacement}') """ for col in self.headers])))
            self.sql_db_conn.execute(replace_stmt)
            return
        else:
            replace_stmt = " ".join(("select",",".join([f""" replace("{col}",'{pattern}','{replacement}') as "{col}" """ for col in self.headers]),f'from "{self.sql_model}"'))
            return self.execute_fetch(replace_stmt, **kwargs)

    def set_header_at_index(self, index:int, new_value:str) -> None:
        """
        Renames a column in the `SQLDataModel` at the specified index with the provided new value.

        Parameters:
            - `index` (int): The index of the column to be renamed.
            - `new_value` (str): The new name for the specified column.

        Raises:
            - `IndexError`: If the provided column index is outside the current column range.
            - `SQLProgrammingError`: If there is an issue with the SQL execution during the column renaming.

        Note:
            - The method allows renaming a column identified by its index in the SQLDataModel.
            - Handles negative indices by adjusting them relative to the end of the column range.
            - If an error occurs during SQL execution, it rolls back the changes and raises a SQLProgrammingError with an informative message.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the model with sample data
        sqldm = SQLDataModel(data,headers)

        # Example: Rename the column at index 1 to 'first_name'
        sqldm.set_header_at_index(1, 'first_name')

        # Get current values
        new_headers = sqldm.get_headers()

        # Outputs ['first_name', 'last', 'age']
        print(new_headers)
        ```
        """
        if index < 0:
            index = self.column_count + (index)
        if (index < 0 or index >= self.column_count):
            raise IndexError(
                SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{index}', provided index is outside of current column range '0:{self.column_count}', use `.get_headers()` to view current valid columns")
            )
        rename_stmts = f"""alter table "{self.sql_model}" rename column "{self.headers[index]}" to "{new_value}" """
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers[index] = new_value # replace old column at specified index with new value to retain original column ordering
        self._update_model_metadata()

    def get_headers(self) -> list[str]:
        """
        Returns the current `SQLDataModel` headers.

        Returns:
            `list`: A list of strings representing the headers.

        Example:
        ```python
        import SQLDataModel

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name', 'Salary'])

        # Get current model headers
        headers = sqldm.get_headers()

        # Display values
        print(headers) # outputs: ['First Name', 'Last Name', 'Salary']
        ```
        """
        return self.headers
    
    def set_headers(self, new_headers:list[str]) -> None:
        """
        Renames the current `SQLDataModel` headers to values provided in `new_headers`. Headers must have the same dimensions
        and match existing headers.

        Parameters:
            `new_headers` (list): A list of new header names. It must have the same dimensions as the existing headers.

        Returns:
            `None`

        Raises:
            - `TypeError`: If the `new_headers` type is not a valid type (list or tuple).
            - `DimensionError`: If the length of `new_headers` does not match the column count.
            - `TypeError`: If the type of the first element in `new_headers` is not a valid type (str, int, or float).

        Example:
        ```python
        import SQLDataModel

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name', 'Salary'])

        # Set new headers
        sqldm.set_headers(['First_Name', 'Last_Name', 'Payment'])
        ```
        """
        if not isinstance(new_headers,list|tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid header types, type \"{type(new_headers).__name__}\" is not a valid type for headers, please provide a tuple or list type...")
                )
        if len(new_headers) != self.column_count:
            raise DimensionError(
                SQLDataModel.ErrorFormat(f"DimensionError: invalid header dimensions, provided headers length \"{len(new_headers)} != {self.column_count}\" column count, please provide correct dimensions...")
                )
        if not isinstance(new_headers[0], str|int|float):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid header values, type \"{type(new_headers[0]).__name__}\" is not a valid type for header values, please provide a string type...")
                )
        rename_stmts = ";".join([f"""alter table "{self.sql_model}" rename column "{self.headers[i]}" to "{new_headers[i]}" """ for i in range(self.column_count)])
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers = new_headers
        self._update_model_metadata()
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: Successfully renamed all model columns'))

    def normalize_headers(self, apply_function:Callable=None) -> None:
        """
        Reformats the current `SQLDataModel` headers into an uncased normalized form using alphanumeric characters only.
        Wraps `.set_headers()`.

        Parameters:
            - `apply_function` (Callable, optional): Specify an alternative normalization pattern. When `None`, the pattern
                `'[^0-9a-z _]+'` will be used on uncased values.

        Returns:
            None

        Example:
        ```python
        import SQLDataModel

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name', 'Salary'])

        # Use default normalization scheme, uncased and strips invalid SQL identifiers
        sqldm.normalize_headers()

        # Get renamed headers after default normalization
        sqldm.get_headers() # now outputs ['first_name', 'last_name', 'salary']

        # Or use custom renaming scheme
        sqldm.normalize_headers(lambda x: x.upper())

        # Get renamed headers again
        sqldm.get_headers() # now outputs ['FIRST_NAME', 'LAST_NAME', 'SALARY']
        ```
        """
        if apply_function is None:
            apply_function = lambda x: "_".join(x.strip() for x in re.sub('[^0-9a-z_]+', '', x.lower().replace(" ","_")).split('_') if x !='')
        new_headers = [apply_function(x) for x in self.get_headers()]
        self.set_headers(new_headers)
        return

    def get_display_max_rows(self) -> int|None:
        """
        Retrieves the current value of the `display_max_rows` property, which determines the maximum rows displayed for `SQLDataModel`.

        Returns:
            - `int|None`: The current value of the 'display_max_rows' property.

        Example:
        ```python
        # Example usage:

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get current value
        display_max_rows = sqldm.get_display_max_rows()

        # By default any display will be limited to 1000 rows max
        print(display_max_rows) # 1000
        ```
        """
        return self.display_max_rows
    
    def set_display_max_rows(self, rows:int|None) -> None:
        """
        Set `display_max_rows` to limit rows displayed when `repr` or `print` is called, or set to `None` for `display_max_rows` to be derived from current terminal height. Using this option will create a table that fits within the dimensions of the current terminal such that the table headers occupy the top-most row with the table caption occupying the lower-most.
        Modifying this attribute does not change the actual number of rows stored in `SQLDataModel`, only the number of rows displayed.

        Parameters:
            - `rows` (int): The maximum number of rows to display.

        Raises:
            - `TypeError`: If the provided argument is not `None` or is not an integer.
            - `IndexError`: If the provided value is an integer less than or equal to 0.

        Example:
        ```python
        import SQLDataModel

        # Create model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Any call to `print` or `repr` will be restricted to 500 max rows
        sdm.set_display_max_rows(500)

        # Alternatively, auto-detect dimensions by setting to `None`
        sdm.set_display_max_rows(None)
        ```
        """
        if not isinstance(rows, int|None):
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid argument type "{type(rows).__name__}", please provide an integer value to set the maximum rows attribute...')
                )
        if isinstance(rows, int) and rows <= 0:
            raise IndexError(
                SQLDataModel.ErrorFormat(f'IndexError: invalid value "{rows}", please provide an integer value >= 1 to set the maximum rows attribute...')
                )
        self.display_max_rows = rows

    def get_min_column_width(self) -> int:
        """
        Returns the current `min_column_width` property value.

        Returns:
            int: The current value of the `min_column_width` property.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get and save the current value
        min_width = sqldm.get_min_column_width()

        # Output
        print(min_width)  # 6
        ```
        """
        return self.min_column_width
    
    def set_min_column_width(self, width:int) -> None:
        """
        Set `min_column_width` as the minimum number of characters per column when `repr` or `print` is called.

        Parameters:
            - `width` (int): The minimum width for each column.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Set a new minimum column width value
        sqldm.set_min_column_width(8)

        # Check updated value
        print(sqldm.get_min_column_width) # 8
        ```
        """
        self.min_column_width = width

    def get_max_column_width(self) -> int:
        """
        Returns the current `max_column_width` property value.

        Returns:
            int: The current value of the `max_column_width` property.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get the current max column width value
        max_width = sqldm.get_max_column_width()

        # Output
        print(max_width)  # 32
        ```
        """
        return self.max_column_width
    
    def set_max_column_width(self, width:int) -> None:
        """
        Set `max_column_width` as the maximum number of characters per column when `repr` or `print` is called.

        Parameters:
            - `width` (int): The maximum width for each column.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Change the max column width for the table representation
        sqldm.set_max_column_width(20)
        ```
        """
        self.max_column_width = width

    def get_column_alignment(self) -> str:
        """
        Returns the current `column_alignment` property value, `None` by default.

        Returns:
            - `str`: The current value of the `column_alignment` property.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get the current alignment value
        alignment = sqldm.get_column_alignment()

        # Output
        print(alignment)  # None
        ```
        """
        return self.column_alignment
    
    def set_column_alignment(self, alignment:Literal['<', '^', '>']=None) -> None:
        """
        Set `column_alignment` as the default alignment behavior when `repr` or `print` is called.
        
        Options:
            - `column_alignment = None`: Default behavior, dynamically aligns columns based on value types.
            - `column_alignment = '<'`: Left-align all column values.
            - `column_alignment = '^'`: Center-align all column values.
            - `column_alignment = '>'`: Right-align all column values.
        
        Default behavior aligns strings left, integers & floats right, with headers matching value alignment.

        Parameters:
            - `alignment` (str | None): The alignment setting.

        Raises:
            - `TypeError`: If the provided alignment is not a valid f-string alignment formatter.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Set to right-align
        sqldm.set_column_alignment('>')

        # Output
        print(sqldm)
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬─────────┐
        │   │  first │    last │    age │ service │
        ├───┼────────┼─────────┼────────┼─────────┤
        │ 0 │   john │   smith │     27 │    1.22 │
        │ 1 │  sarah │    west │     39 │    0.70 │
        │ 2 │   mike │  harlin │     36 │    3.00 │
        │ 3 │    pat │ douglas │     42 │   11.50 │
        └───┴────────┴─────────┴────────┴─────────┘        
        ```
        ```python
        # Set to left-align
        sqldm.set_column_alignment('<')

        # Output
        print(sqldm)
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬─────────┐
        │   │ first  │ last    │ age    │ service │
        ├───┼────────┼─────────┼────────┼─────────┤
        │ 0 │ john   │ smith   │  27    │  1.22   │
        │ 1 │ sarah  │ west    │  39    │  0.70   │
        │ 2 │ mike   │ harlin  │  36    │  3.00   │
        │ 3 │ pat    │ douglas │  42    │  11.50  │
        └───┴────────┴─────────┴────────┴─────────┘        
        ```
        """
        if alignment is None:
            self.column_alignment = alignment
            return
        if (not isinstance(alignment, str)) or (alignment not in ('<', '^', '>')):
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid argument "{alignment}", please provide a valid f-string alignment formatter or set `alignment=None` to use default behaviour...')
                )
        self.column_alignment = alignment
        return

    def get_display_index(self) -> bool:
        """
        Returns the current boolean value for `is_display_index`, which determines
        whether or not the `SQLDataModel` index will be shown in print or repr calls.

        Returns:
            - `bool`: The current value of the `display_index` property.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get the current value for displaying the index
        display_index = sqldm.get_display_index()

        # Outputs `True`
        print(display_index)
        ```
        """
        return self.display_index

    def set_display_index(self, display_index:bool) -> None:
        """
        Sets the `display_index` property to enable or disable the inclusion of the
        `SQLDataModel` index value in print or repr calls, default set to include.

        Parameters:
            - `display_index` (bool): A boolean value (True | False) to determine whether
            to include the index in print or repr calls.

        Raises:
            - `TypeError`: If the provided argument is not a boolean value.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm.set_display_index(False)
        ```
        """
        if not isinstance(display_index, bool):
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid argument "{display_index}", please provide a valid boolean (True | False) value to the `display_index` argument...')
                )
        self.display_index = display_index
    
    def get_shape(self) -> tuple[int]:
        """
        Returns the shape of the data as a tuple of `(rows x columns)`.

        Returns:
            - `tuple[int]`: A tuple representing the number of rows and columns in the SQLDataModel.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        shape = sqldm.get_shape()
        print(shape)  # Output: (10, 3)
        ```
        """
        return (self.row_count,self.column_count)
    
    def get_display_float_precision(self) -> int:
        """
        Retrieves the current float display precision used exclusively for representing the values of real numbers
        in the `repr` method for the `SQLDataModel`. Default value is set to 4 decimal places of precision.

        Returns:
            - `int`: The current float display precision.

        Note:
            - The float display precision is the number of decimal places to include when displaying real numbers
            in the string representation of the `SQLDataModel`.
            - This value is utilized in the `repr` method to control the precision of real number values.
            - The method does not affect the actual value of float dtypes in the underlying `SQLDataModel`
        """
        return self.display_float_precision
    
    def set_display_float_precision(self, float_precision:int) -> None:
        """
        Sets the current float display precision to the specified value for use in the `repr` method of the `SQLDataModel`
        when representing float data types. Note that this precision limit is overridden by the `max_column_width` value
        if the precision limit exceeds the specified maximum width.

        Parameters:
            - `float_precision` (int): The desired float display precision to be used for real number values.

        Raises:
            - `TypeError`: If the `float_precision` argument is not of type 'int'.
            - `ValueError`: If the `float_precision` argument is a negative value, as it must be a valid f-string precision identifier.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service_time']
        data = [
            (0, 'john', 'smith', 27, 1.22)
            ,(1, 'sarah', 'west', 0.7)
            ,(2, 'mike', 'harlin', 3)
            ,(3, 'pat', 'douglas', 11.5)
        ]

        # Create the model with sample data
        sdm = SQLDataModel(data,headers)

        # Example: Set the float display precision to 2
        sdm.set_display_float_precision(2)

        # View model
        print(sdm)

        # Outputs:
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬──────────────┐
        │   │ first  │ last    │    age │ service_time │
        ├───┼────────┼─────────┼────────┼──────────────┤
        │ 0 │ john   │ smith   │     27 │         2.10 │
        │ 1 │ sarah  │ west    │     29 │         0.80 │
        │ 2 │ mike   │ harlin  │     36 │         1.30 │
        │ 3 │ pat    │ douglas │     42 │         7.02 │
        └───┴────────┴─────────┴────────┴──────────────┘
        ```

        ```python
        # Get the updated float display precision
        updated_precision = sdm.get_display_float_precision()

        # Outputs 2
        print(updated_precision)
        ```

        """
        if not isinstance(float_precision, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(float_precision).__name__}' received for `float_precision` argument, expected type 'int'")
            )
        if float_precision < 0:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid value '{float_precision}' received for `float_precision` argument, value must be valid f-string precision identifier")
            )
        self.display_float_precision = float_precision
        
    def describe(self, exclude_columns:str|list=None, exclude_dtypes:list[Literal["str","int","float","date","datetime","bool"]]=None, ignore_na:bool=True, **kwargs) -> SQLDataModel:
        """
        Generates descriptive statistics for columns in the `SQLDataModel` instance based on column dtype including count, unique values, top value, frequency, mean, standard deviation, minimum, 25th, 50th, 75th percentiles, maximum and dtype for specified column.

        Parameters:
            - `exclude_columns` (str | list, optional): Columns to exclude from the analysis. Default is None.
            - `exclude_dtypes` (list[Literal["str", "int", "float", "date", "datetime", "bool"]], optional): Data types to exclude from the analysis. Default is None.
            - `ignore_na` (bool, optional): If True, ignores NA like values ('NA', ' ', 'None') when computing statistics. Default is True.
            - `**kwargs`: Additional keyword arguments to be passed to the `execute_fetch` method.

        Statistics:
            - count: Total number of non-null values for specified column
            - unique: Total number of unique values for specified column
            - top: Top value represented for specified column, ties broken arbitrarily
            - freq: Frequency of corresponding value represented in 'top' metric
            - mean: Mean as calculated by summing all values and dividing by 'count'
            - std: Standard Deviation for specified column
                - Uncorrected sample standard deviation for `int`, `float` dtypes
                - Mean time difference represented in number of days for `date`, `datetime` dtypes
                - 'NaN' for all other dtypes
            - min: Minimum value for specified column
                - Least value for `int`, `float` dtypes
                - Least value sorted by alphabetical ascending for `str` dtypes
                - Earliest date or datetime for `date`, `datetime` dtypes
            - p25: Percentile, 25th
                - Max first bin value as determined by quartered binning of values for `int`, `float` dtypes
                - 'NaN' for all other dtypes
            - p50: Percentile, 50th
                - Max second bin value as determined by quartered binning of values for `int`, `float` dtypes
                - 'NaN' for all other dtypes
            - p75: Percentile, 75th
                - Max third bin value as determined by quartered binning of values for `int`, `float` dtypes
                - 'NaN' for all other dtypes        
            - max: Maximum value for specified column
                - Greatest value for `int`, `float` dtypes
                - Greatest value sorted by alphabetical ascending for `str` dtypes
                - Latest date or datetime for `date`, `datetime` dtypes        
            - dtype: Datatype of specified column
                - Python datatype as determined by relevant class `__name__` attribute, e.g. 'float' or 'int'
                - dtypes can be excluded by using `exclude_dtypes` parameter

        Returns:
            - `SQLDataModel`: A new SQLDataModel containing a comprehensive set of descriptive statistics for selected columns.

        Notes:
            - Standard deviation is calculated using uncorrected sample standard deviation for numeric dtypes, and timediff in days for datetime dtypes
            - Ties in unique, top and freq columns are broken arbitrarily as determined by first ordering of values prior to calling `describe()`
            - Ties encountered when binning for p25, p50, p75 will favor lower bins for data that cannot be quartered cleanly
            - Metrics for count, min, p25, p50, p75 and max include non-null values only
            - Using `ignore_na=True` only affects inclusion of 'NA like' values such as empty strings
            - Floating point precision determined by `display_float_precision` attribute
        
        Examples:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('employees.csv')

        # View all 10 rows
        print(sdm)

        # Output
        ```
        ```shell
        ┌───┬──────────────────┬────────────┬─────────────┬───────────────┬────────┬─────────────────────┐
        │   │ name             │ hire_date  │ country     │ service_years │    age │ last_update         │
        ├───┼──────────────────┼────────────┼─────────────┼───────────────┼────────┼─────────────────────┤
        │ 0 │ Pamela Berg      │ 2007-06-06 │ New Zealand │          3.02 │     56 │ 2023-08-12 17:13:46 │
        │ 1 │ Mason Hoover     │ 2009-04-19 │ Australia   │          5.01 │     41 │ 2023-05-18 01:29:44 │
        │ 2 │ Veda Suarez      │ 2007-07-02 │ Ukraine     │          4.65 │     26 │ 2023-12-09 15:38:01 │
        │ 3 │ John Smith       │ 2017-08-12 │ New Zealand │          3.81 │     35 │ 2023-03-10 18:23:56 │
        │ 4 │ Xavier McCoy     │ 2021-04-03 │ France      │          2.95 │     42 │ 2023-09-27 11:39:08 │
        │ 5 │ John Smith       │ 2020-10-11 │ Germany     │          4.61 │     56 │ 2023-12-09 18:41:52 │
        │ 6 │ Abigail Mays     │ 2021-07-25 │ Costa Rica  │          5.34 │     50 │ 2023-02-11 16:43:07 │
        │ 7 │ Rama Galloway    │ 2009-02-09 │ Italy       │          3.87 │     24 │ 2023-03-13 16:08:48 │
        │ 8 │ Lucas Rodriquez  │ 2018-06-19 │ New Zealand │          2.73 │     28 │ 2023-03-17 01:45:22 │
        │ 9 │ Hunter Donaldson │ 2015-12-18 │ Belgium     │          4.58 │     43 │ 2023-04-06 03:22:54 │
        └───┴──────────────────┴────────────┴─────────────┴───────────────┴────────┴─────────────────────┘      
        [10 rows x 6 columns]  
        ```
        ```python
        # Generate statistics
        sdm_described = sdm.describe()

        # View stats
        print(sdm_described)

        # Output
        ```
        ```shell
        ┌────────┬──────────────┬─────────────┬─────────────┬───────────────┬────────┬─────────────────────┐
        │ metric │         name │   hire_date │     country │ service_years │    age │         last_update │
        ├────────┼──────────────┼─────────────┼─────────────┼───────────────┼────────┼─────────────────────┤
        │ count  │           10 │          10 │          10 │            10 │     10 │                  10 │
        │ unique │            9 │          10 │           8 │            10 │      9 │                  10 │
        │ top    │   John Smith │  2021-07-25 │ New Zealand │          5.34 │     56 │ 2023-12-09 18:41:52 │
        │ freq   │            2 │           1 │           3 │             1 │      2 │                   1 │
        │ mean   │          NaN │  2014-11-24 │         NaN │          4.06 │     40 │ 2023-06-16 19:18:39 │
        │ std    │          NaN │ 2164.4 days │         NaN │          0.92 │     11 │         117.58 days │
        │ min    │ Abigail Mays │  2007-06-06 │   Australia │          2.73 │     24 │ 2023-02-11 16:43:07 │
        │ p25    │          NaN │  2009-02-09 │         NaN │          3.02 │     28 │ 2023-03-13 16:08:48 │
        │ p50    │          NaN │  2017-08-12 │         NaN │          4.58 │     42 │ 2023-05-18 01:29:44 │
        │ p75    │          NaN │  2020-10-11 │         NaN │          4.65 │     50 │ 2023-09-27 11:39:08 │
        │ max    │ Xavier McCoy │  2021-07-25 │     Ukraine │          5.34 │     56 │ 2023-12-09 18:41:52 │
        │ dtype  │          str │        date │         str │         float │    int │            datetime │
        └────────┴──────────────┴─────────────┴─────────────┴───────────────┴────────┴─────────────────────┘
        [12 rows x 7 columns]    
        ```
        ```python
        # Set filters to exclude all str dtypes and the 'hire_date' column:
        sdm_describe = sdm.describe(exclude_dtypes=['str'], exclude_columns=['hire_date'])

        # View statistics
        print(sdm_described)

        # Output
        ```
        ```shell
        ┌────────┬───────────────┬────────┬─────────────────────┐
        │ metric │ service_years │    age │         last_update │
        ├────────┼───────────────┼────────┼─────────────────────┤
        │ count  │            10 │     10 │                  10 │
        │ unique │            10 │      9 │                  10 │
        │ top    │          5.34 │     56 │ 2023-10-28 05:42:43 │
        │ freq   │             1 │      2 │                   1 │
        │ mean   │          4.06 │     40 │ 2023-08-11 23:18:12 │
        │ std    │          0.92 │     11 │          73.15 days │
        │ min    │          2.73 │     24 │ 2023-04-07 23:56:06 │
        │ p25    │          3.02 │     28 │ 2023-06-02 14:36:19 │
        │ p50    │          4.58 │     42 │ 2023-09-09 19:18:38 │
        │ p75    │          4.65 │     50 │ 2023-10-09 19:34:55 │
        │ max    │          5.34 │     56 │ 2023-10-28 05:42:43 │
        │ dtype  │         float │    int │            datetime │
        └────────┴───────────────┴────────┴─────────────────────┘
        [12 rows x 4 columns]
        ```
        Warnings:
            - Generally, do not rely on `SQLDataModel` to do statistics, use `NumPy` or a real library instead
            - Statistics for `date` and `datetime` can be unpredictable if formatting is inconsistent
        """
        if isinstance(exclude_columns, str|None):
            exclude_columns = [exclude_columns]
        if isinstance(exclude_dtypes, str|None):
            exclude_dtypes = [exclude_dtypes]
        desc_cols = [col for col in self.headers if ((col not in exclude_columns) and (self.header_master[col][1] not in exclude_dtypes))]
        has_numeric_dtype = any(map(lambda v: v in ('float','int','date','datetime'), [self.header_master[col][1] for col in desc_cols])) # ('float','int','date','datetime')
        headers_select_literal = [f""" "'{col}'" as "{col}" """ for col in desc_cols]
        headers_select = """ "'metric'" as "metric",""" + ",".join(headers_select_literal)
        headers_sub_literal = [f"'{col}'" for col in desc_cols]
        headers_subselect = " select 'metric'," + ",".join(headers_sub_literal)
        count_subselect = "select 'count'," + ",".join([f""" count("{col}") """ if not ignore_na else f"""(select count("{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL')) """ for col in desc_cols])
        count_subselect = f"""{count_subselect} {f'from "{self.sql_model}"' if not ignore_na else ''}"""
        unique_subselect = "select 'unique'," + ",".join([f""" count(distinct "{col}") """ if not ignore_na else f"""(select count(distinct "{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL')) """ for col in desc_cols])
        unique_subselect = f"""{unique_subselect} {f'from "{self.sql_model}"' if not ignore_na else ''}"""
        top_subselect = "select 'top'," + ",".join([f"""(select max("{col}") from "{self.sql_model}" group by "{col}" order by count(*) desc limit 1) """ if not ignore_na else f"""(select max("{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL') group by "{col}" order by count(*) desc limit 1) """ for col in desc_cols])
        freq_subselect = "select 'freq'," + ",".join([f"""(select count("{col}") from "{self.sql_model}" group by "{col}" order by count(*) desc limit 1) """ if not ignore_na else f"""(select count("{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL') group by "{col}" order by count(*) desc limit 1) """ for col in desc_cols])
        pcte_stmt = f"""with pcte as (select {','.join([f'"{col}",ntile(4) over (order by "{col}") as "p{col}"' for col in desc_cols if self.header_master[col][1] in ('float','int','date','datetime')])} from "{self.sql_model}")""" if has_numeric_dtype else ""
        mean_subselect = "select * from (select 'mean'," + ",".join([f"""round(avg("{col}"),{self.display_float_precision}) """ if self.header_master[col][1] == 'float' else f"""printf('%d',avg("{col}"))""" if self.header_master[col][1] == 'int' else f"""date(avg(julianday("{col}")))""" if self.header_master[col][1] == "date" else f"""datetime(avg(julianday("{col}")))""" if self.header_master[col][1] == "datetime" else "'NaN'" for col in desc_cols]) + f"""from "{self.sql_model}" limit 1) """
        std_subselect = "select * from (select 'std'," + ",".join([f"""round(stdev("{col}"),{self.display_float_precision}) """ if self.header_master[col][1] == 'float' else f"""printf('%d',stdev("{col}"))""" if self.header_master[col][1] == 'int' else f"""round(stdev(julianday("{col}")),{self.display_float_precision})||' days'""" if self.header_master[col][1] in ("date","datetime") else "'NaN'" for col in desc_cols]) + f"""from "{self.sql_model}" limit 1) """
        min_subselect = "select * from (select 'min'," + ",".join([f"""round(min("{col}"),{self.display_float_precision}) """ if self.header_master[col][1] in ('float','int') else f"""min("{col}")""" if (not ignore_na or self.header_master[col][1] in ("date","datetime")) else f"""(select min("{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL')) """ for col in desc_cols]) + f"""from "{self.sql_model}" limit 1) """
        p25_subselect = "select 'p25'," + ",".join([f"""(select round(max("{col}"),{self.display_float_precision}) as "max25{col}" from pcte where "p{col}" = 1 group by "p{col}") as "p25{col}" """ if self.header_master[col][1] in ('float','int') else f"""(select max("{col}") as "max25{col}" from pcte where "p{col}" = 1 group by "p{col}") as "p25{col}" """ if self.header_master[col][1] in ('date','datetime') else "'NaN'" for col in desc_cols])
        p50_subselect = "select 'p50'," + ",".join([f"""(select round(max("{col}"),{self.display_float_precision}) as "max50{col}" from pcte where "p{col}" = 2 group by "p{col}") as "p50{col}" """ if self.header_master[col][1] in ('float','int') else f"""(select max("{col}") as "max50{col}" from pcte where "p{col}" = 2 group by "p{col}") as "p50{col}" """ if self.header_master[col][1] in ('date','datetime') else "'NaN'" for col in desc_cols])
        p75_subselect = "select 'p75'," + ",".join([f"""(select round(max("{col}"),{self.display_float_precision}) as "max75{col}" from pcte where "p{col}" = 3 group by "p{col}") as "p75{col}" """ if self.header_master[col][1] in ('float','int') else f"""(select max("{col}") as "max75{col}" from pcte where "p{col}" = 3 group by "p{col}") as "p75{col}" """ if self.header_master[col][1] in ('date','datetime') else "'NaN'" for col in desc_cols])
        max_subselect = "select * from (select 'max'," + ",".join([f"""round(max("{col}"),{self.display_float_precision}) """ if self.header_master[col][1] in ('float','int') else f"""max("{col}")""" if (not ignore_na or self.header_master[col][1] in ("date","datetime")) else f"""(select max("{col}") from "{self.sql_model}" where trim(upper("{col}")) not in (' ', '', 'NA', 'NONE','NULL')) """ for col in desc_cols]) + f"""from "{self.sql_model}" limit 1) """
        dtype_subselect = "select 'dtype'," + ",".join([f"""'{self.header_master[col][1]}'""" for col in desc_cols])
        full_script = f"""{pcte_stmt} select {headers_select} from ({headers_subselect} UNION ALL {count_subselect} UNION ALL {unique_subselect} UNION ALL {top_subselect} UNION ALL {freq_subselect} UNION ALL {mean_subselect} UNION ALL {std_subselect} UNION ALL {min_subselect} UNION ALL {p25_subselect} UNION ALL {p50_subselect} UNION ALL {p75_subselect} UNION ALL {max_subselect} UNION ALL {dtype_subselect}) limit -1 offset 1"""
        # print(full_script.replace("\\'","'").replace("UNION ALL","\nUNION ALL"))
        return self.execute_fetch(full_script, display_index=False, **kwargs)

    def sample(self, n_samples:float|int=0.05, **kwargs) -> SQLDataModel:
        """
        Return a random sample of size `n_samples` as a new `SQLDataModel`.

        Parameters:
            - `n_samples` (float | int): Number of rows or proportion of rows to sample. Default set to `0.05`, proportional to 5% of the current `row_count`
                - If `n_samples` is an integer, it represents the exact number of rows to sample where `0 < n_samples <= row_count`.
                - If `n_samples` is a float, it represents the proportion of rows to sample where `0.0 < n_samples <= 1.0`.

        Returns:
            - `SQLDataModel`: A new SQLDataModel instance containing the sampled rows.

        Raises:
            - `TypeError`: If the `n_samples` parameter is not of type 'int' or 'float'.
            - `ValueError`: If the `n_samples` value is invalid or out of range.

        This method generates a random sample of rows from the current SQLDataModel. The number of rows to sample
        can be specified either as an integer representing the exact number of rows or as a float representing
        the proportion of rows to sample. The sampled rows are returned as a new SQLDataModel instance.

        Examples:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Amount'])
        
        # Example 1: Sample 10 random rows
        sample_result = sdm.sample(n_samples=10)

        # Create the model
        sdm2 = SQLDataModel.from_csv('another_example.csv', headers=['Code', 'Description', 'Price'])
        
        # Example 2: Sample 20% of rows
        sample_result2 = sdm2.sample(n_samples=0.2)
        ```
        """
        if not isinstance(n_samples, float|int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid `n_samples` type '{type(n_samples).__name__}', `n_samples` parameter type must be one of 'int', 'float' as number of rows or proportion of rows, respectively")
            )            
        if isinstance(n_samples, float):
            if (n_samples <= 0.0) or (n_samples > 1.0):
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid `n_samples` value '{n_samples}', expected value in range '0.0 < n_samples <= 1.0' when using proportional value for `n_samples`")
                )
            n_samples = round(self.row_count * n_samples)
        if n_samples <= 0:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid `n_samples` value '{n_samples}', expected value within current row range '0 < n_samples <= {self.row_count}' when using integer value for `n_samples`")
            ) 
        return self.execute_fetch(self._generate_unordered_sql_stmt(n_rows=n_samples,ordering="random"), **kwargs)

    def infer_dtypes(self, n_samples:int=10, infer_threshold:float=0.5, datetime_format:str="%Y-%m-%d %H:%M:%S", non_inferrable_dtype:Literal["str","int","float","datetime","date","bytes"]="str") -> None:
        """
        Infer and set data types for columns based on random sampling of `n_samples` from the dataset if proportion of data types in returned sample equals or exceeds `infer_threshold` otherwise fallback to data type specified in `non_inferrable_dtype` argument.
        Dateutil library required for datetime parsing, otherwise if module not found `datetime_format` will be used.
        Parameters:
            - `n_samples` (int): The number of random samples to use for data type inference.
            - `infer_threshold` (float): The threshold by which a dtype is selected should count_dtype/count_samples exceed value, default set to 50% of 5 possible dtypes.
            - `datetime_format` (str): The datetime format to use to try and parse datetime.date and datetime.datetime objects from if `dateutil` library not installed.
            - `non_inferrable_dtype` (Literal["str","int","float","datetime","date","bytes"]): The default data type to assign when ties occur, or when no dtype can be inferred.
        
        Raises:
            - `TypeError`: If argument for `n_samples` is not of type `int` or `infer_threshold` is not of type `float`
            - `ValueError`: If value for `infer_threshold` is not a valid range satisfying `0.0 < infer_threshold <= 1.0`
            - `TypeError`: If `non_inferrable_dtype` is not one of "str", "int", "float", "datetime", "date", "bytes"

        Returns:
            - `None`

        Notes:
            - If a single `str` instance is found in the samples, the corresponding column dtype will remain as `str` to avoid data loss.
            - Co-occurences of `int` & `float`, or `date` & `datetime` will favor the superset dtype after `infer_threshold` is met, so `float` and `datetime` respectively.
            - If a single `datetime` instance is found amongst a higher proportion of `date` dtypes, `datetime` will be used according to second rule.
            - If a single `float` instance is found amongst a higher proportion of `int` dtypes, `float` will be used according to second rule.
            - Ties between dtypes are broken according to `non_inferrable_dtype`<`str`<`float`<`int`<`datetime`<`date`<`bytes`
            - This method calls the `set_column_dtypes()` method once the column dtypes have been inferred if they differ from the current dtype.

        ---   

        Examples:
        ```python
        import SQLDataModel

        # Sample data of `str` containing probable datatypes
        headers = ['first', 'last', 'age', 'service', 'hire_date']
        data = [
            ('John', 'Smith', '27', '1.22', '2023-02-01'),
            ('Sarah', 'West', '39', '0.7', '2023-10-01'),
            ('Mike', 'Harlin', '36', '3.9', '2020-08-27'),
            ('Pat', 'Douglas', '42', '11.5', '2015-11-06'),
            ('Kelly', 'Lee', '32', '8.0', '2016-09-18')
        ]     

        # Create the model
        sdm = SQLDataModel(data, headers)
        
        # Infer and set data types based on 10 random samples
        sdm.infer_dtypes(n_samples=10)

        # View updated model
        print(sdm)

        # Numeric dtypes correctly aligned
        ```
        ```shell
        ┌───────┬─────────┬──────┬─────────┬────────────┐
        │ first │ last    │  age │ service │ hire_date  │
        ├───────┼─────────┼──────┼─────────┼────────────┤
        │ John  │ Smith   │   27 │    1.22 │ 2023-02-01 │
        │ Sarah │ West    │   39 │    0.70 │ 2023-10-01 │
        │ Mike  │ Harlin  │   36 │    3.90 │ 2020-08-27 │
        │ Pat   │ Douglas │   42 │   11.50 │ 2015-11-06 │
        │ Kelly │ Lee     │   32 │    8.00 │ 2016-09-18 │
        └───────┴─────────┴──────┴─────────┴────────────┘
        [5 rows x 5 columns]
        ```
        ```python

        # View updated dtypes
        for col in sdm.headers:
            print(f"{col}: {sdm.get_column_dtypes(col)}")
        
        # Outputs
        first: str
        last: str
        age: int
        service: float
        hire_date: datetime.date        
        ```
        """
        if not isinstance(n_samples, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid `n_samples` type '{type(n_samples).__name__}', `n_samples` argument must be of type 'int' for `infer_dtypes()` method")
            )
        if not isinstance(infer_threshold, float):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid `infer_threshold` type '{type(infer_threshold).__name__}', `infer_threshold` argument must be of type 'int' for `infer_dtypes()` method")
            )        
        if (infer_threshold <= 0.0) or (infer_threshold > 1.0):
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid `infer_threshold` value '{infer_threshold}', expected value in range '0.0 < infer_threshold <= 1.0' for `infer_dtypes()` method")
            )   
        if non_inferrable_dtype not in ("str","int","float","datetime","date","bytes"):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid `non_inferrable_dtype` type '{type(non_inferrable_dtype).__name__}', expected one of 'str','int','float','datetime','date','bytes' for `infer_dtypes()` method")
            )
        str_dtype_columns = [col for col in self.headers if self.header_master[col][1] == 'str']
        fetch_str_dtype_stmt = " ".join((f"""select""", ",".join([f'trim("{col}")' for col in str_dtype_columns]),f"""from "{self.sql_model}" where "{self.sql_idx}" in (select "{self.sql_idx}" from "{self.sql_model}" order by random() limit {n_samples}) """))
        samples_dict = {col:tuple(row[j] for row in self.sql_db_conn.execute(fetch_str_dtype_stmt)) for j,col in enumerate(str_dtype_columns)}
        inferred_dtypes = {k:non_inferrable_dtype for k in samples_dict.keys()}
        dtype_labels = ("str", "int", "float", "datetime", "date", "bytes")
        for s_header, s_samples in samples_dict.items():
            count_checked, count_str, count_int, count_float, count_datetime, count_date, count_bytes = 0,0,0,0,0,0,0
            for s_item in s_samples:
                if (s_item is None) or (s_item == ''):
                    continue
                count_checked += 1
                try:
                    s_type = literal_eval(s_item)
                except:
                    try:
                        if _has_dateutil:
                            dt_type = dateparser(s_item, fuzzy=False, fuzzy_with_tokens=False)
                        else:
                            dt_type = datetime.datetime.strptime(s_item, datetime_format)
                        # both datetime.date and datetime.datetime are instances of date, only way it works is date is not instance of datetime.datetime
                        if isinstance(dt_type,datetime.date):
                            total_time_after_date = sum((dt_type.hour,dt_type.minute,dt_type.second))
                            if total_time_after_date == 0:
                                count_date += 1
                            else:
                                count_datetime += 1
                    except:
                        count_str += 1
                    continue
                if isinstance(s_type, int):
                    count_int += 1
                elif isinstance(s_type, float):
                    if s_type.is_integer():
                        count_int += 1
                    else:
                        count_float += 1
                elif isinstance(s_type, bytes):
                    count_bytes += 1
                elif s_type is None:
                    continue
                else:
                    count_str += 1
            dtype_results = (count_str, count_int, count_float, count_datetime, count_date, count_bytes)
            max_number_dtypes_found = max(dtype_results)
            if count_str != 0:
                inferred_dtypes[s_header] = "str"
                continue
            if max_number_dtypes_found > 0:
                dtype_maximum = dtype_labels[dtype_results.index(max_number_dtypes_found)]
                dtype_maximum_ratio = round(max_number_dtypes_found/count_checked,2) if count_checked > 0 else 0.0
                if dtype_maximum_ratio >= infer_threshold:
                    if (dtype_maximum == "date") and (count_datetime != 0):
                        inferred_dtypes[s_header] = "datetime"
                    elif (dtype_maximum == "int") and (count_float != 0):
                        inferred_dtypes[s_header] = "float"                        
                    else:
                        inferred_dtypes[s_header] = dtype_maximum
        for column, dtype in inferred_dtypes.items():
            if dtype != "str":
                self.set_column_dtypes(column,dtype)

#############################################################################################################
############################################### class methods ###############################################
#############################################################################################################

    @classmethod
    def from_shape(cls, n_rows:int, n_cols:int, dtype:Literal['bytes','datetime','float','int','str']=None) -> SQLDataModel:
        """
        Returns an empty SQLDataModel of shape (`n_rows`, `n_cols`) as a convenience method to quickly build a model through an iterative approach. 
        By default, no particular `dtype` is assigned given the flexibility of sqlite3 when it comes to column datatypes, however one can be enforced in the schema be providing a valid datatype identifier to the `dtype` keyword argument.

        Parameters:
            - `n_rows` (int): The number of rows for the new SQLDataModel.
            - `n_cols` (int): The number of columns for the new SQLDataModel.
            - `dtype` (str) (optional): A valid python or SQL datatype to initialize the n-dimensional model with

        Raises:
            - `TypeError`: If `n_rows` is not of type 'int' or if `n_cols` is not of type 'int'.
            - `ValueError`: If `dtype` is not a valid python or SQL convertible datatype to initialize the model with.

        Returns:
            - `SQLDataModel`: Instance with the specified number of rows and columns, initialized with by `dtype` fill values or with `None` values (default).

        Example:
        ```python
        import SQLDataModel

        # Define shape
        n_rows, n_cols = 10, 3

        # Create an empty SQLDataModel with 10 rows and 3 columns
        sdm = SQLDataModel.from_shape(n_rows, n_cols)

        # Assign rows
        for row in range(n_rows):
            sdm[row] = ('john', 'smith', 29)
        
        # Assign specific cells
        for row in range(n_rows):
            for col in range(n_cols):
                sdm[row, col] = f"cell at index {row}, {col}"
        
        # Assign headers if needed
        sdm.set_headers(['first', 'last', 'age'])
        ```
        """
        if not isinstance(n_rows, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(n_rows).__name__}', argument of type 'int' expected for `n_rows` parameter")
            )
        if not isinstance(n_cols, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(n_cols).__name__}', argument of type 'int' expected for `n_cols` parameter")
            )
        if dtype is not None:
            if dtype not in ('bytes','datetime','float','int','str'):
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid argument '{dtype}', `dtype` must be one of 'bytes','datetime','float','int','str'")
                )
            else:
                if dtype == 'bytes':
                    fill_value = b''
                elif dtype == 'datetime':
                    fill_value = datetime.datetime.now()
                elif dtype == 'float':
                    fill_value = 0.0
                elif dtype == 'int':
                    fill_value = 0
                else:
                    fill_value = ''
        else:
            fill_value = None
        return cls([[fill_value for _ in range(n_cols)] for _ in range(n_rows)])
        
    @classmethod
    def from_csv(cls, csv_file:str, encoding:str="Latin1", delimeter:str=',', quotechar:str='"', headers:list[str] = None, **kwargs) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` from the provided CSV file.

        Parameters:
            - `csv_file` (str): The path to the CSV file.
            - `encoding` (str, optional): The encoding used to decode the CSV file. Default is 'Latin1'.
            - `delimiter` (str, optional): The delimiter used in the CSV file. Default is ','.
            - `quotechar` (str, optional): The character used for quoting fields. Default is '"'.
            - `headers` (list[str], optional): List of column headers. If None, the first row of the CSV file is assumed to contain headers.
            - `*args`, `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the provided CSV file.

        Example:
        ```python
        import SQLDataModel

        # CSV file to import
        csvfile = "/user/home/example.csv"

        # Create the model using  the file
        sdm = SQLDataModel.from_csv(csvfile, headers=['ID', 'Name', 'Value'])
        ```
        """
        with open(csv_file, encoding=encoding) as csvfile:
            tmp_all_rows = tuple(list(row) for row in csv.reader(csvfile,delimiter=delimeter,quotechar=quotechar))
        return cls(tmp_all_rows[1:],tmp_all_rows[0] if headers is None else headers, **kwargs)   
    
    @classmethod
    def from_data(cls, data:list[list], headers:list[str]=None, **kwargs) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` from the provided data.

        Parameters:
            - `data` (list[list]): The data to populate the SQLDataModel.
            - `headers` (list[str], optional): List of column headers. If None, no headers are used.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the provided data.

        Example:
        ```python
        data = [[1, 'John', 30], [2, 'Jane', 25], [3, 'Bob', 40]]
        sqldm = SQLDataModel.from_data(data, headers=['ID', 'Name', 'Age'])
        ```
        """
        return cls(data, headers, **kwargs)
    
    @classmethod
    def from_dict(cls, data:dict, **kwargs) -> SQLDataModel:
        """
        Create a new `SQLDataModel` instance from the provided dictionary.

        Parameters:
            - `data` (dict): The dictionary to convert to a SQLDataModel.
            If keys are of type int, they will be used as row indexes; otherwise, keys will be used as headers.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the provided dictionary.

        Raises:
            - `TypeError`: If the provided dictionary values are not of type 'list', 'tuple', or 'dict'.

        Example:
        ```python
        data_dict = {1: [10, 'A'], 2: [20, 'B'], 3: [30, 'C']}
        sdm_obj = SQLDataModel.from_dict(data_dict)
        ```

        Note:
            - The method determines the structure of the SQLDataModel based on the format of the provided dictionary.
            - If the keys are integers, they are used as row indexes; otherwise, keys are used as headers.
        """
        rowwise = True if all(isinstance(x, int) for x in data.keys()) else False
        if rowwise:
            # column_count = len(data[next(iter(data))])
            headers = ['idx',*[f'col_{i}' for i in range(len(data[next(iter(data))]))]] # get column count from first key value pair in provided dict
            return cls([tuple([k,*v]) for k,v in data.items()], headers, **kwargs)
        else:
            first_key_val = data[next(iter(data))]
            if isinstance(first_key_val, dict):
                headers = list(data.keys())
                data = [[data[col][val] for col in headers] for val in data.keys()]
            elif isinstance(first_key_val, list|tuple):
                headers = [k for k in data.keys()]
                column_count = len(headers)
                row_count = len(first_key_val)
                data = [x for x in data.values()]
                data = [tuple([data[j][row] for j in range(column_count)]) for row in range(row_count)]
            else:
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid dict values, received type '{type(first_key_val).__name__}' but expected dict values as one of type 'list', 'tuple' or 'dict'")
                )
            return cls(data, headers, **kwargs)
        
    @classmethod
    def from_numpy(cls, array, headers:list[str]=None, **kwargs) -> SQLDataModel:
        """
        Returns a `SQLDataModel` object created from the provided numpy `array`.

        Parameters:
            - `array` (numpy.ndarray): The numpy array to convert to a SQLDataModel.
            - `headers` (list of str, optional): The list of headers to use for the SQLDataModel. If None, no headers will be used, and the data will be treated as an n-dimensional array. Default is None.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the numpy array.

        Raises:
            - `ModuleNotFoundError`: If the required package `numpy` is not found.

        Example:
        ```python
        import numpy as np
        sdm_obj = SQLDataModel.from_numpy(np.array([[1, 'a'], [2, 'b'], [3, 'c']]), headers=['Number', 'Letter'])
        ```
        """
        if not _has_np:
            raise ModuleNotFoundError(
                SQLDataModel.ErrorFormat(f"""ModuleNotFoundError: required package not found, numpy must be installed in order to use the `from_numpy()` method""")
                )
        return cls(data=array.tolist(),headers=headers, **kwargs)

    @classmethod
    def from_pandas(cls, df, headers:list[str]=None, **kwargs) -> SQLDataModel:
        """
        Returns a `SQLDataModel` object created from the provided pandas `DataFrame`. Note that `pandas` must be installed in order to use this class method.

        Parameters:
            - `df` (pandas.DataFrame): The pandas DataFrame to convert to a SQLDataModel.
            - `headers` (list of str, optional): The list of headers to use for the SQLDataModel. If None, the columns of the DataFrame will be used. Default is None.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the pandas DataFrame.

        Raises:
            - `ModuleNotFoundError`: If the required package `pandas` is not found.

        Example:
        ```python
        import pandas as pd

        sqldm = SQLDataModel.from_pandas(pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']}))
        ```
        """
        if not _has_pd:
            raise ModuleNotFoundError(
                SQLDataModel.ErrorFormat(f"""ModuleNotFoundError: required package not found, pandas must be installed in order to use the `from_pandas()` method""")
                )
        data = [x[1:] for x in df.itertuples()]
        headers = df.columns.tolist() if headers is None else headers
        return cls(data=data,headers=headers, **kwargs)
    
    @classmethod
    def from_pickle(cls, filename:str=None, **kwargs) -> SQLDataModel:
        """
        Returns the `SQLDataModel` object from the provided `filename`. If `None`, the current directory will be scanned for the default `to_pickle()` format.

        Parameters:
            - `filename` (str, optional): The name of the pickle file to load. If None, the current directory will be scanned for the default filename. Default is None.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the loaded pickle file.

        Raises:
            - `FileNotFoundError`: If the provided filename could not be found or does not exist.

        Example:
        ```python
        sdm_obj = SQLDataModel.from_pickle("data.sdm")
        ```
        """
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        if (filename is not None) and (len(filename.split(".")) <= 1):
            print(SQLDataModel.WarnFormat(f"SQLDataModelWarning: file extension missing, provided filename \"{filename}\" did not contain an extension and so \".sdm\" was appended to create a valid filename..."))
            filename += '.sdm'
        if not Path(filename).is_file():
            raise FileNotFoundError(
                SQLDataModel.ErrorFormat(f"FileNotFoundError: file not found, provided filename \"{filename}\" could not be found, please ensure the filename exists in a valid path...")
                )
        with open(filename, 'rb') as f:
            tot_raw = pickle.load(f) # Tuple of Tuples raw data
            return cls(tot_raw[1:],headers=tot_raw[0], **kwargs)

    @classmethod
    def get_supported_sql_connections(cls) -> tuple:
        """
        Returns the currently tested DB API 2.0 dialects for use with `SQLDataModel.from_sql()` method.

        Returns:
            - `tuple`: A tuple of supported DB API 2.0 dialects.

        Example:
        ```python
        import SQLDataModel

        # Get supported dialects
        supported_dialects = SQLDataModel.get_supported_sql_connections()

        # View details
        print(supported_dialects)

        # Outputs
        supported_dialects = ('sqlite3', 'psycopg2', 'pyodbc', 'cx_oracle', 'teradatasql')
        ```
        """
        return ('sqlite3', 'psycopg2', 'pyodbc', 'cx_oracle', 'teradatasql')
    
    @classmethod
    def from_sql(cls, sql_query: str, sql_connection: sqlite3.Connection, **kwargs) -> SQLDataModel:
        """
        Create a `SQLDataModel` object by executing the provided SQL query using the specified SQL connection.
        If a single word is provided as the `sql_query`, the method wraps it and executes a select all treating the text as the target table.

        Parameters:
            - `sql_query` (str): The SQL query to execute and create the SQLDataModel.
            - `sql_connection` (sqlite3.Connection): The SQLite3 database connection object.
            - `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            - `SQLDataModel`: The SQLDataModel object created from the executed SQL query.

        Raises:
            - `WarnFormat`: If the provided SQL connection has not been tested, a warning is issued.
            - `SQLProgrammingError`: If the provided SQL connection is not opened or valid, or the SQL query is invalid or malformed.
            - `DimensionError`: If the provided SQL query returns no data.

        ---
        Example:

        ```python
        import SQLDataModel

        # Single word parameter
        sqldm = SQLDataModel.from_sql("table_name", sqlite3.Connection)
        
        # Equilavent query executed
        sqldm = SQLDataModel.from_sql("select * from table_name", sqlite3.Connection)
        ```

        ---
        Example with sqlite3:
        ```python
        import SQLDataModel, sqlite3

        # Create connection object
        sqlite_db_conn = sqlite3.connect('./database/users.db')

        # Basic usage with a select query
        sqldm = SQLDataModel.from_sql("SELECT * FROM my_table", sqlite_db_conn)

        # When a single word is provided, it is treated as a table name for a select all query
        sqldm_table = SQLDataModel.from_sql("my_table", sqlite_db_conn)
        ```
        ---
        Example with psycopg2:
        ```python
        import SQLDataModel, psycopg2

        # Create connection object
        pg_db_conn = psycopg2.connect('dbname=users user=postgres password=postgres')
        
        # Basic usage with a select query
        sqldm = SQLDataModel.from_sql("SELECT * FROM my_table", pg_db_conn)

        # When a single word is provided, it is treated as a table name for a select all query
        sqldm_table = SQLDataModel.from_sql("my_table", pg_db_conn)
        ```        
        ---
        Example with pyodbc:
        ```python
        import SQLDataModel, pyodbc

        # Create connection object
        sqls_db_conn = pyodbc.connect("DRIVER={SQL Server};SERVER=host;DATABASE=db;UID=user;PWD=pw;")
        
        # Basic usage with a select query
        sqldm = SQLDataModel.from_sql("SELECT * FROM my_table", sqls_db_conn)

        # When a single word is provided, it is treated as a table name for a select all query
        sqldm_table = SQLDataModel.from_sql("my_table", sqls_db_conn)
        ```    
        ---
        Supported database connection APIs:
            - `sqlite3`
            - `psycopg2`
            - `pyodbc`
            - `cx_Oracle`
            - `teradatasql`

        Note:
            - Connections with write access can be used in the `to_sql()` method for writing to the same connection types.
            - Unsupported connection object will output a `SQLDataModelWarning` advising unstable or undefined behaviour.
        """
        ### type checking connection to ensure compatible ###
        db_dialect = type(sql_connection).__module__.split('.')[0].lower()
        if db_dialect not in cls.get_supported_sql_connections():
            print(SQLDataModel.WarnFormat(f"""SQLDataModelWarning: provided SQL connection has not been tested, behavior for "{db_dialect}" may be unpredictable or unstable..."""))
        if len(sql_query.split()) == 1:
            sql_query = f""" select * from {sql_query} """
        try:
            sql_c = sql_connection.cursor()
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f"SQLProgrammingError: provided SQL connection is not opened or valid, failed with: {e}...")
            ) from None
        try:
            sql_c.execute(sql_query)
            data = sql_c.fetchall()
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f"SQLProgrammingError: provided SQL query is invalid or malformed, failed with: {e}...")
            ) from None
        if (len(data) < 1) or (data is None):
            raise DimensionError("DimensionError: provided SQL query returned no data, please provide a valid query with sufficient return data...")
        headers = [x[0] for x in sql_c.description]
        return cls(data, headers, **kwargs)

################################################################################################################
############################################## conversion methods ##############################################
################################################################################################################

    def data(self, include_index:bool=False, include_headers:bool=False) -> list[tuple]:
        """
        Returns the `SQLDataModel` data as a list of tuples for multiple rows or as a single tuple for individual rows. 
        Data is returned without index and headers by default. Use `include_headers=True` or `include_index=True` to modify.

        Parameters:
            - `include_index` (bool, optional): If True, includes the index in the result; if False, excludes the index. Default is False.
            - `include_headers` (bool, optional): If True, includes column headers in the result; if False, excludes headers. Default is False.

        Returns:
            list: The list of tuples representing the SQLDataModel data.

        Example:
        ```python
        result_data = sqldm.data(include_index=True, include_headers=False)
        ```
        """
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index))
        data = res.fetchall()
        if (len(data) == 1) and (not include_headers): # if only single row
            data = data[0]
        if len(data) == 1: # if only single cell
            data = data[0]
        return [tuple([x[0] for x in res.description]),data] if include_headers else data

    def to_csv(self, csv_file:str, delimeter:str=',', quotechar:str='"', include_index:bool=False, **kwargs) -> None:
        """
        Writes `SQLDataModel` to the specified file in the `csv_file` argument.
        The file must have a compatible `.csv` file extension.

        Parameters:
            - `csv_file` (str): The name of the CSV file to which the data will be written.
            - `delimiter` (str, optional): The delimiter to use for separating values. Default is ','.
            - `quotechar` (str, optional): The character used to quote fields. Default is '"'.
            - `include_index` (bool, optional): If True, includes the index in the CSV file; if False, excludes the index. Default is False.
            - `**kwargs`: Additional arguments to be passed to the `csv.writer` constructor.

        Returns:
            None

        Example:
        ```python
        import SQLDataModel

        # Create instance
        sqldm = SQLDataModel.from_csv('raw-data.csv')

        # Save SQLDataModel to csv file in data directory:
        sqldm.to_csv("./data/analysis.csv", include_index=True)

        # Save SQLDataModel as tab separated values instead:
        sqldm.to_csv("./data/analysis.csv", delimiter='\\t', include_index=False)
        ```
        """
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index))
        write_headers = [x[0] for x in res.description]
        with open(csv_file, 'w', newline='') as file:
            csvwriter = csv.writer(file,delimiter=delimeter,quotechar=quotechar,quoting=csv.QUOTE_MINIMAL, **kwargs)
            csvwriter.writerow(write_headers)
            csvwriter.writerows(res.fetchall())
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: csv file "{csv_file}" created'))

    def to_dict(self, rowwise:bool=True) -> dict:
        """
        Convert the `SQLDataModel` to a dictionary, using either index rows or model headers as keys.

        Parameters:
            - `rowwise` (bool, optional): If True, use index rows as keys; if False, use model headers as keys. Default is True.

        Returns:
            `dict`: The dictionary representation of the SQLDataModel.

        Examples:
        ```python
        import SQLDataModel

        # Convert to a dictionary using index rows as keys
        result_dict = sqldm.to_dict(rowwise=True)

        # Example of output format:
        result_dict = {
             0: ['john', 'smith', 'new york']
            ,1: ['sarah', 'west', 'chicago']
        }

        # Convert to a dictionary using model headers as keys
        result_dict = sqldm.to_dict(rowwise=False)

        # Example of output format when not rowwise:
        result_dict = {
            "first_name" : ['john', 'sarah']
            "last_name" : ['smith', 'west']
            "city" : ['new york', 'chicago']
        }
        ```
        Note:
            - The method uses all of the underlying data, to selectively output simple index the SQLDataModel instance.
            - If `rowwise` is True, each index row is a key, and its corresponding values are a tuple of the row data.
            - If `rowwise` is False, each model header is a key, and its corresponding values are a tuple of the column data.
        """
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=True))
        if rowwise:
            return {row[0]:row[1:] for row in res.fetchall()}
        else:
            data = res.fetchall()
            headers = [x[0] for x in res.description]
            return {headers[i]:tuple([x[i] for x in data]) for i in range(len(headers))}
        
    def to_list(self, include_index:bool=True, include_headers:bool=False) -> list[tuple]:
        """
        Returns a list of tuples containing all the `SQLDataModel` data without the headers by default.
        Use `include_headers=True` to return the headers as the first item in the returned sequence.

        Parameters:
            - `include_index` (bool, optional): If True, includes the index in the result; if False, excludes the index. Default is True.
            - `include_headers` (bool, optional): If True, includes column headers in the result; if False, excludes headers. Default is False.

        Returns:
            `list`: The list of tuples representing the SQLDataModel data.

        Example:
        ```python
        import SQLDataModel

        # Output the data with indicies but without headers:
        result_list = sqldm.to_list(include_index=True, include_headers=False)

        # Format of output:
        output_list = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'patrick', 'mcdouglas', 42)
        ]

        # Output the data without indicies and with headers:
        result_list = sqldm.to_list(include_index=False, include_headers=True)

        # Format of output:
        output_list = [
            ('first', 'last', 'age')
            ,('john', 'smith', 27)
            ,('sarah', 'west', 29)
            ,('patrick', 'mcdouglas', 42)
        ]
        ```
        """
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index))
        return [tuple([x[0] for x in res.description]),*res.fetchall()] if include_headers else res.fetchall()
    
    def to_numpy(self, include_index:bool=False, include_headers:bool=False) -> _np.ndarray:
        """
        Converts `SQLDataModel` to a NumPy `ndarray` object of shape (rows, columns).
        Note that NumPy must be installed to use this method.

        Parameters:
            - `include_index` (bool, optional): If True, includes the model index in the result. Default is False.
            - `include_headers` (bool, optional): If True, includes column headers in the result. Default is False.

        Returns:
            `numpy.ndarray`: The model's data converted into a NumPy array.

        Example:
        ```python
        import SQLDataModel, numpy

        # Create the numpy array with default parameters, no indicies or headers
        result_array = sqldm.to_numpy()

        # Example output format
        result_array = numpy.ndarray([
            ['john' 'smith' '27']
            ,['sarah' 'west' '29']
            ,['mike' 'harlin' '36']
            ,['pat' 'douglas' '42']
        ])

        # Create the numpy array with with indicies and headers
        result_array = sqldm.to_numpy(include_index=True, include_headers=True)

        # Example of output format
        result_array = numpy.ndarray([
            ['idx' 'first' 'last' 'age']
            ,['0' 'john' 'smith' '27']
            ,['1' 'sarah' 'west' '29']
            ,['2' 'mike' 'harlin' '36']
            ,['3' 'pat' 'douglas' '42']
        ])
        ```

        Raises:
            - `ModuleNotFoundError`: If NumPy is not installed.
        """
        if not _has_np:
            raise ModuleNotFoundError(
                SQLDataModel.ErrorFormat(f"""ModuleNotFoundError: required package not found, numpy must be installed in order to use `.to_numpy()` method""")
                )            
        fetch_stmt = self._generate_sql_stmt(include_index=include_index)
        res = self.sql_db_conn.execute(fetch_stmt)
        if include_headers:
            return _np.vstack([_np.array([x[0] for x in res.description]),[_np.array(x) for x in res.fetchall()]])
        return _np.array([_np.array(x) for x in res.fetchall()])

    def to_pandas(self, include_index:bool=False, include_headers:bool=True) -> _pd.DataFrame:
        """
        Converts `SQLDataModel` to a Pandas `DataFrame` object.
        Note that Pandas must be installed to use this method.

        Parameters:
            - `include_index` (bool, optional): If True, includes the model index in the result. Default is False.
            - `include_headers` (bool, optional): If True, includes column headers in the result. Default is True.

        Returns:
            `pandas.DataFrame`: The model's data converted to a Pandas DataFrame.

        Example:
        ```python
        import SQLDataModel, pandas

        # Output the model data as a pandas dataframe
        result_df = sqldm.to_pandas(include_index=True, include_headers=False)
        ```

        Raises:
            `ModuleNotFoundError`: If Pandas is not installed.
        """
        if not _has_pd:
            raise ModuleNotFoundError(
                SQLDataModel.ErrorFormat(f"""ModuleNotFoundError: required package not found, pandas must be installed in order to use `.to_pandas()` method""")
                )
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index))
        raw_data = res.fetchall()
        data = [x[1:] for x in raw_data] if include_index else [x for x in raw_data]
        indicies = [x[0] for x in raw_data] if include_index else None
        columns = ([x[0] for x in res.description[1:]] if include_index else [x[0] for x in res.description]) if include_headers else None
        return _pd.DataFrame(data=data,columns=columns,index=indicies)

    def to_pickle(self, filename:str=None) -> None:
        """
        Save the `SQLDataModel` instance to the specified `filename`.

        By default, the name of the invoking Python file will be used.

        Parameters:
            - `filename` (str, optional): The name of the file to which the instance will be saved. If not provided,
            the invoking Python file's name with a ".sdm" extension will be used.

        Returns:
            `None`

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
         (0, 'john', 'smith', 27)
        ,(1, 'sarah', 'west', 29)
        ,(2, 'mike', 'harlin', 36)
        ,(3, 'pat', 'douglas', 42)
        ]
        
        # Create the SQLDataModel object
        sqldm = SQLDataModel(data, headers)

        # Save the model's data as a pickle file "output.sdm"
        sqldm.to_pickle("output.sdm")

        # Alternatively, leave blank to use the current file's name:
        sqldm.to_pickle()

        # This way the same data can be recreated later by calling the from_pickle() method from the same project:
        sqldm = SQLDataModel.from_pickle()
        ```
        """
        if (filename is not None) and (len(filename.split(".")) <= 1):
            print(SQLDataModel.WarnFormat(f"{type(self).__name__}Warning: extension missing, provided filename '{filename}' did not contain an extension and so '.sdm' was appended to create a valid filename"))
            filename += '.sdm'
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        serialized_data = tuple(x for x in self.iter_rows(include_index=True,include_headers=True)) # no need to send sql_store_id aka index to pickle
        with open(filename, 'wb') as handle:
            pickle.dump(serialized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: pickle file "{filename}" created'))

    def to_sql(self, table:str, extern_conn:sqlite3.Connection, replace_existing:bool=True, include_index:bool=True) -> None:
        """
        Insert the `SQLDataModel` into the specified table using the provided SQLite database connection object.

        Two modes are available:
        - `replace_existing=True`: Deletes the existing table and replaces it with the SQLDataModel's data.
        - `replace_existing=False`: Appends to the existing table and performs deduplication immediately after.

        Use `SQLDataModel.get_supported_sql_connections()` to view supported connections.
        Use `include_index=True` to retain the model index in the target table.

        Parameters:
            - `table` (str): The name of the table where data will be inserted.
            - `extern_conn` (sqlite3.Connection): The SQLite database connection object.
            - `replace_existing` (bool, optional): If True, replaces the existing table; if False, appends to the existing table. Default is True.
            - `include_index` (bool, optional): If True, retains the model index in the target table. Default is True.

        Returns:
            `None`

        Raises:
            - `SQLProgrammingError`: If the provided SQL connection is not open.
        ---
        Example:

        ```python
        import SQLDataModel, sqlite3

        # Create connection object
        sqlite_db_conn = sqlite3.connect('./database/users.db')

        # Basic usage with insert, replace existing table, and exclude index
        sqldm.to_sql("my_table", sqlite_db_conn, replace_existing=True, include_index=False)

        # Append to the existing table and perform deduplication
        sqldm.to_sql("my_table", sqlite_db_conn, replace_existing=False, include_index=True)
        ```

        ---
        Supported database connection APIs:
        - `sqlite3`
        - `psycopg2`
        - `pyodbc`
        - `cx_Oracle`
        - `teradatasql`

        Note:
        - Connections with write access can be used in the `to_sql()` method for writing to the same connection types.
        - Unsupported connection objects will output a `SQLDataModelWarning` advising unstable or undefined behavior.
        """
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index))
        model_data = [x for x in res.fetchall()] # using new process
        model_headers = [x[0] for x in res.description]
        created_header_dict = {col:f"{self.header_master[col][0]}" if self.header_master[col][2] else f"{self.header_master[col][0]} PRIMARY KEY" for col in model_headers}
        try:
            extern_c = extern_conn.cursor()
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f"""SQLProgrammingError: provided SQL connection is not open, please reopen the database connection or resolve "{e}"...""")
            ) from None
        if replace_existing:
            extern_c.execute(f"""drop table if exists "{table}" """)
            extern_conn.commit()
        db_dialect = type(extern_conn).__module__.split('.')[0].lower()
        dyn_bind = '?' if db_dialect == 'sqlite3' else '%s'
        sql_dtypes_stmt = ", ".join(f""" "{header}" {created_header_dict[header]}""" for header in model_headers) # generates sql create table statement using type mapping dict
        sql_create_stmt = f"""create table if not exists "{table}" ({sql_dtypes_stmt})"""
        sql_insert_stmt = f"""insert into "{table}" ({','.join([f'"{col}"' for col in model_headers])}) values ({','.join([dyn_bind for _ in model_headers])})""" # changed to string formatter
        extern_c.execute(sql_create_stmt)
        extern_conn.commit()
        extern_c.executemany(sql_insert_stmt,model_data)
        extern_conn.commit()
        if not replace_existing:
            sql_dedupe_stmt = f"""delete from "{table}" where rowid not in (select min(rowid) from "{table}" group by {','.join(f'"{col}"' for col in model_headers)})"""
            extern_c.execute(sql_dedupe_stmt)
            extern_conn.commit()
        return

    def to_text(self, filename:str, include_ts:bool=False) -> None:
        """
        Writes contents of `SQLDataModel` to the specified `filename` as text representation.

        Parameters:
            - `filename` (str): The name of the file to which the contents will be written.
            - `include_ts` (bool, optional): If True, includes a timestamp in the file. Default is False.

        Returns:
            - `None`

        Example:
        ```python
        import SQLDataModel

        sqldm.to_text("output.txt", include_ts=True)
        ```
        """
        contents = f"{datetime.datetime.now().strftime('%B %d %Y %H:%M:%S')} status:\n" + self.__repr__() if include_ts else self.__repr__()
        with open(filename, "w", encoding='utf-8') as file:
            file.write(contents)
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: text file "{filename}" created'))

    def to_local_db(self, db:str=None) -> None:
        """
        Stores the `SQLDataModel` internal in-memory database to a local disk database.

        Parameters:
            - `db` (str, optional): The filename or path of the target local disk database.
            If not provided (None), the current filename will be used as a default target.

        Raises:
            - `sqlite3.Error`: If there is an issue with the SQLite database operations during backup.

        Note:
            - The method connects to the specified local disk database using sqlite3.
            - It performs a backup of the in-memory database to the local disk database.
            - If `db=None`, the current filename is used as the default target.
            - After successful backup, it prints a success message indicating the creation of the local database.

        Example:
        ```python
        # Example 1: Store the in-memory database to a local disk database with a specific filename
        sqldm.to_local_db("local_database.db")

        # Example 2: Store the in-memory database to a local disk database using the default filename
        sqldm.to_local_db()
        ```
        """
        with sqlite3.connect(db) as target:
            self.sql_db_conn.backup(target)
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: local db "{db}" created'))

####################################################################################################################
############################################## dunder special methods ##############################################
####################################################################################################################

    def __add__(self, value:str|int|float|SQLDataModel) -> SQLDataModel:
        """
        Implements the + operator functionality for compatible `SQLDataModel` operations.

        Parameters:
        - `value` (str | int | float): The value to be added to each element in the SQLDataModel.

        Returns:
        - `SQLDataModel`: A new SQLDataModel resulting from the addition operation.

        Raises:
        - `TypeError`: If the provided `value` is not a valid type (str, int, or float).

        Example:
        ```python
        # Example usage for strings:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name'])
        sqldm['Loud Name'] = sqldm['First Name'] + '!'

        # Example usage for integers:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['Age', 'Years of Service'])
        sqldm['Age'] = sqldm['Age'] + 7 # it's a cruel world after all
        ```
        """
        if not isinstance(value, str|int|float|SQLDataModel):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: unsupported operand type '{type(value).__name__}', addition operations can only be performed on types 'str', 'int' or 'float' ")
            )
        if isinstance(value, SQLDataModel):
            value = value.data()
        if isinstance(value, str|int|float):
            return self.apply(lambda x: x + value)

    def __sub__(self, value:int|float|SQLDataModel) -> SQLDataModel:
        """
        Implements the - operator functionality for compatible `SQLDataModel` operations.

        Parameters:
        - `value` (int | float): The value to subtract from each element in the SQLDataModel.

        Returns:
        - `SQLDataModel`: A new SQLDataModel resulting from the subtraction operation.

        Raises:
        - `TypeError`: If the provided `value` is not a valid type (int or float).

        Example:
        ```python
        # Example usage:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Numbers'])
        sqldm['Adjusted Numbers'] = sqldm['Numbers'] - 2.5
        ```
        """
        if not isinstance(value, int|float|SQLDataModel):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: unsupported operand type '{type(value).__name__}', subtraction operations can only be performed on types 'int' or 'float' ")
            )
        if isinstance(value, SQLDataModel):
            value = value.data()
        if isinstance(value, int|float):
            return self.apply(lambda x: x - value)

    def __mul__(self, value:int|float) -> SQLDataModel:
        """
        Implements the * operator functionality for compatible `SQLDataModel` operations.

        Parameters:
        - `value` (int | float): The value to multiply each element in the SQLDataModel by.

        Returns:
        - `SQLDataModel`: A new SQLDataModel resulting from the multiplication operation.

        Raises:
        - `TypeError`: If the provided `value` is not a valid type (int or float).

        Example:
        ```python
        # Example usage:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Monthly Cost'])
        new_sdm['Yearly Cost'] = sqldm['Monthly Cost'] * 12
        ```
        """
        if not isinstance(value, int|float):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: unsupported operand type '{type(value).__name__}', multiplication operations can only be performed on types 'int' or 'float' ")
            )
        if isinstance(value, int|float):
            return self.apply(lambda x: x * value)

    def __truediv__(self, value:int|float) -> SQLDataModel:
        """
        Implements the / operator functionality for compatible `SQLDataModel` operations.

        Parameters:
        - `value` (int | float): The value to divide each element in the SQLDataModel by.

        Returns:
        - `SQLDataModel`: A new SQLDataModel resulting from the division operation.

        Raises:
        - `TypeError`: If the provided `value` is not a valid type (int or float).
        - `ZeroDivisionError`: If `value` is 0.

        Example:
        ```python
        # Example usage:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Yearly Amount'])
        sqldm['Weekly Amount'] = sqldm['Yearly Amount'] / 52
        ```
        """
        if not isinstance(value, int|float):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: unsupported operand type '{type(value).__name__}', division operations can only be performed on types 'int' or 'float' ")
            )
        if isinstance(value, int|float):
            return self.apply(lambda x: x / value)
        
    def __pow__(self, value:int|float) -> SQLDataModel:
        """
        Implements the ** operator functionality for compatible `SQLDataModel` operations.

        Parameters:
        - `value` (int | float): The value to raise each element in the SQLDataModel to.

        Returns:
        - `SQLDataModel`: A new SQLDataModel resulting from the exponential operation.

        Raises:
        - `TypeError`: If the provided `value` is not a valid type (int or float).

        Example:
        ```python
        # Example usage:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Numbers'])
        sqldm['Numbers Squared'] = sqldm['Numbers'] ** 2
        ```
        """
        if not isinstance(value, int|float):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: unsupported operand type '{type(value).__name__}', exponential operations can only be performed on types 'int' or 'float' ")
            )
        if isinstance(value, int|float):
            return self.apply(lambda x: x ** value)        

    def __iadd__(self, value) -> SQLDataModel:
        """
        Implements the += operator functionality for compatible `SQLDataModel` operations.

        Parameters:
            - `value` (str | int | float | SQLDataModel): The value to be added to each element in the SQLDataModel.

        Returns:
            - `SQLDataModel`: The modified SQLDataModel after the addition operation.

        Raises:
            - `TypeError`: If the provided `value` is not a valid type (str, int, float, or SQLDataModel).

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service']
        data = [
            (0, 'john', 'smith', 27, 1.22),
            (1, 'sarah', 'west', 39, 0.7),
            (2, 'mike', 'harlin', 36, 3),
            (3, 'pat', 'douglas', 42, 11.5)
        ]     

        # Create the model
        sdm = SQLDataModel(data, headers)

        # Modifying first name column with a bang!
        sdm['first'] += '!'

        # View model
        print(sdm)

        # Output
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬─────────┐
        │   │ first  │ last    │    age │ service │
        ├───┼────────┼─────────┼────────┼─────────┤
        │ 0 │ john!  │ smith   │     27 │    1.22 │
        │ 1 │ sarah! │ west    │     39 │    0.70 │
        │ 2 │ mike!  │ harlin  │     36 │    3.00 │
        │ 3 │ pat!   │ douglas │     42 │   11.50 │
        └───┴────────┴─────────┴────────┴─────────┘
        [4 rows x 4 columns]
        ```        
        """        
        return self.__add__(value)
    
    def __isub__(self, value) -> SQLDataModel:
        """
        Implements the -= operator functionality for compatible `SQLDataModel` operations.

        Parameters:
            - `value` (int | float | SQLDataModel): The value to subtract from each element in the SQLDataModel.

        Returns:
            - `SQLDataModel`: The modified SQLDataModel after the subtraction operation.

        Raises:
            - `TypeError`: If the provided `value` is not a valid type (int, float, or SQLDataModel).

        Example:        
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service']
        data = [
            (0, 'john', 'smith', 27, 1.22),
            (1, 'sarah', 'west', 39, 0.7),
            (2, 'mike', 'harlin', 36, 3),
            (3, 'pat', 'douglas', 42, 11.5)
        ]     

        # Create the model
        sdm = SQLDataModel(data, headers)

        # Modifying age column in the best direction
        sdm['age'] -= 10

        # View model
        print(sdm)

        # Output
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬─────────┐
        │   │ first  │ last    │    age │ service │
        ├───┼────────┼─────────┼────────┼─────────┤
        │ 0 │ john   │ smith   │     17 │    1.22 │
        │ 1 │ sarah  │ west    │     29 │    0.70 │
        │ 2 │ mike   │ harlin  │     26 │    3.00 │
        │ 3 │ pat    │ douglas │     32 │   11.50 │
        └───┴────────┴─────────┴────────┴─────────┘
        [4 rows x 4 columns]
        ```
        """
        return self.__sub__(value)    
    
    def __imul__(self, value) -> SQLDataModel:
        """
        Implements the *= operator functionality for compatible `SQLDataModel` operations.

        Parameters:
            - `value` (int | float): The value to multiply each element in the SQLDataModel by.

        Returns:
            - `SQLDataModel`: The modified SQLDataModel after the multiplication operation.

        Raises:
            - `TypeError`: If the provided `value` is not a valid type (int or float).

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Salary'])

        # Give raises to all!
        sdm['Salary'] *= 12
        ```
        """        
        return self.__mul__(value)    

    def __idiv__(self, value) -> SQLDataModel:
        """
        Implements the /= operator functionality for compatible `SQLDataModel` operations.

        Parameters:
            - `value` (int | float): The value to divide each element in the SQLDataModel by.

        Returns:
            - `SQLDataModel`: The modified SQLDataModel after the division operation.

        Raises:
            - `TypeError`: If the provided `value` is not a valid type (int or float).
            - `ZeroDivisionError`: If `value` is 0.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Budget'])

        # Adjust existing column
        sdm['Budget'] /= 52
        ```
        """        
        return self.__truediv__(value)
    
    def __ipow__(self, value) -> SQLDataModel:
        """
        Implements the **= operator functionality for compatible `SQLDataModel` operations.

        Parameters:
            - `value` (int | float): The value to raise each element in the SQLDataModel to.

        Returns:
            - `SQLDataModel`: The modified SQLDataModel after the exponential operation.

        Raises:
            - `TypeError`: If the provided `value` is not a valid type (int or float).

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Salary'])

        # More raises!
        sdm['Salary'] **= 2
        ```
        """        
        return self.__pow__(value)
    
    def __iter__(self) -> Iterator:
        """
        Iterates over a range of rows in the `SQLDataModel` based on the current model's row indices.

        Yields:
            - `tuple`: A row fetched from the `SQLDataModel`.

        Notes:
            - This iterator fetches rows from the `SQLDataModel` using a SQL statement generated
            by the `_generate_sql_stmt()` method.
            - The iteration starts from the first row, index 0, and continues until
            'self.row_count' is reached.
        
        Raises:
            - `StopIteration` when there are no more rows to return.

        Example:
        ```python
        import SQLDataModel

        # Iterate over all rows in the SQLDataModel
        for row in sqldm:
            print(row)
        ```
        """  
        yield from (self.sql_db_conn.execute(self._generate_sql_stmt(include_index=True, rows=slice(0,self.row_count))))

    def __getitem__(self, target_indicies) -> SQLDataModel:
        """
        Retrieves a subset of the SQLDataModel based on the specified indices.

        Parameters:
            - `slc`: Indices specifying the rows and columns to be retrieved. This can be an integer, a tuple, a slice, or a combination of these.

        Returns:
            - An instance of SQLDataModel containing the selected subset of data.

        Notes:
            - The `slc` parameter can be an integer, a tuple of disconnected row indices, a slice representing a range of rows, a string or list of strings representing column names, or a tuple combining row and column indices.
            - The returned SQLDataModel instance will contain the specified subset of rows and columns.
        
        Raises:
            - `ValueError` if there are issues with the specified indices, such as invalid row or column names.
            - `TypeError` if the `slc` type is not compatible with indexing SQLDataModel.
            - `IndexError` if the `slc` includes a range or int that is outside of the current row count or column count.

        Example:
        ```python
        # Retrieve a specific row by index
        subset_model = sqldm[3]

        # Retrieve multiple rows and specific columns using a tuple
        subset_model = sqldm[(1, 2, 5), ["first_name", "age", "job"]]

        # Retrieve a range of rows and all columns using a slice
        subset_model = sqldm[2:7]

        # Retrieve a single column by name
        subset_model = sqldm["first_name"]
        ```
        """         
        try:
            validated_rows, validated_columns = self.validate_indicies(target_indicies)
        except ValueError as e:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
            ) from None
        except TypeError as e:
            raise TypeError(
                SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
            ) from None
        except IndexError as e:
            raise IndexError(
                SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
            ) from None 
        # print(f"{validated_rows = }\n{validated_columns = }")
        sql_stmt_generated = self._generate_sql_stmt(rows=validated_rows,columns=validated_columns,include_index=False)
        # print(f"sql_stmt_generated:\n{sql_stmt_generated.replace("\\'","'")}")
        return self.execute_fetch(sql_stmt_generated)

    def __setitem__(self, target_indicies, update_values) -> None:
        """
        Updates specified rows and columns in the SQLDataModel with the provided values.

        Parameters:
            - `target_indicies`: Indices specifying the rows and columns to be updated. This can be an integer, a tuple, a slice, or a combination of these.
            - `update_values`: The values to be assigned to the corresponding model records. It can be of types: str, int, float, bool, bytes, list, tuple, or another SQLDataModel object.

        Notes:
            - If `update_values` is another SQLDataModel object, its data will be normalized using the `data()` method.
            - The `target_indicies` parameter can be an integer, a tuple of disconnected row indices, a slice representing a range of rows, a string or list of strings representing column names, or a tuple combining row and column indices.
            - Values can be single values or iterables matching the specified rows and columns.

        Raises:
            - `TypeError` if the `update_values` type is not compatible with SQL datatypes.
            - `DimensionError` if there is a shape mismatch between targeted indicies and provided update values.
            - `ValueError` if there are issues with the specified indices, such as invalid row or column names.

        Example:
        ```python
        # Update a specific row with new values
        sqldm[3] = ("John", 25, "Engineer")

        # Update multiple rows and columns with a list of values
        sqldm[1:5, ["first_name", "age", "job"]] = [("Alice", 30, "Manager"), ("Bob", 28, "Developer"), ("Charlie", 35, "Designer"), ("David", 32, "Analyst")]

        # Create a new column named "new_column" and set values for specific rows
        sqldm[2:7, "new_column"] = [10, 20, 30, 40, 50]
        ```
        """        
        # first check if target is new column that needs to be created, if so create it and return so long as the target values aren't another sqldatamodel object:
        if isinstance(update_values, SQLDataModel):
            update_values = update_values.data() # normalize data input
        if not isinstance(update_values, str|int|float|bool|bytes|list|tuple|datetime.date) and (update_values is not None):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid values type '{type(update_values).__name__}', update values must be compatible with SQL datatypes such as <'str', 'int', 'float', 'datetime', 'bool', 'bytes'>")
            )
        # normal update values process where target update values is not another SQLDataModel object:
        if isinstance(target_indicies, str) and target_indicies not in self.headers:
            validated_rows, validated_columns = tuple(range(self.row_count)), [target_indicies]
        else:
            try:
                validated_rows, validated_columns = self.validate_indicies(target_indicies)
            except ValueError as e:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
                ) from None
            except TypeError as e:
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
                ) from None
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"{e}") # using existing formatting from validation
                ) from None             
        # convert various row options to be tuple or int
        if isinstance(validated_rows,slice):
            validated_rows = tuple(range(validated_rows.start, validated_rows.stop))
        if isinstance(validated_rows,int):
            validated_rows = (validated_rows,)
        self._update_rows_and_columns_with_values(rows_to_update=validated_rows,columns_to_update=validated_columns,values_to_update=update_values)
        return

    def __len__(self) -> int:
        """
        Returns the `row_count` property for the current `SQLDataModel` which represents the current number of rows in the model.

        Returns:
            - `int`: The total number of rows in the SQLDataModel.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get current length
        num_rows = len(sdm)

        # Output
        print(num_rows)  # 1000
        ```
        """        
        return self.row_count

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the SQLDataModel instance.

        Notes:
            - The representation includes a truncated table view of the SQLDataModel.
            - The output adjusts column widths dynamically and provides ellipses if the table is truncated.
            - The number of displayed rows is limited to either the row count or the specified maximum rows.
            - The output includes column headers, row data, and information about the total number of rows and columns.

        Returns:
            - `str`: The string representation of the SQLDataModel instance.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the model with sample data
        sqldm = SQLDataModel(data,headers)

        # Display the string representation
        print(sqldm)

        # Outputs with default alignment
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┐
        │   │ first  │ last    │    age │
        ├───┼────────┼─────────┼────────┤
        │ 0 │ john   │ smith   │     27 │
        │ 1 │ sarah  │ west    │     29 │
        │ 2 │ mike   │ harlin  │     36 │
        │ 3 │ pat    │ douglas │     42 │
        └───┴────────┴─────────┴────────┘  
        [4 rows x 3 columns]      
        ```
        ```python
        
        # Using left alignment
        sqldm.set_column_alignment("<")
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┐
        │   │ first  │ last    │ age    │
        ├───┼────────┼─────────┼────────┤
        │ 0 │ john   │ smith   │ 27     │
        │ 1 │ sarah  │ west    │ 29     │
        │ 2 │ mike   │ harlin  │ 36     │
        │ 3 │ pat    │ douglas │ 42     │
        └───┴────────┴─────────┴────────┘
        [4 rows x 3 columns]
        ```
        ```python
        
        # Using center alignment
        sqldm.set_column_alignment("^")
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┐
        │   │ first  │  last   │  age   │
        ├───┼────────┼─────────┼────────┤
        │ 0 │  john  │  smith  │   27   │
        │ 1 │ sarah  │  west   │   29   │
        │ 2 │  mike  │ harlin  │   36   │
        │ 3 │  pat   │ douglas │   42   │
        └───┴────────┴─────────┴────────┘
        [4 rows x 3 columns]
        ```
        ```python
        
        # Using right alignment
        sqldm.set_column_alignment(">")
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┐
        │   │  first │    last │    age │
        ├───┼────────┼─────────┼────────┤
        │ 0 │   john │   smith │     27 │
        │ 1 │  sarah │    west │     29 │
        │ 2 │   mike │  harlin │     36 │
        │ 3 │    pat │ douglas │     42 │
        └───┴────────┴─────────┴────────┘
        [4 rows x 3 columns]        
        ```
        Formatting:
            - Default alignment is right-aligned for numeric types and left-aligned for remaining types
            - To override default and set custom alignment, use `set_column_alignment()` method
            - Max displayed rows set to 1,000 by default, use `set_display_max_rows()` to modify
            - Set table color using `set_display_color()` method
        """         
        total_available_width, total_available_height = shutil.get_terminal_size()
        display_max_rows = self.display_max_rows if self.display_max_rows is not None else (total_available_height - 6) if (total_available_height - 6 > 0) else 1
        vertical_truncation_required = display_max_rows < self.row_count
        max_display_rows = display_max_rows if vertical_truncation_required else self.row_count # max rows to display in repr
        display_max_rows_check = 12 # 12 makes sense to get possible second digit incrementation in values
        split_row = max_display_rows // 2
        split_top = split_row
        split_bottom = (self.row_count - 1) - split_row
        display_index = self.display_index
        display_headers = [self.sql_idx] + self.headers if display_index else self.headers
        headers_select_length = f"""select max("{self.sql_idx}") as 'x',""" if display_index else """select """
        header_py_dtype_dict = {col:cmeta[1] for col, cmeta in self.header_master.items()}
        header_printf_modifiers_dict = {col:(f"'% .{self.display_float_precision}f'" if dtype == 'float' else "'%!s'" if dtype != 'bytes' else "'b''%!s'''") for col,dtype in header_py_dtype_dict.items()}
        headers_select_length = f"""{headers_select_length} {",".join([f'''case when max(max(length(printf({header_printf_modifiers_dict[col]},"{col}"))),length('{col}')) > {self.max_column_width} then {self.max_column_width} when max(max(length(printf({header_printf_modifiers_dict[col]},"{col}"))),length('{col}')) < {self.min_column_width} then {self.min_column_width} else max(max(length(printf({header_printf_modifiers_dict[col]},"{col}"))),length('{col}')) end as '{col}' ''' for col in self.headers])} from("""
        headers_sub_select = f""" select(select max(length("{self.sql_idx}")) from (select "{self.sql_idx}" from "{self.sql_model}" where ("{self.sql_idx}" <= {split_top} or "{self.sql_idx}" > {split_bottom}) order by "{self.sql_idx}" desc limit {max_display_rows+1})) "{self.sql_idx}", """ if display_index else """select """
        headers_sub_select = f"""{headers_sub_select} {",".join([f'"{col}"' for col in display_headers])} from "{self.sql_model}" order by "{self.sql_idx}" asc limit {display_max_rows_check} )"""
        headers_full_select = f"{headers_select_length}{headers_sub_select}"
        length_meta = self.sql_db_conn.execute(headers_full_select).fetchone()
        header_length_dict = {display_headers[i]:width for i, width in enumerate(length_meta)}  
        table_repr = """""" # big things...
        table_left_edge = """│ """
        table_left_edge_width = 2
        table_right_edge = """ │"""
        table_right_edge_width = 2
        table_column_interval_width = 3
        table_truncated_ellipses = """⠤⠄"""
        table_truncated_ellipses_width = 3 # added extra space after truncation mark before ellipses, looks better
        table_bare_newline = """\n"""
        total_required_width = table_left_edge_width + sum((table_column_interval_width + length) for length in header_length_dict.values()) + table_right_edge_width - table_column_interval_width
        table_truncation_required = False if total_available_width > total_required_width else True
        # print(f'truncation info: {total_required_width} of {total_available_width}, truncation: {table_truncation_required}')
        if table_truncation_required:
            total_available_width -= table_truncated_ellipses_width
            max_cols, max_width = 0, (table_left_edge_width + table_right_edge_width + table_truncated_ellipses_width) # max width starts with the tax of index and border already included, around 5-7 depending on index width
            for v in header_length_dict.values():
                if max_width < total_available_width:
                    max_width += (v+3)
                    max_cols += 1
                    continue
                break
            if max_width > total_available_width:
                max_cols -= 1
                max_width -= (header_length_dict[display_headers[max_cols]] +3)
            display_headers = display_headers[:max_cols] # +1 to include accounted for column
            table_dynamic_newline = f""" {table_truncated_ellipses}\n"""
        else:
            table_dynamic_newline = """\n"""
        vconcat_column_separator = """|| ' │ ' ||"""
        fetch_idx = SQLDataModel.sqlite_printf_format(self.sql_idx,"index",header_length_dict[self.sql_idx]) + vconcat_column_separator if display_index else ""
        header_fmt_str = vconcat_column_separator.join([f"""{SQLDataModel.sqlite_printf_format(col,header_py_dtype_dict[col],header_length_dict[col],self.display_float_precision,alignment=self.column_alignment)}""" for col in display_headers if col != self.sql_idx])
        vertical_sep_chars = '⠒⠂' # '⠐⠒⠂'
        vertical_sep_fmt_str = vconcat_column_separator.join([f"""printf("%!{header_length_dict[col]}.{header_length_dict[col]}s", printf("%*s%s%*s", ({header_length_dict[col]}-2)/2, "", '{vertical_sep_chars}', ({header_length_dict[col]}-2)/2, ""))""" for col in display_headers])
        if vertical_truncation_required:
            fetch_fmt_stmt = f"""
            with "_repr" as (
                select "{self.sql_idx}" as "_row" from "{self.sql_model}" where "{self.sql_idx}" in 
                    (select "{self.sql_idx}" from "{self.sql_model}" order by "{self.sql_idx}" asc limit {split_row}+1)
                        or "{self.sql_idx}" in
                    (select "{self.sql_idx}" from "{self.sql_model}" order by "{self.sql_idx}" desc limit {split_row})
                order by "{self.sql_idx}" asc limit {max_display_rows}+1)
                ,"_trigger" as (select "{self.sql_idx}" as "_sep" from "{self.sql_model}" order by "{self.sql_idx}" asc limit 1 offset {split_row})
            select CASE WHEN "{self.sql_idx}" <> (select "_sep" from "_trigger") THEN "_full_row" 
            ELSE '{table_left_edge}' || {vertical_sep_fmt_str} ||' │{table_dynamic_newline}' 
            END from (select "{self.sql_idx}",'{table_left_edge}' || {fetch_idx}{header_fmt_str}||' │{table_dynamic_newline}' as "_full_row" from "{self.sql_model}" where "{self.sql_idx}" in (select "_row" from "_repr") order by "{self.sql_idx}" asc)"""
        else:
            fetch_fmt_stmt = f"""select '{table_left_edge}' || {fetch_idx}{header_fmt_str}||' │{table_dynamic_newline}' as "_full_row" from "{self.sql_model}" order by "{self.sql_idx}" asc limit {max_display_rows}"""
        # print(f"fetch_fmt_stmt:\n{'-'*total_available_width}\n{fetch_fmt_stmt}\n{'-'*total_available_width}")
        formatted_response = self.sql_db_conn.execute(fetch_fmt_stmt)
        if self.column_alignment is None:
            formatted_headers = [f"""{(col if len(col) <= header_length_dict[col] else f"{col[:(header_length_dict[col]-2)]}⠤⠄"):{'>' if header_py_dtype_dict[col] in ('int','float') else '<'}{header_length_dict[col]}}""" if col != self.sql_idx else f"""{' ':>{header_length_dict[col]}}"""for col in display_headers]
        else:
            formatted_headers = [(f"""{col:{self.column_alignment}{header_length_dict[col]}}""" if len(col) <= header_length_dict[col] else f"""{col[:(header_length_dict[col]-2)]}⠤⠄""") if col != self.sql_idx else f"""{' ':>{header_length_dict[col]}}"""for col in display_headers]
        table_cross_bar = """┌─""" + """─┬─""".join(["""─""" * header_length_dict[col] for col in display_headers]) + """─┐""" + table_bare_newline
        table_repr = "".join([table_repr, table_cross_bar])
        table_repr = "".join([table_repr, table_left_edge + """ │ """.join(formatted_headers) + table_right_edge + table_dynamic_newline])
        table_repr = "".join([table_repr, table_cross_bar.replace("┌","├").replace("┬","┼").replace("┐","┤")])
        table_repr = "".join([table_repr,*[row[0] for row in formatted_response]])
        table_repr = "".join([table_repr, table_cross_bar.replace("┌","└").replace("┬","┴").replace("┐","┘")])
        table_caption = f"""[{self.row_count} rows x {self.column_count} columns]"""
        table_repr = "".join([table_repr, table_caption])
        return table_repr if self.display_color is None else self.display_color.wrap(table_repr)

##################################################################################################################
############################################## sqldatamodel methods ##############################################
##################################################################################################################

    def iter_rows(self, min_row:int=None, max_row:int=None, include_index:bool=True, include_headers:bool=False) -> Generator:
        """
        Returns a generator object of the rows in the model from `min_row` to `max_row`.

        Parameters:
            - `min_row` (int, optional): The minimum row index to start iterating from (inclusive). Defaults to None.
            - `max_row` (int, optional): The maximum row index to iterate up to (exclusive). Defaults to None.
            - `include_index` (bool, optional): Whether to include the row index in the output. Defaults to True.
            - `include_headers` (bool, optional): Whether to include headers as the first row. Defaults to False.

        Yields:
            - `Generator`: Rows from the specified range, including headers if specified.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name', 'Salary'])
        for row in sqldm.iter_rows(min_row=2, max_row=4):
            print(row)
        ```
        """
        min_row, max_row = min_row if min_row is not None else 0, max_row if max_row is not None else self.row_count
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_index, rows=slice(min_row,max_row)))
        if include_headers:
            yield tuple(x[0] for x in res.description)
        yield from (res)
    
    def iter_tuples(self, include_idx_col:bool=False) -> Generator:
        """
        Returns a generator object of the `SQLDataModel` as namedtuples using current headers as field names.

        Parameters:
            - `include_idx_col` (bool, optional): Whether to include the index column in the namedtuples. Defaults to False.

        Yields:
            - `Generator`: Namedtuples representing rows with field names based on current headers.

        Raises:
            - `ValueError`: Raised if headers are not valid Python identifiers. Use `normalize_headers()` method to fix.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['First Name', 'Last Name', 'Salary'])
        for row_tuple in sqldm.iter_tuples(include_idx_col=True):
            print(row_tuple)
        ```
        """
        try:
            Row = namedtuple('Row', [self.sql_idx] + self.headers if include_idx_col else self.headers)
        except ValueError as e:
            raise ValueError(
                SQLDataModel.ErrorFormat(f'ValueError: {e}, rename header or use `normalize_headers()` method to fix')
            ) from None
        res = self.sql_db_conn.execute(self._generate_sql_stmt(include_index=include_idx_col))
        yield from (Row(*x) for x in res.fetchall())

    def head(self, n_rows:int=5) -> SQLDataModel:
        """
        Returns the first `n_rows` of the current `SQLDataModel`.

        Parameters:
            - `n_rows` (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the specified number of rows.

        Example:
        ```python
        import SQLDataModel

        # Create the  model
        sdm = SQLDataModel.from_csv('example.csv', headers=['Name', 'Age', 'Salary'])

        # Generate a new model with top 3 rows
        head_result = sdm.head(3)
        ```
        """
        return self.execute_fetch(self._generate_unordered_sql_stmt(n_rows, ordering="asc"))
    
    def tail(self, n_rows:int=5) -> SQLDataModel:
        """
        Returns the last `n_rows` of the current `SQLDataModel`.

        Parameters:
            - `n_rows` (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the specified number of rows.

        Example:
        ```python
        import SQLDataModel

        # Create the  model
        sdm = SQLDataModel.from_csv('example.csv', headers=['Name', 'Age', 'Salary'])

        # Generate a new model with bottom 3 rows
        head_result = sdm.tail(3)
        ```
        """
        return self.execute_fetch(self._generate_unordered_sql_stmt(n_rows, ordering="desc"))
    
    def set_display_color(self, color:str|tuple):
        """
        Sets the table string representation color when `SQLDataModel` is displayed in the terminal.

        Parameters:
            - `color` (str or tuple): Color to set. Accepts hex value (e.g., '#A6D7E8') or tuple of RGB values (e.g., (166, 215, 232)).

        Returns:
            - `None`

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['Name', 'Age', 'Salary'])
        sqldm.set_display_color('#A6D7E8')
        ```

        Note:
            By default, no color styling is applied.
        """
        try:
            pen = ANSIColor(color)
            self.display_color = pen
            if SQLDataModel.debug:
                print(SQLDataModel.SuccessFormat(f"""SQLDataModel: display color changed, the terminal display color has been changed to {pen.wrap(f"color {pen.text_color_str}")}"""))
        except:
            print(SQLDataModel.WarnFormat(f"{type(self).__name__}Warning: invalid color, the terminal display color could not be changed, please provide a valid hex value or rgb color code..."))

    def sort(self, by:str|list[str]=None, asc:bool=True) -> SQLDataModel:
        """
        Sort columns in the dataset by the specified ordering. If no value is specified, the current `sql_idx` column is used with the default ordering `asc = True`.

        Parameters:
            - `by` (str | list[str]): The column or list of columns by which to sort the dataset. Defaults to sorting by the dataset's index.
            - `asc` (bool): If True, sort in ascending order; if False, sort in descending order. Defaults to ascending order.

        Raises:
            - `TypeError`: If value for `by` argument is not one of type 'str' or 'list'
            - `ValueError`: If a specified column in `by` is not found in the current dataset or is an invalid column.

        Returns:
            - `SQLDataModel`: A new instance of SQLDataModel with columns sorted according to the specified ordering.
        
        Example:
        ```python
        import SQLDataModel

        headers = ['first', 'last', 'age', 'service', 'hire_date']
        data = [
            ('John', 'Smith', 27, 1.22, '2023-02-01'),
            ('Sarah', 'West', 39, 0.7, '2023-10-01'),
            ('Mike', 'Harlin', 36, 3.9, '2020-08-27'),
            ('Pat', 'Douglas', 42, 11.5, '2015-11-06'),
            ('Kelly', 'Lee', 32, 8.0, '2016-09-18')
        ]     

        # Create the model
        sdm = SQLDataModel(data, headers)

        # Sort by last name column
        sorted_sdm = sdm.sort('last')

        # View sorted model
        print(sorted_sdm)

        # Output
        ```
        ```shell
        ┌───┬───────┬─────────┬──────┬─────────┬────────────┐
        │   │ first │ last    │  age │ service │ hire_date  │
        ├───┼───────┼─────────┼──────┼─────────┼────────────┤
        │ 0 │ Pat   │ Douglas │   42 │   11.50 │ 2015-11-06 │
        │ 1 │ Mike  │ Harlin  │   36 │    3.90 │ 2020-08-27 │
        │ 2 │ Kelly │ Lee     │   32 │    8.00 │ 2016-09-18 │
        │ 3 │ John  │ Smith   │   27 │    1.22 │ 2023-02-01 │
        │ 4 │ Sarah │ West    │   39 │    0.70 │ 2023-10-01 │
        └───┴───────┴─────────┴──────┴─────────┴────────────┘
        [5 rows x 5 columns]
        ```
        ```python

        # Sort by multiple columns in descending order
        sorted_sdm = sdm.sort(['age','hire_date'], asc=False)

        # View sorted
        print(sorted_sdm)

        # Output
        ```
        ```shell
        ┌───┬───────┬─────────┬──────┬─────────┬────────────┐
        │   │ first │ last    │  age │ service │ hire_date  │
        ├───┼───────┼─────────┼──────┼─────────┼────────────┤
        │ 0 │ Pat   │ Douglas │   42 │   11.50 │ 2015-11-06 │
        │ 1 │ Sarah │ West    │   39 │    0.70 │ 2023-10-01 │
        │ 2 │ Mike  │ Harlin  │   36 │    3.90 │ 2020-08-27 │
        │ 3 │ Kelly │ Lee     │   32 │    8.00 │ 2016-09-18 │
        │ 4 │ John  │ Smith   │   27 │    1.22 │ 2023-02-01 │
        └───┴───────┴─────────┴──────┴─────────┴────────────┘
        [5 rows x 5 columns]
        ```
        """
        if by is not None:
            if not isinstance(by, str|list):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid argument type '{type(by).__name__}', `by` argument for `sort()` must be one of 'str', 'list'")
                )
            if isinstance(by,str):
                by = [by]
            for col in by:
                if col not in self.headers:
                    raise ValueError(
                        SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', valid columns required for `sort()`, use `get_headers()` to view current valid headers")
                    )
        else:
            by = [self.sql_idx]
        sort_ord = "asc" if asc else "desc"
        sort_by_str = ",".join([f'"{x}" {sort_ord}' for x in by])
        headers_str = ",".join([f'"{col}"' for col in self.headers])
        sort_stmt = " ".join(("select",headers_str,f'from "{self.sql_model}" order by {sort_by_str}'))
        return self.execute_fetch(sort_stmt)

    def where(self, predicate:str) -> SQLDataModel:
        """
        Filters the rows of the current `SQLDataModel` object based on the specified SQL predicate and returns a
        new `SQLDataModel` containing only the rows that satisfy the condition. Only the predicates are needed as the statement prepends the select clause as "select [current model columns] where [`predicate`]", see below for detailed examples.

        Parameters:
            - `predicate` (str): The SQL predicate used for filtering rows that follows the 'where' keyword in a normal SQL statement.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` containing rows that satisfy the specified predicate.

        Raises:
            - `TypeError`: If the provided `predicate` argument is not of type `str`.
            - `SQLProgrammingError`: If the provided string is invalid or malformed SQL when executed against the model

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the model with sample data
        sqldm = SQLDataModel(data,headers)

        # Filter model by 'age' < 30
        sqldm_filtered = sqldm.where('age < 30')

        # Filter by first name and age
        sqldm_johns = sqldm.where("first = 'john' and age >= 45")
        ```
        Notes:
            - `predicate` can be any valid SQL, for example ordering can be acheived without any filtering by simple using the argument '(1=1) order by "age" asc'
            - This method allows you to filter rows in the SQLDataModel based on a specified SQL predicate. The resulting
            - SQLDataModel contains only the rows that meet the condition specified in the `predicate`. 
            - It is essential to provide a valid SQL predicate as a string for proper filtering. 
            - If the predicate is not a valid string, a `TypeError` is raised.
        """        
        if not isinstance(predicate, str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid predicate type '{type(predicate).__name__}' received, argument must be of type 'str'")
            )
        fetch_stmt = f""" select * from "{self.sql_model}" where {predicate} """
        return self.execute_fetch(fetch_stmt)
        
    def reset_index(self, start_index:int=0) -> None:
        """
        Resets the index of the `SQLDataModel` instance inplace to zero-based sequential autoincrement, or to specified `start_index` base with sequential incrementation.

        Parameters:
            - `start_index` (int, optional): The starting index for the reset operation. Defaults to 0.

        Raises:
            - `TypeError`: If provided `start_index` argument is not of type `int`
            - `ValueError`: If the specified `start_index` is greater than the minimum index in the current model.
            - `SQLProgrammingError`: If reset index execution results in constraint violation or programming error.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service']
        data = [
            (0, 'john', 'smith', 27, 1.22),
            (1, 'sarah', 'west', 39, 0.7),
            (2, 'mike', 'harlin', 36, 3),
            (3, 'pat', 'douglas', 42, 11.5)
        ]        

        # Create the model
        sdm = SQLDataModel(data, headers)

        # View current state
        print(sdm)

        # Output
        ```
        ```shell
        ┌─────┬────────┬─────────┬────────┬─────────┐
        │     │ first  │ last    │    age │ service │
        ├─────┼────────┼─────────┼────────┼─────────┤
        │ 994 │ john   │ smith   │     27 │    1.22 │
        │ 995 │ sarah  │ west    │     39 │    0.70 │
        │ 996 │ mike   │ harlin  │     36 │    3.00 │
        │ 997 │ pat    │ douglas │     42 │   11.50 │
        └─────┴────────┴─────────┴────────┴─────────┘
        [4 rows x 4 columns]        
        ```

        ```python

        # Reset the index with default start value
        sdm.reset_index()

        # View updated model
        print(sdm)

        # Output
        ```
        ```shell
        ┌───┬────────┬─────────┬────────┬─────────┐
        │   │ first  │ last    │    age │ service │
        ├───┼────────┼─────────┼────────┼─────────┤
        │ 0 │ john   │ smith   │     27 │    1.22 │
        │ 1 │ sarah  │ west    │     39 │    0.70 │
        │ 2 │ mike   │ harlin  │     36 │    3.00 │
        │ 3 │ pat    │ douglas │     42 │   11.50 │
        └───┴────────┴─────────┴────────┴─────────┘
        [4 rows x 4 columns]        
        ```
        ```python

        # Reset the index with a different value
        sdm.reset_index(start_index = -3)

        # View updated model
        print(sdm)

        # Output
        ```
        ```shell
        ┌────┬────────┬─────────┬────────┬─────────┐
        │    │ first  │ last    │    age │ service │
        ├────┼────────┼─────────┼────────┼─────────┤
        │ -3 │ john   │ smith   │     27 │    1.22 │
        │ -2 │ sarah  │ west    │     39 │    0.70 │
        │ -1 │ mike   │ harlin  │     36 │    3.00 │
        │  0 │ pat    │ douglas │     42 │   11.50 │
        └────┴────────┴─────────┴────────┴─────────┘
        [4 rows x 4 columns]        
        ```
        """
        if not isinstance(start_index,int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"""TypeError: invalid start index type '{type(start_index).__name__}', start index must be type 'int'""")
            )            
        if start_index > 0:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"""ValueError: invalid start index '{start_index}', constraints require start index <= minimum index, which is '0' in the current model""")
            )
        start_index = start_index - 1
        tmp_table_name = f"_temp_{self.sql_model}"
        created_headers = [self.sql_idx,*self.headers]
        headers_str = ",".join([f'"{col}"' for col in created_headers[1:]])
        headers_dtypes_str = ",".join([f'"{col}" {self.header_master[col][0]}' if self.header_master[col][2] else f'"{col}" {self.header_master[col][0]} PRIMARY KEY' for col in created_headers])
        reset_idx_stmt = f"""drop table if exists "{tmp_table_name}";create table "{tmp_table_name}"({headers_dtypes_str}); 
        insert into "{tmp_table_name}"("{self.sql_idx}",{headers_str})
        select row_number() over (order by "{self.sql_idx}" asc) + {start_index} as "{self.sql_idx}",{headers_str} from "{self.sql_model}" order by "{self.sql_idx}" asc;
        drop table if exists "{self.sql_model}"; alter table "{tmp_table_name}" rename to "{self.sql_model}";"""
        self.execute_transaction(reset_idx_stmt)
        self._update_model_metadata(update_row_meta=True)        
        return

##############################################################################################################
################################################ sql commands ################################################
##############################################################################################################

    def apply(self, func:Callable) -> SQLDataModel:
        """
        Applies `func` to the current `SQLDataModel` object and returns a modified `SQLDataModel` by passing its
        current values to the argument of `func` updated with the output.

        Parameters:
            - `func` (Callable): A callable function to apply to the `SQLDataModel`.

        Returns:
            - `SQLDataModel`: A modified `SQLDataModel` resulting from the application of `func`.

        Raises:
            - `TypeError`: If the provided argument for `func` is not a valid callable.
            - `SQLProgrammingError`: If the provided function is not valid based on the current SQL datatypes.
        ---
        #### Example 1: Applying to a single column:

        ```python
        import SQLDataModel

        # Create the SQLDataModel:
        sqldm = SQLDataModel.from_csv('employees.csv', headers=['First Name', 'Last Name', 'City', 'State'])

        # Create the function:
        def uncase_name(x):
            return x.lower()
        
        # Apply to existing column:
        sqldm['First Name'] = sqldm['First Name'].apply(uncase_name) # existing column will be updated with new values

        # Or create new one by passing in a new column name:
        sqldm['New Column'] = sqldm['First Name'].apply(uncase_name) # new column will be created with returned values
        ```

        ---
        #### Example 2: Applying to multiple columns:

        ```python
        import SQLDataModel
        
        # Create the function, note that `func` must have the same number of args as the model `.apply()` is called on:
        def summarize_employee(first, last, city, state)
            summary = f"{first} {last} is from {city}, {state}"
        
        # Create a new 'Employee Summary' column for the returned values:
        sqldm['Employee Summary'] = sqldm.apply(summarize_employee) # new column after all fields passed `summarize_employee` function arg
        ```

        ---
        #### Example 3: Applying a built-in function (e.g., math.sqrt) to each row:

        ```python
        import SQLDataModel, math

        # Create the SQLDataModel:
        sqldm = SQLDataModel.from_csv('number-data.csv', headers=['Number'])

        # Apply the math.sqrt function to the original 'Number' column:
        sqldm_sqrt = sqldm.apply(math.sqrt)
        ```
        
        ---
        #### Example 4: Applying a lambda function to create a new calculated column:

        ```python
        import SQLDataModel

        # Create the SQLDataModel:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['Column1', 'Column2'])
        
        # Create a new 'Column3' using the values returned from the lambda function:
        sqldm['Column3'] = sqldm.apply(lambda x, y: x + y, new_column_name='Sum_Columns')
        ```

        ---
        #### Note:
            - The number of `args` in the inspected signature of `func` must equal the current number of `SQLDataModel` columns.
            - The number of `func` args must match the current number of columns in the model, or an `Exception` will be raised.
            - Use `generate_apply_function_stub()` method to return a preconfigured template using current `SQLDataModel`
            columns and dtypes to assist.
        """        
        ### get column name from str or index ###
        if not isinstance(func, Callable):
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid argument for `func`, expected type "Callable" but type "{type(func).__name__}" was provided, please provide a valid python "Callable"...')
            )
        try:
            func_name = func.__name__.replace('<','').replace('>','')
            func_argcount = func.__code__.co_argcount
            self.sql_db_conn.create_function(func_name, func_argcount, func)
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to create function with provided callable "{func}", SQL process failed with: {e}')
            ) from None
        input_columns = ",".join([f"\"{col}\"" for col in self.headers])
        derived_query = f"""select {func_name}({input_columns}) as "{func_name}" from "{self.sql_model}" """
        return self.execute_fetch(derived_query)

    def get_column_dtypes(self,columns:str|int|list=None, dtypes:Literal["python","sql"]="python") -> dict:
        """
        Get the data types of specified columns as either Python or SQL datatypes as a `dict` in the format of `{'column': 'dtype'}`.

        Parameters: 
            - `columns` (str | int | list): The column or columns for which to retrieve data types. Defaults to all columns.
            - `dtypes` (Literal["python", "sql"]): The format in which to retrieve data types. Defaults to "python".

        Raises:
            - `TypeError`: If `columns` is not of type `str`, `int`, or `list`.
            - `IndexError`: If `columns` is of type `int` and the index is outside the valid range.
            - `ValueError`: If a specified column in `columns` is not found in the current dataset. Use `get_headers()` to view valid columns.

        Returns:
            - `dict`: A dictionary mapping column names to their data types.

        Notes:
            - SQLDataModel index column is not included, only columns specified in the `headers` attribute are in scope
            - Only the dtypes are returned, any primary key references are removed to ensure compatability with external calls
            - Python datatypes are returned in lower case, while SQL dtypes are returned in upper case to reflect convention
        ---
        Examples:
        
        ```python
        import SQLDataModel

        headers = ['first', 'last', 'age', 'service', 'hire_date']
        data = [
            ('John', 'Smith', 27, 1.22, '2023-02-01'),
            ('Sarah', 'West', 39, 0.7, '2023-10-01'),
            ('Mike', 'Harlin', 36, 3.9, '2020-08-27'),
            ('Pat', 'Douglas', 42, 11.5, '2015-11-06'),
            ('Kelly', 'Lee', 32, 8.0, '2016-09-18')
        ]  

        # Create the model
        sdm = SQLDataModel(data, headers)
        
        # Get all column python dtypes
        sdm_dtypes = sdm.get_column_dtypes()

        # View dict items
        for col, dtype in sdm_dtypes.items():
            print(f"{col}: {dtype}")

        # Outputs:
        ```
        ```shell
        first: str
        last: str
        age: int
        service: float
        hire_date: date
        ```
        ```python

        # Get specific column sql dtypes
        sdm_dtypes = sdm.get_column_dtypes(columns=['first','age','service'], dtypes="sql")

        # View dict items
        for col, dtype in sdm_dtypes.items():
            print(f"{col}: {dtype}")

        # Outputs:
        ```
        ```shell
        first: TEXT
        age: INTEGER
        service: REAL

        ```
        """        
        dtypes = 1 if dtypes == "python" else 0
        if columns is None:
            return {col:self.header_master[col][dtypes] for col in self.headers}
        if not isinstance(columns, str|int|list):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(columns).__name__}', `columns` must be one of 'str', 'int' or 'list', use `get_headers()` to view current valid arguments")
            )
        if isinstance(columns, int):
            try:
                columns = self.headers[columns]
            except IndexError:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{columns}', column index is outside of current model range '0:{self.column_count}', use `get_headers()` to veiw current valid arguments")
                ) from None
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if col not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', column must be in current model, use `get_headers()` to view current valid arguments")
                )
        return {col:self.header_master[col][dtypes] for col in columns}

    def set_column_dtypes(self, column:str|int, dtype:Literal['bytes','date','datetime','float','int','str']) -> None:
        """
        Casts the specified `column` into the provided python `dtype`. The datatype must be a valid convertable python datatype to map to an equivalent SQL datatype.

        Parameters:
            - `column` (str or int): The name or index of the column to be cast, must be current header or within range of current `column_count`
            - `dtype` (Literal['bytes', 'datetime', 'float', 'int', 'str']): The target python data type for the specified column.

        Raises:
            - `TypeError`: If `column` is not of type 'str' or 'int'.
            - `IndexError`: If `column` is an integer and the index is outside of the current model range.
            - `ValueError`: If `column` is a string and the column is not found in the current model.

        Returns:
            - `None`

        Note:
            - This method modifies the SQLDataModel by altering the data type of the specified column in the underlying SQL table.
            - The method performs a series of SQL statements to create a temporary column with the desired data type,
            copies the data from the original column to the temporary column, drops the original column,
            and finally, renames the temporary column to the original column name.

        Example:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
         (0, 'john', 'smith', 27)
        ,(1, 'sarah', 'west', 29)
        ,(2, 'mike', 'harlin', 36)
        ,(3, 'pat', 'douglas', 42)
        ]
        
        # Create the SQLDataModel object
        sqldm = SQLDataModel(data, headers)        

        # Set the data type of the 'age' column to 'float'
        sqldm.set_column_dtypes('age', 'float')

        # Confirm column dtype
        new_dtype = sqldm.get_column_dtypes('age')

        # Outputs 'float'
        print(new_dtype)
        ```
        """
        if not isinstance(column, str|int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(column).__name__}', `column` must be one of 'str' or 'int', use `get_headers()` to view current valid arguments")
            )
        if dtype not in ('bytes','date','datetime','float','int','str'):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid argument type '{type(column).__name__}', `dtype` must be one of 'bytes','date','datetime','float','int','str' use `get_column_dtypes()` to view current column datatypes")
            )        
        if isinstance(column, int):
            try:
                column = self.headers[column]
            except IndexError:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{column}', column index is outside of current model range '0:{self.column_count}', use `get_headers()` to veiw current valid arguments")
                ) from None
        if isinstance(column, str):
            if column not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{column}', column must be in current model, use `get_headers()` to view valid arguments")
                )
        col_sql_dtype = self.static_py_to_sql_map_dict[dtype]
        dyn_dtype_cast = f"""cast("{column}" as {col_sql_dtype})""" if dtype not in ("date","datetime") else f"""{dtype}(trim("{column}"))"""
        update_col_sql = f"""alter table "{self.sql_model}" add column "{column}_x" {col_sql_dtype}; update "{self.sql_model}" set "{column}_x" = {dyn_dtype_cast}; alter table "{self.sql_model}" drop column "{column}"; alter table "{self.sql_model}" rename column "{column}_x" to "{column}";"""
        self.execute_transaction(update_col_sql)
        
    def get_model_name(self) -> str:
        """
        Returns the `SQLDataModel` table name currently being used by the model as an alias for any SQL queries executed by the user and internally.

        Returns:
            - `str`: The current `SQLDataModel` table name.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['Column1', 'Column2'])
        model_name = sqldm.get_model_name()
        print(f'The model is currently using the table name: {model_name}')
        ```
        """
        return self.sql_model
    
    def set_model_name(self, new_name:str) -> None:
        """
        Sets the new `SQLDataModel` table name that will be used as an alias for any SQL queries executed by the user or internally.

        Parameters:
            - `new_name` (str): The new table name for the `SQLDataModel`.

        Raises:
            - `SQLProgrammingError`: If unable to rename the model table due to SQL execution failure.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['Column1', 'Column2'])
        sqldm.set_model_name('custom_table')
        ```

        Note:
            - The provided value must be a valid SQL table name.
            - This alias will be reset to the default value for any new `SQLDataModel` instances: 'sdm'.
        """
        full_stmt = f"""begin transaction; alter table "{self.sql_model}" rename to {new_name}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename model table, SQL execution failed with: "{e}"')
            ) from None
        self.sql_model = new_name
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: Successfully renamed primary model alias to "{new_name}"'))

    def deduplicate(self, subset:list[str]=None, keep_first:bool=True, inplace:bool=True) -> None|SQLDataModel:
        """
        Removes duplicate rows from the SQLDataModel based on the specified subset of columns. Deduplication occurs inplace by default, otherwise use `inplace=False` to return a new `SQLDataModel`.

        Parameters:
            - `subset` (list[str], optional): List of columns to consider when identifying duplicates.
            If None, all columns are considered. Defaults to None.
            - `keep_first` (bool, optional): If True, keeps the first occurrence of each duplicated row; otherwise,
            keeps the last occurrence. Defaults to True.
            - `inplace` (bool, optional): If True, modifies the current SQLDataModel in-place; otherwise, returns
            a new SQLDataModel without duplicates. Defaults to True.

        Returns:
            - `None`: If `inplace` is True, the method modifies the current SQLDataModel in-place.
            - `SQLDataModel`: If `inplace` is False, returns a new SQLDataModel without duplicates.

        Raises:
            - `ValueError`: If a column specified in `subset` is not found in the SQLDataModel.

        Example:
        ```python
        import SQLDataModel
        # Example 1: Deduplicate based on a specific column
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm.deduplicate(subset='ID', keep_first=True, inplace=True)

        # Example 2: Deduplicate based on multiple columns
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm_deduped = sqldm.deduplicate(subset=['ID', 'Name'], keep_first=False, inplace=False)
        ```
        """        
        dyn_keep_order = 'min' if keep_first else 'max'
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            for col in subset:
                if col not in self.headers:
                    raise ValueError(
                        SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', provided columns in `subset` must be valid columns, use `get_headers()` to view current valid column names")
                    )
        else:
            subset = self.headers
        if inplace:
            sql_stmt = f"""delete from "{self.sql_model}" where rowid not in (select {dyn_keep_order}(rowid) from "{self.sql_model}" group by {','.join(f'"{col}"' for col in subset)})"""
            self.sql_db_conn.execute(sql_stmt)
            self.sql_db_conn.commit()
            self._update_model_metadata(update_row_meta=True)
            return
        else:
            sql_stmt = f"""select * from "{self.sql_model}" where rowid in (select {dyn_keep_order}(rowid) from "{self.sql_model}" group by {','.join(f'"{col}"' for col in subset)})"""
            return self.execute_fetch(sql_stmt)

    def group_by(self, columns:str|list[str], order_by_count:bool=True) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` after performing a group by operation on specified columns.

        Parameters:
            - `columns` (str, list, tuple): Columns to group by. Accepts either individual strings or a list/tuple of strings.

        Keyword Args:
            - `order_by_count` (bool, optional): If True (default), orders the result by count. If False, orders by the specified columns.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the result of the group by operation.

        Raises:
            - `TypeError`: If the columns argument is not of type str, list, or tuple.
            - `ValueError`: If any specified column does not exist in the current model.
            - `SQLProgrammingError`: If any specified columns or aggregate keywords are invalid or incompatible with the current model.

        ---
        Example:
        ```python
        import SQLDataModel

        headers = ['first', 'last', 'age', 'service', 'hire_date', 'gender']
        data = [
            ('John', 'Smith', 27, 1.22, '2023-02-01', 'Male'),
            ('Sarah', 'West', 39, 0.7, '2023-10-01', 'Female'),
            ('Mike', 'Harlin', 36, 3.9, '2020-08-27', 'Male'),
            ('Pat', 'Douglas', 42, 11.5, '2015-11-06', 'Male'),
            ('Kelly', 'Lee', 32, 8.0, '2016-09-18', 'Female')
        ]        
        # Create the model
        sdm = SQLDataModel(data, headers, display_float_precision=2, display_index=True)

        # Group by 'gender' column
        sdm_gender = sdm.group_by("gender")

        # View model
        print(sdm_gender)

        # Output
        ```
        ```shell
        ┌───┬────────┬───────┐
        │   │ gender │ count │
        ├───┼────────┼───────┤
        │ 0 │ Male   │     3 │
        │ 1 │ Female │     2 │
        └───┴────────┴───────┘
        [2 rows x 2 columns]        
        ```
        ---

        Example 2:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('data.csv')

        # Group by multiple columns
        sdm.group_by(["country", "state", "city"])

        ```
        Notes:
            - Use `order_by_count=False` to change ordering from count to column arguments.
        """
        if not isinstance(columns, str|list|tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(columns).__name__}', arguments for `columns` must be one of 'str', 'list' or 'tuple'")
                )
        if isinstance(columns,str):
            columns = [columns]
        for col in columns:
            if col not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', valid columns are required for grouping, use `get_headers()` to view current valid arguments")
                    )
        columns_group_by = ",".join(f'"{col}"' for col in columns)
        order_by = "count(*)" if order_by_count else columns_group_by
        group_by_stmt = f"""select {columns_group_by}, count(*) as count from "{self.sql_model}" group by {columns_group_by} order by {order_by} desc"""
        return self.execute_fetch(group_by_stmt)
    
    def join_model(self, model:SQLDataModel, left:bool=True, on_column:str=None) -> SQLDataModel:
        """
        Performs a left or right join using the caller `SQLDataModel` as the base table and another `model` of type `SQLDataModel` instance as the joined table.

        Parameters:
            - `model` (SQLDataModel): The `SQLDataModel` instance to join with.
            - `left` (bool, optional): If True (default), performs a left join. If False, performs a right join.
            - `on_column` (str, optional): The shared column used for joining. If None, attempts to automatically find a matching column.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the result of the join operation.

        Raises:
            - `DimensionError`: If no shared column is found, or if the provided column is not present in both models.

        Example:
        ```python
        import SQLDataModel

        sqldm_1 = SQLDataModel.from_csv('base-data.csv')  # create first model from data
        sqldm_2 = SQLDataModel.from_csv('join-data.csv')  # create second model from data
        result = sqldm_1.join_model(sqldm_2, left=True, on_column='shared_column')  # perform left join on a shared column
        ```
        """
        validated_join_col = False
        join_tablename = 'f_table'
        join_cols = model.headers
        if on_column is None:
            for col in join_cols:
                if col in self.headers:
                    on_column = col
                    validated_join_col = True
                    break
        else:
            if (on_column in self.headers) and (on_column in join_cols):
                validated_join_col = True
        if not validated_join_col:
            raise DimensionError(
                SQLDataModel.ErrorFormat(f"DimensionError: no shared column, no matching join column was found in the provided model, ensure one is available or specify one explicitly with on_column='shared_column'")
                )
        model.to_sql(join_tablename, self.sql_db_conn)
        join_cols = [x for x in join_cols if x != on_column] # removing shared join column
        sql_join_stmt = self._generate_sql_fetch_for_joining_tables(self.headers, join_tablename, join_column=on_column, join_headers=join_cols, join_type='left' if left else 'right')
        res = self.sql_db_conn.execute(sql_join_stmt)
        return type(self)(data=res.fetchall(), headers=[x[0] for x in res.description], display_max_rows=self.display_max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)

    def execute_fetch(self, sql_query:str, **kwargs) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` object after executing the provided SQL query using the current `SQLDataModel`. 
        This method is called by other methods which expect results to be returned from their execution.

        Parameters:
            - `sql_query` (str): The SQL query to execute with the expectation of rows returned.
            - `**kwargs` (optional): Additional keyword args to pass to `SQLDataModel` constructor

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the result of the SQL query.

        Raises:
            - `SQLProgrammingError`: If the provided SQL query is invalid or malformed.
            - `ValueError`: If the provided SQL query was valid but returned 0 rows, which is insufficient to return a new model.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['Column1', 'Column2'])

        # Create the fetch query to use
        query = 'SELECT * FROM sdm WHERE Column1 > 10'

        # Fetch and save the result to a new instance
        result_model = sqldm.execute_fetch(query)
        ```

        Important:
            - The default table name is 'sdm', or you can use `SQLDataModel.get_model_name()` to get the current model alias.
            - This function is the primary method used by `SQLDataModel` methods that are expected to return a new instance.
        """
        try:
            res = self.sql_db_conn.execute(sql_query)
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, provided query failed with error "{e}"')
            ) from None
        fetch_result = res.fetchall()
        fetch_headers = [x[0] for x in res.description]
        if (rows_returned := len(fetch_result)) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: nothing to return, provided query returned '{rows_returned}' rows which is insufficient to return or generate a new model from")
            )
        if not kwargs:
            return type(self)(fetch_result, headers=fetch_headers, display_max_rows=self.display_max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)
        else:
            params = {"display_max_rows":self.display_max_rows, "min_column_width":self.min_column_width, "max_column_width":self.max_column_width, "column_alignment":self.column_alignment, "display_color":self.display_color, "display_index":self.display_index, "display_float_precision":self.display_float_precision}
            params.update({k:v for k,v in kwargs.items()})
            return type(self)(fetch_result, headers=fetch_headers, **params)

    def execute_statement(self, sql_stmt:str) -> None:
        """
        Executes an arbitrary SQL query against the current model without the expectation of selection or returned rows.

        Parameters:
            - `sql_stmt` (str): The SQL query to execute.
            
        Raises:
            - `SQLProgrammingError`: If the SQL execution fails.

        Notes:
            - To execute a query with the expectation of results, see `execute_fetch()` method
            - To execute multiple queries within a single transaction, see `execute_transaction()` method

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sdm = SQLDataModel.from_csv('data.csv')

        # Execute statement without results, modifying column in place
        sdm.execute_statement('UPDATE table SET column = value WHERE condition')

        ```
        """
        try:
            self.sql_db_conn.execute(sql_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, unable to execute provided SQL query with error "{e}"...')
            ) from None
        self._update_model_metadata(update_row_meta=True)
    
    def execute_transaction(self, sql_script:str) -> None:
        """
        Executes a prepared SQL script wrapped in a transaction against the current model without the expectation of selection or returned rows.

        Parameters:
            - `sql_script` (str): The SQL script to execute within a transaction.

        Raises:
            - `SQLProgrammingError`: If the SQL execution fails.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('data.csv')  # create model from data
        transaction_script = '''
            UPDATE table1 SET column1 = value1 WHERE condition1;
            UPDATE table2 SET column2 = value2 WHERE condition2;
        '''
        sqldm.execute_transaction(transaction_script)  # execute a transaction with multiple SQL statements
        ```
        """
        full_stmt = f"""begin transaction; {sql_script}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to execute provided transaction, SQL execution failed with: "{e}"')
            ) from None
        self._update_model_metadata(update_row_meta=True)        

    def freeze_index(self, column_name:str=None) -> None:
        """
        Freeze the current index as a new column, expanding it into the current model.
        The new column is unaffected by any future changes to the primary index column.

        Parameters:
            - `column_name` (str, optional): The name for the new frozen index column.
            If not provided, a default name 'frzn_id' will be used.

        Raises:
            - `TypeError`: If the provided `column_name` is not of type 'str'.

        Returns:
            - `None`

        Examples:

        ---

        ```python
        import SQLDataModel

        headers = ['first', 'last', 'age', 'service', 'hire_date']
        data = [
            ('John', 'Smith', 27, 1.22, '2023-02-01'),
            ('Sarah', 'West', 39, 0.7, '2023-10-01'),
            ('Mike', 'Harlin', 36, 3.9, '2020-08-27'),
            ('Pat', 'Douglas', 42, 11.5, '2015-11-06'),
            ('Kelly', 'Lee', 32, 8.0, '2016-09-18')
        ]   
            
        # Create the model
        sdm = SQLDataModel(data, headers)

        # Freeze index as new column 'id'
        sdm.freeze_index("id")    

        # View model
        print(sdm)

        # Output    
        ```
        ```shell
        ┌───┬───────┬─────────┬──────┬─────────┬────────────┬──────┐
        │   │ first │ last    │  age │ service │ hire_date  │   id │
        ├───┼───────┼─────────┼──────┼─────────┼────────────┼──────┤
        │ 0 │ John  │ Smith   │   27 │    1.22 │ 2023-02-01 │    0 │
        │ 1 │ Sarah │ West    │   39 │    0.70 │ 2023-10-01 │    1 │
        │ 2 │ Mike  │ Harlin  │   36 │    3.90 │ 2020-08-27 │    2 │
        │ 3 │ Pat   │ Douglas │   42 │   11.50 │ 2015-11-06 │    3 │
        │ 4 │ Kelly │ Lee     │   32 │    8.00 │ 2016-09-18 │    4 │
        └───┴───────┴─────────┴──────┴─────────┴────────────┴──────┘
        [5 rows x 6 columns]
        ```
        """
        if not isinstance(column_name, str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(column_name).__name__}', argument for `column_name` must be of type 'str'")
            )
        if column_name is None:
            column_name = "frzn_id"
        column_name = tuple(SQLDataModel.alias_duplicates([self.sql_idx,*self.headers,column_name]))[-1]
        sql_script = f"""alter table "{self.sql_model}" add column "{column_name}" integer;update "{self.sql_model}" set "{column_name}" = "{self.sql_idx}";"""
        self.execute_transaction(sql_script)

    def add_column_with_values(self, column_name:str, value=None) -> None:
        """
        Adds a new column with the specified `column_name` to the `SQLDataModel`. The new column is populated with the values provided in the `value` argument. If `value` is not provided (default), the new column is populated with NULL values.

        Parameters:
            - `column_name` (str): The name of the new column to be added.
            - `value`: The value to populate the new column. If None (default), the column is populated with NULL values. If a valid column name is provided, the values of that column will be used to fill the new column.

        Raises:
            - `DimensionError`: If the length of the provided values does not match the number of rows in the model.
            - `TypeError`: If the data type of the provided values is not supported or translatable to an SQL data type.

        Example:
        ```python
        import SQLDataModel

        # Create model from data
        sqldm = SQLDataModel.from_csv('data.csv')

        # Add new column with default value 42
        sqldm.add_column_with_values('new_column', value=42)

        # Add new column by copying values from an existing column
        sqldm.add_column_with_values('new_column', value='existing_column')
        ```
        """
        create_col_stmt = f"""alter table "{self.sql_model}" add column "{column_name}" """
        if (value is not None) and (value in self.headers):
            dyn_dtype_default_value = f"""{self.header_master[value][0]}"""
            dyn_copy_existing = f"""update "{self.sql_model}" set "{column_name}" = "{value}";"""
            sql_script = f"""{create_col_stmt} {dyn_dtype_default_value};{dyn_copy_existing};"""
            self.execute_transaction(sql_script)
            return
        if value is None:
            sql_script = create_col_stmt
            self.execute_transaction(sql_script)
            return
        if isinstance(value, str|int|float|bool):
            value = f"'{value}'" if isinstance(value,str) else value
            dyn_dtype_default_value = f"""{self.static_py_to_sql_map_dict[type(value).__name__]} not null default {value}""" if value is not None else "TEXT"
            sql_script = f"""{create_col_stmt} {dyn_dtype_default_value};"""
            self.execute_transaction(sql_script)
            return
        if isinstance(value, datetime.date):
            dyn_dtype_default_value = f"""{self.static_py_to_sql_map_dict[type(value).__name__]}"""
            create_col_stmt = f"""{create_col_stmt} {dyn_dtype_default_value};"""
            update_val_stmt = f"""update "{self.sql_model}" set "{column_name}" = {f"datetime('{value}')" if isinstance(value, datetime.datetime) else f"date('{value}')"};"""
            sql_script = f"""{create_col_stmt} {update_val_stmt}"""
            self.execute_transaction(sql_script)
            return 
        if isinstance(value, bytes):
            dyn_dtype_default_value = f"""{self.static_py_to_sql_map_dict[type(value).__name__]} not null default X'{value.hex()}'""" if value is not None else "TEXT"
            sql_script = f"""{create_col_stmt} {dyn_dtype_default_value};"""
            self.execute_transaction(sql_script)
            return                 
        if isinstance(value, list|tuple):
            if (len_values := len(value)) != self.row_count:
                raise DimensionError(
                    SQLDataModel.ErrorFormat(f"DimensionError: invalid dimensions '{len_values} != {self.row_count}', provided values have length '{len_values}' while current row count is '{self.row_count}'")
                )
            try:
                seq_dtype = self.static_py_to_sql_map_dict[type(value[0]).__name__]
            except:
                raise TypeError(
                    SQLDataModel.ErrorFormat(f'TypeError: invalid datatype "{type(value[0]).__name__}", please provide a valid and SQL translatable datatype...')
                ) from None
            sql_script = f"""{create_col_stmt} {seq_dtype};"""
            all_model_idxs = tuple(range(self.row_count))
            for i,val in enumerate(value):
                sql_script += f"""update {self.sql_model} set "{column_name}" = "{val}" where {self.sql_idx} = {all_model_idxs[i]};"""
            self.execute_transaction(sql_script)
            return

    def apply_function_to_column(self, func:Callable, column:str|int) -> None:
        """
        Applies the specified callable function (`func`) to the provided `SQLDataModel` column. The function's output is used to update the values in the column. For broader uses or more input flexibility, see related method `apply()`.

        Parameters:
            - `func` (Callable): The callable function to apply to the column.
            - `column` (str | int): The name or index of the column to which the function will be applied.

        Raises:
            - `TypeError`: If the provided column argument is not a valid type (str or int).
            - `IndexError`: If the provided column index is outside the valid range of column indices.
            - `ValueError`: If the provided column name is not valid for the current model.
            - `SQLProgrammingError`: If the provided function return types or arg count is invalid or incompatible to SQL types.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('data.csv')

        # Apply upper() method using lambda function to column `name`
        sqldm.apply_function_to_column(lambda x: x.upper(), column='name')

        # Apply addition through lambda function to column at index 1
        sqldm.apply_function_to_column(lambda x, y: x + y, column=1)
        ```
        """
        ### get column name from str or index ###
        if (not isinstance(column, int)) and (not isinstance(column, str)):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column argument, \"{type(column).__name__}\" is not a valid target, provide column index or column name as a string...")
            )
        if isinstance(column, int):
            try:
                column = self.headers[column]
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index provided, {column} is not a valid column index, use `.column_count` property to get valid range...")
                ) from None
        if isinstance(column, str):
            if column not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid column provided, {column} is not valid for current model, use `.get_headers()` method to get model headers...")
                    )
            else:
                column = column
        target_column = column
        try:
            func_name = func.__name__
            func_argcount = func.__code__.co_argcount
            self.sql_db_conn.create_function(func_name, func_argcount, func)
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to create function with provided callable "{func}", SQL process failed with: {e}')
            ) from None
        if func_argcount == 1:
            input_columns = target_column
        elif func_argcount == self.column_count:
            input_columns = ",".join([f"\"{col}\"" for col in self.headers])
        else:
            raise ValueError(
                SQLDataModel.ErrorFormat(f'ValueError: invalid function arg count: {func_argcount}, input args to "{func_name}" must be 1 or {self.column_count} based on the current models structure, ie...\n{self.generate_apply_function_stub()}')
                )
        sql_apply_update_stmt = f"""update {self.sql_model} set {target_column} = {func_name}({input_columns})"""
        full_stmt = f"""begin transaction; {sql_apply_update_stmt}; end transaction;"""
        try:
            self.sql_db_conn.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to apply function, SQL execution failed with: {e}')
            ) from None
        self._update_model_metadata()
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: applied function "{func_name}()" to current model')        )

    def generate_apply_function_stub(self) -> str:
        """
        Generates a function template using the current `SQLDataModel` to format function arguments for the `apply_function_to_column()` method.

        Returns:
            - `str`: A string representing the function template.

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('data.csv')

        # Create and print the function signature template
        stub = sqldm.generate_apply_function_stub()
        print(stub)
        ```
        Output:
        ```python
        def func(user_name:str, user_age:int, user_salaray:float):
            # apply logic and return value
            return
        ```
        """
        func_signature = ", ".join([f"""{k.replace(" ","_")}:{v[1]}""" for k,v in self.header_master.items() if k != self.sql_idx])
        return f"""def func({func_signature}):\n    # apply logic and return value\n    return"""
    
    def insert_row(self, values:list|tuple=None) -> None:
        """
        Inserts a row in the `SQLDataModel` at index `self.rowcount+1` with provided `values`.
        If `values=None`, an empty row with SQL `null` values will be used.

        Parameters:
            - `values` (list or tuple, optional): The values to be inserted into the row. 
            If not provided or set to None, an empty row with SQL `null` values will be inserted.

        Raises:
            - `TypeError`: If `values` is provided and is not of type list or tuple.
            - `DimensionError`: If the number of values provided does not match the current column count.
            - `SQLProgrammingError`: If there is an issue with the SQL execution during the insertion.

        Note:
            - The method handles the insertion of rows into the SQLDataModel, updates metadata, and commits the changes to the database.
            - If an error occurs during SQL execution, it rolls back the changes and raises a SQLProgrammingError with an informative message.

        Usage:
        ```python
        import SQLDataModel

        # Example 1: Insert a row with values
        sqldm.insert_row([1, 'John Doe', 25])

        # Example 2: Insert an empty row with SQL null values
        sqldm.insert_row()
        ```
        """
        if values is not None:
            if not isinstance(values,list|tuple):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f'TypeError: invalid type provided \"{type(values).__name__}\", insert values must be of type list or tuple...')
                    )
            if isinstance(values,list):
                values = tuple(values)
            if (len_val := len(values)) != self.column_count:
                raise DimensionError(
                    SQLDataModel.ErrorFormat(f'DimensionError: invalid dimensions \"{len_val} != {self.column_count}\", the number of values provided ({len_val}) must match the current column count ({self.column_count})...')
                    )
        else:
            values = tuple(None for _ in range(self.column_count))
        insert_cols = ",".join([f'"{col}"' for col in self.headers])
        insert_vals = ",".join(["?" if not isinstance(val,datetime.date) else "datetime(?)" if isinstance(val, datetime.datetime) else "date(?)" for val in values])
        insert_stmt = f"""insert into {self.sql_model}({insert_cols}) values ({insert_vals})"""
        try:
            self.sql_db_conn.execute(insert_stmt, values)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: "{e}"')
            ) from None
        self._update_model_metadata(update_row_meta=True)        

    def update_index_at(self, row_index:int, column_index:int|str, value=None) -> None:
        """
        Updates a specific cell in the `SQLDataModel` at the given row and column indices with the provided value.

        Parameters:
            - `row_index` (int): The index of the row to be updated.
            - `column_index` (int or str): The index or name of the column to be updated.
            - `value`: The new value to be assigned to the specified cell.

        Raises:
            - `TypeError`: If row_index is not of type 'int' or if column_index is not of type 'int' or 'str'.
            - `ValueError`: If the provided row index is outside the current model range.
            - `IndexError`: If the provided column index (when specified as an integer) is outside of the current model range.
            - `SQLProgrammingError`: If there is an issue with the SQL execution during the update.

        Note:
            - The method allows updating cells identified by row and column indices in the SQLDataModel.
            - Handles different index types for rows and columns (int or str).
            - If an error occurs during SQL execution, it rolls back the changes and raises a SQLProgrammingError with an informative message.
            - After successful execution, it prints a success message with the number of modified rows.

        Usage:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the model with sample data
        sqldm = SQLDataModel(data,headers)

        # Example 1: Update a cell in the first row and second column
        sqldm.update_index_at(0, 1, 'NewValue')

        # Example 2: Update a cell in the 'Name' column of the third row
        sqldm.update_index_at(2, 'Name', 'John Doe')
        ```
        """        
        if not isinstance(row_index, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid row index type '{type(row_index).__name__}', rows must be indexed by type 'int'")
            )
        if not isinstance(column_index, int|str):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column index type '{type(row_index).__name__}', columns must be indexed by type 'int' or 'str', use `.get_headers()` to view current model headers")
            )
        if row_index < 0:
            row_index = self.row_count + row_index
        if row_index < 0 or row_index > self.row_count:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}', provided row index is outisde of current model range '0:{self.row_count}'")
            )
        if isinstance(column_index, int):
            try:
                column_index = self.headers[column_index]
            except IndexError:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{column_index}', provided column index is outside of current model range '0:{self.column_count}'")
                ) from None
        if column_index not in self.headers:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: column not found '{column_index}', use `.get_headers()` to view current valid model headers")
            )
        if not isinstance(value,tuple):
            value = (value,)
        if isinstance(value[0], datetime.datetime):
            val_binding = "datetime(?)"
        elif isinstance(value[0], datetime.date):
            val_binding = "date(?)"
        else:
            val_binding = "?"
        update_stmt = f"""update "{self.sql_model}" set "{column_index}" = {val_binding} where "{self.sql_idx}" = {row_index}"""
        try:
            self.sql_db_conn.execute(update_stmt, value)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: {e}')
            ) from None
        self._update_model_metadata()        

    def _update_model_metadata(self, update_row_meta:bool=False) -> None:
        """
        Generates and updates metadata information about the columns and optionally the rows in the SQLDataModel instance based on the current model. 

        Parameters:
            - `update_row_meta` (bool, optional): If True, updates row metadata information; otherwise, retrieves column metadata only (default).
        
        Attributes updated:
            - `self.header_master`: Master dictionary of column metadata.
            - `self.headers`: List of current model headers, order retained.
            - `self.column_count`: Number of columns in current model.
                - `self.row_count`: Optionally updated, represents current row count.
            
        Returns:
            - `None`
        
        ---

        Example Format:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service_time']
        data = [
            (0, 'john', 'smith', 27, 1.22),
            (1, 'sarah', 'west', 39, 0.7),
            (2, 'mike', 'harlin', 36, 3),
            (3, 'pat', 'douglas', 42, 11.5)
        ]

        # Create the model with sample data
        sdm = VF(data, headers)

        # View header master
        print(sdm.header_master)

        # Output:
        header_master = {
            'first': ('TEXT', 'str', True, '<'),
            'last': ('TEXT', 'str', True, '<'),
            'age': ('INTEGER', 'int', True, '>'),
            'service_time': ('REAL', 'float', True, '>'),
            'idx': ('INTEGER', 'int', False, '>')
        }
        ```
        
        ---
        Example Attributes Modified:

        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age', 'service_time']
        data = [
            (0, 'john', 'smith', 27, 1.22),
            (1, 'sarah', 'west', 0.7),
            (2, 'mike', 'harlin', 3),
            (3, 'pat', 'douglas', 11.5)
        ]

        # Create the model with sample data
        sdm = SQLDataModel(data, headers)

        # Get current column count
        num_cols_before = sdm.column_count

        # Add new column
        sdm['new_column'] = 'empty'

        # Method is called behind the scenes
        sdm._update_model_metadata()

        # Get new column count
        num_cols_after = sdm.column_count

        # View difference
        print(f"cols before: {num_cols_before}, cols after: {num_cols_after}")
        ```
        """        
        if update_row_meta:
            fetch_metadata = f"""select "_ordered_name","_ordered_type","_is_regular_column","_def_alignment" from (
                select '_rowmeta' as "_ordered_name", 0 as "_ordered_type", 0 as "_is_regular_column", count("{self.sql_idx}") as "_def_alignment" from "{self.sql_model}"
                union all
                select "name" as "_ordered_name","type" as "_ordered_type","pk" as "_is_regular_column",case when ("type"='INTEGER' or "type"='REAL') then '>' else '<' end as "_def_alignment" from pragma_table_info('{self.sql_model}') 
            ) order by "_ordered_name"='_rowmeta' desc, {",".join([f'"_ordered_name"="{col}" desc' for col in self.headers])}"""
        else:
            fetch_metadata = f"""select "name" as "_ordered_name","type" as "_ordered_type","pk" as "_is_regular_column",case when ("type"='INTEGER' or "type"='REAL') then '>' else '<' end as "_def_alignment" from pragma_table_info('{self.sql_model}') order by {",".join([f'''"_ordered_name"='{col}' desc''' for col in self.headers])}"""
        # print(fetch_metadata.replace("\\'","'"))
        metadata = self.sql_db_conn.execute(fetch_metadata).fetchall()
        if update_row_meta:
            self.row_count = metadata[0][-1]
            metadata = metadata[1:]
        header_master = {m[0]:(m[1], self.static_sql_to_py_map_dict[m[1]],True if m[2] == 0 else False,m[3]) for m in metadata}
        self.headers = list(dict.fromkeys([k for k,v in header_master.items() if v[2]]))
        self.column_count = len(self.headers)
        self.header_master = header_master # format: 'column_name': ('sql_dtype', 'py_dtype', is_regular_column, 'default_alignment')

    def _generate_sql_stmt(self, columns:list[str]=None, rows:int|slice|tuple=None, include_index:bool=True) -> str:
        """
        Generate an SQL statement for fetching specific columns and rows from the model, duplicate column references are aliased in order of appearance.

        Parameters:
            - `columns` (list of str, optional): The list of columns to include in the SQL statement.
            If not provided, all columns from the model will be included.
            - `rows` (int, slice, tuple, optional): The rows to include in the SQL statement. It can be an integer for a single row,
            a slice for a range of rows, or a tuple for specific row indices. If not provided, all rows will be included.
            - `include_index` (bool, optional): If True, include the primary index column in the SQL statement.

        Returns:
            - `str`: The generated SQL statement.   

        Notes:
            - No validation is performed on row or column indicies, see `validate_indicies()` for usage     
        """
        if columns is None:
            columns = self.headers
        if include_index:
            columns = [self.sql_idx,*columns]
        headers_selection_str = ",".join([f'"{col}" as "{col_alias}"' for col,col_alias in zip(columns,SQLDataModel.alias_duplicates(columns))])
        if isinstance(rows, int):
            row_selection_str = f"""where "{self.sql_idx}" = {rows}"""
        elif isinstance(rows, slice):
            row_selection_str = f"""where "{self.sql_idx}" >= {rows.start} and "{self.sql_idx}" < {rows.stop}"""
        elif isinstance(rows, tuple):
            row_selection_str = f"""where "{self.sql_idx}" in {f'{rows}' if len(rows) != 1 else f'({rows[0]})'}"""
        else:
            row_selection_str = """"""
        order_by_str = f"""order by "{self.sql_idx}" asc"""
        fetch_stmt = f"""select {headers_selection_str} from "{self.sql_model}" {row_selection_str} {order_by_str}"""
        # print(f"final fetch_stmt generated:\n{fetch_stmt.replace("\\'","'")}")
        return fetch_stmt
    
    def _generate_unordered_sql_stmt(self, n_rows:int=None, columns:list[str]=None, include_index:bool=True, ordering:Literal["asc","desc","random"]="asc") -> str:
        """
        Generates an SQL statement for fetching unordered rows from the SQLDataModel, used by `head()`, `tail()` and `sample()` methods to fetch specified number of rows.

        Parameters:
            - `n_rows` (int): The number of rows to fetch. If `None`, fetches all rows.
            - `columns` (list[str]): The list of columns to include in the SELECT statement. If `None`, includes all columns.
            - `include_index` (bool): If True, includes the index column in the SELECT statement.
            - `ordering` (Literal["asc", "desc", "random"]): The ordering of the rows. Can be 'asc' (ascending), 'desc' (descending), or 'random'.

        Returns:
            - `str`: The SQL statement for fetching unordered rows with specified ordering and limit.

        Notes:
            - No argument type validation or out of scope indexing checking occurs.
            - Inputs are assumed validated, no exceptions are raised by this method.
        """
        if n_rows is None:
            n_rows = self.row_count
        if columns is None:
            columns = self.headers
        if isinstance(columns,str):
            columns = [columns]
        columns_str = ",".join([f'"{col}"' for col in columns])
        ordering_str = f"""order by "{self.sql_idx}" {ordering}""" if ordering in ("asc","desc") else "order by random()"
        fetch_stmt = " ".join(("select",f'"{self.sql_idx}",' if include_index else '',columns_str,f'from "{self.sql_model}"', ordering_str, f"limit {n_rows}"))
        return fetch_stmt

    def _update_rows_and_columns_with_values(self, rows_to_update:tuple[int]=None, columns_to_update:list[str]=None, values_to_update:list[tuple]=None) -> None:
        """
        Generates and executes a SQL update statement to modify specific rows and columns with provided values in the SQLDataModel.

        Parameters:
            - `rows_to_update`: A tuple of row indices to be updated. If set to None, it defaults to all rows in the SQLDataModel.
            - `columns_to_update`: A list of column names to be updated. If set to None, it defaults to all columns in the SQLDataModel.
            - `values_to_update`: A list of tuples representing values to update in the specified rows and columns.

        Raises:
            - `DimensionError` if the shape of the provided values does not match the specified rows and columns.
            - `TypeError` if the `values_to_update` parameter is not a list or tuple.
            - `SQLProgrammingError` if the `values_to_update` parameter contains invalid or SQL incompatible data.

        Notes:
            - To create a new column, pass a single header item in a list to the `columns_to_update` parameter.
            - To copy an existing column, pass the corresponding data is a list of tuples to the `values_to_update` parameter.

        Example:
        ```python
        import SQLDataModel

        # Update specific rows and columns with provided values
        sqldm._update_rows_and_columns_with_values(
            rows_to_update=(1, 2, 3),
            columns_to_update=["column1", "column2"],
            values_to_update=[(10, 'A'), (20, 'B'), (30, 'C')]
        )

        # Create a new column named "new_column" with default values
        sqldm._update_rows_and_columns_with_values(
            columns_to_update=["new_column"],
            values_to_update=[(None,)] * sqldm.row_count
        )
        ```
        """
        update_sql_script = None
        rows_to_update = rows_to_update if rows_to_update is not None else tuple(range(self.row_count))
        columns_to_update = columns_to_update if columns_to_update is not None else self.headers
        if not isinstance(values_to_update, tuple|list):
            values_to_update = (values_to_update,)
            rowwise_update = False
        else:
            rowwise_update = True
        if isinstance(values_to_update, list):
            if not isinstance(values_to_update[0], tuple|list):
                values_to_update = tuple(values_to_update)
        if isinstance(values_to_update, tuple):
            values_to_update = [values_to_update]
        num_rows_to_update = len(rows_to_update)
        num_columns_to_update = len(columns_to_update)
        num_value_rows_to_update = len(values_to_update)
        num_value_columns_to_update = len(values_to_update[0])
        create_new_column = True if (num_columns_to_update == 1 and columns_to_update[0] not in self.headers) else False
        if create_new_column:
            new_column = columns_to_update[0]
            new_column_py_dtype = type(values_to_update[0][0]).__name__ # should always have an item
            new_column_sql_dtype = self.static_py_to_sql_map_dict.get(new_column_py_dtype, 'TEXT')
            update_sql_script = f"""alter table "{self.sql_model}" add column "{new_column}" {new_column_sql_dtype};"""
        if not rowwise_update:
            values_to_update = values_to_update[0][0]
            values_to_update = "null" if values_to_update is None else f"""{values_to_update}""" if not isinstance(values_to_update,str|bytes|datetime.date) else f"datetime('{values_to_update}')" if isinstance(values_to_update,datetime.datetime) else f"date('{values_to_update}')" if isinstance(values_to_update,datetime.date) else f"'{values_to_update.replace("'","''")}'" if not isinstance(values_to_update,bytes) else f"""X'{values_to_update.hex()}'"""
            col_val_param = ','.join([f""" "{column}" = {values_to_update} """ for column in columns_to_update]) 
            if update_sql_script is None:
                update_sql_script = f"""update "{self.sql_model}" set {col_val_param} where {self.sql_idx} in {f'{rows_to_update}' if num_rows_to_update > 1 else f'({rows_to_update[0]})'};"""
            else:
                update_sql_script += f"""update "{self.sql_model}" set {col_val_param} where {self.sql_idx} in {f'{rows_to_update}' if num_rows_to_update > 1 else f'({rows_to_update[0]})'};"""
            # print(f'final update script generated:\n{update_sql_script}')
            self.execute_transaction(update_sql_script)
            return            
        if num_rows_to_update != num_value_rows_to_update:
            raise DimensionError(
                SQLDataModel.ErrorFormat(f"DimensionError: shape mismatch '{num_rows_to_update} != {num_value_rows_to_update}', number of rows to update '{num_rows_to_update}' must match provided number of value rows to update '{num_value_rows_to_update}'")
            )
        if num_columns_to_update != num_value_columns_to_update:
            raise DimensionError(
                SQLDataModel.ErrorFormat(f"DimensionError: shape mismatch '{num_columns_to_update} != {num_value_columns_to_update}', number of columns to update '{num_columns_to_update}' must match provided number of value columns to update '{num_value_columns_to_update}'")
            )
        if update_sql_script is not None:
            self.execute_transaction(update_sql_script)
        col_val_param = ','.join([f""" "{column}" = ? """ for column in columns_to_update]) 
        update_stmt = f"""update "{self.sql_model}" set {col_val_param} where {self.sql_idx} = ?"""
        update_params = [(*values_to_update[i], row) for i,row in enumerate(rows_to_update)]
        try:
            self.sql_db_conn.executemany(update_stmt, update_params)
            self.sql_db_conn.commit()
        except sqlite3.ProgrammingError as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f"SQLProgrammingError: invalid update values, SQL execution failed with \"{e}\"")
            ) from None
        self._update_model_metadata()
        return

    def _generate_sql_fetch_for_joining_tables(self, base_headers:list[str], join_table:str, join_column:str, join_headers:list[str], join_type:str='left') -> str:
        """
        Generates a SQL SELECT statement for joining tables, called by the `join_model()` method for performning left or right joins.

        Usage:
            Constructs a SQL SELECT statement to join the base table (specified by 'base_headers') with another table
            (specified by 'join_table') based on a common column ('join_column'). The columns to be selected from
            the base and joined tables are specified by 'base_headers' and 'join_headers', respectively.

        Parameters:
            - `base_headers` (list[str]): List of column names to be selected from the base table.
            - `join_table` (str): Name of the table to be joined with the base table.
            - `join_column` (str): Common column used as the join predicate between the base and joined tables.
            - `join_headers` (list[str]): List of column names to be selected from the joined table.
            - `join_type` (str, optional): Type of join ('left' by default). Can be 'left' or 'left outer'.

        Returns:
            str: A SQL SELECT statement for joining tables.

        Example:
        ```python
        import SQLDataModel

        # Example usage:
        sql_stmt = sqldm._generate_sql_fetch_for_joining_tables(
            base_headers=['ID', 'Name', 'Value'],
            join_table='other_table',
            join_column='ID',
            join_headers=['Description', 'Category'],
            join_type='left'
        )
        print(sql_stmt)
        ```
        """
        # base_headers_str = ",".join([f"""a.\"{v[0]}\" as \"{v[0]}\"""" for v in self.header_idx_dtype_dict.values() if v[0] in base_headers])
        base_headers_str = ",".join([f"""a.\"{col}\" as \"{col}\"""" for col in base_headers])
        join_headers_str = ",".join([f"""b.\"{col}\" as \"{col}\" """ for col in join_headers])
        join_type_str = "left join" if join_type == 'left' else 'left outer join'
        join_predicate_str = f"""from {self.sql_model} a {join_type_str} \"{join_table}\" b on a.\"{join_column}\" = b.\"{join_column}\" """
        sql_join_stmt = f"""select {base_headers_str}, {join_headers_str} {join_predicate_str}"""
        return sql_join_stmt

    def _get_sql_create_stmt(self) -> str:
        """
        Retrieves the SQL CREATE statement used to create the SQLDataModel database.
        Queries the sqlite_master table to fetch the SQL CREATE statement that was used
        to create the underlying database of the SQLDataModel.

        Returns:
            - `str`: The SQL CREATE statement for the SQLDataModel database.

        Notes:
            - This method provides insight into the structure of the database schema.

        Example:
        ```python
        import SQLDataModel
        
        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model
        sqldm = SQLDataModel(data,headers)

        # Retrieve the SQL CREATE statement for the SQLDataModel database
        create_stmt = sqldm._get_sql_create_stmt()

        # Print the returned statement
        print(create_stmt)
        ```
        """
        self.sql_db_conn.execute("select sql from sqlite_master")
        return self.sql_db_conn.fetchone()[0]

    def validate_indicies(self, indicies) -> tuple[int|slice, list[str]]:
        """
        Validates and returns indices for accessing rows and columns in the `SQLDataModel`.

        Parameters:
            - `indicies`: Specifies the indices for rows and columns. It can be of various types:
            - int: Single row index.
            - slice: Range of row indices.
            - tuple: Tuple of disconnected row indices.
            - str: Single column name.
            - list: List of column names.
            - tuple[int|slice, str|list]: Two-dimensional indexing with rows and columns.
            - `strict_validation` (bool, optional): If True, performs strict validation for column names against the current model headers. Default is True.

        Returns:
            - `tuple` containing validated row indices and column indices.

        Raises:
            - `TypeError`: If the type of indices is invalid such as a float for row index or a boolean for a column name index.
            - `ValueError`: If the indices are outside the current model range or if a column is not found in the current model headers when indexed by column name as `str`.
            - `IndexError`: If the column indices are outside the current column range or if a column is not found in the current model headers when indexed by `int`.

        Usage:
        ```python
        import SQLDataModel

        headers = ['idx', 'first', 'last', 'age']
        data = [
            (0, 'john', 'smith', 27)
            ,(1, 'sarah', 'west', 29)
            ,(2, 'mike', 'harlin', 36)
            ,(3, 'pat', 'douglas', 42)
        ]

        # Create the sample model
        sqldm = SQLDataModel(data,headers)

        # Example 1: Validate a single row index
        validated_row_index, validated_columns = model.validate_indicies(3)

        # Example 2: Validate a range of row indices and a list of column names
        validated_row_indices, validated_columns = model.validate_indicies((0, 2, 3), ['first', 'last'])

        # Example 3: Validate a slice for row indices and a single column name
        validated_row_indices, validated_columns = model.validate_indicies(slice(1, 2), 'col_3')

        # Example 4: Validate two-dimensional indexing with rows and columns
        validated_row_indices, validated_columns = model.validate_indicies((slice(0, 3), ['first', 'last']))
        ```

        Note:
            - For two-dimensional indexing, the first element represents rows, and the second element represents columns.
            - Strict validation ensures that column names are checked against the current model headers.
        """        
        ### single row index ###
        if isinstance(indicies, int):
            row_index = indicies
            if row_index < 0:
                row_index = self.row_count + row_index
            if row_index < 0:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}', index must be within current model row range of '0:{self.row_count}'")
                )
            if row_index >= self.row_count:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}', index must be within current model row range of '0:{self.row_count}'")
                )
            return (row_index, self.headers)
        ### single row slice index ###
        if isinstance(indicies, slice):
            row_slice = indicies
            start_idx = row_slice.start if row_slice.start is not None else 0
            stop_idx = row_slice.stop if row_slice.stop is not None else self.row_count
            start_idx = start_idx if start_idx >= 0 else self.row_count + start_idx
            stop_idx = stop_idx if stop_idx > 0 else self.row_count + stop_idx
            if start_idx < 0:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '0:{self.row_count}'")
                )
            if stop_idx <= start_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: insufficient rows '{start_idx-stop_idx}', provided row index returns no valid rows within current model range '0:{self.row_count}'")
                )
            if row_slice.step is None:
                return (slice(start_idx,stop_idx), self.headers)
            else:
                rows_in_scope = tuple(range(self.row_count))[slice(start_idx,stop_idx,row_slice.step)]
                if (num_rows_in_scope := len(rows_in_scope)) < 1:
                    raise IndexError(
                        SQLDataModel.ErrorFormat(f"IndexError: insufficient rows '{num_rows_in_scope}', provided row slice returned no valid row indicies within current model range '0:{self.row_count}'")
                    )
                return (rows_in_scope, self.headers)
        ### columns by str or list of str ###
        if isinstance(indicies, str|list):
            col_index = indicies
            if isinstance(indicies, str):
                col_index = [col_index]
            if not all(isinstance(col, str) for col in col_index):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid column index type '{type(col_index[0].__name__)}' received, use `.get_headers()` to view valid column arguments")
                )
            for col in col_index:
                if col not in self.headers and col != self.sql_idx:
                    raise ValueError(
                        SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', valid columns indicies are required, use `get_headers()` to view current valid arguments")
                    )
            return (slice(0,self.row_count), col_index)
        ### indexing by rows and columns ###        
        if not isinstance(indicies, tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid indexing type '{type(indicies).__name__}', indexing the model must be done using two-dimensional [rows,columns] parameters with 'int' or 'slice' types")
            )
        if (arg_length := len(indicies)) != 2:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid indexing args, expected no more than 2 indicies for [row, column] but '{arg_length}' were received")
            )
        row_indicies, col_indicies = indicies
        ### rows first ###
        if not isinstance(row_indicies, int|tuple|slice):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type for row indexing '{type(row_indicies).__name__}', rows must be indexed by type 'int' or 'slice'")
            )
        if isinstance(row_indicies, int):
            if row_indicies < 0:
                row_indicies = self.row_count + row_indicies
            if row_indicies < 0:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '0:{self.row_count}'")
                )
            if row_indicies > self.row_count:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '0:{self.row_count}'")
                )
            validated_row_indicies = row_indicies
        elif isinstance(row_indicies, tuple): # tuple of disconnected row indicies
            if not all(isinstance(row, int) for row in row_indicies):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid row index type '{type(row_indicies[0]).__name__}', rows must be indexed by type 'int'")
                )
            min_row_idx, max_row_idx = min(row_indicies), max(row_indicies)
            if min_row_idx < 0:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{min_row_idx}' outside of current model range '0:{self.row_count}'")
                )
            if max_row_idx > self.row_count:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{max_row_idx}' outside of current model range '0:{self.row_count}'")
                )
            validated_row_indicies = row_indicies
        else: # is slice
            start_idx = 0 if row_indicies.start is None else row_indicies.start if row_indicies.start >= 0 else (self.row_count + row_indicies.start)
            stop_idx = self.row_count if row_indicies.stop is None else row_indicies.stop if row_indicies.stop >= 0 else (self.row_count + row_indicies.stop)
            if start_idx < 0:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '0:{self.row_count}'")
                )
            if stop_idx <= start_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: insufficient rows '{stop_idx - start_idx}', provided row slice returned no valid row indicies within current model range '0:{self.row_count}'")
                )    
            if row_indicies.step is None:
                validated_row_indicies = slice(start_idx, stop_idx)
            else:
                rows_in_scope = tuple(range(self.row_count))[slice(start_idx,stop_idx,row_indicies.step)]
                if (num_rows_in_scope := len(rows_in_scope)) < 1:
                    raise IndexError(
                        SQLDataModel.ErrorFormat(f"IndexError: insufficient rows '{num_rows_in_scope}', provided row slice returned no valid row indicies within current model range '0:{self.row_count}'")
                    )
                validated_row_indicies = rows_in_scope
        ### then columns ###
        if not isinstance(col_indicies, int|slice|tuple|str|list):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies).__name__}', for column indexing a slice, list or str type is required")
                )        
        if isinstance(col_indicies, int):
            try:
                col_indicies = [self.headers[col_indicies]]
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `get_headers()` to view current valid arguments")
                ) from None
        elif isinstance(col_indicies, slice):
            col_indicies = self.headers[col_indicies]
            if (len_col_args := len(col_indicies)) < 1:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: insufficient columns '{len_col_args}', provided column slice returned no valid column indicies within current model range '0:{self.column_count}'")
                )
        elif isinstance(col_indicies, tuple):
            col_indicies = list(col_indicies)
        elif isinstance(col_indicies, str):
            col_indicies = [col_indicies]
        if not all(isinstance(col, int|str) for col in col_indicies):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies[0].__name__)}', column indexing must be done by 'int' or 'str' types, use `.get_headers()` to view current valid arguments")
            )
        if all(isinstance(col, int) for col in col_indicies):
            try:
                col_indicies = [self.headers[i] for i in col_indicies]
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `get_headers()` to view current valid arguments")
                ) from None                
        ### columns validated to be a list of str, not neccessarily currnet columns, but thats ok for setitem method which should be allowed to create columns ###
        for col in col_indicies:
            if col not in self.headers and col != self.sql_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', valid columns indicies are required, use `get_headers()` to view current valid arguments")
                )
        validated_column_indicies = col_indicies
        return (validated_row_indicies, validated_column_indicies)

