from __future__ import annotations
import sqlite3, os, csv, sys, datetime, pickle, warnings, re, shutil
from typing import Generator, Callable, Literal, Iterator
from collections import namedtuple
from pathlib import Path

# in ./src/SQLDataModel/SQLDataModel.py
try:
    from SQLDataModel.exceptions import DimensionError, SQLProgrammingError
    from SQLDataModel.ANSIColor import ANSIColor
# in ./future/SQLDataModel_future.py
except:
    from exceptions import DimensionError, SQLProgrammingError
    from ANSIColor import ANSIColor

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

###################################################################################################################################
################### header master and row metadata targets for deletion after adjusting to use master dict only ###################
###################################################################################################################################

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
    sdm = sdm.fetch_query("select first_name, last_name, dob, job_title from sdm where job_title = "sales" ")
    sdm.set_headers(["first", "last", "dob", "position"])

    # iterate through the data
    for row in sdm.iterrows():
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
    ┌───┬─────────┬────────┬─────────┬────────┬────────┬─────────────────────┐
    │   │ string  │   ints │ value_2 │ floats │   bits │ datetime            │
    ├───┼─────────┼────────┼─────────┼────────┼────────┼─────────────────────┤
    │ 1 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 2 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 3 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 4 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 5 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 6 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 7 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    │ 8 │ "names" │      1 │ value 2 │ 3.1415 │ b'bit' │ 2023-11-24 15:45:00 │
    └───┴─────────┴────────┴─────────┴────────┴────────┴─────────────────────┘
    ```
    ---
    ### Notes
    use `SQLDM.get_supported_sql_connections()` to view supported databases, please reach out with any issues or questions, thanks!
    """
    __slots__ = ('sql_idx','sql_model','max_rows','min_column_width','max_column_width','column_alignment','display_color','display_index','row_count','min_idx','max_idx','max_out_of_bounds','headers','column_count','static_py_to_sql_map_dict','static_sql_to_py_map_dict','sql_c','sql_db_conn','display_float_precision','header_master')

    debug:bool=False

    def __init__(self, data:list[list], headers:list[str]=None, max_rows:int=1_000, min_column_width:int=6, max_column_width:int=32, column_alignment:str=None, display_color:str=None, display_index:bool=True, display_float_precision:int=4):
        if not isinstance(data, list|tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: type mismatch, \"{type(data).__name__}\" is not a valid type for data, which must be of type list or tuple...")
                )
        if len(data) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: data not found, data of length \"{len(data)}\" is insufficient to construct a valid model, additional rows of data required...")
                )
        try:
            _ = data[0]
        except Exception as e:
            raise IndexError(
                SQLDataModel.ErrorFormat(f"IndexError: data index error, data index provided does not exist for length \"{len(data)}\" due to \"{e}\"...")
                ) from None
        if not isinstance(data[0], list|tuple):
            if type(data[0]).__module__ != 'pyodbc': # check for pyodbc.Row which is acceptable
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: type mismatch, \"{type(data[0]).__name__}\" is not a valid type for data rows, which must be of type list or tuple...")
                    )
        if len(data[0]) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: data rows not found, data rows of length \"{len(data[0])}\" are insufficient to construct a valid model, at least one row is required...")
                )
        if headers is not None:
            if not isinstance(headers, list|tuple):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid header types, \"{type(headers).__name__}\" is not a valid type for headers, please provide a tuple or list type...")
                    )
            if len(headers) != len(data[0]):
                raise DimensionError(
                    SQLDataModel.ErrorFormat(f"DimensionError: invalid data dimensions, provided headers length \"{len(headers)} != {len(data[0])}\", the implied column count for data provided...")
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
        self.max_rows = max_rows
        self.min_column_width = min_column_width
        self.max_column_width = max_column_width
        self.column_alignment = column_alignment
        self.display_color = display_color
        self.display_index = display_index
        self.display_float_precision = display_float_precision
        self.row_count = len(data)
        had_idx = True if headers[0] == self.sql_idx else False
        self.min_idx = min([row[0] for row in data]) if had_idx else 0
        self.max_idx = max([row[0] for row in data]) if had_idx else (self.row_count -1) # since indexing starts at zero now, these are the lower and upper bound indicies
        self.max_out_of_bounds = self.max_idx + 1 # to upper bound for sql statement generation while retaining conventional slicing methods
        dyn_idx_offset,dyn_idx_bind,dyn_add_idx_insert = (1, "?,", f"\"{self.sql_idx}\",") if had_idx else (0, "", "")
        headers = headers[dyn_idx_offset:]
        self.headers = headers
        self.column_count = len(self.headers)
        self.static_py_to_sql_map_dict = {'None': 'NULL','int': 'INTEGER','float': 'REAL','str': 'TEXT','bytes': 'BLOB', 'datetime': 'TIMESTAMP', 'NoneType':'TEXT', 'bool':'INTEGER'}
        self.static_sql_to_py_map_dict = {'NULL': 'None','INTEGER': 'int','REAL': 'float','TEXT': 'str','BLOB': 'bytes', 'TIMESTAMP': 'datetime','':'str'}
        headers_to_py_dtypes_dict = {self.headers[i]:type(data[0][i+dyn_idx_offset]).__name__ if type(data[0][i+dyn_idx_offset]).__name__ != 'NoneType' else 'str' for i in range(self.column_count)}
        headers_to_sql_dtypes_dict = {k:self.static_py_to_sql_map_dict[v] for (k,v) in headers_to_py_dtypes_dict.items()}
        headers_with_sql_dtypes_str = ", ".join(f"""\"{col}\" {type}""" for col,type in headers_to_sql_dtypes_dict.items())
        sql_create_stmt = f"""create table if not exists \"{self.sql_model}\" (\"{self.sql_idx}\" INTEGER PRIMARY KEY,{headers_with_sql_dtypes_str})"""
        sql_insert_stmt = f"""insert into "{self.sql_model}" ({dyn_add_idx_insert}{','.join([f'"{col}"' for col in self.headers])}) values ({dyn_idx_bind}{','.join(['?' for _ in self.headers])})"""
        self.sql_db_conn = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
        self.sql_db_conn.execute(sql_create_stmt)
        self.sql_c = self.sql_db_conn.cursor()
        self._generate_header_master()
        if not had_idx:
            first_row_insert_stmt = f"""insert into "{self.sql_model}" ({self.sql_idx},{','.join([f'"{col}"' for col in self.headers])}) values (?,{','.join(['?' for _ in self.headers])})"""
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
                select_item_fmt = f"""printf('%.{max_pad_width}s', printf('% {max_pad_width}.{float_precision}f',"{column}")) """ # does the most precision then treats as string to trim
            elif dtype == 'int':
                select_item_fmt = f"""printf('% {max_pad_width}s', CASE WHEN length("{column}") <= ({max_pad_width}) THEN "{column}" ELSE substr("{column}",1,({max_pad_width})-2)||'⠤⠄' END) """
            elif dtype == 'index':
                select_item_fmt = f"""printf('%{max_pad_width}s', "{column}") """
            else:
                select_item_fmt = f"""printf('%!-{max_pad_width}s', CASE WHEN length("{column}") <= ({max_pad_width}) THEN "{column}" ELSE substr("{column}",1,({max_pad_width})-2)||'⠤⠄' END) """
            return select_item_fmt
        else:
            if alignment in ("<", ">"):
                numeric_discriminator = """ """ if dtype in ("int","float") else """"""
                float_int_discriminator = f""".{float_precision}f""" if dtype == 'float' else """d""" if dtype == 'int' else """s"""
                printf_alignment = "-" if alignment == "<" else ""
                select_item_fmt = f"""CASE WHEN length(printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',"{column}")) <= {max_pad_width} THEN
                printf('%.{max_pad_width}s', printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',"{column}")) 
                ELSE substr(printf('%{numeric_discriminator}{printf_alignment}{max_pad_width}{float_int_discriminator}',"{column}"),1,({max_pad_width})-2)||'⠤⠄'END """
            else:
                float_int_discriminator = f"""printf('% .{float_precision}f',"{column}")""" if dtype == 'float' else f"""printf('% d',"{column}")""" if dtype == 'int' else f"""printf('%s',"{column}")"""
                select_item_fmt = f"""CASE WHEN length({float_int_discriminator}) <= {max_pad_width} THEN
                printf("%-{max_pad_width}.{max_pad_width}s", /* change minus operator before max_pad_width to favor right alignment when pad width is odd number, right now set to favor left side */
                printf("%*s%s%*s", (({max_pad_width}-length({float_int_discriminator}))/2), "", {float_int_discriminator}, (({max_pad_width}-length({float_int_discriminator}))/2), ""))
                ELSE substr({float_int_discriminator},1,({max_pad_width})-2)||'⠤⠄' END """                 
        return select_item_fmt 
    
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
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers[self.headers.index(column)] = new_column_name # replace old column with new column in same position as original column
        self._generate_header_master()

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
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers[index] = new_value # replace old column at specified index with new value to retain original column ordering
        self._generate_header_master()

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
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self.headers = new_headers
        self._generate_header_master()
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
            apply_function = lambda x: "_".join(x.strip() for x in re.sub('[^0-9a-z _]+', '', x.lower()).split('_') if x !='')
        new_headers = [apply_function(x) for x in self.get_headers()]
        self.set_headers(new_headers)
        return

    def _generate_header_master(self, update_row_meta:bool=False):
        """
        Generates and updates metadata information about the columns and optionally the rows in the SQLDataModel instance based on the current model. 

        Parameters:
            - `update_row_meta` (bool, optional): If True, updates row metadata information; otherwise, retrieves column metadata only (default).
        
        Attributes updated:
            - `self.header_master`: Master dictionary of column metadata.
            - `self.headers`: List of current model headers, order retained.
            - `self.column_count`: Number of columns in current model.
                - `self.min_idx`: Optionally updated, minimum valid model index.
                - `self.max_idx`: Optionally updated, maximum valid model index.
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
        sdm._generate_header_master()

        # Get new column count
        num_cols_after = sdm.column_count

        # View difference
        print(f"cols before: {num_cols_before}, cols after: {num_cols_after}")
        ```
        """        
        if update_row_meta:
            fetch_metadata = f"""select "name","type","is_regular_column","alignment" from (
                select '_rowmeta' as "name", min("{self.sql_idx}") as "type", max("{self.sql_idx}") as "is_regular_column", count("{self.sql_idx}") as "alignment" from "{self.sql_model}"
                union all
                select "name" as "name","type" as "type","pk" as "is_regular_column",case when ("type"='INTEGER' or "type"='REAL') then '>' else '<' end as "alignment" from pragma_table_info('{self.sql_model}') 
            ) order by "name"='_rowmeta' desc, {",".join([f"name=\"{col}\" desc" for col in self.headers])}"""
        else:
            fetch_metadata = f"""select "name" as "name","type" as "type","pk" as "is_regular_column",case when ("type"='INTEGER' or "type"='REAL') then '>' else '<' end as "alignment" from pragma_table_info('{self.sql_model}') order by {",".join([f"name=\"{col}\" desc" for col in self.headers])}"""
        metadata = self.sql_db_conn.execute(fetch_metadata)
        if update_row_meta:
            self.min_idx, self.max_idx, self.row_count = next(metadata)[1:]
            self.max_out_of_bounds = self.max_idx + 1
        header_master = {m[0]:(m[1], self.static_sql_to_py_map_dict[m[1]],True if m[2] == 0 else False,m[3]) for m in metadata}
        self.headers = list(dict.fromkeys([k for k,v in header_master.items() if v[2]]))
        self.column_count = len(self.headers)
        self.header_master = header_master # format: 'column_name': ('sql_dtype', 'py_dtype', is_regular_column, 'default_alignment')

    def get_max_rows(self) -> int:
        """
        Retrieves the current value of the `max_rows` property, which determines the maximum rows displayed for `SQLDataModel`.

        Returns:
            - `int`: The current value of the 'max_rows' property.

        Example:
        ```python
        # Example usage:

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Get current value
        max_rows = sqldm.get_max_rows()

        # By default any display will be limited to 1000 rows max
        print(max_rows) # 1000
        ```
        """
        return self.max_rows
    
    def set_max_rows(self, rows:int) -> None:
        """
        Set `max_rows` to limit rows displayed when `repr` or `print` is called, does not change the maximum rows stored in `SQLDataModel`.

        Parameters:
            - `rows` (int): The maximum number of rows to display.

        Raises:
            - `TypeError`: If the provided argument is not an integer.
            - `IndexError`: If the provided value is less than or equal to 0.

        Example:
        ```python
        import SQLDataModel

        # Create model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])

        # Any call to `print` or `repr` will be restricted to 500 max rows
        sqldm.set_max_rows(500)
        ```
        """
        # if type(rows) != int:
        if not isinstance(rows, int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid argument type "{type(rows).__name__}", please provide an integer value to set the maximum rows attribute...')
                )
        if rows <= 0:
            raise IndexError(
                SQLDataModel.ErrorFormat(f'IndexError: invalid value "{rows}", please provide an integer value >= 1 to set the maximum rows attribute...')
                )
        self.max_rows = rows

    def get_min_column_width(self) -> int:
        """
        Returns the current `min_column_width` property value.

        Returns:
            int: The current value of the `min_column_width` property.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        min_width = sqldm.get_min_column_width()
        print(min_width)  # Output: 6
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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm.set_min_column_width(8)
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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        max_width = sqldm.get_max_column_width()
        print(max_width)  # Output: 32
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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm.set_max_column_width(20)
        ```
        """
        self.max_column_width = width

    def get_column_alignment(self) -> str:
        """
        Returns the current `column_alignment` property value, `None` by default.

        Returns:
            str: The current value of the `column_alignment` property.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        alignment = sqldm.get_column_alignment()
        print(alignment)  # Output: None
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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        sqldm.set_column_alignment('<')
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
            `bool`: The current value of the `display_index` property.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        display_index = sqldm.get_display_index()
        print(display_index)  # Output: True
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
            tuple[int]: A tuple representing the number of rows and columns in the SQLDataModel.

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
    def from_csv(cls, csv_file:str, delimeter:str=',', quotechar:str='"', headers:list[str] = None, **kwargs) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` from the provided CSV file.

        Parameters:
            - `csv_file` (str): The path to the CSV file.
            - `delimiter` (str, optional): The delimiter used in the CSV file. Default is ','.
            - `quotechar` (str, optional): The character used for quoting fields. Default is '"'.
            - `headers` (list[str], optional): List of column headers. If None, the first row of the CSV file is assumed to contain headers.
            - `*args`, `**kwargs`: Additional arguments to be passed to the SQLDataModel constructor.

        Returns:
            `SQLDataModel`: The SQLDataModel object created from the provided CSV file.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        ```
        """
        with open(csv_file) as csvfile:
            tmp_all_rows = tuple(list(row) for row in csv.reader(csvfile, delimiter=delimeter,quotechar=quotechar))
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
            tuple: A tuple of supported DB API 2.0 dialects.

        Example:
        ```python
        supported_dialects = SQLDataModel.get_supported_sql_connections()
        ```
        """
        return ('sqlite3', 'psycopg2', 'pyodbc', 'cx_oracle', 'teradatasql')
    
    @classmethod
    def from_sql(cls, sql_query: str, sql_connection: sqlite3.Connection, **kwargs) -> SQLDataModel:
        """
        Create a `SQLDataModel` object by executing the provided SQL query using the specified SQL connection.

        If a single word is provided as the `sql_query`, the method wraps it and executes a select all:
        ```python
        sqldm = SQLDataModel.from_sql("table_name", sqlite3.Connection)
        ```
        This is equivalent to:
        ```python
        sqldm = SQLDataModel.from_sql("select * from table_name", sqlite3.Connection)
        ```

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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        data = self.sql_c.fetchall()
        if (len(data) == 1) and (not include_headers): # if only single row
            data = data[0]
        if len(data) == 1: # if only single cell
            data = data[0]
        return [tuple([x[0] for x in self.sql_c.description]),data] if include_headers else data

    def to_csv(self, csv_file:str, delimeter:str=',', quotechar:str='"', include_index:bool=False, **kwargs):
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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        write_headers = [x[0] for x in self.sql_c.description]
        with open(csv_file, 'w', newline='') as file:
            csvwriter = csv.writer(file,delimiter=delimeter,quotechar=quotechar,quoting=csv.QUOTE_MINIMAL, **kwargs)
            csvwriter.writerow(write_headers)
            csvwriter.writerows(self.sql_c.fetchall())
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
        self.sql_c.execute(self._generate_sql_stmt(include_index=True))
        if rowwise:
            return {row[0]:row[1:] for row in self.sql_c.fetchall()}
        else:
            data = self.sql_c.fetchall()
            headers = [x[0] for x in self.sql_c.description]
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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        return [tuple([x[0] for x in self.sql_c.description]),*self.sql_c.fetchall()] if include_headers else self.sql_c.fetchall()
    
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
        self.sql_c.execute(fetch_stmt)
        if include_headers:
            return _np.vstack([_np.array([x[0] for x in self.sql_c.description]),[_np.array(x) for x in self.sql_c.fetchall()]])
        return _np.array([_np.array(x) for x in self.sql_c.fetchall()])

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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        raw_data = self.sql_c.fetchall()
        data = [x[1:] for x in raw_data] if include_index else [x for x in raw_data]
        indicies = [x[0] for x in raw_data] if include_index else None
        columns = ([x[0] for x in self.sql_c.description[1:]] if include_index else [x[0] for x in self.sql_c.description]) if include_headers else None
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
            print(SQLDataModel.WarnFormat(f"{type(self).__name__}Warning: extension missing, provided filename \"{filename}\" did not contain an extension and so \".sdm\" was appended to create a valid filename..."))
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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        model_data = [x for x in self.sql_c.fetchall()] # using new process
        model_headers = [x[0] for x in self.sql_c.description]
        created_header_dict = {col:self.header_master[col][0] for col in model_headers}
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
        sql_dtypes_stmt = ", ".join(f"""\"{header}\" {created_header_dict[header]}""" for header in model_headers) # generates sql create table statement using type mapping dict
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
            None

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

    def to_local_db(self, db:str=None):
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
        return self.__add__(value)
    
    def __isub__(self, value) -> SQLDataModel:
        return self.__sub__(value)    
    
    def __imul__(self, value) -> SQLDataModel:
        return self.__mul__(value)    

    def __idiv__(self, value) -> SQLDataModel:
        return self.__truediv__(value)
    
    def __ipow__(self, value) -> SQLDataModel:
        return self.__pow__(value)
    
    def __iter__(self) -> Iterator:
        """
        Iterates over a range of rows in the `SQLDataModel` based on the current model's row indices.

        Yields:
            `tuple`: A row fetched from the `SQLDataModel`.

        Notes:
            - This iterator fetches rows from the `SQLDataModel` using a SQL statement generated
            by the `_generate_sql_stmt()` method.
            - The iteration starts from the 'min_idx' value and continues until
            'max_out_of_bounds' is reached.
        
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
        iter_idx = self.min_idx
        self.sql_c.execute(self._generate_sql_stmt(include_index=True, rows=slice(iter_idx,None)))
        while iter_idx < self.max_out_of_bounds:
            yield from (x for x in self.sql_c.fetchall())
            iter_idx += 1

    def __getitem__(self, slc) -> SQLDataModel:
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
            validated_indicies = self.validate_indicies(slc)
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
        validated_rows, validated_columns = validated_indicies
        # print(f'parsed as:\nrows: {validated_rows}\ncols: {validated_columns}')
        return self._generate_sql_stmt(columns=validated_columns,rows=validated_rows, execute_fetch=True)

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
        if not isinstance(update_values, str|int|float|bool|bytes|list|tuple) and (update_values is not None):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid values type '{type(update_values).__name__}', update values must be compatible with SQL datatypes such as <'str', 'int', 'float', 'bool', 'bytes'>")
            )
        # normal update values process where target update values is not another SQLDataModel object:
        try:
            validated_indicies = self.validate_indicies(target_indicies,strict_validation=False)
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
        validated_rows, validated_columns = validated_indicies
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
            `int`: The total number of rows in the SQLDataModel.

        Example:
        ```python
        # Example usage:
        sqldm = SQLDataModel.from_csv('example.csv', headers=['ID', 'Name', 'Value'])
        num_rows = len(sqldm)
        print(num_rows)  # Output: 1000
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
            - Max displayed rows set to 1,000 by default, use `set_max_rows()` to modify
            - Set table color using `set_display_color()` method
        """        
        max_display_rows = self.row_count if self.row_count < self.max_rows else self.max_rows
        max_rows_check = self.row_count if self.row_count < 12 else 12 # 12 makes sense to get possible second digit incrementation in values
        display_index = self.display_index
        display_headers = [self.sql_idx] + self.headers if display_index else self.headers
        headers_select_length = f"""select max("{self.sql_idx}") as 'x',""" if display_index else """select """
        header_py_dtype_dict = {col:cmeta[1] for col, cmeta in self.header_master.items()}
        header_printf_modifiers_dict = {col:(f'\'% .{self.display_float_precision}f\'' if dtype == 'float' else '\'% d\'' if dtype == 'int' else '\'%!s\'') for col,dtype in header_py_dtype_dict.items()}
        headers_select_length = f"""{headers_select_length} {",".join([f"case when max(max(length(printf({header_printf_modifiers_dict[col]},\"{col}\"))),length('{col}')) > {self.max_column_width} then {self.max_column_width} when max(max(length(printf({header_printf_modifiers_dict[col]},\"{col}\"))),length('{col}')) < {self.min_column_width} then {self.min_column_width} else max(max(length(printf({header_printf_modifiers_dict[col]},\"{col}\"))),length('{col}')) end as '{col}'" for col in self.headers])} from("""
        headers_sub_select = f""" select(select max(length("{self.sql_idx}")) from (select "{self.sql_idx}" from "{self.sql_model}" limit {max_display_rows})) "{self.sql_idx}", """ if display_index else """select """
        headers_sub_select = f"""{headers_sub_select} {",".join([f"\"{col}\"" for col in display_headers])} from \"{self.sql_model}\" order by "{self.sql_idx}" asc limit {max_rows_check} )"""
        headers_full_select = f"{headers_select_length}{headers_sub_select}"
        length_meta = self.sql_db_conn.execute(headers_full_select).fetchone()
        header_length_dict = {display_headers[i]:width for i, width in enumerate(length_meta)}  
        table_repr = """""" # big things...
        table_left_edge = """│ """
        table_left_edge_width = 2           # """│ """
        table_right_edge = """ │"""
        table_right_edge_width = 2          # """ │"""
        table_column_interval_width = 3    # """ │ """
        table_truncated_ellipses = """⠤⠄"""
        table_truncated_ellipses_width = 3 # added extra space after truncation mark before ellipses, looks better
        table_bare_newline = """\n"""
        total_required_width = table_left_edge_width + sum((table_column_interval_width + length) for length in header_length_dict.values()) + table_right_edge_width - table_column_interval_width
        total_available_width = shutil.get_terminal_size()[0]
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
        fetch_fmt_stmt = f"""select '{table_left_edge}' || {fetch_idx}{header_fmt_str}||' │{table_dynamic_newline}' as "full_row" from "{self.sql_model}" limit {self.max_rows}"""
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
        table_repr = "".join([table_repr, f"""[{max_display_rows} rows x {self.column_count} columns]"""])
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
        min_row, max_row = min_row if min_row is not None else self.min_idx, max_row if max_row is not None else self.row_count
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
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_idx_col))
        yield from (Row(*x) for x in self.sql_c.fetchall())

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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['Name', 'Age', 'Salary'])
        head_result = sqldm.head(3)
        print(head_result)
        ```
        """
        return self._generate_sql_stmt(fetch_limit=n_rows, execute_fetch=True)
    
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

        sqldm = SQLDataModel.from_csv('example.csv', headers=['Name', 'Age', 'Salary'])
        tail_result = sqldm.tail(3)
        print(tail_result)
        ```
        """
        rows = slice((self.max_idx-n_rows), self.max_idx + 1)
        return self._generate_sql_stmt(rows=rows, execute_fetch=True)
    
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
        return self.fetch_query(fetch_stmt)
        
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
        return self.fetch_query(derived_query)

    def get_column_dtype(self, column:str|int) -> str:
        """
        Returns the data type of the specified `column` by column name or column index.

        Parameters:
            - `column` (str or int): The name or index of the column to retrieve the data type.

        Raises:
            - `TypeError`: If `column` is not of type 'str' or 'int'.
            - `IndexError`: If `column` is an integer and the index is outside of the current model range.
            - `ValueError`: If `column` is a string and the column is not found in the current model.

        Returns:
            - str: The data type of the specified column.

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
                
        # Get the data type of the 'age' column
        dtype = sqldm.get_column_dtype('age')

        # Outputs 'int'
        print(dtype)

        # Get the data type of the 'last' column by index
        dtype = sqldm.get_column_dtype(2)

        # Outputs 'str'
        print(dtype)
        ```

        Note:
            - This method retrieves the data type of the specified column from the SQLDataModel.
        """
        if not isinstance(column, str|int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(column).__name__}', `column` must be one of 'str' or 'int', use `get_headers()` to view current valid arguments")
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
        return self.header_master[column][1]

    def set_column_dtype(self, column:str|int, dtype:Literal['bytes','datetime','float','int','str']) -> None:
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
        sqldm.set_column_dtype('age', 'float')

        # Confirm column dtype
        new_dtype = sqldm.get_column_dtype('age')

        # Outputs 'float'
        print(new_dtype)
        ```
        """
        if not isinstance(column, str|int):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type '{type(column).__name__}', `column` must be one of 'str' or 'int', use `get_headers()` to view current valid arguments")
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
        update_col_sql = f"""alter table "{self.sql_model}" add column "{column}_x" {col_sql_dtype}; update "{self.sql_model}" set "{column}_x" = cast("{column}" as {col_sql_dtype}); alter table "{self.sql_model}" drop column "{column}"; alter table "{self.sql_model}" rename column "{column}_x" to "{column}"; """
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
            self.sql_c.executescript(full_stmt)
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
            self.sql_c.execute(sql_stmt)
            self.sql_db_conn.commit()
            self._generate_header_master(update_row_meta=True)
            return
        else:
            sql_stmt = f"""select * from "{self.sql_model}" where rowid in (select {dyn_keep_order}(rowid) from "{self.sql_model}" group by {','.join(f'"{col}"' for col in subset)})"""
            return self.fetch_query(sql_stmt)

    def fetch_query(self, sql_query:str) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` object after executing the provided SQL query using the current `SQLDataModel`.

        Parameters:
            - `sql_query` (str): The SQL query to execute.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the result of the SQL query.

        Raises:
            - `SQLProgrammingError`: If the provided SQL query is invalid or malformed.
            - `ValueError`: If the provided SQL query was valid but returned 0 rows, which is insufficient to return a new model from

        Example:
        ```python
        import SQLDataModel

        # Create the model
        sqldm = SQLDataModel.from_csv('example.csv', headers=['Column1', 'Column2'])

        # Create the fetch query to use
        query = 'SELECT * FROM sdm WHERE Column1 > 10'

        # Fetch and save the result to a new instance
        result_model = sqldm.fetch_query(query)
        ```

        Important:
            - The default table name is 'sdm', or you can use `SQLDataModel.get_model_name()` to get the current model alias.
            - This function is the primary method used by `SQLDataModel` methods that are expected to return a new instance.
        """
        try:
            self.sql_c.execute(sql_query)
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, provided query failed with error "{e}"...')
            ) from None
        fetch_result = self.sql_c.fetchall()
        fetch_headers = [x[0] for x in self.sql_c.description]
        if (rows_returned := len(fetch_result)) < 1:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: nothing to return, provided query returned '{rows_returned}' rows which is insufficient to return or generate a new model from")
            )
        return type(self)(fetch_result, headers=fetch_headers, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)

    def group_by(self, *columns:str, order_by_count:bool=True) -> SQLDataModel:
        """
        Returns a new `SQLDataModel` after performing a group by operation on specified columns.

        Parameters:
            - `*columns` (str, list, tuple): Columns to group by. Accepts either individual strings or a list/tuple of strings.

        Keyword Args:
            - `order_by_count` (bool, optional): If True (default), orders the result by count. If False, orders by the specified columns.

        Returns:
            - `SQLDataModel`: A new `SQLDataModel` instance containing the result of the group by operation.

        Raises:
            - `TypeError`: If the columns argument is not of type str, list, or tuple.
            - `ValueError`: If any specified column does not exist in the current model.
            - `SQLProgrammingError`: If any specified columns or aggregate keywords are invalid or incompatible with the current model.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('data.csv')  # create model from data
        sqldm.group_by("country")  # by single string
        sqldm.group_by("country", "state", "city")  # by multiple strings
        sqldm.group_by(["country", "state", "city"])  # by multiple list
        ```

        Notes:
            - Use `order_by_count=False` to change ordering from count to column arguments.
        """
        if type(columns[0]) == str:
            columns = columns
        elif type(columns[0]) in (list, tuple):
            columns = columns[0]
        else:
            raise TypeError(
                SQLDataModel.ErrorFormat(f'TypeError: invalid columns argument, provided type {type(columns[0]).__name__} is invalid, please provide str, list or tuple type...')
                )
        for col in columns:
            if col not in self.headers:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f'ValueError: invalid group by targets, provided column \"{col}\" does not exist in current model, valid targets:\n{self.headers}')
                    )
        columns_group_by = ",".join(f'"{col}"' for col in columns)
        order_by = "count(*)" if order_by_count else columns_group_by
        group_by_stmt = f"""select {columns_group_by}, count(*) as count from "{self.sql_model}" group by {columns_group_by} order by {order_by} desc"""
        try:
            self.sql_c.execute(group_by_stmt)
        except Exception as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, provided query failed with error "{e}"...')
            ) from None
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)
    
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
        self.sql_c.execute(sql_join_stmt)
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)

    def execute_query(self, sql_query:str) -> None:
        """
        Executes an arbitrary SQL query against the current model without the expectation of selection or returned rows.

        Parameters:
            - `sql_query` (str): The SQL query to execute.
            
        Raises:
            - `SQLProgrammingError`: If the SQL execution fails.

        Example:
        ```python
        import SQLDataModel

        sqldm = SQLDataModel.from_csv('data.csv')  # create model from data
        sqldm.execute_query('UPDATE table SET column = value WHERE condition')  # execute an update query
        ```
        """
        try:
            self.sql_c.execute(sql_query)
            self.sql_db_conn.commit()
            self._generate_header_master(update_row_meta=True)
            if SQLDataModel.debug:
                print(SQLDataModel.SuccessFormat(f'SQLDataModel: Executed SQL, provided query executed with {self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0} rows modified'))
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, unable to execute provided SQL query with error "{e}"...')
            ) from None
    
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
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
            rows_modified = self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to execute provided transaction, SQL execution failed with: "{e}"')
            ) from None
        self._generate_header_master(update_row_meta=True)        
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: Executed SQL, provided query executed with {rows_modified} rows modified')  )

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
        create_col_stmt = f"""alter table {self.sql_model} add column \"{column_name}\""""
        if (value is not None) and (value in self.headers):
            dyn_dtype_default_value = f"""{self.header_master[value][0]}"""
            dyn_copy_existing = f"""update {self.sql_model} set \"{column_name}\" = \"{value}\";"""
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
        if isinstance(value, list|tuple):
            if (len_values := len(value)) != self.row_count:
                raise DimensionError(
                    SQLDataModel.ErrorFormat(f'DimensionError: invalid dimensions "{len_values} != {self.row_count}", provided values have length {len_values} while current row count is {self.row_count}...')
                )
            try:
                seq_dtype = self.static_py_to_sql_map_dict[type(value[0]).__name__]
            except:
                raise TypeError(
                    SQLDataModel.ErrorFormat(f'TypeError: invalid datatype "{type(value[0]).__name__}", please provide a valid and SQL translatable datatype...')
                ) from None
            sql_script = f"""{create_col_stmt} {seq_dtype};"""
            all_model_idxs = tuple(range(self.min_idx, self.max_idx+1)) # plus 1 otherwise range excludes last item and out of range error is triggered during for loop below
            for i,val in enumerate(value):
                sql_script += f"""update {self.sql_model} set \"{column_name}\" = \"{val}\" where {self.sql_idx} = {all_model_idxs[i]};"""
            print(sql_script)
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
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to apply function, SQL execution failed with: {e}')
            ) from None
        self._generate_header_master()
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
        insert_cols = ",".join([f"\"{col}\"" for col in self.headers])
        insert_vals = ",".join(["?" for _ in values])
        insert_stmt = f"""insert into {self.sql_model}({insert_cols}) values ({insert_vals})"""
        try:
            self.sql_c.execute(insert_stmt, values)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: {e}')
            ) from None
        self._generate_header_master(update_row_meta=True)        

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
            row_index = self.max_out_of_bounds + row_index
        if row_index < self.min_idx or row_index > self.max_idx:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}', provided row index is outisde of current model range '{self.min_idx}:{self.max_idx}'")
            )
        if isinstance(column_index, int):
            try:
                column_index = self.headers[column_index]
            except IndexError:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid column index '{column_index}', provided column index is outside of current model range '0:{self.column_count-1}'")
                ) from None
        if column_index not in self.headers:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid column name '{column_index}', use `.get_headers()` to view current valid model headers")
            )
        update_stmt = f"""update \"{self.sql_model}\" set {column_index} = ? where {self.sql_idx} = {row_index}"""
        if not isinstance(value,tuple):
            value = (value,)
        try:
            self.sql_c.execute(update_stmt, value)
            self.sql_db_conn.commit()
            rows_modified = self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: {e}')
            ) from None
        self._generate_header_master()        
        if SQLDataModel.debug:
            print(SQLDataModel.SuccessFormat(f'SQLDataModel: Executed SQL, provided query executed with {rows_modified} rows modified'))

    def _generate_sql_stmt(self, columns:None|slice|tuple|int|str|list[str]=None, rows:None|slice|tuple|int=None, include_index:bool=True, fetch_limit:int=None, execute_fetch:bool=False) -> str|SQLDataModel:
        """
        Generates and optionally executes an SQL statement for querying the `SQLDataModel` based on specified columns and rows.

        Parameters:
            - columns (None, slice, tuple, int, str, list[str], optional): Specifies the columns to include in the SQL statement.
            - None: All columns will be used.
            - slice: Range of columns using a slice.
            - tuple: Discontiguous columns using indices as ints.
            - int: Single column using index as an int.
            - str: Single column using column name.
            - list[str]: List of column names.
            - rows (None, slice, tuple, int, optional): Specifies the rows to include in the SQL statement.
            - None: All rows will be used.
            - slice: Range of rows using a slice.
            - tuple: Discontiguous rows using indices as ints.
            - int: Single row using index as an int.
            - include_index (bool, optional): Whether to include the index column in the SQL statement. Default is True.
            - fetch_limit (int, optional): Limits the number of rows fetched. Default is None (no limit).
            - execute_fetch (bool, optional): If True, executes the generated SQL statement and returns a new SQLDataModel. If False, returns the generated SQL statement only.

        Returns:
            - If `execute_fetch` is False, returns the generated SQL statement as a string.
            - If `execute_fetch` is True, executes the SQL statement, creates a new SQLDataModel, and returns it.

        Raises:
            - `SQLProgrammingError`: If there is an issue with the SQL execution during fetch.

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
        
        # Create the model using sample data
        sqldm = SQLDataModel(data,headers)

        # Generates statement and executes, returning values as new SQLDataModel
        sqldm = self._generate_sql_stmt(columns=['first'], rows=(0,2,3), execute_fetch=True)

        # Generates only statement for data at row index 2
        sql_stmt = self._generate_sql_stmt(rows=2)

        # Ordering of columns determines ordering of data
        columns = ['first', 'last']

        # Indexing can be done by the following parameters, types:
        rows = slice(1,10) # slice for range of rows
        rows = (1,3,5,9) # tuple for discontiguous rows
        rows = 2 # int for single row
        columns = ['first', 'last']
        columns = slice(1,3)
        columns = 3
        ```

        Note:
            - Use `execute_fetch=False` to generate the SQL statement string only.
            - Use `execute_fetch=True` to execute the generated SQL statement and return its result.
        """
        if isinstance(columns, slice): # slice of column range
            min_col = columns.start if columns.start is not None else 0
            max_col = columns.stop if columns.stop is not None else self.column_count
            columns = [self.headers[i] for i in range(min_col,max_col)]
        elif isinstance(columns, tuple): # column indicies as ints
            columns = [self.headers[i] for i in columns]
        elif isinstance(columns, int):
            columns = [self.headers[columns]]
        elif isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, list):
            columns = columns
        else:
            columns = self.headers
        headers_str = ",".join([f'"{col}"' for col in columns])
        if isinstance(rows, slice):
            min_row = rows.start if rows.start is not None else self.min_idx
            max_row = rows.stop if rows.stop is not None else self.max_out_of_bounds
            rows_str = f"where {self.sql_idx} >= {min_row} and {self.sql_idx} < {max_row}"
        elif isinstance(rows, tuple):
            rows_str = f"where {self.sql_idx} in {rows}" if len(rows) > 1 else f"where {self.sql_idx} in ({rows[0]})"
        elif isinstance(rows, int):
            if rows < 0:
                rows = self.max_out_of_bounds + rows
            rows_str = f"where {self.sql_idx} = {rows}"
        else:
            rows_str = ""
        index_str = f'"{self.sql_idx}",' if include_index else "" # if index included
        limit_str = f"limit {fetch_limit}" if fetch_limit is not None else "" # if limit included
        fetch_stmt = f"""select {index_str}{headers_str} from "{self.sql_model}" {rows_str} order by {self.sql_idx} asc {limit_str}"""
        if not execute_fetch:
            return fetch_stmt # return generated sql only
        else: # execute and return model
            try:
                self.sql_c.execute(fetch_stmt)
            except Exception as e:
                raise SQLProgrammingError(
                    SQLDataModel.ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, unable to execute provided SQL query with error "{e}"...')
                ) from None
            return type(self)(self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, display_float_precision=self.display_float_precision)

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
        rows_to_update = rows_to_update if rows_to_update is not None else tuple(range(self.min_idx, self.max_idx+1))
        columns_to_update = columns_to_update if columns_to_update is not None else self.headers
        if not isinstance(values_to_update, list|tuple):
            values_to_update = (values_to_update,)
            rowwise_update = False
        else:
            rowwise_update = True
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
            values_to_update = "null" if values_to_update is None else f"""{values_to_update}""" if not isinstance(values_to_update,str|bytes) else f"'{values_to_update}'" if not isinstance(values_to_update,bytes) else f"""'{values_to_update.decode('utf-8')}'"""
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
            self.sql_c.executemany(update_stmt, update_params)
            self.sql_db_conn.commit()
        except sqlite3.ProgrammingError as e:
            raise SQLProgrammingError(
                SQLDataModel.ErrorFormat(f"SQLProgrammingError: invalid update values, SQL execution failed with \"{e}\"")
            ) from None
        self._generate_header_master()
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
        self.sql_c.execute("select sql from sqlite_master")
        return self.sql_c.fetchone()[0]

    def validate_indicies(self, indicies, strict_validation:bool=True) -> tuple[int|slice, list[str]]:
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
                row_index = self.max_out_of_bounds + row_index
            if row_index < self.min_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            if row_index > self.max_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_index}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            return (row_index, self.headers)
        ### single row slice index ###
        if isinstance(indicies, slice):
            row_slice = indicies
            start_idx = row_slice.start if row_slice.start is not None else self.min_idx
            stop_idx = row_slice.stop if row_slice.stop is not None else self.max_out_of_bounds
            if start_idx < self.min_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if stop_idx > self.max_out_of_bounds:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{stop_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )                
            return (slice(start_idx,stop_idx), self.headers)        
        ### columns by str or list of str ###
        if isinstance(indicies, str|list):
            col_index = indicies
            if isinstance(indicies, str):
                col_index = [col_index]
            if not all(isinstance(col, str) for col in col_index):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid column index type '{type(col_index[0].__name__)}' received, use `.get_headers()` to view valid column arguments...")
                )
            for col in col_index:
                if (col not in self.headers) and (strict_validation):
                    raise ValueError(
                        SQLDataModel.ErrorFormat(f"ValueError: '{col}' is not one of the current model headers, use `.get_headers()` method to view current valid headers...")
                    )
            return (slice(self.min_idx,self.max_out_of_bounds), col_index)
        ### indexing by rows and columns ###        
        if not isinstance(indicies, tuple):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid indexing type '{type(indicies).__name__}', indexing the model must be done using two-dimensional [rows,columns] parameters with 'int' or 'slice' types...")
            )
        if (arg_length := len(indicies)) != 2:
            raise ValueError(
                SQLDataModel.ErrorFormat(f"ValueError: invalid indexing args, expected no more than 2 indicies for [row, column] but '{arg_length}' were received")
            )
        row_indicies, col_indicies = indicies
        ### rows first ###
        if not isinstance(row_indicies, int|tuple|slice):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid type for row indexing '{type(row_indicies).__name__}', rows must be indexed by type 'int' or 'slice'...")
            )
        if isinstance(row_indicies, int):
            if row_indicies < 0:
                row_indicies = self.max_out_of_bounds + row_indicies
            if row_indicies < self.min_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            if row_indicies > self.max_out_of_bounds:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            validated_row_indicies = row_indicies
        elif isinstance(row_indicies, tuple): # tuple of disconnected row indicies
            if not all(isinstance(row, int) for row in row_indicies):
                raise TypeError(
                    SQLDataModel.ErrorFormat(f"TypeError: invalid row index type '{type(row_indicies[0]).__name__}', rows must be indexed by type 'int'")
                )
            min_row_idx, max_row_idx = min(row_indicies), max(row_indicies)
            if min_row_idx < self.min_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{min_row_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if max_row_idx > self.max_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{max_row_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            validated_row_indicies = row_indicies
        else: # is slice
            start_idx = row_indicies.start if row_indicies.start is not None else self.min_idx
            stop_idx = row_indicies.stop if row_indicies.stop is not None else self.max_out_of_bounds
            if start_idx < self.min_idx:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if stop_idx > self.max_out_of_bounds:
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: provided row index '{stop_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )    
            validated_row_indicies = slice(start_idx, stop_idx)
        ### then columns ###
        if not isinstance(col_indicies, int|slice|tuple|str|list):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies).__name__}', for column indexing a slice, list or str type is required...")
                )        
        if isinstance(col_indicies, int):
            try:
                col_indicies = [self.headers[col_indicies]]
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                ) from None
        elif isinstance(col_indicies, slice):
            col_start = col_indicies.start if col_indicies.start is not None else 0
            col_stop = col_indicies.stop if col_indicies.stop is not None else self.column_count
            if (col_start < 0) or (col_stop > self.column_count):
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: invalid column index '{col_start}', columns index must be inside of current model range: '0:{self.column_count-1}'")
                )                  
            col_indicies = [self.headers[i] for i in tuple(range(col_start,col_stop))]
        elif isinstance(col_indicies, tuple):
            col_indicies = list(col_indicies)
        elif isinstance(col_indicies, str):
            col_indicies = [col_indicies]
        if not all(isinstance(col, int|str) for col in col_indicies):
            raise TypeError(
                SQLDataModel.ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies[0].__name__)}', column indexing must be done by 'int' or 'str' types, use `.get_headers()` to view current valid arguments...")
            )
        if all(isinstance(col, int) for col in col_indicies):
            try:
                col_indicies = [self.headers[i] for i in col_indicies]
            except IndexError as e:
                raise IndexError(
                    SQLDataModel.ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                ) from None                
        ### columns validated to be a list of str, not neccessarily currnet columns, but thats ok for setitem method which should be allowed to create columns ###
        for col in col_indicies:
            if (col not in self.headers) and (strict_validation):
                raise ValueError(
                    SQLDataModel.ErrorFormat(f"ValueError: column not found '{col}', use `.get_headers()` to view current valid headers...")
                )
        validated_column_indicies = col_indicies
        return (validated_row_indicies, validated_column_indicies)
