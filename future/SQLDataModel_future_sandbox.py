from __future__ import annotations
import sqlite3, os, csv, sys, datetime, pickle, warnings, re
from typing import Generator, Callable
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
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

def ErrorFormat(error:str) -> str:
    """returns an ansi formatted error message, coloring the error type in bold red and displaying the rest of the error message alongside"""
    error_type, error_description = error.split(':',1)
    return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""

def WarnFormat(warn:str) -> str:
    """returns an ansi formatted warning message, coloring the class name in bold yellow and displaying the rest of the warning message alongside"""
    warned_by, warning_description = warn.split(':',1)
    return f"""\r\033[1m\033[38;2;246;221;109m{warned_by}:\033[0m\033[39m\033[49m{warning_description}"""

def create_placeholder_data(n_rows:int, n_cols:int) -> list[list]:
    return [[f"value {i}" if i%2==0 else i**2 for i in range(n_cols-6)] + [3.1415, b'bit', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True, '', None] for _ in range(n_rows)]

@dataclass # speeds up instantiation ~4% as dataclass
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
    sdm = sdm.fetch_query("select first_name, last_name, dob, job_title from sdmmodel where job_title = "sales" ")
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
    ```python
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
    def __init__(self, data:list[list], headers:list[str]=None, max_rows:int=1_000, min_column_width:int=6, max_column_width:int=32, column_alignment:str=None, display_color:str=None, display_index:bool=True, *args, **kwargs):
        self.clsname = type(self).__name__
        self.SDMError = ANSIColor(text_color=(247,141,160),text_bold=True)
        self.clserror = ANSIColor(text_bold=True).alert(self.clsname, 'E')
        self.clswarning = ANSIColor(text_bold=True).alert(self.clsname, 'W')
        self.clssuccess = ANSIColor(text_bold=True).alert(self.clsname, 'S')
        if not isinstance(data, list|tuple):
            raise TypeError(
                ErrorFormat(f"TypeError: type mismatch, \"{type(data).__name__}\" is not a valid type for data, which must be of type list or tuple...")
                )
        if len(data) < 1:
            raise ValueError(
                ErrorFormat(f"ValueError: data not found, data of length \"{len(data)}\" is insufficient to construct a valid model, additional rows of data required...")
                )
        try:
            _ = data[0]
        except Exception as e:
            raise IndexError(
                ErrorFormat(f"IndexError: data index error, data index provided does not exist for length \"{len(data)}\" due to \"{e}\"...")
                ) from None
        if not isinstance(data[0], list|tuple):
            if type(data[0]).__module__ != 'pyodbc': # check for pyodbc.Row which is acceptable
                raise TypeError(
                    ErrorFormat(f"TypeError: type mismatch, \"{type(data[0]).__name__}\" is not a valid type for data rows, which must be of type list or tuple...")
                    )
        if len(data[0]) < 1:
            raise ValueError(
                ErrorFormat(f"ValueError: data rows not found, data rows of length \"{len(data[0])}\" are insufficient to construct a valid model, at least one row is required...")
                )
        if headers is not None:
            if not isinstance(headers, list|tuple):
                raise TypeError(
                    ErrorFormat(f"TypeError: invalid header types, \"{type(headers).__name__}\" is not a valid type for headers, please provide a tuple or list type...")
                    )
            if len(headers) != len(data[0]):
                raise DimensionError(
                    ErrorFormat(f"DimensionError: invalid data dimensions, provided headers length \"{len(headers)} != {len(data[0])}\", the implied column count for data provided...")
                    )                
            if isinstance(headers,tuple):
                try:
                    headers = list(headers)
                except:
                    raise TypeError(
                        ErrorFormat(f"TypeError: failed header conversion, unable to convert provided headers tuple to list type, please provide headers as a list type...")
                        ) from None
            if not all(isinstance(x, str) for x in headers):
                try:
                    headers = [str(x) for x in headers]
                except:
                    raise TypeError(
                        ErrorFormat(f"TypeError: invalid header values, all headers provided must be of type string...")
                        ) from None
        else:
            headers = [f"col_{x}" for x in range(len(data[0]))]
        self.sql_idx = "sdmidx"
        self.sql_model = "sdmmodel"
        self.max_rows = max_rows
        self.min_column_width = min_column_width
        self.max_column_width = max_column_width
        self.column_alignment = column_alignment
        self.display_color = display_color
        self.display_index = display_index
        self.row_count = len(data)
        had_idx = True if headers[0] == self.sql_idx else False
        self.max_idx = max([row[0] for row in data]) if had_idx else self.row_count
        self.min_idx = min([row[0] for row in data]) if had_idx else 1
        dyn_idx_offset,dyn_idx_bind,dyn_add_idx_insert = (1, "?,", f"\"{self.sql_idx}\",") if had_idx else (0, "", "")
        headers = headers[dyn_idx_offset:]
        self.headers = headers
        self.column_count = len(self.headers)
        self.static_py_to_sql_map_dict = {'None': 'NULL','int': 'INTEGER','float': 'REAL','str': 'TEXT','bytes': 'BLOB', 'TIMESTAMP': 'datetime', 'NoneType':'NULL', 'bool':'INTEGER'}
        self.static_sql_to_py_map_dict = {'NULL': 'None','INTEGER': 'int','REAL': 'float','TEXT': 'str','BLOB': 'bytes', 'datetime': 'TIMESTAMP'}
        self.headers_to_py_dtypes_dict = {self.headers[i]:type(data[0][i+dyn_idx_offset]).__name__ if type(data[0][i+dyn_idx_offset]).__name__ != 'NoneType' else 'str' for i in range(self.column_count)}
        self.headers_to_sql_dtypes_dict = {k:self.static_py_to_sql_map_dict[v] for (k,v) in self.headers_to_py_dtypes_dict.items()}
        self.headers_with_sql_dtypes_str = ", ".join(f"""\"{col}\" {type}""" for col,type in self.headers_to_sql_dtypes_dict.items())
        sql_create_stmt = f"""create table if not exists \"{self.sql_model}\" (\"{self.sql_idx}\" INTEGER PRIMARY KEY,{self.headers_with_sql_dtypes_str})"""
        sql_insert_stmt = f"""insert into "{self.sql_model}" ({dyn_add_idx_insert}{','.join([f'"{col}"' for col in self.headers])}) values ({dyn_idx_bind}{','.join(['?' for _ in self.headers])})"""
        self.sql_db_conn = sqlite3.connect(":memory:",uri=True)
        self.sql_c = self.sql_db_conn.cursor()
        self.sql_c.execute(sql_create_stmt)
        self.sql_db_conn.commit()
        self._set_updated_sql_metadata()
        self.sql_fetch_all_no_idx = self._generate_sql_stmt(include_index=False)
        self.sql_fetch_all_with_idx = self._generate_sql_stmt(include_index=True)
        if kwargs:
            self.__dict__.update(kwargs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sql_c.executemany(sql_insert_stmt,data)
            self.sql_db_conn.commit()
        except sqlite3.ProgrammingError as e:
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: invalid or inconsistent data, failed with "{e}"...')
                ) from None

    def get_header_at_index(self, index:int) -> str:
        """returns the header at specified index"""
        if (index < 0) or (index >= self.column_count):
            print(f"{self.clswarning}: provided index of {index} is outside of current column range 0:{self.column_count-1}, no header to return...")
            return
        return self.headers[index]
    
    def set_header_at_index(self, index:int, new_value:str) -> None:
        if (index < 0) or (index >= self.column_count):
            print(f"{self.clswarning}: provided index of {index} is outside of current column range 0:{self.column_count-1}, headers unchanged...")
            return
        rename_stmts = f"""alter table "{self.sql_model}" rename column "{self.headers[index]}" to "{new_value}" """
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} Successfully renamed column at index {index}')

    def get_headers(self) -> list[str]:
        """returns the current `SQLDataModel` headers"""
        return self.headers
    
    def set_headers(self, new_headers:list[str]) -> None:
        """renames the current `SQLDataModel` headers to values provided in `new_headers`, must have the same dimensions and existing headers"""
        if not isinstance(new_headers,list|tuple):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid header types, type \"{type(new_headers).__name__}\" is not a valid type for headers, please provide a tuple or list type...")
                )
        if len(new_headers) != self.column_count:
            raise DimensionError(
                ErrorFormat(f"DimensionError: invalid header dimensions, provided headers length \"{len(new_headers)} != {self.column_count}\" column count, please provide correct dimensions...")
                )
        if not isinstance(new_headers[0], str|int|float):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid header values, type \"{type(new_headers[0]).__name__}\" is not a valid type for header values, please provide a string type...")
                )
        rename_stmts = ";".join([f"""alter table "{self.sql_model}" rename column "{self.headers[i]}" to "{new_headers[i]}" """ for i in range(self.column_count)])
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to rename columns, SQL execution failed with: "{e}"')
            ) from None
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} Successfully renamed all model columns')

    def normalize_headers(self, apply_function:Callable=None) -> None:
        """reformats the current `SQLDataModel` headers into an uncased normalized form using alphanumeric characters only, wraps `.set_headers()`\n\nuse `apply_function` to specify alternative normalization pattern, when `None` the pattern `'[^0-9a-z _]+'` will be used on uncased values"""
        if apply_function is None:
            apply_function = lambda x: "_".join(x.strip() for x in re.sub('[^0-9a-z _]+', '', x.lower()).split('_') if x !='')
        new_headers = [apply_function(x) for x in self.get_headers()]
        self.set_headers(new_headers)
        return

    def _set_updated_sql_row_metadata(self):
        rowmeta = self.sql_db_conn.execute(f""" select min({self.sql_idx}), max({self.sql_idx}), count({self.sql_idx}) from {self.sql_model} """).fetchone()
        self.min_idx, self.max_idx, self.row_count = rowmeta

    def _set_updated_sql_metadata(self, return_data:bool=False) -> tuple[list, dict, dict]:
        """sets and optionally returns the header indicies, names and current sql data types from the sqlite pragma function\n\nreturn format (updated_headers, updated_header_dtypes, updated_metadata_dict)"""
        meta = self.sql_db_conn.execute(f""" select cid,name,type from pragma_table_info('{self.sql_model}')""").fetchall()
        self.headers = [h[1] for h in meta if h[0] > 0] # ignore idx column
        self.column_count = len(self.headers)
        self.header_dtype_dict = {d[1]: d[2] for d in meta}
        self.headers_to_py_dtypes_dict = {k:self.static_sql_to_py_map_dict[v] if v in self.static_sql_to_py_map_dict.keys() else "str" for (k,v) in self.header_dtype_dict.items() if k != self.sql_idx}
        self.headers_to_sql_dtypes_dict = {k:"TEXT" if v=='str' else "INTEGER" if v=='int' else "REAL" if v=='float' else "TIMESTAMP" if v=='datetime' else "NULL" if v=='NoneType' else "BLOB" for (k,v) in self.headers_to_py_dtypes_dict.items()}
        self.header_idx_dtype_dict = {(m[0]-1): (m[1], m[2]) for m in meta if m[1] != self.sql_idx}
        
        if return_data:
            return (self.headers,self.header_dtype_dict,self.header_idx_dtype_dict) # format of {header_idx: (header_name, header_dtype)}

    def get_max_rows(self) -> int:
        """returns the current `max_rows` property value"""
        return self.max_rows
    
    def set_max_rows(self, rows:int) -> None:
        """set `max_rows` to limit rows displayed when `repr` or `print` is called, does not change the maximum rows stored in `SQLDataModel`"""
        # if type(rows) != int:
        if not isinstance(rows, int):
            raise TypeError(
                ErrorFormat(f'TypeError: invalid argument type "{type(rows).__name__}", please provide an integer value to set the maximum rows attribute...')
                )
        if rows <= 0:
            raise IndexError(
                ErrorFormat(f'IndexError: invalid value "{rows}", please provide an integer value >= 1 to set the maximum rows attribute...')
                )
        self.max_rows = rows

    def get_min_column_width(self) -> int:
        """returns the current `min_column_width` property value"""
        return self.min_column_width
    
    def set_min_column_width(self, width:int) -> None:
        """set `min_column_width` as minimum number of characters per column when `repr` or `print` is called"""
        self.min_column_width = width

    def get_max_column_width(self) -> int:
        """returns the current `max_column_width` property value"""
        return self.max_column_width
    
    def set_max_column_width(self, width:int) -> None:
        """set `max_column_width` as maximum number of characters per column when `repr` or `print` is called"""
        self.max_column_width = width

    def get_column_alignment(self) -> str:
        """returns the current `column_alignment` property value, `None` by default"""
        return self.column_alignment
    
    def set_column_alignment(self, alignment:str|None) -> None:
        """set `column_alignment` as default alignment behavior when `repr` or `print` is called, options:
        \n`column_alignment = None` default behavior, dynamically aligns columns based on value types
        \n`column_alignment = '<'` left align all column values
        \n`column_alignment = '^'` center align all column values
        \n`column_alignment = '>'` right align all column values
        \ndefault behavior aligns strings left, integers & floats right, with headers matching value alignment
        """
        if alignment is None:
            self.column_alignment = alignment
            return
        if (not isinstance(alignment, str)) or (alignment not in ('<', '^', '>')):
            raise TypeError(
                ErrorFormat(f'TypeError: invalid argument "{alignment}", please provide a valid f-string alignment formatter or set `alignment=None` to use default behaviour...')
                )
        self.column_alignment = alignment
        return

    def get_display_index(self) -> bool:
        """returns the current boolean value for `is_display_index`, which determines whether or not the `SQLDataModel` index will be shown in print or repr calls"""
        return self.display_index

    def set_display_index(self, display_index:bool) -> None:
        """sets the `display_index` property to enable or disable the inclusion of the `SQLDataModel` index value in print or repr calls, default set to include"""
        if not isinstance(display_index, bool):
            raise TypeError(
                ErrorFormat(f'TypeError: invalid argument "{display_index}", please provide a valid boolean (True | False) value to the `display_index` argument...')
                )
        self.display_index = display_index
    
    def get_shape(self) -> tuple[int]:
        """returns the shape of the data as a tuple of `(rows x columns)`"""
        return (self.row_count,self.column_count)
    
#############################################################################################################
############################################### class methods ###############################################
#############################################################################################################

    @classmethod
    def from_csv(cls, csv_file:str, delimeter:str=',', quotechar:str='"', headers:list[str] = None, *args, **kwargs) -> SQLDataModel:
        """returns a new `SQLDataModel` from the provided csv file by wrapping the from_data method after grabbing the rows and headers by assuming first row represents column headers"""
        with open(csv_file) as csvfile:
            tmp_all_rows = tuple(list(row) for row in csv.reader(csvfile, delimiter=delimeter,quotechar=quotechar))
        return cls.from_data(tmp_all_rows[1:],tmp_all_rows[0] if headers is None else headers, *args, **kwargs)   
    
    @classmethod
    def from_data(cls, data:list[list], headers:list[str]=None, max_rows:int=1_000, min_column_width:int=6, max_column_width:int=32, *args, **kwargs) -> SQLDataModel:
        """convienence function to route all external class methods through single helper function to wrap defaults before instantiating main `SQLDataModel`"""
        return cls(data, headers, max_rows=max_rows, min_column_width=min_column_width, max_column_width=max_column_width, *args, **kwargs)
    
    @classmethod
    def from_dict(cls, data:dict, max_rows:int=1_000, min_column_width:int=6, max_column_width:int=32, *args, **kwargs) -> SQLDataModel:
        """returns a new `SQLDataModel` instance from the dict provided using the keys as row indexes if keys are of type int, otherwise using the keys as headers"""
        rowwise = True if all(isinstance(x, int) for x in data.keys()) else False
        if rowwise:
            # column_count = len(data[next(iter(data))])
            headers = ['sdmidx',*[f'col_{i}' for i in range(len(data[next(iter(data))]))]] # get column count from first key value pair in provided dict
            return cls([tuple([k,*v]) for k,v in data.items()], headers, max_rows=max_rows, min_column_width=min_column_width, max_column_width=max_column_width, *args, **kwargs)
        else:
            headers = [k for k in data.keys()]
            column_count = len(headers)
            row_count = len(data[next(iter(data))])
            data = [x for x in data.values()]
            data = [tuple([data[j][row] for j in range(column_count)]) for row in range(row_count)]
            return cls(data, headers, max_rows=max_rows, min_column_width=min_column_width, max_column_width=max_column_width, *args, **kwargs)
        
    @classmethod
    def from_numpy(cls, array, headers:list[str]=None, *args, **kwargs) -> SQLDataModel:
        """returns a `SQLDataModel` object created from the provided numpy `array`
        \nuse `headers` to use a specific header row, default set to assume no headers and treats data as n-dimensional array
        \nnote that `numpy` must be installed in order to use this class method"""
        if not _has_np:
            raise ModuleNotFoundError(
                ErrorFormat(f"""ModuleNotFoundError: required package not found, numpy must be installed in order to use the `from_numpy()` method""")
                )
        return cls.from_data(data=array.tolist(),headers=headers, *args, **kwargs)

    @classmethod
    def from_pandas(cls, df, headers:list[str]=None, *args, **kwargs) -> SQLDataModel:
        """returns a `SQLDataModel` object created from the provided pandas `DataFrame`, note that `pandas` must be installed in order to use this class method"""
        if not _has_pd:
            raise ModuleNotFoundError(
                ErrorFormat(f"""ModuleNotFoundError: required package not found, pandas must be installed in order to use the `from_pandas()` method""")
                )
        data = [x[1:] for x in df.itertuples()]
        headers = df.columns.tolist() if headers is None else headers
        return cls.from_data(data=data,headers=headers, *args, **kwargs)
    
    @classmethod
    def from_pickle(cls, filename:str=None, *args, **kwargs) -> SQLDataModel:
        """returns the `SQLDataModel` object from `filename` if provided, if `None` the current directory will be scanned for the default `to_pickle()` format"""
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        if (filename is not None) and (len(filename.split(".")) <= 1):
            print(f"{ANSIColor().alert(cls.__name__, 'W')} file extension missing, provided filename \"{filename}\" did not contain an extension and so \".sdm\" was appended to create a valid filename...")
            filename += '.sdm'
        if not Path(filename).is_file():
            raise FileNotFoundError(
                ErrorFormat(f"FileNotFoundError: file not found, provided filename \"{filename}\" could not be found, please ensure the filename exists in a valid path...")
                )
        with open(filename, 'rb') as f:
            tot_raw = pickle.load(f) # Tuple of Tuples raw data
            return cls.from_data(tot_raw[1:],headers=tot_raw[0], *args, **kwargs)

    @classmethod
    def get_supported_sql_connections(cls) -> tuple:
        """returns the currently tested DB API 2.0 dialects for use with `SQLDataModel.from_sql()` method"""
        return ('sqlite3', 'psycopg2', 'pyodbc', 'cx_oracle', 'teradatasql')
    
    @classmethod
    def from_sql(cls, sql_query: str, sql_connection: sqlite3.Connection, *args, **kwargs) -> SQLDataModel:
        """returns a `SQLDataModel` object created from executing the `sql_query` using the provided `sql_connection`
        \nif a single word is provided, the query will be wrapped and executed as a select all:
        \n\t`sdm_obj = SQLDataModel.from_sql("table_name", sqlite3.Connection)` # will be executed as:
        \n\t`sdm_obj = SQLDataModel.from_sql("select * from table_name", sqlite3.Connection)`
        \notherwise the full query passed to `sql_query` will be executed verbatim.
        \nnote that `sql_connection` can be any valid DB API 2.0 compliant connection.
        """
        ### type checking connection to ensure compatible ###
        db_dialect = type(sql_connection).__module__.split('.')[0].lower()
        if db_dialect not in cls.get_supported_sql_connections():
            print(WarnFormat(f"""{cls.__name__}Warning: provided SQL connection has not been tested, behavior for "{db_dialect}" may be unpredictable or unstable..."""))
        if len(sql_query.split()) == 1:
            sql_query = f""" select * from {sql_query} """
        try:
            sql_c = sql_connection.cursor()
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f"SQLProgrammingError: provided SQL connection is not opened or valid, failed with: {e}...")
            ) from None
        try:
            sql_c.execute(sql_query)
            data = sql_c.fetchall()
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f"SQLProgrammingError: provided SQL query is invalid or malformed, failed with: {e}...")
            ) from None
        if (len(data) < 1) or (data is None):
            raise DimensionError("DimensionError: provided SQL query returned no data, please provide a valid query with sufficient return data...")
        headers = [x[0] for x in sql_c.description]
        return cls.from_data(data, headers, *args, **kwargs)

################################################################################################################
############################################## conversion methods ##############################################
################################################################################################################

    def data(self, include_index:bool=False, include_headers:bool=False) -> list[tuple]:
        """returns the `SQLDataModel` data as a list of tuples for multiple rows, or as a single tuple for individual rows\n\ndata returned without index and headers by default, use `include_headers=True` or `include_index=True` to modify"""
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        data = self.sql_c.fetchall()
        if (len(data) == 1) and (not include_headers): # if only single row
            data = data[0]
        if len(data) == 1: # if only single column
            data = data[0]
        return [tuple([x[0] for x in self.sql_c.description]),data] if include_headers else data

    def to_csv(self, csv_file:str, delimeter:str=',', quotechar:str='"', include_index:bool=False, *args, **kwargs):
        """writes `SQLDataModel` to specified file in `csv_file` argument, must be compatible `.csv` file extension"""
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        write_headers = [x[0] for x in self.sql_c.description]
        with open(csv_file, 'w', newline='') as file:
            csvwriter = csv.writer(file,delimiter=delimeter,quotechar=quotechar,quoting=csv.QUOTE_MINIMAL, *args, **kwargs)
            csvwriter.writerow(write_headers)
            csvwriter.writerows(self.sql_c.fetchall())
        print(f'{self.clssuccess} csv file "{csv_file}" created')

    def to_dict(self, rowwise:bool=True) -> dict:
        """returns a dict from `SQLDataModel` using index rows as keys, set `rowwise=False` to use model headers as keys instead"""
        self.sql_c.execute(self._generate_sql_stmt(include_index=True))
        if rowwise:
            return {row[0]:row[1:] for row in self.sql_c.fetchall()}
        else:
            data = self.sql_c.fetchall()
            headers = [x[0] for x in self.sql_c.description]
            return {headers[i]:tuple([x[i] for x in data]) for i in range(len(headers))}
        
    def to_list(self, include_index:bool=True, include_headers:bool=False) -> list[tuple]:
        """returns a list of tuples containing all the `SQLDataModel` data without the headers by default
        \nuse `include_headers = True` to return the headers as the first item in returned sequence"""
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        return [tuple([x[0] for x in self.sql_c.description]),*self.sql_c.fetchall()] if include_headers else self.sql_c.fetchall()
    
    def to_numpy(self, include_index:bool=False, include_headers:bool=False):
        """converts `SQLDataModel` to numpy `array` object of shape (rows, columns), note that `numpy` must be installed to use this method"""
        if not _has_np:
            raise ModuleNotFoundError(
                ErrorFormat(f"""ModuleNotFoundError: required package not found, numpy must be installed in order to use `.to_numpy()` method""")
                )            
        fetch_stmt = self._generate_sql_stmt(include_index=include_index)
        self.sql_c.execute(fetch_stmt)
        if include_headers:
            return _np.vstack([_np.array([x[0] for x in self.sql_c.description]),[_np.array(x) for x in self.sql_c.fetchall()]])
        return _np.array([_np.array(x) for x in self.sql_c.fetchall()])

    def to_pandas(self, include_index:bool=False, include_headers:bool=True):
        """converts `SQLDataModel` to pandas `DataFrame` object, note that `pandas` must be installed to use this method"""
        if not _has_pd:
            raise ModuleNotFoundError(
                ErrorFormat(f"""ModuleNotFoundError: required package not found, pandas must be installed in order to use `.to_pandas()` method""")
                )
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        raw_data = self.sql_c.fetchall()
        data = [x[1:] for x in raw_data] if include_index else [x for x in raw_data]
        indicies = [x[0] for x in raw_data] if include_index else None
        columns = ([x[0] for x in self.sql_c.description[1:]] if include_index else [x[0] for x in self.sql_c.description]) if include_headers else None
        return _pd.DataFrame(data=data,columns=columns,index=indicies)

    def to_pickle(self, filename:str=None) -> None:
        """save the `SQLDataModel` instance to the specified `filename`
        \nby default the name of the invoking python file will be used"""
        if (filename is not None) and (len(filename.split(".")) <= 1):
            print(f"{self.clswarning} File extension missing, provided filename \"{filename}\" did not contain an extension and so \".sdm\" was appended to create a valid filename...")
            filename += '.sdm'
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        serialized_data = tuple(x for x in self.iter_rows(include_index=True,include_headers=True)) # no need to send sql_store_id aka index to pickle
        with open(filename, 'wb') as handle:
            pickle.dump(serialized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'{self.clssuccess} pickle file "{filename}" created')

    def to_sql(self, table:str, extern_conn:sqlite3.Connection, replace_existing:bool=True, include_index:bool=True) -> None:
        """inserts `SQLDataModel` into specified table using the sqlite database connection object provided in one of two modes:
        \n`replace_existing = True:` deletes the existing table and replaces it with the SQLDataModel's
        \n`replace_existing = False:` append to the existing table and executes a deduplication statement immediately after
        \nuse `SQLDataModel.get_supported_sql_connections()` to view supported connections, use `include_index=True` to retain model index in target table
        """
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index))
        model_data = [x for x in self.sql_c.fetchall()] # using new process
        model_headers = [x[0] for x in self.sql_c.description]
        try:
            extern_c = extern_conn.cursor()
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f"""SQLProgrammingError: provided SQL connection is not open, please reopen the database connection or resolve "{e}"...""")
            ) from None
        if replace_existing:
            extern_c.execute(f"""drop table if exists "{table}" """)
            extern_conn.commit()
        db_dialect = type(extern_conn).__module__.split('.')[0].lower()
        dyn_bind = '?' if db_dialect == 'sqlite3' else '%s'
        sql_dtypes_stmt = ", ".join(f"""\"{header}\" {self.header_dtype_dict[header]}""" for header in model_headers) # generates sql create table statement using type mapping dict
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
        """writes contents of `SQLDataModel` to specified `filename` as text repr, use `include_ts = True` to include timestamp"""
        contents = f"{datetime.datetime.now().strftime('%B %d %Y %H:%M:%S')} status:\n" + self.__repr__() if include_ts else self.__repr__()
        with open(filename, "w", encoding='utf-8') as file:
            file.write(contents)
        print(f'{self.clssuccess} text file "{filename}" created')

    def to_local_db(self, db:str=None):
        """stores the `SQLDataModel` internal in-memory database to local disk database, if `db=None`, the current filename will be used as a default target"""
        with sqlite3.connect(db) as target:
            self.sql_db_conn.backup(target)
        print(f'{self.clssuccess} local db "{db}" created')

####################################################################################################################
############################################## dunder special methods ##############################################
####################################################################################################################

    def __add__(self, value:int) -> SQLDataModel:
        """implements + operator functionality for compatible `SQLDataModel` operations"""

    def __getitem__(self, slc) -> SQLDataModel:
        validated_indicies = self.validate_indicies(slc)
        print(f'validated indicies: {validated_indicies}')
        if isinstance(slc, tuple):
            if len(slc) != 2:
                raise DimensionError(
                    ErrorFormat(f"DimensionError: \"{len(slc)}\" is not a valid number of arguments for row and column indexes...")
                    )
            row_idxs, col_idxs = slc[0], slc[1]
            if not isinstance(row_idxs, slice|int|tuple):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(row_idxs).__name__}\" is not a valid type for row indexing, please provide a slice type to index rows correctly...")
                    )
            if not isinstance(col_idxs, slice|int|tuple|str|list):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(col_idxs).__name__}\" is not a valid type for column indexing, please provide a slice, list or str type to index columns correctly...")
                    )
            if isinstance(row_idxs, int):
                if row_idxs < 0:
                    val_row = self.max_idx + (row_idxs + 1) # negative indexing
                else:
                    val_row = row_idxs
                if (val_row < self.min_idx) or (val_row > self.max_idx):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid row index \"{val_row}\", row index must be inside of current model range: \"{self.min_idx}:{self.max_idx}\"")
                    )
            if isinstance(row_idxs, slice):
                row_start = row_idxs.start if row_idxs.start is not None else self.min_idx
                row_stop = row_idxs.stop if row_idxs.stop is not None else self.max_idx
                val_row = slice(row_start,row_stop)
            if isinstance(col_idxs, int): # multiple discontiguous columns
                if col_idxs < 0:
                    val_columns = self.column_count + (col_idxs)
                else:
                    val_columns = col_idxs
                if (val_columns < 0) or (val_columns >= self.column_count):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column index \"{val_columns}\", columns index must be inside of current model range: \"0:{self.column_count-1}\"")
                    )                
            if isinstance(col_idxs, slice): # multiple column range
                col_start = col_idxs.start if col_idxs.start is not None else 0
                col_stop = col_idxs.stop if col_idxs.stop is not None else self.column_count
                col_start = 0 if col_start < 0 else col_start
                col_stop = self.column_count if col_stop > self.column_count else col_stop
                if (col_start < 0) or (col_stop > self.column_count):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column index \"{col_start}\", columns index must be inside of current model range: \"0:{self.column_count-1}\"")
                    )                  
                val_columns = slice(col_start,col_stop)
            if isinstance(col_idxs, str): # single column as string
                if col_idxs not in self.headers:
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column provided, \"{col_idxs}\" is not a valid header, use `.get_headers()` method to get current model headers...")
                        )
                val_columns = col_idxs
            if isinstance(col_idxs, list|tuple): # multiple assumed discontiguous columns as list of strings
                if isinstance(col_idxs[0],str):
                    for col in col_idxs:
                        if col not in self.headers:
                            raise ValueError(
                                ErrorFormat(f"ValueError: invalid column provided \"{col}\" not recognized, use `.get_headers()` method to get current model headers...")
                            )
                    val_columns = col_idxs
                elif isinstance(col_idxs[0],int):
                    try:
                        val_columns = [self.headers[i] for i in col_idxs]
                    except IndexError as e:
                        raise IndexError(
                            ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                            ) from None
            if isinstance(row_idxs, tuple):
                if not isinstance(row_idxs[0], int):
                    raise TypeError(
                        ErrorFormat(f"TypeError: invalid row index type \"{type(row_idxs[0]).__name__}\", rows must be indexed by type \"int\"")
                    )
                min_row_idx, max_row_idx = min(row_idxs), max(row_idxs)
                if min_row_idx < self.min_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{min_row_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                if max_row_idx > self.max_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{max_row_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                val_row = row_idxs
            return self._generate_sql_stmt(columns=val_columns,rows=val_row,execute_fetch=True)
        
        ### single column index ###
        if isinstance(slc, str|list):
            ### column indexes by str or list ###
            if isinstance(slc,str):
                slc = [slc]
            if isinstance(slc,list):
                if not isinstance(slc[0],str):
                    raise TypeError(
                        ErrorFormat(f"TypeError: invalid argument type \"{type(slc[0].__name__)}\" for column indexing, columns must be indexed by type \"str\" or \"slice\"")
                    )
            val_columns = slc
            for col in val_columns:
                if col not in self.headers:
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column provided \"{col}\", use `get_headers()` method to view current & valid model headers")
                    )
            return self._generate_sql_stmt(columns=val_columns, execute_fetch=True)                
        ### single row index ###
        if isinstance(slc, slice|int):
            ### row indexes slice ###
            if isinstance(slc, slice):
                start_idx = slc.start if slc.start is not None else self.min_idx
                stop_idx = slc.stop if slc.stop is not None else self.max_idx
                if start_idx < self.min_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{start_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                if stop_idx > self.max_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{stop_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )                
                val_row = slice(start_idx,stop_idx)
            ### single row idx by int ###
            if isinstance(slc, int):
                if slc < 0:
                    slc = self.max_idx + (slc + 1)
                if (slc < self.min_idx) or (slc > self.max_idx):
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{slc}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                val_row = slc
            return self._generate_sql_stmt(rows=val_row,execute_fetch=True)

        # if isinstance(slc, str) or isinstance(slc, list):
        #     if type(slc) == str:
        #         slc = [slc]
        #     try:
        #         col_idxs = tuple([self.headers.index(col) for col in slc])
        #     except ValueError as e:
        #         raise ValueError(
        #             ErrorFormat(f"ValueError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
        #             ) from None
        #     row_idxs = None
        #     return self._get_discontiguous_rows_and_cols(rows=row_idxs, cols=col_idxs, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)

        # ### row indexes slice ###
        # if isinstance(slc, slice):
        #     start_idx = slc.start if slc.start is not None else self.min_idx
        #     stop_idx = slc.stop if slc.stop is not None else self.max_idx
        #     return self.get_rows_and_cols_at_index_range(start_index=start_idx,stop_index=stop_idx, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)
        
        # ### single row index ###
        # if isinstance(slc, int):
        #     single_idx = slc
        #     return self.get_rows_and_cols_at_index_range(start_index=single_idx, stop_index=single_idx, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index) 

    def __setitem__(self, key_idxs, value) -> None:
        """retrieves the row indicies and column indicies and assigns the values to the corresponding model records using the `update_at()` method with value arg"""
        self.validate_indicies(key_idxs)
        if isinstance(value, SQLDataModel):
            sdm_other = value # rename it to reflect its another SQLDataModel object
            if isinstance(key_idxs, str):
                if key_idxs not in self.headers: # is new column
                    if sdm_other.headers[0] not in self.headers: # no need to copy over
                        if sdm_other.row_count == 1:
                            value = sdm_other.data()
                        elif sdm_other.row_count == self.row_count:
                            self.add_column_with_values(key_idxs,[x[0] for x in sdm_other.data()])
                            return
                        else:
                            raise DimensionError(
                                ErrorFormat(f'DimensionError: assignment target {key_idxs} != {value.row_count} rows of target, provided values must have the same shape for assignment...')
                                )                            
                        return
                    else: # just copying over data
                        self.add_column_with_values(key_idxs, sdm_other.headers[0])
                        return
            # if sdm_other.row_count == self.row_count:
            #     if (key_idxs not in self.headers) and (sdm_other.headers[0] in self.headers): # if new column provided and value is a current column: in sdm["key_idxs"]="current_column"
            #         self.add_column_with_values(key_idxs, sdm_other.headers[0]) # create new column 
            #         return
            #     else:
            #         print(f'PATH ENTERED!')
            #         self.add_column_with_values(key_idxs, sdm_other.data()) # create new column 
            #         return
            elif isinstance(key_idxs, int):
                if sdm_other.row_count == 1:
                    value = sdm_other.data()
            else:
                raise DimensionError(
                    ErrorFormat(f'DimensionError: assignment target {key_idxs} != {value.row_count} rows of target, provided values must have the same shape for assignment...')
                    )
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            value = (value,) # convert to tuple
        if isinstance(key_idxs, tuple):
            if len(key_idxs) != 2:
                raise DimensionError(
                    ErrorFormat(f"DimensionError: expected at most 2 arguments but \"{len(key_idxs)}\" were provided, please use at most 2 arguments for row and column indexing...")
                    )
            row_idxs, col_idxs = key_idxs[0], key_idxs[1]
            if not isinstance(row_idxs, slice|int|tuple):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(row_idxs)}\" is not a valid type for row indexing, please provide a slice type to index rows correctly...")
                    )
            if not isinstance(col_idxs, slice|int|tuple|str|list):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(col_idxs)}\" is not a valid type for column indexing, please provide a slice, list or str type to index columns correctly...")
                    )
                
            if isinstance(row_idxs, int):
                row_idxs = (row_idxs,) if row_idxs >= 0 else ((self.max_idx + row_idxs) + 1,) # allows for negative reverse indexing like -1
            if isinstance(row_idxs, slice):
                row_start = row_idxs.start if row_idxs.start is not None else self.min_idx
                row_stop = row_idxs.stop if row_idxs.stop is not None else (self.max_idx+1)
                row_idxs = tuple(range(row_start, row_stop))
            if isinstance(col_idxs, int): # multiple discontiguous columns
                col_idxs = (col_idxs,) if col_idxs <= self.column_count else (self.column_count,)
            if isinstance(col_idxs, slice): # multiple column range
                col_start = col_idxs.start if col_idxs.start is not None else 0
                col_stop = col_idxs.stop if col_idxs.stop is not None else self.column_count
                col_idxs = tuple(range(col_start, col_stop))
            if isinstance(col_idxs, str): # single column as string
                if col_idxs not in self.headers:
                    raise ValueError(
                        ErrorFormat(f"""ValueError: invalid columns provided, \"{col_idxs}\" of current model headers, use `.get_headers()` method to get model headers...""")
                    )
                col_target = self.headers.index(col_idxs)
                col_idxs = (col_target,)
            if isinstance(col_idxs, list): # multiple assumed discontiguous columns as list of strings
                try:
                    col_idxs = tuple([self.headers.index(col) for col in col_idxs])
                except ValueError as e:
                    raise ValueError(
                        ErrorFormat(f"""ValueError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...""")
                    ) from None
            try:
                columns = [self.headers[col_idx] for col_idx in col_idxs]
            except IndexError as e:
                raise IndexError(
                    ErrorFormat(f'IndexError: \"{col_idxs}\" is an invalid index outside of the current column index range of \"0:{self.column_count}\", use `.get_headers()` to view current columns...')
                    ) from None          
            if (len_val := len(value)) != (len_col := len(columns)):
                raise DimensionError(
                    ErrorFormat(f'DimensionError: "{len_val} != {len_col}", provided value dimension {len_val} must equal target assignment dimension {len_col} to update model with value...')
                    )
            self.update_at(row_idxs=row_idxs,columns=columns, value=value)
            return
        ### column indexes by str or list ###
        if isinstance(key_idxs, str) or isinstance(key_idxs, list):
            if type(key_idxs) == str:
                ### create new column if single string provided and is not a current column ###
                if key_idxs not in self.headers:
                    if len(value) == 1:
                        self.add_column_with_values(key_idxs, value=value[0])
                        return
                    elif len(value) == self.row_count:
                        self.add_column_with_values(key_idxs, value=value)
                        return
                else:
                    key_idxs = [key_idxs]
            for col in key_idxs:
                if col not in self.headers:
                    raise ValueError(
                    ErrorFormat(f"ValueError: \"{col}\" is not one of the current model headers, use `.get_headers()` method to get current headers...")
                    )
                col_idxs = tuple([self.headers.index(col) for col in key_idxs])
            row_idxs = tuple(range(self.min_idx, self.max_idx+1))
            columns = [self.headers[col_idx] for col_idx in col_idxs]
            if (len_val := len(value)) != (len_col := len(columns)):
                raise DimensionError(
                    ErrorFormat(f'DimensionError: "{len_val} != {len_col}", provided value dimension {len_val} must equal target assignment dimension {len_col} to update model with value...')
                    )
            self.update_at(row_idxs=row_idxs,columns=columns, value=value)
            return

        ### row indexes slice ###
        if isinstance(key_idxs, slice):
            start_idx = key_idxs.start if key_idxs.start is not None else self.min_idx
            stop_idx = key_idxs.stop if key_idxs.stop is not None else self.max_idx
            row_idxs = tuple(range(start_idx, stop_idx))
            columns = self.headers
            if (len_val := len(value)) != (len_col := len(columns)):
                raise DimensionError(
                    ErrorFormat(f'DimensionError: "{len_val} != {len_col}", provided value dimension {len_val} must equal target assignment dimension {len_col} to update model with value...')
                    )
            self.update_at(row_idxs=row_idxs,columns=columns, value=value)
        
        ### single row index ###
        if isinstance(key_idxs, int):
            if key_idxs == self.row_count+1: # if index is + 1 of current row count, add values as new row
                self.insert_row(value)
                return
            row_idxs = (key_idxs,) if key_idxs >= 0 else ((self.max_idx + key_idxs) + 1,)
            columns = self.headers
            if (len_val := len(value)) != (len_col := len(columns)):
                raise DimensionError(
                    ErrorFormat(f'DimensionError: "{len_val} != {len_col}", provided value dimension {len_val} must equal target assignment dimension {len_col} to update model with value...')
                    )
            self.update_at(row_idxs=row_idxs,columns=columns, value=value)

    def __len__(self):
        return self.row_count

    def __repr__(self):
        display_headers = self.headers
        include_idx = self.display_index
        fetch_stmt = self._generate_sql_stmt(include_index=include_idx)
        self.sql_c.execute(f"{fetch_stmt} limit {self.max_rows}")
        table_data = self.sql_c.fetchall()
        index_width = len(str(max(row[0] for row in table_data))) + 1 if include_idx else 2
        dyn_idx_include = 1 if include_idx else 0
        table_body = "" # big things can have small beginnings...
        table_newline = "\n"
        display_rows = self.row_count if (self.max_rows is None or self.max_rows > self.row_count) else self.max_rows
        right_border_width = 3
        max_rows_to_check = display_rows if display_rows < 15 else 15 # updated as exception to 15
        col_length_dict = {col:len(str(x)) for col,x in enumerate(display_headers)} # for first row populate all col lengths
        col_alignment_dict = {i:'<' if v == 'str' else '>' if v != 'float' else '<' for i,v in enumerate(self.headers_to_py_dtypes_dict.values())}
        for row in range(max_rows_to_check): # each row is indexed in row col length dict and each one will contain its own dict of col lengths
            current_row = {col:len(str(x)) for col,x in enumerate(table_data[row][dyn_idx_include:])} # start at one to enusre index is skipped and column lengths correctly counted
            for col_i in range(self.column_count):
                if current_row[col_i] > col_length_dict[col_i]:
                    col_length_dict.update({col_i:current_row[col_i]})
        for col,width in col_length_dict.items():
            if width < self.min_column_width:
                col_length_dict[col] = self.min_column_width
            elif width > self.max_column_width:
                col_length_dict[col] = self.max_column_width
        index_fmt = f'│{{:>{index_width}}} │ ' if include_idx else '│ '
        right_border_fmt = ' │'
        col_alignment = self.column_alignment # if None columns will be dynmaic aligned based on dtypes
        columns_fmt = " │ ".join([f"""{{:{col_alignment_dict[i] if col_alignment is None else col_alignment}{col_length}}}""" for i,col_length in col_length_dict.items()]) # col alignment var determines left or right align
        table_abstract_template = """{index}{columns}{right_border}""" # assumption is each column will handle its right side border only and the last one will be stripped
        fmt_dict = {'index':index_fmt,'columns':columns_fmt,'right_border':right_border_fmt}
        table_row_fmt = table_abstract_template.format(**fmt_dict)
        total_required_width = index_width + sum(col_length_dict.values()) + (self.column_count*3) + right_border_width # extra 2 for right border width
        try:
            total_available_width = os.get_terminal_size()[0]
        except OSError:
            total_available_width = 100
        table_truncation_required = False if total_required_width <= total_available_width else True
        max_cols = self.column_count + 1 # plus 1 for newly added index col in sqlite table
        max_width = total_required_width
        if table_truncation_required:
            ellipsis_suffix_width = 4
            max_cols, max_width = 0, (index_width + right_border_width + ellipsis_suffix_width) # max width starts with the tax of index and border already included, around 5-7 depending on index width
            for v in col_length_dict.values():
                if max_width < total_available_width:
                    max_width += (v+3)
                    max_cols += 1
            if max_width > total_available_width:
                max_cols -= 1
                max_width -= (col_length_dict[max_cols] +3)
            table_row_fmt = """ │ """.join(table_row_fmt.split(""" │ """)[:max_cols+dyn_idx_include]) + """ │""" # no longer required, maybe...? +1 required on max columns since index fmt is included after split leaving the format missing two slots right away if you simply decrease it by 1
        table_dynamic_newline = f' ...\n' if table_truncation_required else '\n'
        table_top_bar = table_row_fmt.replace(" │ ","─┬─").replace("│{","┌{").replace(" │","─┐") if include_idx else table_row_fmt.replace(" │ ","─┬─").replace("│ {","┌─{",1).replace(" │","─┐")
        table_top_bar = table_top_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) if include_idx else table_top_bar.format(*['─' * length for length in col_length_dict.values()])
        if include_idx:
            header_row = table_row_fmt.format("",*[str(x)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(x)) > col_length_dict[k] else str(x) for k,x in enumerate(display_headers)]) # for header the first arg will be empty as no index will be used, for the rest it will be the data col key
        else:
            header_row = table_row_fmt.format(*[str(x)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(x)) > col_length_dict[k] else str(x) for k,x in enumerate(display_headers)]) # for header the first arg will be empty as no index will be used, for the rest it will be the data col key
        header_sub_bar = table_row_fmt.replace(" │ ","─┼─").replace("│{","├{").replace(" │","─┤") if include_idx else table_row_fmt.replace(" │ ","─┼─").replace("│ {","├─{",1).replace(" │","─┤")
        header_sub_bar = header_sub_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) if include_idx else header_sub_bar.format(*['─' * length for length in col_length_dict.values()])
        table_body += table_top_bar + table_newline
        table_body += header_row + table_dynamic_newline
        table_body += header_sub_bar + table_newline
        for i,row in enumerate(table_data):
            if i < display_rows:
                if include_idx:
                    table_body += table_row_fmt.format(row[0],*[str(cell)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(cell)) > (col_length_dict[k]) else str(cell) for k,cell in enumerate(row[1:max_cols+1])]) # start at 1 to avoid index col
                else:
                    table_body += table_row_fmt.format(*[str(cell)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(cell)) > (col_length_dict[k]) else str(cell) for k,cell in enumerate(row)]) # start at 1 to avoid index col
                table_body +=  table_dynamic_newline
        table_bottom_bar = table_row_fmt.replace(" │ ","─┴─").replace("│{","└{").replace(" │","─┘") if include_idx else table_row_fmt.replace(" │ ","─┴─").replace("│ {","└─{",1).replace(" │","─┘")
        table_bottom_bar = table_bottom_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) if include_idx else table_bottom_bar.format(*['─' * length for length in col_length_dict.values()])
        table_signature = f"""\n[{display_rows} rows x {self.column_count} columns]"""
        # width_truncation_debug_details = f"""\t({max_width} of {total_available_width} available width used with {max_cols -1 if not table_truncation_required else max_cols} columns)""" # include additional debug info with: \ncol dytpes dictionary: {self.column_dtypes}\ncol alignment dict: {col_alignment_dict}"""
        # table_body += table_bottom_bar + table_signature + width_truncation_debug_details
        table_body += table_bottom_bar + table_signature # exception change to have less details
        return table_body if self.display_color is None else self.display_color.wrap(table_body)  

##################################################################################################################
############################################## sqldatamodel methods ##############################################
##################################################################################################################

    def iter_rows(self, min_row:int=None, max_row:int=None, include_index:bool=False, include_headers:bool=False) -> Generator:
        """returns a generator object of the rows in the model from `min_row` to `max_row`, usage:
        \n`for row in sm_obj.iter_rows(min_row=2, max_row=4):`
        \n\t`print(row)`
        \n`min_row` and `max_row` are both inclusive to values specified
        \n`include_headers=True` to include headers as first row
        \nto return rows as namedtuples, use `iter_tuples()` method
        """
        min_row, max_row = min_row if min_row is not None else self.min_idx, max_row if max_row is not None else self.max_idx
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_index, rows=slice(min_row,max_row)))
        if include_headers:
            yield tuple([x[0] for x in self.sql_c.description])
        yield from (x for x in self.sql_c.fetchall())
    
    def iter_tuples(self, include_idx_col:bool=False):
        """returns a generator object of the `SQLDataModel` as namedtuples using current headers as field names, note that headers must be valid python identifiers for this method, headers can be converted to valid identifiers using the `normalized_headers()` method"""
        try:
            Row = namedtuple('Row', [self.sql_idx] + self.headers if include_idx_col else self.headers)
        except ValueError as e:
            raise ValueError(
                ErrorFormat(f'ValueError: {e}, rename header or use `normalize_headers()` method to fix')
            ) from None
        self.sql_c.execute(self._generate_sql_stmt(include_index=include_idx_col))
        yield from (Row(*x) for x in self.sql_c.fetchall())

    def _generate_sql_fetch_for_joining_tables(self, base_headers:list[str], join_table:str, join_column:str, join_headers:list[str], join_type:str='left') -> str:
        base_headers_str = ",".join([f"""a.\"{v[0]}\" as \"{v[0]}\"""" for v in self.header_idx_dtype_dict.values() if v[0] in base_headers])
        join_headers_str = ",".join([f"""b.\"{col}\" as \"{col}\" """ for col in join_headers])
        join_type_str = "left join" if join_type == 'left' else 'left outer join'
        join_predicate_str = f"""from {self.sql_model} a {join_type_str} \"{join_table}\" b on a.\"{join_column}\" = b.\"{join_column}\" """
        sql_join_stmt = f"""select {base_headers_str}, {join_headers_str} {join_predicate_str}"""
        return sql_join_stmt

    # def get_rows_and_cols_at_index_range(self, start_index:int=None, stop_index:int=None, start_col:int=None, stop_col:int=None, *args, **kwargs) -> SQLDataModel:
    #     """returns `SQLDataModel` rows from specified `start_index` to specified `stop_index` ranging from `start_col` up to and including `stop_col`
    #     \nas a new `SQLDataModel`, if `stop_index` is not specified or the value exceeds `row_count`, then last row or largest valid index will be used"""
    #     row_idx_start = start_index if start_index is not None else self.min_idx
    #     row_idx_stop = stop_index if stop_index is not None else self.max_idx
    #     col_idx_start = start_col if start_col is not None else 0
    #     col_idx_stop = stop_col if stop_col is not None else self.column_count
    #     if (row_idx_start > self.max_idx) or (row_idx_stop < self.min_idx):
    #         raise IndexError(
    #             ErrorFormat(f'IndexError: row index out of range, row indicies ({row_idx_start}:{row_idx_stop}) out of range for current row indicies of ({self.min_idx}:{self.max_idx})')
    #             )
    #     if  (col_idx_start > self.column_count) or (col_idx_start > col_idx_stop) or (col_idx_stop > self.column_count):
    #         raise IndexError(
    #             ErrorFormat(f'IndexError: column index out of range, column indicies ({col_idx_start}:{col_idx_stop}) out of range for current column indicies of (0:{self.column_count})')
    #             )
    #     else:
    #         idx_args = tuple(range(col_idx_start,col_idx_stop)) if col_idx_start != col_idx_stop else (col_idx_start,)
    #         fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(idx_args,include_idx_col=True)
    #     self.sql_c.execute(f"""{fetch_stmt} where {self.sql_idx} >= {row_idx_start} and {self.sql_idx} <= {row_idx_stop} order by {self.sql_idx} asc""")
    #     return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description],*args,**kwargs)    

    def head(self, n_rows:int=5) -> SQLDataModel:
        """returns the first `n_rows` of the current `SQLDataModel` or less if `n_rows` exceeds current `rowcount` property, by default the first 5 rows are returned"""
        return self._generate_sql_stmt(fetch_limit=n_rows, execute_fetch=True)
    
    def tail(self, n_rows:int=5) -> SQLDataModel:
        """returns the last `n_rows` of the current `SQLDataModel` or less if `n_rows` exceeds current `rowcount` property, by default the last 5 rows are returned"""
        rows = slice((self.max_idx-n_rows+1), self.max_idx)
        return self._generate_sql_stmt(rows=rows, execute_fetch=True)

    def _get_sql_create_stmt(self) -> str:
        """returns the current sql create statement stored in the sqlite_master table of the model database"""
        self.sql_c.execute("select sql from sqlite_master")
        return self.sql_c.fetchone()[0]
    
    # def _get_discontiguous_rows_and_cols(self, rows:tuple=None, cols:tuple=None, *args,**kwargs) -> SQLDataModel:
    #     if rows is not None:
    #         if type(rows) == int:
    #             dyn_rows_stmt = f"where {self.sql_idx} = {rows}"
    #         if type(rows) == slice:
    #             dyn_rows_stmt = f"where {self.sql_idx} >= {rows.start if rows.start is not None else self.min_idx} and {self.sql_idx} <= {rows.stop if rows.stop is not None else self.max_idx}"
    #         if type(rows) == tuple:
    #             dyn_rows_stmt = f"where {self.sql_idx} in {rows}"
    #     else:
    #         dyn_rows_stmt = ""
    #     if cols is not None:
    #         if type(cols) == slice:
    #             cols = tuple(range(cols.start if cols.start is not None else 0, cols.stop if cols.stop is not None else self.column_count))
    #         if type(cols) == int:
    #             cols = (cols,)
    #     fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(idxs=cols, include_idx_col=True) + dyn_rows_stmt # none cols handled by this function
    #     self.sql_c.execute(fetch_stmt)
    #     return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description],*args,**kwargs)    

    def set_display_color(self, color:str|tuple):
        """sets the table string representation color when `SQLDataModel` is displayed in the terminal, use hex value or a tuple of rgb values:
        \n\t`color='#A6D7E8'` # string with hex value\n\n\t`color=(166, 215, 232)` # tuple of rgb values\n\nNote: by default no color styling is applied"""
        try:
            pen = ANSIColor(color)
            self.display_color = pen
            print(f"""{self.clssuccess} Display color changed, the terminal display color has been changed to {pen.wrap(f"color {pen.text_color_str}")}""")
        except:
            print(WarnFormat(f"{type(self).__name__}Warning: invalid color, the terminal display color could not be changed, please provide a valid hex value or rgb color code..."))

##############################################################################################################
################################################ sql commands ################################################
##############################################################################################################

    ### needs more testing, still unstable ###
    def apply(self, func:Callable) -> SQLDataModel:
        """applies `func` to the current `SQLDataModel` object and returns a modified `SQLDataModel` by passing its current values to the argument of `func`\n\nnote that the number of `args` in the inspected signature of `func` must equal the current number of `SQLDataModel` columns as all are passed as input to `func`.\n\nThe number of `func` args must match the current number of columns in the model or an `Exception` will be raised\n\nuse `generate_apply_function_stub()` method to return a preconfigured template using current `SQLDataModel` columns and dtypes to assist"""
        ### get column name from str or index ###
        if not isinstance(func, Callable):
            raise TypeError(
                ErrorFormat(f'TypeError: invalid argument for `func`, expected type "Callable" but type "{type(func).__name__}" was provided, please provide a valid python "Callable"...')
            )
        try:
            func_name = func.__name__
            func_argcount = func.__code__.co_argcount
            self.sql_db_conn.create_function(func_name, func_argcount, func)
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to create function with provided callable "{func}", SQL process failed with: {e}')
            ) from None
        input_columns = ",".join([f"\"{col}\"" for col in self.headers])
        derived_query = f"""select {func_name}({input_columns}) as "{func_name}" from "{self.sql_model}" """
        return self.fetch_query(derived_query)

    def get_model_name(self) -> str:
        """returns the `SQLDataModel` table name currently being used by the model as an alias for any SQL queries executed by the user and internally"""
        return self.sql_model
    
    def set_model_name(self, new_name:str) -> None:
        """sets the new `SQLDataModel` table name that will be used as an alias for any SQL queries executed by the user or internally\n\nnote: value must be a valid SQL table name, this alias will be reset to the default value for any new `SQLDataModel` instances: 'sdmmodel'"""
        full_stmt = f"""begin transaction; alter table "{self.sql_model}" rename to {new_name}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to rename model table, SQL execution failed with: "{e}"')
            ) from None
        self.sql_model = new_name
        print(f'{self.clssuccess} Successfully renamed primary model alias to "{new_name}"')

    def deduplicate(self, subset:list[str]=None, keep_first:bool=True, print_results:bool=True) -> None:
        """deduplicates the current `SQLDataModel` using all the columns and keeping the first unique row by default\n\nset `subset=['column_1', 'column_2'... 'column_n']` to provide alternate deduplicate targets\n\nuse `keep_first=False` to keep the last unique record instead of the first"""
        dyn_keep_order = 'min' if keep_first else 'max'
        subset_columns = self.headers if subset is None else subset
        if type(subset_columns) == str:
            subset_columns = [subset_columns]
        sql_dedupe_stmt = f"""delete from "{self.sql_model}" where rowid not in (select {dyn_keep_order}(rowid) from "{self.sql_model}" group by {','.join(f'"{col}"' for col in subset_columns)})"""
        self.sql_c.execute(sql_dedupe_stmt)
        rows_deleted = self.sql_c.rowcount
        self.sql_db_conn.commit()
        new_row_count = self.row_count - rows_deleted
        self.row_count = new_row_count
        if print_results:
            print(f'{self.clssuccess} Deduplicated model, {rows_deleted} duplicate rows have been removed with {new_row_count} rows remaining...')
        return

    def fetch_query(self, sql_query:str, **kwargs) -> SQLDataModel:
        """returns a new `SQLDataModel` object after executing provided `sql_query` arg using the current `SQLDataModel`\n\nuse default table name 'sdmmodel' or `SQLDataModel.get_model_name()` to get current model alias\n\nimportant: this function is the primary method used by `SQLDataModel` methods that are expected to return a new instance"""
        try:
            self.sql_c.execute(sql_query)
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, provided query failed with error "{e}"...')
            ) from None
        return type(self)(self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, **kwargs)

    def group_by(self, *columns:str, order_by_count:bool=True, **kwargs) -> SQLDataModel:
        """
        #### Overview:
        returns a new `SQLDataModel` after performing group by on single or multiple columns specified
        #### Usage:
        ```python
        sdm = SQLDataModel.from_csv('data.csv') # create model from data
        sdm.group_by("country") # by single str
        sdm.group_by("country","state","city") # by multiple str
        sdm.group_by(["country","state","city"]) # by multiple list
        ```
        ---
        #### Notes:
        use `order_by_count = False` to change ordering from count to column args
        """
        if type(columns[0]) == str:
            columns = columns
        elif type(columns[0]) in (list, tuple):
            columns = columns[0]
        else:
            raise TypeError(
                ErrorFormat(f'TypeError: invalid columns argument, provided type {type(columns[0]).__name__} is invalid, please provide str, list or tuple type...')
                )
        for col in columns:
            if col not in self.headers:
                raise ValueError(
                    ErrorFormat(f'ValueError: invalid group by targets, provided column \"{col}\" does not exist in current model, valid targets:\n{self.headers}')
                    )
        columns_group_by = ",".join(f'"{col}"' for col in columns)
        order_by = "count(*)" if order_by_count else columns_group_by
        group_by_stmt = f"""select {columns_group_by}, count(*) as count from "{self.sql_model}" group by {columns_group_by} order by {order_by} desc"""
        try:
            self.sql_c.execute(group_by_stmt)
        except Exception as e:
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, provided query failed with error "{e}"...')
            ) from None
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, **kwargs)
    
    def join_model(self, sqldatamodel:SQLDataModel, left:bool=True, on_column:str=None, *args, **kwargs) -> SQLDataModel:
        """performs a left join using the caller `SQLDataModel` as the base table and a single shared column with the `sqldatamodel` instance provided in the argument\n\nuse `left=False` to perform a right join, and set `on_column='target_column'` to specify a unique column if no single shared one exists"""
        validated_join_col = False
        join_tablename = 'f_table'
        join_cols = sqldatamodel.headers
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
                ErrorFormat(f"DimensionError: no shared column, no matching join column was found in the provided model, ensure one is available or specify one explicitly with on_column='shared_column'")
                )
        sqldatamodel.to_sql(join_tablename, self.sql_db_conn)
        join_cols = [x for x in join_cols if x != on_column] # removing shared join column
        sql_join_stmt = self._generate_sql_fetch_for_joining_tables(self.headers, join_tablename, join_column=on_column, join_headers=join_cols, join_type='left' if left else 'right')
        self.sql_c.execute(sql_join_stmt)
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, **kwargs)

    def execute_query(self, sql_query:str) -> None:
        """executes arbitrary `sql_query` against the current model without the expectation of selection or returned rows"""
        try:
            self.sql_c.execute(sql_query)
            self.sql_db_conn.commit()
            print(f'{self.clssuccess} Executed SQL, provided query executed with {self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0} rows modified')
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, unable to execute provided SQL query with error "{e}"...')
            ) from None
    
    def execute_transaction(self, sql_script:str) -> None:
        """executes prepared `sql_script` script wrapped in a transaction against the current model without the expectation of selection or returned rows"""
        full_stmt = f"""begin transaction; {sql_script}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
            rows_modified = self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to execute provided transaction, SQL execution failed with: "{e}"')
            ) from None
        self._set_updated_sql_metadata()        
        print(f'{self.clssuccess} Executed SQL, provided query executed with {rows_modified} rows modified')  

    # def add_column(self, column_name:str, value=None) -> None:
    #     """adds `column_name` argument as a new column to `SQLDataModel`, populating with the `value` provided or with SQLs null field if `value=None` by default\n\nnote that if a current column name is passed to `value`, that columns values will be used to fill the new column, effectively copying it"""
    #     create_col_stmt = f"""alter table {self.sql_model} add column \"{column_name}\""""
    #     if (value is not None) and (value in self.headers):
    #         dyn_default_value = f"""{self.headers_to_sql_dtypes_dict[value]}"""
    #         dyn_copy_existing = f"""update {self.sql_model} set \"{column_name}\" = \"{value}\";"""
    #     else:
    #         if isinstance(value, str):
    #             value = f"'{value}'"
    #         dyn_default_value = f"""{self.static_py_to_sql_map_dict[type(value).__name__]} not null default {value}""" if value is not None else "TEXT"
    #         dyn_copy_existing = ""
    #     full_stmt = f"""begin transaction; {create_col_stmt} {dyn_default_value};{dyn_copy_existing} end transaction;"""
    #     print(full_stmt)
    #     try:
    #         self.sql_c.executescript(full_stmt)
    #         self.sql_db_conn.commit()
    #     except Exception as e:
    #         self.sql_db_conn.rollback()
    #         trace_back = sys.exc_info()[2]
    #         line = trace_back.tb_lineno
    #         print(ErrorFormat(f'SQLProgrammingError: unable to apply function, SQL execution failed with: {e} (line {line})'))
    #         sys.exit()
    #     self._set_updated_sql_metadata()
    #     print(f'{self.clssuccess} added new column "{column_name}" to model')

    def add_column_with_values(self, column_name:str, value=None) -> None:
        """adds `column_name` argument as a new column to `SQLDataModel`, populating with the `value` provided or with SQLs null field if `value=None` by default\n\nnote that if a current column name is passed to `value`, that columns values will be used to fill the new column, effectively copying it"""
        create_col_stmt = f"""alter table {self.sql_model} add column \"{column_name}\""""
        if (value is not None) and (value in self.headers):
            dyn_dtype_default_value = f"""{self.headers_to_sql_dtypes_dict[value]}"""
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
                    ErrorFormat(f'DimensionError: invalid dimensions "{len_values} != {self.row_count}", provided values have length {len_values} while current row count is {self.row_count}...')
                )
            try:
                seq_dtype = self.static_py_to_sql_map_dict[type(value[0]).__name__]
            except:
                raise TypeError(
                    ErrorFormat(f'TypeError: invalid datatype "{type(value[0]).__name__}", please provide a valid and SQL translatable datatype...')
                ) from None
            sql_script = f"""{create_col_stmt} {seq_dtype};"""
            all_model_idxs = tuple(range(self.min_idx, self.max_idx+1)) # plus 1 otherwise range excludes last item and out of range error is triggered during for loop below
            for i,val in enumerate(value):
                sql_script += f"""update {self.sql_model} set \"{column_name}\" = \"{val}\" where {self.sql_idx} = {all_model_idxs[i]};"""
            print(sql_script)
            self.execute_transaction(sql_script)
            return

    def apply_function_to_column(self, func:Callable, column:str|int) -> None:
        """applies `func` to provided `SQLDataModel` column by passing its current value to the argument of `func`, updating the columns values `func` output\n\nnote that if the number of `args` in the inspected signature of `func`, is more than 1, all of `SQLDataModel` current columns will be provided as input and consequently, the number of `func` args must match the current number of columns in the model or an `Exception` will be raised\n\nuse `generate_apply_function_stub()` method to return a preconfigured template using current `SQLDataModel` columns and dtypes to assist"""
        ### get column name from str or index ###
        if (not isinstance(column, int)) and (not isinstance(column, str)):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid column argument, \"{type(column).__name__}\" is not a valid target, provide column index or column name as a string...")
            )
        if isinstance(column, int):
            try:
                column = self.headers[column]
            except IndexError as e:
                raise IndexError(
                    ErrorFormat(f"IndexError: invalid column index provided, {column} is not a valid column index, use `.column_count` property to get valid range...")
                ) from None
        if isinstance(column, str):
            if column not in self.headers:
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid column provided, {column} is not valid for current model, use `.get_headers()` method to get model headers...")
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
                ErrorFormat(f'SQLProgrammingError: unable to create function with provided callable "{func}", SQL process failed with: {e}')
            ) from None
        if func_argcount == 1:
            input_columns = target_column
        elif func_argcount == self.column_count:
            input_columns = ",".join([f"\"{col}\"" for col in self.headers])
        else:
            raise ValueError(
                ErrorFormat(f'ValueError: invalid function arg count: {func_argcount}, input args to "{func_name}" must be 1 or {self.column_count} based on the current models structure, ie...\n{self.generate_apply_function_stub()}')
                )
        sql_apply_update_stmt = f"""update {self.sql_model} set {target_column} = {func_name}({input_columns})"""
        full_stmt = f"""begin transaction; {sql_apply_update_stmt}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to apply function, SQL execution failed with: {e}')
            ) from None
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} applied function "{func_name}()" to current model')        

    def generate_apply_function_stub(self) -> str:
        """generates a function template using the current `SQLDataModel` to format function arguments to pass to `apply_function_to_column()` method"""
        func_signature = ", ".join([f"""{k.replace(" ","_")}:{v}""" for k,v in self.headers_to_py_dtypes_dict.items() if k != self.sql_idx])
        return f"""def func({func_signature}):\n    # apply logic and return value\n    return"""
    
    def insert_row(self, values:list|tuple=None) -> None:
        """inserts a row in the `SQLDataModel` at index `self.rowcount+1` with provided `values`, if `values=None`, an empty row with SQL `null` values will be used"""
        if values is not None:
            if not isinstance(values,list|tuple):
                raise TypeError(
                    ErrorFormat(f'TypeError: invalid type provided \"{type(values).__name__}\", insert values must be of type list or tuple...')
                    )
            if isinstance(values,list):
                values = tuple(values)
            if (len_val := len(values)) != self.column_count:
                raise DimensionError(
                    ErrorFormat(f'DimensionError: invalid dimensions \"{len_val} != {self.column_count}\", the number of values provided ({len_val}) must match the current column count ({self.column_count})...')
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
                ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: {e}')
            ) from None
        self._set_updated_sql_row_metadata()
        self._set_updated_sql_metadata()        

    def update_at(self, row_idxs:tuple[int], columns:list[str], value:list=None) -> None:
        """updates `SQLDataModel` at `row_idxs`, `columns` with `value` provided"""
        if row_idxs is None:
            row_idxs = tuple(range(self.min_idx, self.max_idx))
        if columns is None:
            columns = self.headers
        col_param_str = ",".join([f""" \"{col}\"=?""" for col in columns])
        row_idxs = row_idxs if len(row_idxs) != 1 else f"({row_idxs[0]})"
        update_stmt = f"""update \"{self.sql_model}\" set {col_param_str} where {self.sql_idx} in {row_idxs}"""
        try:
            self.sql_c.execute(update_stmt, value)
            self.sql_db_conn.commit()
            rows_modified = self.sql_c.rowcount if self.sql_c.rowcount >= 0 else 0
        except Exception as e:
            self.sql_db_conn.rollback()
            raise SQLProgrammingError(
                ErrorFormat(f'SQLProgrammingError: unable to update values, SQL execution failed with: {e}')
            ) from None
        self._set_updated_sql_metadata()        
        print(f'{self.clssuccess} Executed SQL, provided query executed with {rows_modified} rows modified')        
        
    def _generate_sql_stmt(self, columns:None|slice|tuple|int|str|list[str]=None, rows:None|slice|tuple|int=None, include_index:bool=True, fetch_limit:int=None, execute_fetch:bool=False, **kwargs) -> str|SQLDataModel:
        """
        #### Usage:
        ```python
        sdm = self._generate_sql_stmt(columns=['col_1'], rows=(1,4,9), execute_fetch=True) # executes and returns SQLDataModel
        sql_stmt = self._generate_sql_stmt(rows=2) # generates and returns fetch stmt for row at index 2 only
        columns = ['col_1', 'col_2'] # ordering relevant
        rows = slice(1,10) # slice for range of rows
        rows = (1,3,5,9) # tuple for discontiguous rows
        rows = 2 # int for single row
        columns, rows = None, None # all columns and rows will be used
        ```
        ---
        #### Note: 
        - use `execute_fetch=False` to generate sql statement string only
        - use `execute_fetch=True` to execute the generated sql statement and return its result
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
            max_row = rows.stop if rows.stop is not None else self.max_idx
            rows_str = f"where {self.sql_idx} >= {min_row} and {self.sql_idx} <= {max_row}"
        elif isinstance(rows, tuple):
            rows_str = f"where {self.sql_idx} in {rows}" if len(rows) > 1 else f"where {self.sql_idx} in ({rows[0]})"
        elif isinstance(rows, int):
            if rows < 0:
                rows = self.max_idx + (rows + 1)
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
                    ErrorFormat(f'SQLProgrammingError: invalid or malformed SQL, unable to execute provided SQL query with error "{e}"...')
                ) from None
            return type(self)(self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, **kwargs)

    def validate_indicies(self, indicies, strict_validation:bool=True) -> tuple[int|slice, list[str]]:
        """helper function to validate indicies used in getitem and setitem dunder methods, use `strict_validation=True` to require all indexed items to exist, otherwise if `strict_validation=False`, new headers will be allowed"""
        ### single row index ###
        if isinstance(indicies, int):
            row_index = indicies
            if row_index < 0:
                row_index = self.max_idx + (row_index + 1)
            if row_index < self.min_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid row index '{row_index}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            if row_index > self.max_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid row index '{row_index}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            return (row_index, self.headers)
        ### single row slice index ###
        if isinstance(indicies, slice):
            row_slice = indicies
            start_idx = row_slice.start if row_slice.start is not None else self.min_idx
            stop_idx = row_slice.stop if row_slice.stop is not None else self.max_idx
            if start_idx < self.min_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if stop_idx > self.max_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{stop_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )                
            return (slice(start_idx,stop_idx), self.headers)        
        ### columns by str or list of str ###
        if isinstance(indicies, str|list):
            col_index = indicies
            if isinstance(indicies, str):
                col_index = [col_index]
            if not all(isinstance(col, str) for col in col_index):
                raise TypeError(
                    ErrorFormat(f"TypeError: invalid column index type '{type(col_index[0].__name__)}' received, use `.get_headers()` to view valid column arguments...")
                )
            for col in col_index:
                if (col not in self.headers) and (strict_validation):
                    raise ValueError(
                        ErrorFormat(f"ValueError: '{col}' is not one of the current model headers, use `.get_headers()` method to view current valid headers...")
                    )
            return (slice(self.min_idx,self.max_idx), col_index)
        ### indexing by rows and columns ###        
        if not isinstance(indicies, tuple):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid indexing type '{type(indicies).__name__}', indexing the model must be done using two-dimensional [rows,columns] parameters with 'int' or 'slice' types...")
            )
        if (arg_length := len(indicies)) != 2:
            raise ValueError(
                ErrorFormat(f"ValueError: invalid indexing args, expected no more than 2 indicies for [row, column] but '{arg_length}' were received")
            )
        row_indicies, col_indicies = indicies
        ### rows first ###
        if not isinstance(row_indicies, int|slice|tuple):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid type for row indexing '{type(row_indicies).__name__}', rows must be indexed by type 'int' or 'slice'...")
            )
        if isinstance(row_indicies, int):
            if row_indicies < 0:
                row_indicies = self.max_idx + (row_indicies + 1)
            if row_indicies < self.min_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            if row_indicies > self.max_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid row index '{row_indicies}' is outside of current model row indicies of '{self.min_idx}:{self.max_idx}'...")
                )
            validated_row_indicies = row_indicies
        elif isinstance(row_indicies, tuple): # tuple of disconnected row indicies
            if not all(isinstance(row, int) for row in row_indicies):
                raise TypeError(
                    ErrorFormat(f"TypeError: invalid row index type '{type(row_indicies[0]).__name__}', rows must be indexed by type 'int'")
                )
            min_row_idx, max_row_idx = min(row_indicies), max(row_indicies)
            if min_row_idx < self.min_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{min_row_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if max_row_idx > self.max_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{max_row_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            validated_row_indicies = row_indicies
        else: # is slice
            start_idx = row_indicies.start if row_indicies.start is not None else self.min_idx
            stop_idx = row_indicies.stop if row_indicies.stop is not None else self.max_idx
            if start_idx < self.min_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{start_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )
            if stop_idx > self.max_idx:
                raise ValueError(
                    ErrorFormat(f"ValueError: provided row index '{stop_idx}' outside of current model range '{self.min_idx}:{self.max_idx}'")
                )    
            validated_row_indicies = slice(start_idx, stop_idx)
        ### then columns ###
        if not isinstance(col_indicies, int|slice|tuple|str|list):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies).__name__}', for column indexing a slice, list or str type is required...")
                )        
        if isinstance(col_indicies, int):
            try:
                col_indicies = [self.headers[col_indicies]]
            except IndexError as e:
                raise IndexError(
                    ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                ) from None
        elif isinstance(col_indicies, slice):
            col_start = col_indicies.start if col_indicies.start is not None else 0
            col_stop = col_indicies.stop if col_indicies.stop is not None else self.column_count
            if (col_start < 0) or (col_stop > self.column_count):
                raise ValueError(
                    ErrorFormat(f"ValueError: invalid column index '{col_start}', columns index must be inside of current model range: '0:{self.column_count-1}'")
                )                  
            col_indicies = [self.headers[i] for i in tuple(range(col_start,col_stop))]
        elif isinstance(col_indicies, tuple):
            col_indicies = list(col_indicies)
        elif isinstance(col_indicies, str):
            col_indicies = [col_indicies]
        if not all(isinstance(col, int|str) for col in col_indicies):
            raise TypeError(
                ErrorFormat(f"TypeError: invalid column indexing type '{type(col_indicies[0].__name__)}', column indexing must be done by 'int' or 'str' types, use `.get_headers()` to view current valid arguments...")
            )
        if all(isinstance(col, int) for col in col_indicies):
            try:
                col_indicies = [self.headers[i] for i in col_indicies]
            except IndexError as e:
                raise IndexError(
                    ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                ) from None                
        ### columns validated to be a list of str, not neccessarily currnet columns, but thats ok for setitem method which should be allowed to create columns ###
        for col in col_indicies:
            if (col not in self.headers) and (strict_validation):
                raise ValueError(
                    ErrorFormat(f"ValueError: column not found '{col}', use `.get_headers()` to view current valid headers...")
                )
        validated_column_indicies = col_indicies
        return (validated_row_indicies, validated_column_indicies)


########################################## run locally ##########################################
if __name__ == '__main__':
    cols = ['string', 'integer', 'float', 'bit', 'datetime', 'bool', 'empty', 'null']
    data = create_placeholder_data(10,8)
    sdm = SQLDataModel(data,cols)
    sdm.set_display_color('#A6D7E8')
    print(sdm)
    rows_tuple = (1,3,4,9)
    row_slice = slice(0,5)
    from_master_query = sdm._generate_sql_stmt(columns=None,rows=rows_tuple,execute_fetch=True)
    ### single row index ###
    print(sdm[2])
    print(sdm[-1])
    print(sdm[:])
    print(sdm[2:])
    print(sdm[:7])
    ### single column index ###
    print(sdm['datetime'])
    print(sdm[:,['datetime','empty','float']])
    ### both combined ###
    print(sdm[1,['bit','integer']])
    print(sdm[:3,['bit','integer']])
    print(sdm[5:,['bit','integer']])
    print(sdm[:,3])
    print(sdm[:,-1])
    print(sdm[:,:])
    print(sdm[-1,-1])
    print(sdm[-1,(2,1)])
    print(sdm[(9,3,1,4),(2,1,6)])

"""
        if isinstance(slc, tuple):
            if len(slc) != 2:
                raise DimensionError(
                    ErrorFormat(f"DimensionError: \"{len(slc)}\" is not a valid number of arguments for row and column indexes...")
                    )
            row_idxs, col_idxs = slc[0], slc[1]
            if not isinstance(row_idxs, slice|int|tuple):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(row_idxs).__name__}\" is not a valid type for row indexing, please provide a slice type to index rows correctly...")
                    )
            if not isinstance(col_idxs, slice|int|tuple|str|list):
                raise TypeError(
                    ErrorFormat(f"TypeError: \"{type(col_idxs).__name__}\" is not a valid type for column indexing, please provide a slice, list or str type to index columns correctly...")
                    )
            if isinstance(row_idxs, int):
                if row_idxs < 0:
                    val_row = self.max_idx + (row_idxs + 1) # negative indexing
                else:
                    val_row = row_idxs
                if (val_row < self.min_idx) or (val_row > self.max_idx):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid row index \"{val_row}\", row index must be inside of current model range: \"{self.min_idx}:{self.max_idx}\"")
                    )
            if isinstance(row_idxs, slice):
                row_start = row_idxs.start if row_idxs.start is not None else self.min_idx
                row_stop = row_idxs.stop if row_idxs.stop is not None else self.max_idx
                val_row = slice(row_start,row_stop)
            if isinstance(col_idxs, int): # multiple discontiguous columns
                if col_idxs < 0:
                    val_columns = self.column_count + (col_idxs)
                else:
                    val_columns = col_idxs
                if (val_columns < 0) or (val_columns >= self.column_count):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column index \"{val_columns}\", columns index must be inside of current model range: \"0:{self.column_count-1}\"")
                    )                
            if isinstance(col_idxs, slice): # multiple column range
                col_start = col_idxs.start if col_idxs.start is not None else 0
                col_stop = col_idxs.stop if col_idxs.stop is not None else self.column_count
                col_start = 0 if col_start < 0 else col_start
                col_stop = self.column_count if col_stop > self.column_count else col_stop
                if (col_start < 0) or (col_stop > self.column_count):
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column index \"{col_start}\", columns index must be inside of current model range: \"0:{self.column_count-1}\"")
                    )                  
                val_columns = slice(col_start,col_stop)
            if isinstance(col_idxs, str): # single column as string
                if col_idxs not in self.headers:
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column provided, \"{col_idxs}\" is not a valid header, use `.get_headers()` method to get current model headers...")
                        )
                val_columns = col_idxs
            if isinstance(col_idxs, list|tuple): # multiple assumed discontiguous columns as list of strings
                if isinstance(col_idxs[0],str):
                    for col in col_idxs:
                        if col not in self.headers:
                            raise ValueError(
                                ErrorFormat(f"ValueError: invalid column provided \"{col}\" not recognized, use `.get_headers()` method to get current model headers...")
                            )
                    val_columns = col_idxs
                elif isinstance(col_idxs[0],int):
                    try:
                        val_columns = [self.headers[i] for i in col_idxs]
                    except IndexError as e:
                        raise IndexError(
                            ErrorFormat(f"IndexError: invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                            ) from None
            if isinstance(row_idxs, tuple):
                if not isinstance(row_idxs[0], int):
                    raise TypeError(
                        ErrorFormat(f"TypeError: invalid row index type \"{type(row_idxs[0]).__name__}\", rows must be indexed by type \"int\"")
                    )
                min_row_idx, max_row_idx = min(row_idxs), max(row_idxs)
                if min_row_idx < self.min_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{min_row_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                if max_row_idx > self.max_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{max_row_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                val_row = row_idxs
            return self._generate_sql_stmt(columns=val_columns,rows=val_row,execute_fetch=True)
        
        ### single column index ###
        if isinstance(slc, str|list):
            ### column indexes by str or list ###
            if isinstance(slc,str):
                slc = [slc]
            if isinstance(slc,list):
                if not isinstance(slc[0],str):
                    raise TypeError(
                        ErrorFormat(f"TypeError: invalid argument type \"{type(slc[0].__name__)}\" for column indexing, columns must be indexed by type \"str\" or \"slice\"")
                    )
            val_columns = slc
            for col in val_columns:
                if col not in self.headers:
                    raise ValueError(
                        ErrorFormat(f"ValueError: invalid column provided \"{col}\", use `get_headers()` method to view current & valid model headers")
                    )
            return self._generate_sql_stmt(columns=val_columns, execute_fetch=True)                
        ### single row index ###
        if isinstance(slc, slice|int):
            ### row indexes slice ###
            if isinstance(slc, slice):
                start_idx = slc.start if slc.start is not None else self.min_idx
                stop_idx = slc.stop if slc.stop is not None else self.max_idx
                if start_idx < self.min_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{start_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                if stop_idx > self.max_idx:
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{stop_idx}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )                
                val_row = slice(start_idx,stop_idx)
            ### single row idx by int ###
            if isinstance(slc, int):
                if slc < 0:
                    slc = self.max_idx + (slc + 1)
                if (slc < self.min_idx) or (slc > self.max_idx):
                    raise ValueError(
                        ErrorFormat(f"ValueError: provided row index \"{slc}\" outside of current model range \"{self.min_idx}:{self.max_idx}\"")
                    )
                val_row = slc
"""