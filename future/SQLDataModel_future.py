from __future__ import annotations
import sqlite3, os, csv, sys, datetime, pickle, warnings, re
from typing import Generator, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as _np
except:
    _has_np = False
else:
    _has_np = True

try:
    import pandas as _pd
except:
    _has_pd = False
else:
    _has_pd = True

def create_placeholder_data(n_rows:int, n_cols:int) -> list[list]:
    return [[f"value {i}" if i%2==0 else i**2 for i in range(n_cols-3)] + [3.1415, b'bit', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')] for _ in range(n_rows)]

@dataclass
class ANSIColor:
    """
    Creates an ANSI style terminal color using provided hex color or rgb values
    """
    def __init__(self, text_color:str|tuple=None, text_bold:bool=False):
        """creates a pen styling tool using ansi terminal colors, text_color and background_color must be in rgb or hex format, text_bold is off by default"""
        text_color = (95, 226, 197) if text_color is None else text_color # default teal color
        self.text_bold = "\033[1m" if text_bold else ""
        if type(text_color) == str: # assume hex
            fg_r, fg_g, fg_b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            self.text_color_str = text_color
            self.text_color_hex = text_color
            self.text_color_rgb = (fg_r, fg_g, fg_b)
        if type(text_color) == tuple: # assume rgb
            fg_r, fg_g, fg_b = text_color
            self.text_color_str = str(text_color)
            self.text_color_hex = f"#{fg_r:02x}{fg_g:02x}{fg_b:02x}"
            self.text_color_rgb = (fg_r, fg_g, fg_b)
        self._ansi_start = f"""{self.text_bold}\033[38;2;{fg_r};{fg_g};{fg_b}m"""
        self._ansi_stop = "\033[0m\033[39m\033[49m"

    def __repr__(self) -> str:
        return f"""{self._ansi_start}{type(self).__name__}({self.text_color_str}){self._ansi_stop}"""

    def to_rgb(self) -> tuple:
        """returns text color attribute as tuple in format of (r, g, b)"""
        return self.text_color_rgb
    
    def alert(self, alerter:str, alert_type:str, bold_alert:bool=False) -> str:
        """issues ANSI color alert on behalf of alerter using specified preset"""
        match alert_type:
            case 'S': # success
                return f"""{self.text_bold}\033[38;2;108;211;118m{alerter} Success:\033[0m\033[39m\033[49m""" # changed to 108;211;118
            case 'W': # warn
                return f"""{self.text_bold}\033[38;2;246;221;109m{alerter} Warning:\033[0m\033[39m\033[49m"""
            case 'E': # error
                return f"""{self.text_bold}\033[38;2;247;141;160m{alerter} Error:\033[0m\033[39m\033[49m"""
            case other:
                return None

    def wrap(self, text:str) -> str:
        """wraps the provided text in the style of the pen"""
        return f"""{self._ansi_start}{text}{self._ansi_stop}"""

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
        self.clserror = ANSIColor(text_bold=True).alert(self.clsname, 'E')
        self.clswarning = ANSIColor(text_bold=True).alert(self.clsname, 'W')
        self.clssuccess = ANSIColor(text_bold=True).alert(self.clsname, 'S')
        try:
            if type(data) not in (list,tuple):
                raise TypeError(f"{self.clserror} Type mismatch, {type(data)} is not a valid type for data, which must be of type list or tuple...")
            if len(data) < 1:
                raise ValueError(f"{self.clserror} Data not found, data with length of {len(data)} is insufficient to construct a valid model, additional rows of data required...")
            _ = data[0]
            if type(data[0]) not in (list,tuple):
                if type(data[0]).__module__ != 'pyodbc': # check for pyodbc.Row which is acceptable
                    raise TypeError(f"{self.clserror} Type mismatch, {type(data[0])} is not a valid type for data rows, which must be of type list or tuple...")
            if len(data[0]) < 1:
                raise ValueError(f"{self.clserror} Data rows not found, data rows with length of {len(data[0])} are insufficient to construct a valid model, at least one row is required...")
        except (TypeError, ValueError) as e:
            print(e)
            sys.exit()
        except (IndexError) as e:
            print(f"{self.clserror} Data index error, data index provided does not exist for length {len(data)}, {e}...")
            sys.exit()
        if headers is not None:
            try:
                if type(headers) not in (list, tuple):
                    raise TypeError(f"{self.clserror} Invalid header types, {type(headers)} is not a valid type for headers, please provide a tuple or list type...")
                if len(headers) != len(data[0]):
                    raise ValueError(f"{self.clserror} Invalid header dimensions, provided headers length {len(headers)} != {len(data[0])} column count, please provide correct dimensions...")                
                if type(headers) == tuple:
                    try:
                        headers = list(headers)
                    except:
                        raise TypeError(f"{self.clserror} Failed header conversion, unable to convert provided headers tuple to list type, please provide headers as a list type...")
                if not all(isinstance(x, str) for x in headers):
                    try:
                        headers = [str(x) for x in headers]
                    except:
                        raise TypeError(f"{self.clserror} Invalid header values, all headers provided must be of type string...")
            except (TypeError, ValueError) as e:
                print(e)
                sys.exit()
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
        self.static_py_to_sql_map_dict = {'None': 'NULL','int': 'INTEGER','float': 'REAL','str': 'TEXT','bytes': 'BLOB', 'TIMESTAMP': 'datetime', 'NoneType':'NULL'}
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
        self.sql_fetch_all_no_idx = self._generate_sql_fetch_stmt_for_idxs(include_idx_col=False)
        self.sql_fetch_all_with_idx = self._generate_sql_fetch_stmt_for_idxs(include_idx_col=True)
        if kwargs:
            self.__dict__.update(kwargs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sql_c.executemany(sql_insert_stmt,data)
            self.sql_db_conn.commit()
        except sqlite3.ProgrammingError:
            print(f'{self.clserror} Invalid or inconsistent data, the provided data dimensions or values are inconsistent or incompatible with sqlite3...')
            sys.exit()

    def get_header_at_index(self, index:int) -> str:
        """returns the header at specified index"""
        if (index < 0) or (index >= self.column_count):
            print(f"{self.clswarning}: Provided index of {index} is outside of current column range 0:{self.column_count-1}, no header to return...")
            return
        return self.headers[index]
    
    def set_header_at_index(self, index:int, new_value:str) -> None:
        if (index < 0) or (index >= self.column_count):
            print(f"{self.clswarning}: Provided index of {index} is outside of current column range 0:{self.column_count-1}, headers unchanged...")
            return
        rename_stmts = f"""alter table "{self.sql_model}" rename column "{self.headers[index]}" to "{new_value}" """
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            self.sql_db_conn.commit()
            print(f'{self.clserror} Unable to rename columns, SQL execution failed with: {e}')
            return
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} Successfully renamed column at index {index}')

    def get_headers(self) -> list[str]:
        """returns the current `SQLDataModel` headers"""
        return self.headers
    
    def set_headers(self, new_headers:list[str]) -> None:
        """renames the current `SQLDataModel` headers to values provided in `new_headers`, must have the same dimensions and existing headers"""
        try:
            if type(new_headers) not in (list, tuple):
                raise TypeError(f"{self.clswarning} Invalid header types, type {type(new_headers)} is not a valid type for headers, please provide a tuple or list type...")
            if len(new_headers) != self.column_count:
                raise ValueError(f"{self.clswarning} Invalid header dimensions, provided headers length {len(new_headers)} != {self.column_count} column count, please provide correct dimensions...")
            if type(new_headers[0]) not in (str, int, float):
                raise TypeError(f"{self.clswarning} Invalid header values, type {type(new_headers[0])} is not a valid type for header values, please provide a string type...")
        except (ValueError, TypeError) as e:
            print(e)
            return
        rename_stmts = ";".join([f"""alter table "{self.sql_model}" rename column "{self.headers[i]}" to "{new_headers[i]}" """ for i in range(self.column_count)])
        full_stmt = f"""begin transaction; {rename_stmts}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            self.sql_db_conn.commit()
            print(f'{self.clserror} Unable to rename columns, SQL execution failed with: {e}')
            return
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} Successfully renamed all model columns')

    def normalize_headers(self, apply_function:Callable=None) -> None:
        """reformats the current `SQLDataModel` headers into an uncased normalized form using alphanumeric characters only, wraps `.set_headers()`\n\nuse `apply_function` to specify alternative normalization pattern, when `None` the pattern `'[^0-9a-z _]+'` will be used on uncased values"""
        if apply_function is None:
            apply_function = lambda x: "_".join(x.strip() for x in re.sub('[^0-9a-z _]+', '', x.lower()).split('_') if x !='')
        new_headers = [apply_function(x) for x in self.get_headers()]
        self.set_headers(new_headers)
        return

    def _set_updated_sql_metadata(self, return_data:bool=False) -> tuple[list, dict, dict]:
        """sets and optionally returns the header indicies, names and current sql data types from the sqlite pragma function\n\nreturn format (updated_headers, updated_header_dtypes, updated_metadata_dict)"""
        meta = self.sql_db_conn.execute(f"select cid, name, type from pragma_table_info('{self.sql_model}') order by cid asc").fetchall()
        self.headers = [h[1] for h in meta if h[0] > 0] # ignore idx column
        self.column_count = len(self.headers)
        self.header_dtype_dict = {d[1]: d[2] for d in meta}
        self.headers_to_py_dtypes_dict = {k:self.static_sql_to_py_map_dict[v] if v in self.static_sql_to_py_map_dict.keys() else "str" for (k,v) in self.header_dtype_dict.items()}
        self.headers_to_sql_dtypes_dict = {k:"TEXT" if v=='str' else "INTEGER" if v=='int' else "REAL" if v=='float' else "TIMESTAMP" if v=='datetime' else "NULL" if v=='NoneType' else "BLOB" for (k,v) in self.headers_to_py_dtypes_dict.items()}
        self.header_idx_dtype_dict = {(m[0]-1): (m[1], m[2]) for m in meta if m[1] != self.sql_idx}
        if return_data:
            return (self.headers,self.header_dtype_dict,self.header_idx_dtype_dict) # format of {header_idx: (header_name, header_dtype)}

    def get_max_rows(self) -> int:
        """returns the current `max_rows` property value"""
        return self.max_rows
    
    def set_max_rows(self, rows:int) -> None:
        """set `max_rows` to limit rows displayed when `repr` or `print` is called, does not change the maximum rows stored in `SQLDataModel`"""
        if type(rows) != int:
            print(f'{self.clswarning} Invalid argument "{rows}", please provide an integer value to set the maximum rows attribute...')
            return
        if rows <= 0:
            print(f'{self.clswarning} Invalid value "{rows}", please provide an integer value >= 1 to set the maximum rows attribute...')
            return
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
        if  (type(alignment) != str) or (alignment not in ('<', '^', '>')):
            print(f'{self.clswarning} Invalid argument "{alignment}", please provide a valid f-string alignment formatter or set `alignment=None` to use default behaviour...')
            return
        self.column_alignment = alignment
        return

    def get_display_index(self) -> bool:
        """returns the current boolean value for `is_display_index`, which determines whether or not the `SQLDataModel` index will be shown in print or repr calls"""
        return self.display_index

    def set_display_index(self, display_index:bool) -> None:
        """sets the `display_index` property to enable or disable the inclusion of the `SQLDataModel` index value in print or repr calls, default set to include"""
        if not isinstance(display_index, bool):
            print(f'{self.clswarning} Invalid argument "{display_index}", please provide a valid boolean (True | False) value to the `display_index` argument...')
            return
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
            print(f"""{ANSIColor().alert(cls.__name__, 'E')} Required package not found, numpy must be installed in order to use the from_numpy method""")
            sys.exit()
        return cls.from_data(data=array.tolist(),headers=headers, *args, **kwargs)

    @classmethod
    def from_pandas(cls, df, headers:list[str]=None, *args, **kwargs) -> SQLDataModel:
        """returns a `SQLDataModel` object created from the provided pandas `DataFrame`, note that `pandas` must be installed in order to use this class method"""
        if not _has_pd:
            print(f"""{ANSIColor().alert(cls.__name__, 'E')} Required package not found, pandas must be installed in order to use the from_pandas method""")
            sys.exit()
        data = [x[1:] for x in df.itertuples()]
        headers = df.columns.tolist() if headers is None else headers
        return cls.from_data(data=data,headers=headers, *args, **kwargs)
    
    @classmethod
    def from_pickle(cls, filename:str=None, *args, **kwargs) -> SQLDataModel:
        """returns the `SQLDataModel` object from `filename` if provided, if `None` the current directory will be scanned for the default `to_pickle()` format"""
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        if (filename is not None) and (len(filename.split(".")) <= 1):
            print(f"{ANSIColor().alert(cls.__name__, 'W')} File extension missing, provided filename \"{filename}\" did not contain an extension and so \".sdm\" was appended to create a valid filename...")
            filename += '.sdm'
        if not Path(filename).is_file():
            print(f"{ANSIColor().alert(cls.__name__, 'E')} File not found, provided filename \"{filename}\" could not be found, please ensure the filename exists in a valid path...")
            sys.exit()
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
            print(f"""{ANSIColor().alert(cls.__name__, 'W')} Provided SQL Connection has not been tested, behavior for {db_dialect} may be unpredictable or unstable...""")
            pass
        if len(sql_query.split()) == 1:
            sql_query = f""" select * from {sql_query} """
        try:
            sql_c = sql_connection.cursor()
        except:
            print(f"""{ANSIColor().alert(cls.__name__, 'E')} Provided SQL Connection is not open, please reopen the database connection or provide a valid SQL Connection object...""")
            sys.exit()
        try:
            sql_c.execute(sql_query)
            data = sql_c.fetchall()
        except Exception as e:
            print(f"""{ANSIColor().alert(cls.__name__, 'E')} Provided SQL query is invalid or malformed, please check query and resolve {e}...""")
            sys.exit()
        if (len(data) < 1) or (data is None):
            print(f"""{ANSIColor().alert(cls.__name__, 'W')} Provided SQL query returned no data, please provide a valid query with sufficient return data...""")
            return
        headers = [x[0] for x in sql_c.description]
        return cls.from_data(data, headers, *args, **kwargs)

##############################################################################################################
############################################## instance methods ##############################################
##############################################################################################################

    def data(self, include_index:bool=False, include_headers:bool=False) -> list[tuple]:
        """returns the `SQLDataModel` data as a list of tuples for multiple rows, or as a single tuple for individual rows\n\ndata returned without index and headers by default, use `include_headers=True` or `include_index=True` to modify"""
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index))
        data = self.sql_c.fetchall()
        if (len(data) == 1) and (not include_headers): # if only single row
            data = data[0]
        if len(data) == 1: # if only single column
            data = data[0]
        return [tuple([x[0] for x in self.sql_c.description]),data] if include_headers else data

    def to_csv(self, csv_file:str, delimeter:str=',', quotechar:str='"', include_index:bool=False, *args, **kwargs):
        """writes `SQLDataModel` to specified file in `csv_file` argument, must be compatible `.csv` file extension"""
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index))
        write_headers = [x[0] for x in self.sql_c.description]
        with open(csv_file, 'w', newline='') as file:
            csvwriter = csv.writer(file,delimiter=delimeter,quotechar=quotechar,quoting=csv.QUOTE_MINIMAL, *args, **kwargs)
            csvwriter.writerow(write_headers)
            csvwriter.writerows(self.sql_c.fetchall())
        print(f'{self.clssuccess} csv file "{csv_file}" created')

    def to_dict(self, rowwise:bool=True) -> dict:
        """returns a dict from `SQLDataModel` using index rows as keys, set `rowwise=False` to use model headers as keys instead"""
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=True))
        if rowwise:
            return {row[0]:row[1:] for row in self.sql_c.fetchall()}
        else:
            data = self.sql_c.fetchall()
            headers = [x[0] for x in self.sql_c.description]
            return {headers[i]:tuple([x[i] for x in data]) for i in range(len(headers))}
        
    def to_list(self, include_index:bool=True, include_headers:bool=False) -> list[tuple]:
        """returns a list of tuples containing all the `SQLDataModel` data without the headers by default
        \nuse `include_headers = True` to return the headers as the first item in returned sequence"""
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index))
        return [tuple([x[0] for x in self.sql_c.description]),*self.sql_c.fetchall()] if include_headers else self.sql_c.fetchall()
    
    def to_numpy(self, include_index:bool=False, include_headers:bool=False):
        """converts `SQLDataModel` to numpy `array` object of shape (rows, columns), note that `numpy` must be installed to use this method"""
        if not _has_np:
            print(f"""{self.clserror} Required package not found, numpy must be installed in order to use to_numpy method""")
            sys.exit()
        fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index)
        self.sql_c.execute(fetch_stmt)
        if include_headers:
            return _np.vstack([_np.array([x[0] for x in self.sql_c.description]),[_np.array(x) for x in self.sql_c.fetchall()]])
        return _np.array([_np.array(x) for x in self.sql_c.fetchall()])

    def to_pandas(self, include_index:bool=False, include_headers:bool=True):
        """converts `SQLDataModel` to pandas `DataFrame` object, note that `pandas` must be installed to use this method"""
        if not _has_pd:
            print(f"""{self.clserror} Required package not found, pandas must be installed in order to use to_pandas method""")
            sys.exit()
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index))
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
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index))
        model_data = [x for x in self.sql_c.fetchall()] # using new process
        model_headers = [x[0] for x in self.sql_c.description]
        try:
            extern_c = extern_conn.cursor()
        except:
            print(f"""{self.clserror} Provided SQL Connection is not open, please reopen the database connection or provide a valid SQL Connection object...""")
            sys.exit()        
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

    def iter_rows(self, min_row:int=None, max_row:int=None, include_index:bool=False, include_headers:bool=False) -> Generator:
        """returns a generator object of the rows in the model from `min_row` to `max_row`, usage:
        \n`for row in sm_obj.iter_rows(min_row=2, max_row=4):`
        \n\t`print(row)`
        \n`min_row` and `max_row` are both inclusive to values specified
        \n`include_headers=True` to include headers as first row
        """
        min_row, max_row = min_row if min_row is not None else self.min_idx, max_row if max_row is not None else self.max_idx
        self.sql_c.execute(self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_index, restrict_to_row_idxs=(min_row,max_row)))
        if include_headers:
            yield tuple([x[0] for x in self.sql_c.description])
        yield from (x for x in self.sql_c.fetchall())

    def _generate_sql_fetch_stmt_for_idxs(self, idxs:tuple[int]=None, include_idx_col:bool=True, restrict_to_row_idxs:tuple[int]=None) -> str:
        """returns a constructed fetch stmt using the columns and aliases in range of the provided idxs and optionally includes index columnm, `restrict_to_row_idxs` should be tuple of (min_idx, max_idx)"""
        dyn_idx_include_str = f"\"{self.sql_idx}\"," if include_idx_col else ""
        # headers_str = ",".join([f"""\"{v[0]}\"""" for v in self.header_idx_dtype_dict.values()]) if idxs is None else ",".join([f"""\"{v[0]}\"""" for k,v in self.header_idx_dtype_dict.items() if k in idxs])
        headers_str = ",".join([f"""\"{v[0]}\"""" for v in self.header_idx_dtype_dict.values()]) if idxs is None else ",".join([f"""\"{self.headers[i]}\"""" for i in idxs])
        if restrict_to_row_idxs is None:
            return f"""select {dyn_idx_include_str}{headers_str} from {self.sql_model} """
        min_idx, max_idx = restrict_to_row_idxs
        return f"""select {dyn_idx_include_str}{headers_str} from {self.sql_model} where {self.sql_idx} >= {min_idx} and {self.sql_idx} <= {max_idx} order by {self.sql_idx} asc"""

    def _generate_sql_fetch_for_joining_tables(self, base_headers:list[str], join_table:str, join_column:str, join_headers:list[str], join_type:str='left') -> str:
        base_headers_str = ",".join([f"""a.\"{v[0]}\" as \"{v[0]}\"""" for v in self.header_idx_dtype_dict.values() if v[0] in base_headers])
        join_headers_str = ",".join([f"""b.\"{col}\" as \"{col}\" """ for col in join_headers])
        join_type_str = "left join" if join_type == 'left' else 'left outer join'
        join_predicate_str = f"""from {self.sql_model} a {join_type_str} \"{join_table}\" b on a.\"{join_column}\" = b.\"{join_column}\" """
        sql_join_stmt = f"""select {base_headers_str}, {join_headers_str} {join_predicate_str}"""
        return sql_join_stmt

    def get_rows_and_cols_at_index_range(self, start_index: int = None, stop_index: int = None, start_col:int=None, stop_col:int=None, *args, **kwargs) -> SQLDataModel:
        """returns `SQLDataModel` rows from specified `start_index` to specified `stop_index` ranging from `start_col` up to and including `stop_col`
        \nas a new `SQLDataModel`, if `stop_index` is not specified or the value exceeds `row_count`, then last row or largest valid index will be used"""
        row_idx_start = start_index if start_index is not None else self.min_idx
        row_idx_stop = stop_index if stop_index is not None else self.max_idx
        col_idx_start = start_col if start_col is not None else 0
        col_idx_stop = stop_col if stop_col is not None else self.column_count
        if (row_idx_start > self.max_idx) or (row_idx_stop < self.min_idx):
            print(f'{self.clswarning} Row index out of range, row indicies ({row_idx_start}:{row_idx_stop}) out of range for current row indicies of ({self.min_idx}:{self.max_idx})')
            sys.exit()
        if  (col_idx_start > self.column_count) or (col_idx_start > col_idx_stop) or (col_idx_stop > self.column_count):
            print(f'{self.clswarning} Column index out of range, column indicies ({col_idx_start}:{col_idx_stop}) out of range for current column indicies of (0:{self.column_count})')
            sys.exit()
        else:
            idx_args = tuple(range(col_idx_start,col_idx_stop)) if col_idx_start != col_idx_stop else (col_idx_start,)
            fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(idx_args,include_idx_col=True)
        self.sql_c.execute(f"""{fetch_stmt} where {self.sql_idx} >= {row_idx_start} and {self.sql_idx} <= {row_idx_stop} order by {self.sql_idx} asc""")
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description],*args,**kwargs)    

    def to_local_db(self, db:str=None):
        """stores the `SQLDataModel` internal in-memory database to local disk database, if `db=None`, the current filename will be used as a default target"""
        with sqlite3.connect(db) as target:
            self.sql_db_conn.backup(target)
        print(f'{self.clssuccess} local db "{db}" created')

    def _get_sql_create_stmt(self) -> str:
        """returns the current sql create statement stored in the sqlite_master table of the model database"""
        self.sql_c.execute("select sql from sqlite_master")
        return self.sql_c.fetchone()[0]
    
    def _get_discontiguous_rows_and_cols(self, rows:tuple=None, cols:tuple=None, *args,**kwargs) -> SQLDataModel:
        if rows is not None:
            if type(rows) == int:
                dyn_rows_stmt = f"where {self.sql_idx} = {rows}"
            if type(rows) == slice:
                dyn_rows_stmt = f"where {self.sql_idx} >= {rows.start if rows.start is not None else self.min_idx} and {self.sql_idx} <= {rows.stop if rows.stop is not None else self.max_idx}"
            if type(rows) == tuple:
                dyn_rows_stmt = f"where {self.sql_idx} in {rows}"
        else:
            dyn_rows_stmt = ""
        if cols is not None:
            if type(cols) == slice:
                cols = tuple(range(cols.start if cols.start is not None else 0, cols.stop if cols.stop is not None else self.column_count))
            if type(cols) == int:
                cols = (cols,)
        fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(idxs=cols, include_idx_col=True) + dyn_rows_stmt # none cols handled by this function
        self.sql_c.execute(fetch_stmt)
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description],*args,**kwargs)    
    
    def __getitem__(self, slc) -> SQLDataModel:
        if isinstance(slc, tuple):
            try:
                if len(slc) != 2:
                    raise ValueError(f"{self.clserror} Dimension mismatch: {len(slc)} is not a valid number of arguments for row and column indexes...")
                row_idxs, col_idxs = slc[0], slc[1]
                if not isinstance(row_idxs, slice) and not isinstance(row_idxs, int) and not isinstance(row_idxs, tuple):
                    raise TypeError(f"{self.clserror} Type mismatch: {type(row_idxs)} is not a valid type for row indexing, please provide a slice type to index rows correctly...")
                if not isinstance(col_idxs, slice) and not isinstance(col_idxs, int) and not isinstance(col_idxs, tuple) and not isinstance(col_idxs, str) and not isinstance(col_idxs, list):
                    raise TypeError(f"{self.clserror} Type mismatch: {type(col_idxs)} is not a valid type for column indexing, please provide a slice, list or str type to index columns correctly...")
            except (ValueError,TypeError) as e:
                print(e)
                sys.exit()
            if isinstance(row_idxs, int):
                row_target = row_idxs
                row_start = row_target
                row_stop = row_target
            if isinstance(row_idxs, slice):
                row_start = row_idxs.start if row_idxs.start is not None else self.min_idx
                row_stop = row_idxs.stop if row_idxs.stop is not None else self.max_idx
            if isinstance(col_idxs, int): # multiple discontiguous columns
                col_target = col_idxs if col_idxs <= self.column_count else self.column_count
                col_start = col_target
                col_stop = col_target
            if isinstance(col_idxs, slice): # multiple column range
                col_start = col_idxs.start if col_idxs.start is not None else 0
                col_stop = col_idxs.stop if col_idxs.stop is not None else self.column_count
            if isinstance(col_idxs, str): # single column as string
                try:
                    col_target = self.headers.index(col_idxs)
                except ValueError as e:
                    print(f"{self.clserror} Invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                    sys.exit()                
                col_start = col_target
                col_stop = col_target       
            if isinstance(col_idxs, list): # multiple assumed discontiguous columns as list of strings
                try:
                    col_idxs = tuple([self.headers.index(col) for col in col_idxs])
                except ValueError as e:
                    print(f"{self.clserror} Invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                    sys.exit()
            if not isinstance(col_idxs, tuple) and not isinstance(row_idxs, tuple):
                return self.get_rows_and_cols_at_index_range(start_index=row_start, stop_index=row_stop, start_col=col_start, stop_col=col_stop, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)
            else:
                return self._get_discontiguous_rows_and_cols(rows=row_idxs, cols=col_idxs, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)
        
        ### column indexes by str or list ###
        if isinstance(slc, str) or isinstance(slc, list):
            if type(slc) == str:
                slc = [slc]
            try:
                col_idxs = tuple([self.headers.index(col) for col in slc])
            except ValueError as e:
                print(f"{self.clserror} Invalid columns provided, {e} of current model headers, use `.get_headers()` method to get model headers...")
                sys.exit()
            row_idxs = None
            return self._get_discontiguous_rows_and_cols(rows=row_idxs, cols=col_idxs, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)

        ### row indexes slice ###
        if isinstance(slc, slice):
            start_idx = slc.start if slc.start is not None else self.min_idx
            stop_idx = slc.stop if slc.stop is not None else self.max_idx
            return self.get_rows_and_cols_at_index_range(start_index=start_idx,stop_index=stop_idx, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index)
        
        ### single row index ###
        if isinstance(slc, int):
            single_idx = slc
            return self.get_rows_and_cols_at_index_range(start_index=single_idx, stop_index=single_idx, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index) 
        
    def set_display_color(self, color:str|tuple):
        """sets the table string representation color when `SQLDataModel` is displayed in the terminal, use hex value or a tuple of rgb values:
        \n\t`color='#A6D7E8'` # string with hex value\n\n\t`color=(166, 215, 232)` # tuple of rgb values\n\nNote: by default no color styling is applied"""
        try:
            pen = ANSIColor(color)
            self.display_color = pen
            print(f"""{self.clssuccess} Display color changed, the terminal display color has been changed to {pen.wrap(f"color {pen.text_color_str}")}""")
        except:
            print(f"{self.clswarning} Invalid color, the terminal display color could not be changed, please provide a valid hex value or rgb color code...")

##############################################################################################################
################################################ sql commands ################################################
##############################################################################################################

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
            print(f'{self.clserror} Unable to rename model table, SQL execution failed with: {e}')
            return
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

    def fetch_query(self, sql_query:str, **kwargs):
        """returns a new `SQLDataModel` object after executing provided `sql_query` arg using the current `SQLDataModel`\n\nuse default table name 'sdmmodel' or `SQLDataModel.get_model_name()` to get current model alias"""
        try:
            self.sql_c.execute(sql_query)
        except Exception as e:
            print(f'{self.clserror} Invalid or malformed SQL, provided query failed with error {e}...')
            sys.exit()
        data = self.sql_c.fetchall()
        headers = [x[0] for x in self.sql_c.description]
        return type(self)(data, headers=headers, max_rows=self.max_rows, min_column_width=self.min_column_width, max_column_width=self.max_column_width, column_alignment=self.column_alignment, display_color=self.display_color, display_index=self.display_index, **kwargs)

    def group_by(self, *columns:str, order_by_count:bool=True, **kwargs):
        """returns a new `SQLDataModel` after performing group by on columns specified, example:
        \n`dm_obj.group_by("country")` # by single str
        \n`dm_obj.group_by("country","state","city")` # by multiple str
        \n`dm_obj.group_by(["country","state","city"])` # by multiple list
        \nuse `order_by_count = False` to change ordering from count to column args"""
        if type(columns[0]) == str:
            columns = columns
        elif type(columns[0]) in (list, tuple):
            columns = columns[0]
        else:
            print(f'{self.clserror} Invalid columns argument, provided type {type(columns[0]).__name__} is invalid, please provide str, list or tuple type...')
            sys.exit()
        for col in columns:
            if col not in self.headers:
                print(f'{self.clserror} Invalid group by targets, provided column \"{col}\" does not exist in current model, valid targets:\n{self.headers}')
                sys.exit()
        columns_group_by = ",".join(f'"{col}"' for col in columns)
        order_by = "count(*)" if order_by_count else columns_group_by
        group_by_stmt = f"""select {columns_group_by}, count(*) as count from "{self.sql_model}" group by {columns_group_by} order by {order_by} desc"""
        try:
            self.sql_c.execute(group_by_stmt)
        except Exception as e:
            print(f'{self.clserror} Invalid or malformed SQL, provided query failed with error {e}...')
            sys.exit()
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
            print(f"{self.clserror} No shared column found, no matching join column was found in the provided model, ensure one is available or specify one explicitly with on_column='shared_column'")
            sys.exit()
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
            print(f'{self.clserror} Invalid or malformed SQL, unable to execute provided SQL query with error {e}...')
            sys.exit()
    
    def __repr__(self):
        display_headers = self.headers
        include_idx = self.display_index
        fetch_stmt = self._generate_sql_fetch_stmt_for_idxs(include_idx_col=include_idx)
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

    def add_column(self, column_name:str, value=None) -> None:
        """adds `column_name` argument as a new column to `SQLDataModel`, populating with the `value` provided or with SQLs null field if `value=None` by default\n\nnote that if a current column name is passed to `value`, that columns values will be used to fill the new column, effectively copying it"""
        create_col_stmt = f"""alter table {self.sql_model} add column \"{column_name}\""""
        if (value is not None) and (value in self.headers):
            dyn_default_value = f"""{self.headers_to_sql_dtypes_dict[value]}"""
            dyn_copy_existing = f"""update {self.sql_model} set \"{column_name}\" = \"{value}\";"""
        else:
            dyn_default_value = f"""{self.static_py_to_sql_map_dict[type(value).__name__]} not null default {value}""" if value is not None else "TEXT"
            dyn_copy_existing = ""
        full_stmt = f"""begin transaction; {create_col_stmt} {dyn_default_value};{dyn_copy_existing} end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            print(f'{self.clserror} Unable to apply function, SQL execution failed with: {e}')
            sys.exit()
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} added new column "{column_name}" to model')

    def apply_function_to_column(self, func:Callable, column:str|int) -> None:
        """applies `func` to provided `SQLDataModel` column by passing its current value to the argument of `func`, updating the columns values `func` output\n\nnote that if the number of `args` in the inspected signature of `func`, is more than 1, all of `SQLDataModel` current columns will be provided as input and consequently, the number of `func` args must match the current number of columns in the model or an `Exception` will be raised\n\nuse `generate_apply_function_stub()` method to return a preconfigured template using current `SQLDataModel` columns and dtypes to assist"""
        ### get column name from str or index ###
        if (not isinstance(column, int)) and (not isinstance(column, str)):
            print(f"{self.clserror} Invalid column argument provided, {column} is not a valid target, use column index or column name as a string...")
            sys.exit()            
        if isinstance(column, int):
            try:
                column = self.headers[column]
            except Exception as e:
                print(f"{self.clserror} Invalid column index provided, {column} is not a valid column index, use `.column_count` property to get valid range...")
                sys.exit()
        if isinstance(column, str):
            try:
                if column not in self.headers:
                    raise Exception
                else:
                    column = column
            except Exception as e:
                print(f"{self.clserror} Invalid column provided, {column} is not valid for current model, use `.get_headers()` method to get model headers...")
                sys.exit()
        target_column = column
        try:
            func_name = func.__name__
            func_argcount = func.__code__.co_argcount
            self.sql_db_conn.create_function(func_name, func_argcount, func)
        except Exception as e:
            print(f'{self.clserror} Unable to create function with provided callable "{func}", SQL process failed with: {e}')
        if func_argcount == 1:
            input_columns = target_column
        elif func_argcount == self.column_count:
            input_columns = ",".join([f"\"{col}\"" for col in self.headers])
        else:
            print(f'{self.clserror} Invalid function arg count: {func_argcount}, input args to "{func_name}" must be 1 or {self.column_count} based on the current models structure, ie...\n{self.generate_apply_function_stub()}')
            sys.exit()
        sql_apply_update_stmt = f"""update {self.sql_model} set {target_column} = {func_name}({input_columns})"""
        full_stmt = f"""begin transaction; {sql_apply_update_stmt}; end transaction;"""
        try:
            self.sql_c.executescript(full_stmt)
            self.sql_db_conn.commit()
        except Exception as e:
            self.sql_db_conn.rollback()
            print(f'{self.clserror} Unable to apply function, SQL execution failed with: {e}')
            sys.exit()
        self._set_updated_sql_metadata()
        print(f'{self.clssuccess} applied function "{func_name}()" to current model')        

    def generate_apply_function_stub(self) -> str:
        """generates a function template using the current `SQLDataModel` to format function arguments to pass to `apply_function_to_column()` method"""
        func_signature = ", ".join([f"""{k.replace(" ","_")}:{v}""" for k,v in self.headers_to_py_dtypes_dict.items() if k != self.sql_idx])
        return f"""def func({func_signature}):\n    # apply logic and return value\n    return"""
    