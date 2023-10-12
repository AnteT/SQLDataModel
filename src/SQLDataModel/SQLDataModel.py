from dataclasses import dataclass
import sqlite3, os, csv, sys, datetime, pickle
from typing import ClassVar, Generator, Tuple, Self   

@dataclass
class DMColorPen:
    """Base representation of ansi styling object, use `DMColorPen.create()` to specify styling for terminal
    \nCan be used explicitly with start and stop:
    \n\t`f"{DMColorPen.start}I will be styled now{DMColorPen.stop}"`
    \nCan also be used by wrapping the text:
    \n\t`DMColorPen.wrap("I will also be styled now")`
    \nInstance objects only store the ansi literals used for styling the terminal
    """
    start: str
    stop: ClassVar[str] = "\033[0m\033[39m\033[49m" # stop all styling

    def __init__(self, start:str):
        self.start = start

    def __repr__(self):
        return self.start
        # return f"""\033[38;2;{fg_r};{fg_g};{fg_b};48;2;{bg_r};{bg_g};{bg_b}m""" if self.text_style is None else f"""\033[1m\033[38;2;{fg_r};{fg_g};{fg_b};48;2;{bg_r};{bg_g};{bg_b}m"""
    
    @classmethod
    def create(cls, text_color:str, background_color:str=None, text_bold:bool=False):
        """creates a pen styling tool using ansi terminal colors, text_color and background_color must be in rgb or hex format, text_bold is off by default"""
        ansi_bold = f'\033[1m' if text_bold else ''
        if type(text_color) == str: # assume hex
            fg_r, fg_g, fg_b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if type(text_color) == tuple: # assume rgb
            fg_r, fg_g, fg_b = text_color
        ansi_color = f"""{ansi_bold}\033[38;2;{fg_r};{fg_g};{fg_b}"""
        if background_color is None:
            return cls(f"""{ansi_color}m""")
        if type(background_color) == str:
            bg_r, bg_g, bg_b = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if type(background_color) == tuple: # assume rgb
            bg_r, bg_g, bg_b = background_color
        ansi_color = f"""{ansi_color};48;2;{bg_r};{bg_g};{bg_b}m"""
        return cls(ansi_color)

    def wrap(self,text:str) -> str:
        """wraps the provided text in the style of the pen"""
        return f"""{self.start}{text}{self.stop}"""
    
@dataclass
class SQLDataModel:
    data: list[tuple]
    headers: list[str] = None
    def __init__(self, data: list[tuple], headers: list[str] = None, max_rows: int = 1_000, min_column_width: int = 6, max_column_width: int = 32,  retain_index:bool = False, *args, **kwargs):
        if headers is not None:
            try:
                if type(headers) not in (list,tuple):
                    raise TypeError(f"{type(self).__name__} Error:\nType mismatch, {type(headers)} is not a valid type for headers, which must be of type list or tuple...")
                if len(headers) != len(data[0]):
                    raise ValueError(f"{type(self).__name__} Error:\nDimension mismatch, {len(data[0])} data values != {len(headers)} header columns, data and headers must have the same dimension...")
            except (ValueError,TypeError) as e:
                print(e)
                sys.exit()
        else:
            headers = [*(f"column_{i}" for i in range(len(data[0])))]
        if retain_index:
            self.headers = headers[1:]
            dtype_slice = 1
        else:
            self.headers = headers
            dtype_slice = 0
        self.max_rows = max_rows
        self.min_column_width = min_column_width
        self.max_column_width = max_column_width
        self.row_count = len(data)
        self.column_count = len(data[0]) if not retain_index else (len(data[0])-1)
        self.column_alignment = None
        self.sql_conn = sqlite3.connect(":memory:",uri=True)
        self.sql_c = self.sql_conn.cursor()
        self.sql_store = "model"
        self.sql_store_id = "model_id"
        self.column_dtypes = {nt_index:(type(field).__name__) for nt_index, field in zip(range(self.column_count), data[0][dtype_slice:])} # dont think i actually need this, everytime i reference the nt is by index so maybe unnecessary?
        sql_dtypes = ['TEXT' if col not in ('float','int') else 'INTEGER' if col == 'int' else 'REAL' for col in self.column_dtypes.values()]
        sql_dtypes_stmt = ", ".join(f"""\"{col}\" {type}""" for col,type in dict(zip(self.headers,sql_dtypes)).items()) # generates sql create table statement using type mapping dict
        if retain_index:
            sql_create_stmt = f"""create table if not exists "{self.sql_store}" ({self.sql_store_id} INTEGER PRIMARY KEY, {sql_dtypes_stmt})""" # added integer primary key as pseudo index
            sql_insert_stmt = f"""insert into "{self.sql_store}" ({self.sql_store_id},{','.join([f'"{col}"' for col in self.headers if col != self.sql_store_id])}) values (?,{','.join(['?' for _ in self.headers])})"""
        else:
            sql_create_stmt = f"""create table if not exists "{self.sql_store}" ({self.sql_store_id} INTEGER PRIMARY KEY, {sql_dtypes_stmt})""" # added integer primary key as pseudo index
            sql_insert_stmt = f"""insert into "{self.sql_store}" ({','.join([f'"{col}"' for col in self.headers])}) values ({','.join(['?' for _ in self.headers])})"""
        self.model_fetch_all_stmt = f"""select * from "{self.sql_store}" """
        self.model_fetch_all_no_index_stmt = f"""select {','.join([f'"{col}"' for col in self.headers])} from "{self.sql_store}" """
        self.model_fetch_all_explicit_stmt = f"""select "{self.sql_store_id}",{','.join([f'"{col}"' for col in self.headers])} from "{self.sql_store}" """
        self.sql_c.execute(sql_create_stmt)
        self.sql_conn.commit()
        self.sql_c.executemany(sql_insert_stmt,data)
        self.sql_conn.commit()
        if kwargs:
            self.__dict__.update(kwargs)

    @classmethod
    def from_data(cls, data: list[tuple], headers: list[str] = None, max_rows: int = 1_000, min_column_width: int = 6, max_column_width: int = 32, *args, **kwargs) -> Self:
        return cls(data, headers, max_rows, min_column_width, max_column_width, *args, **kwargs)

    @classmethod
    def from_csv(cls, csv_file:str, delimeter:str=',', quotechar:str='"', headers:list[str] = None, *args, **kwargs) -> Self:
        """returns a new `SQLDataModel` from the provided csv file by wrapping the from_data method after grabbing the rows and headers by assuming first row represents column headers"""
        with open(csv_file) as csvfile:
            tmp_all_rows = tuple(tuple(row) for row in csv.reader(csvfile, delimiter=delimeter,quotechar=quotechar))
        # return cls(tmp_all_rows[1:],tmp_all_rows[0] if headers is None else headers, max_rows, min_column_width, max_column_width, *args, **kwargs)
        return cls.from_data(tmp_all_rows[1:],tmp_all_rows[0] if headers is None else headers, *args, **kwargs)    
    
    @classmethod
    def from_pickle(cls, filename: str = None, *args, **kwargs) -> Self:
        """returns the `SQLDataModel` object referenced in the `filename` argument if provided
        \nif `None` the current directory will be scanned for the default `to_pickle()` filename format"""
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        with open(filename, 'rb') as f:
            tot_raw = pickle.load(f) # Tuple of Tuples raw data
            return cls.from_data(tot_raw[1:],headers=tot_raw[0], *args, **kwargs)
               
    @classmethod
    def from_sql(cls, sql_query: str, sql_connection: sqlite3.Connection, max_rows: int = 1_000, min_column_width: int = 4, max_column_width: int = 32, *args, **kwargs) -> Self:
        sql_c = sql_connection.cursor()
        sql_c.execute(sql_query)
        data = sql_c.fetchall()
        if data is None:
            return
        headers = [x[0] for x in sql_c.description]
        return cls.from_data(data, headers, max_rows, min_column_width, max_column_width, *args, **kwargs)
    
    def to_data(self, include_headers: bool = False) -> list[tuple]:
        """returns a list of tuples containing all the `SQLDataModel` data without the headers by default
        \nuse `include_headers = True` to return the headers as the first item in returned sequence"""
        self.sql_c.execute(f"select * from {self.sql_store}")
        return [tuple(self.headers),*self.sql_c.fetchall()] if include_headers else self.sql_c.fetchall()
    
    def to_csv(self, csv_file:str, delimeter:str=',', quotechar:str='"', *args, **kwargs):
        """writes `SQLDataModel` to specified file in `csv_file` argument, must be compatible `.csv` file extension"""
        self.sql_c.execute(f"select * from {self.sql_store}")
        with open(csv_file, 'w', newline='') as csv_file:
            csvwriter = csv.writer(csv_file,delimiter=delimeter,quotechar=quotechar,quoting=csv.QUOTE_MINIMAL, *args, **kwargs)
            csvwriter.writerow(self.headers)
            csvwriter.writerows(self.sql_c.fetchall())
    
    def to_pickle(self, filename:str=None) -> None:
        """save the `SQLDataModel` instance to the specified `filename`
        \nby default the name of the invoking python file will be used"""
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.sdm'
        serialized_data = tuple(x[1:] for x in self.iter_rows(include_headers=True)) # no need to send sql_store_id aka index to pickle
        with open(filename, 'wb') as handle:
            pickle.dump(serialized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    def to_sql(self, table:str, extern_conn: sqlite3.Connection, replace_existing:bool = True):
        """inserts `SQLDataModel` into specified table using the sqlite database connection object provided in one of two modes:
        \n`replace_existing = True:` deletes the existing table and replaces it with the SQLDataModel's
        \n`replace_existing = False:` append to the existing table and executes a deduplication statement immediately after
        \ncurrently only `sqlite3.Connection` types are supported
        """
        self.sql_c.execute(f"select * from {self.sql_store}")
        model_data = [x[1:] for x in self.sql_c.fetchall()] # ignore index in this case
        sql_dtypes = ['TEXT' if col not in ('float','int') else 'INTEGER' if col == 'int' else 'REAL' for col in self.column_dtypes.values()]
        sql_dtypes_stmt = ", ".join(f"""\"{col}\" {type}""" for col,type in dict(zip(self.headers,sql_dtypes)).items()) # generates sql create table statement using type mapping dict
        extern_c = extern_conn.cursor()
        if replace_existing:
            sql_drop_stmt = f"""drop table if exists "{table}" """
            extern_c.execute(sql_drop_stmt)
            extern_conn.commit()
        sql_create_stmt = f"""create table if not exists "{table}" ({sql_dtypes_stmt})"""
        sql_insert_stmt = f"""insert into "{table}" ({','.join([f'"{col}"' for col in self.headers])}) values ({','.join(['?' for _ in self.headers])})"""
        extern_c.execute(sql_create_stmt)
        extern_conn.commit()
        extern_c.executemany(sql_insert_stmt,model_data)
        extern_conn.commit()
        if not replace_existing:
            sql_dedupe_stmt = f"""delete from "{table}" where rowid not in (select min(rowid) from "{table}" group by {','.join(f'"{col}"' for col in self.headers)})"""
            extern_c.execute(sql_dedupe_stmt)
            extern_conn.commit()
        extern_conn.close()

    def to_text(self, filename:str, include_ts:bool=False) -> None:
        """writes contents of `SQLDataModel` to specified `filename`, use `include_ts = True` to include timestamp"""
        contents = f"{datetime.datetime.now().strftime('%B %d %Y %H:%M:%S')} status:\n" + self.__repr__() if include_ts else self.__repr__()
        with open(filename, "w", encoding='utf-8') as text_file:
            text_file.write(contents)

    def get_row_at_index(self, index: int) -> tuple:
        """returns `SQLDataModel` row at specified `index` as a tuple"""
        if index > self.row_count:
            return
        self.sql_c.execute(f"""{self.model_fetch_all_stmt} where {self.sql_store_id} = {index}""")
        return self.sql_c.fetchone()
    
    def get_rows_at_index_range(self, start_index: int, stop_index: int = None, retain_index:bool=True, **kwargs) -> Self:
        """returns `SQLDataModel` rows from specified `start_index` to specified `stop_index` as a new `SQLDataModel`
        \nif `stop_index` is not specified or the value exceeds `row_count`, then last row or largest valid index will be used"""
        idx_start = start_index if start_index < self.row_count else 1
        idx_stop = stop_index if stop_index is not None else self.row_count
        fetch_stmt = self.model_fetch_all_explicit_stmt if retain_index else self.model_fetch_all_no_index_stmt
        # self.sql_c.execute(f"""{self.model_fetch_all_no_index_stmt} where {self.sql_store_id} >= {idx_start} and {self.sql_store_id} <= {idx_stop} order by {self.sql_store_id} asc""")
        self.sql_c.execute(f"""{fetch_stmt} where {self.sql_store_id} >= {idx_start} and {self.sql_store_id} <= {idx_stop} order by {self.sql_store_id} asc""")
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], retain_index=retain_index,**kwargs)
    
    def get_max_rows(self) -> int:
        """returns the current `max_rows` property value"""
        return self.max_rows
    
    def set_max_rows(self, rows:int):
        """set `max_rows` to limit rows displayed when `repr` or `print` is called"""
        self.max_rows = rows

    def get_min_column_width(self) -> int:
        """returns the current `min_column_width` property value"""
        return self.min_column_width
    
    def set_min_column_width(self, width:int):
        """set `min_column_width` as minimum number of characters per column when `repr` or `print` is called"""
        self.min_column_width = width

    def get_max_column_width(self) -> int:
        """returns the current `max_column_width` property value"""
        return self.max_column_width
    
    def set_max_column_width(self, width:int):
        """set `max_column_width` as maximum number of characters per column when `repr` or `print` is called"""
        self.max_column_width = width

    def get_column_alignment(self) -> str:
        """returns the current `column_alignment` property value, `None` by default"""
        return self.column_alignment
    
    def set_column_alignment(self, alignment:str):
        """set `column_alignment` as default alignment behavior when `repr` or `print` is called, options:
        \n`column_alignment = None` default behavior, dynamically aligns columns based on value types
        \n`column_alignment = '<'` left align all column values
        \n`column_alignment = '^'` center align all column values
        \n`column_alignment = '>'` right align all column values
        \ndefault behavior aligns strings left, integers & floats right, with headers matching value alignment
        """
        self.column_alignment = alignment

    def get_shape(self) -> Tuple[int, int]:
        """returns the shape of the data as a tuple of `(rows x columns)`"""
        return (self.row_count,self.column_count)
    
    def __getitem__(self, slc):
        if isinstance(slc, slice):
            start_idx = slc.start
            stop_idx = slc.stop
            return self.get_rows_at_index_range(start_index=start_idx,stop_index=stop_idx)
        if isinstance(slc, int):
            single_idx = slc
            return self.get_row_at_index(index=single_idx) 
               
    def __repr__(self):
        self.sql_c.execute(f"select * from {self.sql_store} limit {self.max_rows}")
        table_data = self.sql_c.fetchall()
        table_body = "" # big things can have small beginnings...
        table_newline = "\n"
        display_rows = self.row_count if (self.max_rows is None or self.max_rows > self.row_count) else self.max_rows
        index_width = 2 if display_rows <= 10 else 3 if display_rows <= 100 else 4 if display_rows <= 10_000 else 5 # size required by index
        right_border_width = 3
        max_rows_to_check = display_rows if display_rows < 15 else 15 # updated as exception to 15
        col_length_dict = {col:len(str(x)) for col,x in enumerate(self.headers)} # for first row populate all col lengths
        col_alignment_dict = {i:'<' if self.column_dtypes[i] == 'str' else '>' if self.column_dtypes[i] != 'float' else '<' for i in range(self.column_count)}

        for row in range(max_rows_to_check): # each row is indexed in row col length dict and each one will contain its own dict of col lengths
            current_row = {col:len(str(x)) for col,x in enumerate(table_data[row][1:])} # start at one to enusre index is skipped and column lengths correctly counted
            for col_i in range(self.column_count):
                if current_row[col_i] > col_length_dict[col_i]:
                    col_length_dict.update({col_i:current_row[col_i]})
        for col,width in col_length_dict.items():
            if width < self.min_column_width:
                col_length_dict[col] = self.min_column_width
            elif width > self.max_column_width:
                col_length_dict[col] = self.max_column_width
        index_fmt = f'│{{:>{index_width}}} │ '
        right_border_fmt = ' │'
        col_alignment = self.column_alignment # if None columns will be dynmaic aligned based on dtypes
        columns_fmt = " │ ".join([f"""{{:{col_alignment_dict[i] if col_alignment is None else col_alignment}{col_length}}}""" for i,col_length in col_length_dict.items()]) # col alignment var determines left or right align
        table_abstract_template = """{index}{columns}{right_border}""" # assumption is each column will handle its right side border only and the last one will be stripped
        fmt_dict = {
            'index':index_fmt
            ,'columns':columns_fmt
            ,'right_border':right_border_fmt
            }
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
            for k,v in col_length_dict.items():
                if max_width < total_available_width:
                    max_width += (v+3)
                    max_cols += 1
            if max_width > total_available_width:
                max_cols -= 1
                max_width -= (col_length_dict[max_cols] +3)
            table_row_fmt = """ │ """.join(table_row_fmt.split(""" │ """)[:max_cols+1]) + """ │""" # no longer required, maybe...? +1 required on max columns since index fmt is included after split leaving the format missing two slots right away if you simply decrease it by 1
        table_dynamic_newline = f' ...\n' if table_truncation_required else '\n'
        table_top_bar = table_row_fmt.replace(" │ ","─┬─").replace("│{","┌{").replace(" │","─┐") # first replace col seps and then the top left index corner then the ending right border corner
        table_top_bar = table_top_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        header_row = table_row_fmt.format("",*[str(x)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(x)) > col_length_dict[k] else str(x) for k,x in enumerate(self.headers)]) # for header the first arg will be empty as no index will be used, for the rest it will be the data col key
        header_sub_bar = table_row_fmt.replace(" │ ","─┼─").replace("│{","├{").replace(" │","─┤")
        header_sub_bar = header_sub_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        table_body += table_top_bar + table_newline
        table_body += header_row + table_dynamic_newline
        table_body += header_sub_bar + table_newline
        for i,row in enumerate(table_data):
            if i < display_rows:
                table_body += table_row_fmt.format(row[0],*[str(cell)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(cell)) > (col_length_dict[k]) else str(cell) for k,cell in enumerate(row[1:max_cols+1])]) # start at 1 to avoid index col
                table_body +=  table_dynamic_newline
        table_bottom_bar = table_row_fmt.replace(" │ ","─┴─").replace("│{","└{").replace(" │","─┘") # first replace col seps and then the top left index corner then the ending right border corner
        table_bottom_bar = table_bottom_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        table_signature = f"""\n[{display_rows} rows x {self.column_count} columns]"""
        # width_truncation_debug_details = f"""\t({max_width} of {total_available_width} available width used with {max_cols -1 if not table_truncation_required else max_cols} columns)""" # include additional debug info with: \ncol dytpes dictionary: {self.column_dtypes}\ncol alignment dict: {col_alignment_dict}"""
        # table_body += table_bottom_bar + table_signature + width_truncation_debug_details
        table_body += table_bottom_bar + table_signature # exception change to have less details
        return table_body

    def colorful(self,color:str=None):
        """returns a colorful `repr` of the `SQLDataModel` object using the hex `color` specified"""
        if color is None:
            color = "#78bed7"
        color_pen = DMColorPen.create(color)
        self.sql_c.execute(f"select * from {self.sql_store} limit {self.max_rows}")
        table_data = self.sql_c.fetchall()
        table_body = "" # big things can have small beginnings...
        table_newline = "\n"
        display_rows = self.row_count if (self.max_rows is None or self.max_rows > self.row_count) else self.max_rows
        index_width = 2 if display_rows <= 10 else 3 if display_rows <= 100 else 4 if display_rows <= 10_000 else 5 # size required by index
        right_border_width = 3
        max_rows_to_check = display_rows if display_rows < 15 else 15 # updated as exception to 15
        col_length_dict = {col:len(str(x)) for col,x in enumerate(self.headers)} # for first row populate all col lengths
        col_alignment_dict = {i:'<' if self.column_dtypes[i] == 'str' else '>' if self.column_dtypes[i] != 'float' else '<' for i in range(self.column_count)}

        for row in range(max_rows_to_check): # each row is indexed in row col length dict and each one will contain its own dict of col lengths
            current_row = {col:len(str(x)) for col,x in enumerate(table_data[row][1:])} # start at one to enusre index is skipped and column lengths correctly counted
            for col_i in range(self.column_count):
                if current_row[col_i] > col_length_dict[col_i]:
                    col_length_dict.update({col_i:current_row[col_i]})
        for col,width in col_length_dict.items():
            if width < self.min_column_width:
                col_length_dict[col] = self.min_column_width
            elif width > self.max_column_width:
                col_length_dict[col] = self.max_column_width
        index_fmt = f'│{{:>{index_width}}} │ '
        right_border_fmt = ' │'
        col_alignment = self.column_alignment # if None columns will be dynmaic aligned based on dtypes
        columns_fmt = " │ ".join([f"""{{:{col_alignment_dict[i] if col_alignment is None else col_alignment}{col_length}}}""" for i,col_length in col_length_dict.items()]) # col alignment var determines left or right align
        table_abstract_template = """{index}{columns}{right_border}""" # assumption is each column will handle its right side border only and the last one will be stripped
        fmt_dict = {
            'index':index_fmt
            ,'columns':columns_fmt
            ,'right_border':right_border_fmt
            }
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
            for k,v in col_length_dict.items():
                if max_width < total_available_width:
                    max_width += (v+3)
                    max_cols += 1
            if max_width > total_available_width:
                max_cols -= 1
                max_width -= (col_length_dict[max_cols] +3)
            table_row_fmt = """ │ """.join(table_row_fmt.split(""" │ """)[:max_cols+1]) + """ │""" # no longer required, maybe...? +1 required on max columns since index fmt is included after split leaving the format missing two slots right away if you simply decrease it by 1
        table_dynamic_newline = f' ...\n' if table_truncation_required else '\n'
        table_top_bar = table_row_fmt.replace(" │ ","─┬─").replace("│{","┌{").replace(" │","─┐") # first replace col seps and then the top left index corner then the ending right border corner
        table_top_bar = table_top_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        header_row = table_row_fmt.format("",*[str(x)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(x)) > col_length_dict[k] else str(x) for k,x in enumerate(self.headers)]) # for header the first arg will be empty as no index will be used, for the rest it will be the data col key
        header_sub_bar = table_row_fmt.replace(" │ ","─┼─").replace("│{","├{").replace(" │","─┤")
        header_sub_bar = header_sub_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        table_body += table_top_bar + table_newline
        table_body += header_row + table_dynamic_newline
        table_body += header_sub_bar + table_newline
        for i,row in enumerate(table_data):
            if i < display_rows:
                table_body += table_row_fmt.format(row[0],*[str(cell)[:(col_length_dict[k])-2] + '⠤⠄' if len(str(cell)) > (col_length_dict[k]) else str(cell) for k,cell in enumerate(row[1:max_cols+1])]) # start at 1 to avoid index col
                table_body +=  table_dynamic_newline
        table_bottom_bar = table_row_fmt.replace(" │ ","─┴─").replace("│{","└{").replace(" │","─┘") # first replace col seps and then the top left index corner then the ending right border corner
        table_bottom_bar = table_bottom_bar.format((index_width*'─'),*['─' * length for length in col_length_dict.values()]) # +1 to cover the index format placeholder
        table_signature = f"""\n[{display_rows} rows x {self.column_count} columns]"""
        # width_truncation_debug_details = f"""\t({max_width} of {total_available_width} available width used with {max_cols -1 if not table_truncation_required else max_cols} columns)""" # include additional debug info with: \ncol dytpes dictionary: {self.column_dtypes}\ncol alignment dict: {col_alignment_dict}"""
        # table_body += table_bottom_bar + table_signature + width_truncation_debug_details
        table_body += table_bottom_bar + table_signature # exception change to have less details
        return color_pen.wrap(table_body)
       
    def query(self, sql_query:str, **kwargs):
        """returns a new SQLDataModel object after executing provided sql_query arg"""
        self.sql_c.execute(sql_query)
        data = self.sql_c.fetchall()
        headers = [x[0] for x in self.sql_c.description]
        return type(self)(data, headers, **kwargs)

    def group_by(self, *columns:str, order_by_count=True, **kwargs):
        """returns a new `SQLDataModel` after performing group by on columns specified, example:
        \n`dm_obj.group_by("country")` # by single str
        \n`dm_obj.group_by("country","state","city")` # by multiple str
        \n`dm_obj.group_by(["country","state","city"])` # by multiple list
        \nuse `order_by_count = False` to change ordering from count to column args"""
        if type(columns[0]) == str:
            columns_group_by = ",".join(f'"{col}"' for col in columns)
        elif type(columns[0]) in (list, tuple):
            columns_group_by = ",".join(f'"{col}"' for col in columns[0])
        else:
            return None
        order_by = "count(*)" if order_by_count else columns_group_by
        group_by_stmt = f"""select {columns_group_by}, count(*) as count from "{self.sql_store}" group by {columns_group_by} order by {order_by} desc"""
        self.sql_c.execute(group_by_stmt)
        return type(self)(data=self.sql_c.fetchall(), headers=[x[0] for x in self.sql_c.description], **kwargs)
    
    def iter_rows(self, min_row:int=None, max_row:int=None, include_headers:bool=False) -> Generator:
        """returns a generator object of the rows in the model from `min_row` to `max_row`, usage:
        \n`for row in sm_obj.iter_rows(min_row=2, max_row=4):`
        \n\t`print(row)`
        \n`min_row` and `max_row` are inclusive to values specified
        \n`include_headers=True` to include headers as first row
        """
        if max_row is None:
            max_row = self.row_count
        if min_row is None:
            min_row = 0
        if include_headers:
            self.sql_c.execute(f"""select 0 "{self.sql_store_id}",""" + ", ".join(f"""'{x}' "{x}" """ for x in self.headers) + f""" UNION {self.model_fetch_all_stmt} where {self.sql_store_id} >= {min_row} and {self.sql_store_id} <= {max_row} order by {self.sql_store_id} asc""")
        else:
            self.sql_c.execute(f"""{self.model_fetch_all_stmt} where {self.sql_store_id} >= {min_row} and {self.sql_store_id} <= {max_row} order by {self.sql_store_id} asc""")
        return (x for x in self.sql_c.fetchall())

    # def iter_rows(self) -> Generator[tuple,None,None]:
    #     """returns an iterator through all the available data"""
    #     self.sql_c.execute(self.model_fetch_all_explicit_stmt)
    #     return (x for x in self.sql_c.fetchall())