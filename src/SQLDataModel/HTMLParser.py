from __future__ import annotations
from html.parser import HTMLParser
from .exceptions import ErrorFormat

class HTMLParser(HTMLParser):
    """
    An HTML parser designed to extract tables from HTML content.

    This parser subclasses HTMLParser from the standard library to parse HTML content.
    It extracts tables from the HTML and provides methods to access the table data.

    Attributes:
        ``convert_charrefs`` (bool): Flag indicating whether to convert character references to Unicode characters. Default is True.
        ``cell_sep`` (str): Separator string to separate cells within a row. Default is an empty string.
        ``table_identifier`` (int or str): Identifier used to locate the target table. It can be either an integer representing the table index, or a string representing the HTML 'name' or 'id' attribute of the table.
        ``_in_td`` (bool): Internal flag indicating whether the parser is currently inside a <td> tag.
        ``_in_th`` (bool): Internal flag indicating whether the parser is currently inside a <th> tag.
        ``_current_table`` (list): List to hold the current table being parsed.
        ``_current_row`` (list): List to hold the current row being parsed.
        ``_current_cell`` (list): List to hold the current cell being parsed.
        ``_ignore_next`` (bool): Internal flag indicating whether the next token should be ignored.
        ``found_target`` (bool): Flag indicating whether the target table has been found.
        ``_is_finished`` (bool): Internal flag indicating whether parsing is finished.
        ``table_counter`` (int): Counter to keep track of the number of tables encountered during parsing.
        ``target_table`` (list): List to hold the data of the target table once found.
    
    Change Log:
        - Version 0.9.0 (2024-06-26):
            - Modified integer indexing of table elements found to use one-based indexing instead of zero-based indexing to align with similar method usage across package.
    """    
    def __init__(self, *, convert_charrefs: bool = True, cell_sep:str=" ", table_identifier:int|str=1) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        if table_identifier is None:
            table_identifier = 0
        if not isinstance(table_identifier, (str,int)):
            msg = ErrorFormat(f"TypeError: invalid type '{type(table_identifier).__name__}', argument for `table_identifier` must be one of 'int' or 'str' representing table index location or HTML 'name' or 'id' attribute")
            raise TypeError(msg)
        self.table_identifier = table_identifier
        self._cell_sep = cell_sep
        self._in_td = False
        self._in_th = False
        self._current_table = []
        self._current_row = []
        self._current_cell = []
        self._ignore_next = False
        self.found_target = False
        self._is_finished = False
        self.table_counter = 0
        self.target_table = []
    
    def handle_starttag(self, tag: str, attrs: list[str]) -> None:
        """
        Handle the start of an HTML tag during parsing.

        Parameters:
            ``tag`` (str): The name of the HTML tag encountered.
            ``attrs`` (list[str]): A list of (name, value) pairs representing the attributes of the tag.
        """        
        if self._is_finished:
            return
        if tag == "table":
            self.table_counter += 1
            if isinstance(self.table_identifier, int):
                if self.table_counter == self.table_identifier:
                    self.found_target = True
            elif isinstance(self.table_identifier, str):
                for param, attribute in attrs:
                    if param.lower().strip() in ("id", "name"):
                        if attribute.lower().strip() == self.table_identifier.lower().strip():
                            self.found_target = True
        if not self.found_target:
            return
        if tag == 'td':
            self._in_td = True
        elif tag == 'th':
            self._in_th = True
        elif tag == "style":
            self._ignore_next = True

    def handle_data(self, data: str) -> None:
        """
        Handle the data within an HTML tag during parsing.

        Parameters:
            ``data`` (str): The data contained within the HTML tag.
        """        
        if self._is_finished:
            return
        if not self.found_target or self._ignore_next:
            return
        if self._in_td or self._in_th:
            self._current_cell.append(data.strip())
    
    def handle_endtag(self, tag: str) -> None:
        """
        Handle the end of an HTML tag during parsing and modify the parsing tags accordingly.

        Parameters:
            ``tag`` (str): The name of the HTML tag encountered.
        """        
        if self._is_finished:
            return
        if tag == 'style':
            self._ignore_next = False
        elif tag == 'td':
            self._in_td = False
        elif tag == 'th':
            self._in_th = False
        if tag in ['td', 'th']:
            if self.found_target:
                final_cell = self._cell_sep.join(self._current_cell).strip()
                final_cell = final_cell if final_cell != '' else None
                self._current_row.append(final_cell)
                self._current_cell = []
        elif tag == 'tr':
            if self.found_target:
                self._current_table.append(self._current_row)
                self._current_row = []
        elif tag == 'table':
            if self.found_target:
                self.target_table = self._current_table
                self._is_finished = True

    def validate_table(self) -> None:
        """
        Validate and retrieve the target HTML table data based on ``table_identifier`` used for parsing.

        Returns:
            ``tuple[list, list|None]``: A tuple containing the table data and headers (if present).

        Raises:
            ``ValueError``: If the target table is not found or cannot be parsed.

        Note:
            - :py:mod:`SQLDataModel.from_html() <sqldatamodel.sqldatamodel.SQLDataModel.from_html>` uses this class to extract valid HTML tables from either web or file content.
            - If a row is found with mismatched dimensions, it will be filled with ``None`` values to ensure tabular output.
        """   
        if not self.found_target:
            if (num_tables_found := self.table_counter) < 1:
                if num_tables_found < 1:
                    msg = ErrorFormat(f"ValueError: zero table elements found in provided source, confirm `html_source` is valid HTML or check integrity of data")
                    raise ValueError(msg)
            else:
                if isinstance(self.table_identifier, int):
                    msg = ErrorFormat(f"ValueError: found '{num_tables_found}' tables in source within range '0:{num_tables_found}' but none found at provided index '{self.table_identifier}'")
                    raise ValueError(msg)
                else:
                    msg = ErrorFormat(f"ValueError: found '{num_tables_found}' tables in source within range '0:{num_tables_found}' but none found with 'id' or 'name' attribute matching provided indentifier '{self.table_identifier}'")
                    raise ValueError(msg)
        if len(self.target_table) < 1:
            msg = ErrorFormat(f"ValueError: found potential match for table identifier '{self.table_identifier}' but failed to parse table from data, check integrity of source")
            raise ValueError(msg)
        if len(self.target_table) == 1:
            return self.target_table, None
        data = []
        column_count = max([len(row) for row in self.target_table])
        for t_row in self.target_table:
            len_row = len(t_row)
            if len_row < 1:
                continue
            elif len_row == column_count:
                data.append(t_row)
            elif len_row < column_count:
                data.append([*t_row,*[None for _ in range(column_count-len_row)]])
        data, headers = data[1:], data[0]
        headers = [col if col is not None else f"col_{i}" for i,col in enumerate(headers)]
        return data, headers