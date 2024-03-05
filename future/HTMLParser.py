from __future__ import annotations
from html.parser import HTMLParser

class HTMLParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = True, cell_sep:str="", table_identifier:int|str=0) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        if table_identifier is None:
            table_identifier = 0
        if not isinstance(table_identifier, (str,int)):
            raise TypeError(
                HTMLParser.ErrorFormat(f"TypeError: invalid type '{type(table_identifier).__name__}', argument for `table_identifier` must be one of 'int' or 'str' representing table index location or HTML 'name' or 'id' attribute")
            )
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
        self.table_counter = -1
        self.target_table = []

    @staticmethod
    def ErrorFormat(error:str) -> str:
        error_type, error_description = error.split(':',1)
        return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""
    
    def handle_starttag(self, tag: str, attrs: list[str]) -> None:
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
        if self._is_finished:
            return
        if not self.found_target or self._ignore_next:
            return
        if self._in_td or self._in_th:
            self._current_cell.append(data)
    
    def handle_endtag(self, tag: str) -> None:
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
        if not self.found_target:
            if (num_tables_found := self.table_counter + 1) < 1:
                if num_tables_found < 1:
                    raise ValueError(
                        HTMLParser.ErrorFormat(f"ValueError: zero table elements found in provided source, confirm `html_source` is valid HTML or check integrity of data")
                    )
            else:
                if isinstance(self.table_identifier, int):
                    raise ValueError(
                        HTMLParser.ErrorFormat(f"ValueError: found '{num_tables_found}' tables in source within range '0:{num_tables_found}' but none found at provided index '{self.table_identifier}'")
                    )
                else:
                    raise ValueError(
                        HTMLParser.ErrorFormat(f"ValueError: found '{num_tables_found}' tables in source within range '0:{num_tables_found}' but none found with 'id' or 'name' attribute matching provided indentifier '{self.table_identifier}'")
                    )
        if len(self.target_table) < 1:
            raise ValueError(
                HTMLParser.ErrorFormat(f"ValueError: found potential match for table identifier '{self.table_identifier}' but failed to parse table from data, check integrity of source")
            )
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