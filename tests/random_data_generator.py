import datetime, random
from typing import Literal
from itertools import cycle, islice
from collections import Counter

def number_duplicates(values):
    counter = Counter()
    for v in values:
        counter[v] += 1
        if counter[v]>1:
            yield v+f'_{counter[v]}'
        else:
            yield v

def data_generator(num_rows:int=12, num_columns:int=8, min_numeric_value:int=-1_000, max_numeric_value:int=1_000, min_literal_length:int=2, max_literal_length:int=12, float_precision:int=6, seed:int=None, return_header_type:Literal['list','tuple']='list', return_row_type:Literal['list','tuple']='list', return_format:Literal["separate","combined"]="separate") -> tuple[list[list], list[str]]:
    """
    Returns randomly generated data for each Python type.

    Parameters:
        - `num_rows` (int): Number of rows in the generated data. Defaults to 12.
        - `num_columns` (int): Number of columns in the generated data. Defaults to 8.
        - `min_numeric_value` (int): Minimum value for numeric types (int and float). Defaults to -1000.
        - `max_numeric_value` (int): Maximum value for numeric types (int and float). Defaults to 1000.
        - `min_literal_length` (int): Minimum length for string and bytes literals. Defaults to 2.
        - `max_literal_length` (int): Maximum length for string and bytes literals. Defaults to 12.
        - `float_precision` (int): Number of decimal places for floating-point numbers. Defaults to 6.
        - `seed` (int): Seed for random number generation. Defaults to None.
        - `return_header_type` (Literal['list','tuple']): Return format of headers:
            - `'list'`: Returns headers as `list[str]`
            - `'tuple'`: Returns headers as `tuple[str]`
        - `return_row_type` (Literal['list','tuple']): Return format of data:
            - `'list'`: Returns headers and data rows as `list[row]`
            - `'tuple'`: Returns headers and data rows as `tuple[row]`
        - `return_format` (Literal['separate','combined']): Return format of data:
            - `'separate'`: Returns in the format of `(data, headers)`
            - `'combined'`: Returns in the format of `(data)` with headers at `data[0]`

    Returns:
        - `tuple[list[list], list[str]]`: A tuple containing the generated data as a list of lists and headers as a list of strings.
    
    ---
    
    Examples:
    
    #### Returning Different Formats

    ```python

    # Combined return format as tuple
    data = data_generator(num_rows = 3, num_columns = 3, seed = 42, return_row_type='tuple', return_format='combined')

    # Result
    ```
    ```shell
    data = [('string', 'int', 'float'), ('hTpigTHKcfoK', -265, 207.452063), ('mNHnKXaXQv', 735, 614.256547), ('nVgx', -296, 459.463573)]
    ```
    ```python
    # Separate return format as list
    data, headers = data_generator(num_rows = 3, num_columns = 4, seed = 42, return_row_type='list', return_format='separate')

    # Result
    ```
    ```shell
    headers = ['string', 'int', 'float']
    data = [['hTpigTHKcfoK', -265, 207.452063], ['mNHnKXaXQv', 735, 614.256547], ['nVgx', -296, 459.463573]]
    ```

    ---

    #### Creating SQLDataModel

    ```python
    # Generate the data
    data, headers = data_generator(num_rows = 3, num_columns = 4, seed = 42)

    # Create the model
    sdm = SQLDataModel(data, headers)

    # Loop over the resulting data
    for row in data:
        print(row)
    
    # Output
    ```
    ```shell
    ['string', 'bool', 'bytes', 'date']
    ['hTpigTHKcfoK', False, b'SG', datetime.date(1920, 6, 12)]
    ['mNHnKXaXQv', False, b'esM2wmeO', datetime.date(1926, 11, 9)]
    ['nVgx', False, b'901xBZ', datetime.date(1989, 11, 21)]
    ```

    ---

    Note:
        - The data consists of random values for each Python type, then types are repeated if `num_columns > 8`.
        - Repeated types are aliased with the numeric suffix such that the max suffix for each type represents the number of occurences.
    """
    if seed is not None:
        random.seed(seed)
    pytypes = ['string', 'int', 'float', 'bool', 'date', 'bytes', 'nonetype', 'datetime']
    dtypes = list(islice(cycle(pytypes), num_columns))
    columns = [[] for _ in range(num_columns)]
    for col_index, dtype in enumerate(dtypes):
        for _ in range(num_rows):
            if dtype == 'int':
                columns[col_index].append(random.randint(min_numeric_value, max_numeric_value))
            elif dtype == 'float':
                columns[col_index].append(round(random.uniform(float(min_numeric_value), float(max_numeric_value)),float_precision))
            elif dtype == 'bool':
                columns[col_index].append(random.choice([True, False]))
            elif dtype == 'string':
                columns[col_index].append(''.join(random.choices('abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(min_literal_length, max_literal_length))))
            elif dtype == 'datetime':
                year = random.randint(1900, 2022)
                month = random.randint(1, 12)
                day = random.randint(1, 28)  # Assuming February has at most 28 days
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                columns[col_index].append(datetime.datetime(year, month, day, hour, minute, second))
            elif dtype == 'bytes':
                columns[col_index].append(bytes(''.join(random.choices('abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(min_literal_length, max_literal_length))),encoding='utf-8'))
            elif dtype == 'date':
                year = random.randint(1900, 2022)
                month = random.randint(1, 12)
                day = random.randint(1, 28)  # Assuming February has at most 28 days
                columns[col_index].append(datetime.date(year, month, day))                
            elif dtype == 'nonetype':
                columns[col_index].append(None)
    row_type = list if return_row_type == 'list' else tuple
    data = list(map(row_type, zip(*columns)))
    header_type = list if return_header_type == 'list' else tuple
    headers = header_type(number_duplicates(dtypes))
    if return_format == 'combined':
        return [headers, *data]
    return data, headers