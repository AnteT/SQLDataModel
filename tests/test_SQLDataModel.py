import datetime, shutil, os, tempfile, csv, random
from itertools import cycle, islice
from collections import Counter
import pytest
import pandas as pd
import numpy as np
from src.SQLDataModel.SQLDataModel import SQLDataModel

def number_duplicates(values):
    counter = Counter()
    for v in values:
        counter[v] += 1
        if counter[v]>1:
            yield v+f'_{counter[v]}'
        else:
            yield v

def data_generator(num_rows:int=12, num_columns:int=8, min_numeric_value:int=-1_000, max_numeric_value:int=1_000, min_literal_length:int=2, max_literal_length:int=12, float_precision:int=6, seed:int=None) -> tuple[list[list], list[str]]:
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

    Returns:
        - `tuple[list[list], list[str]]`: A tuple containing the generated data as a list of lists and headers as a list of strings.
    
    Note:
        - The data consists of random values for each Python type, then types are repeated if `num_columns > 8`.
        - Repeated types are aliased with the numeric suffix such that the max suffix for each type represents the number of occurences.
    
    ---
    
    Example:
    
    ```python
    # Generate the data
    data, headers = data_generator(num_rows = 3, num_columns = 4, seed = 42)

    # Print the headers first
    print(headers)  

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
    data = list(map(list, zip(*columns)))
    headers = list(number_duplicates(dtypes))
    return data, headers

@pytest.fixture
def sample_data() -> tuple[list[list], list[str]]:
    """Returns sample data in format `(data, headers)` to use for testing."""
    return data_generator(num_rows=12, num_columns=8, seed=42)

@pytest.mark.core
def test_init(sample_data):
    input_data, input_headers = sample_data
    input_rows, input_cols = len(input_data), len(input_data[0])
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    assert sdm.row_count == input_rows and sdm.column_count == input_cols
    assert sdm.headers == input_headers

@pytest.mark.core
def test_data(sample_data):
    input_data, input_headers = sample_data
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    out_data = sdm.data()
    assert all([input_data[i][j] == out_data[i][j] for j in range(len(input_data[0])) for i in range(len(input_data))])

@pytest.mark.core
def test_headers(sample_data):
    input_data, input_headers = sample_data
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    out_headers = sdm.get_headers()
    assert all([input_headers[i] == out_headers[i] for i in range(len(input_headers))])
    new_headers = [f"new_{i}" for i in range(len(input_headers))]
    sdm.set_headers(new_headers=new_headers)
    out_headers = sdm.get_headers()
    assert all([new_headers[i] == out_headers[i] for i in range(len(new_headers))])
    rename_headers = [f"rename_{i}" for i in range(len(out_headers))]
    for i, header in enumerate(rename_headers):
        sdm.rename_column(i, header)
    out_headers = sdm.get_headers()
    assert all([rename_headers[i] == out_headers[i] for i in range(len(rename_headers))])

@pytest.mark.core
def test_set_display_properties(sample_data):
    input_data, input_headers = sample_data
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    input_int = 10
    input_bool = False
    input_alignments = ['dynamic', 'left', 'center', 'right']
    sdm.set_display_max_rows(input_int)
    assert input_int == sdm.get_display_max_rows()
    sdm.set_min_column_width(input_int)
    assert input_int == sdm.get_min_column_width()
    sdm.set_max_column_width(input_int)
    assert input_int == sdm.get_max_column_width()
    sdm.set_display_float_precision(input_int)
    assert input_int == sdm.get_display_float_precision()
    sdm.set_display_index(input_bool)
    assert input_bool == sdm.get_display_index()
    for alignment in input_alignments:
        sdm.set_column_alignment(alignment=alignment)
        assert alignment == sdm.get_column_alignment()

@pytest.mark.core
def test_replace():
    input_data, input_headers = [['A',1,'A-X'],['B',2,'B-X'],['C',3,'C-X'],['D',4,'D-X']], ['Letter','Number','Target']
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    sdm.replace('-X', '+Y', inplace=True)
    output_data = sdm.data()
    for i, row in enumerate(input_data):
        input_target = row[-1].replace('-X','+Y')
        output_target = output_data[i][-1]
        assert input_target == output_target    

@pytest.mark.core
def test_repr():
    ### test data type appearance ###
    input_headers = ['string', 'integer', 'float', 'bool', 'datetime']
    input_data = [
        ('ej6JF', -934, 935.961423, 0, datetime.datetime(1946, 8, 27, 7, 30, 50)),
        ('hYFu', -145, -621.435864, 0, datetime.datetime(1963, 1, 14, 17, 19, 38)),
        ('TwBgAiOuPJ1G', 798, 796.610811, 0, datetime.datetime(1982, 10, 2, 15, 45, 28)),
        ('QOkz', -819, -607.55434, 0, datetime.datetime(2008, 5, 9, 22, 22, 46)),
        ('YeANVOB98', 928, 254.282519, 0, datetime.datetime(2017, 3, 24, 12, 45, 16)),
        ('5TegglJ_Tg', -195, -815.716148, 0, datetime.datetime(1932, 11, 13, 6, 57, 27)),
        ('BHEepCKAT', -50, 403.332838, 0, datetime.datetime(1972, 8, 26, 13, 34, 31)),
        ('B9Z4Ry6r7', 600, -684.776982, 1, datetime.datetime(1970, 4, 24, 1, 16, 48)),
        ('6fzZy', 286, 41.511884, 1, datetime.datetime(1933, 4, 5, 9, 46, 24)),
        ('q6mCe7', -467, 779.570168, 0, datetime.datetime(1920, 2, 21, 15, 45, 12)),
        ('MyJ', 365, -757.353885, 0, datetime.datetime(1932, 10, 23, 8, 57, 53)),
        ('Nz', -568, -755.723708, 1, datetime.datetime(1953, 4, 25, 6, 15, 25))
    ]
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='dynamic', min_column_width=4, max_column_width=32, display_index=True, display_float_precision=4, display_max_rows=12)    
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌────┬──────────────┬─────────┬───────────┬──────┬─────────────────────┐
    │    │ string       │ integer │     float │ bool │ datetime            │
    ├────┼──────────────┼─────────┼───────────┼──────┼─────────────────────┤
    │  0 │ ej6JF        │    -934 │  935.9614 │    0 │ 1946-08-27 07:30:50 │
    │  1 │ hYFu         │    -145 │ -621.4359 │    0 │ 1963-01-14 17:19:38 │
    │  2 │ TwBgAiOuPJ1G │     798 │  796.6108 │    0 │ 1982-10-02 15:45:28 │
    │  3 │ QOkz         │    -819 │ -607.5543 │    0 │ 2008-05-09 22:22:46 │
    │  4 │ YeANVOB98    │     928 │  254.2825 │    0 │ 2017-03-24 12:45:16 │
    │  5 │ 5TegglJ_Tg   │    -195 │ -815.7161 │    0 │ 1932-11-13 06:57:27 │
    │  6 │ BHEepCKAT    │     -50 │  403.3328 │    0 │ 1972-08-26 13:34:31 │
    │  7 │ B9Z4Ry6r7    │     600 │ -684.7770 │    1 │ 1970-04-24 01:16:48 │
    │  8 │ 6fzZy        │     286 │   41.5119 │    1 │ 1933-04-05 09:46:24 │
    │  9 │ q6mCe7       │    -467 │  779.5702 │    0 │ 1920-02-21 15:45:12 │
    │ 10 │ MyJ          │     365 │ -757.3539 │    0 │ 1932-10-23 08:57:53 │
    │ 11 │ Nz           │    -568 │ -755.7237 │    1 │ 1953-04-25 06:15:25 │
    └────┴──────────────┴─────────┴───────────┴──────┴─────────────────────┘
    [12 rows x 5 columns]
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]
    ### test short data appearance ###
    input_headers = ['string','integer','bytes','float','datetime']
    input_data = [['string', 12_345, b'binary', 3.14159, datetime.datetime(1989, 11, 9, 6, 15, 25)]]
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='dynamic', min_column_width=4, max_column_width=32, display_index=True, display_float_precision=4, display_max_rows=2)
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌───┬────────┬─────────┬───────────┬─────────┬─────────────────────┐
    │   │ string │ integer │ bytes     │   float │ datetime            │
    ├───┼────────┼─────────┼───────────┼─────────┼─────────────────────┤
    │ 0 │ string │   12345 │ b'binary' │  3.1416 │ 1989-11-09 06:15:25 │
    └───┴────────┴─────────┴───────────┴─────────┴─────────────────────┘
    [1 rows x 5 columns]   
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]   
    ### test truncated data appearance ###
    input_headers = ['string','integer','bytes','float','datetime']
    input_data = [['string', 12_345, b'binary', 3.14159, datetime.datetime(1989, 11, 9, 6, 15, 25)]]
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='dynamic', min_column_width=4, max_column_width=4, display_index=True, display_float_precision=4, display_max_rows=2)
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌───┬──────┬──────┬──────┬──────┬──────┐
    │   │ st⠤⠄ │ in⠤⠄ │ by⠤⠄ │ fl⠤⠄ │ da⠤⠄ │
    ├───┼──────┼──────┼──────┼──────┼──────┤
    │ 0 │ st⠤⠄ │ 12⠤⠄ │ b'⠤⠄ │  3⠤⠄ │ 19⠤⠄ │
    └───┴──────┴──────┴──────┴──────┴──────┘
    [1 rows x 5 columns]
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]   
    ### left column alignment ###    
    input_headers = ['1', '2', '3']
    input_data = [['X','X','X'],['X','X','X'],['X','X','X']]        
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='left', min_column_width=5, max_column_width=5, display_index=True, display_float_precision=4, display_max_rows=4)
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌───┬───────┬───────┬───────┐
    │   │ 1     │ 2     │ 3     │
    ├───┼───────┼───────┼───────┤
    │ 0 │ X     │ X     │ X     │
    │ 1 │ X     │ X     │ X     │
    │ 2 │ X     │ X     │ X     │
    └───┴───────┴───────┴───────┘
    [3 rows x 3 columns]
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]      
    ### center column alignment ###    
    input_headers = ['1', '2', '3']
    input_data = [['X','X','X'],['X','X','X'],['X','X','X']]        
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='center', min_column_width=5, max_column_width=5, display_index=True, display_float_precision=4, display_max_rows=4)
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌───┬───────┬───────┬───────┐
    │   │   1   │   2   │   3   │
    ├───┼───────┼───────┼───────┤
    │ 0 │   X   │   X   │   X   │
    │ 1 │   X   │   X   │   X   │
    │ 2 │   X   │   X   │   X   │
    └───┴───────┴───────┴───────┘
    [3 rows x 3 columns]
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]          
    ### right column alignment ###    
    input_headers = ['1', '2', '3']
    input_data = [['X','X','X'],['X','X','X'],['X','X','X']]        
    sdm = SQLDataModel(data=input_data, headers=input_headers, column_alignment='right', min_column_width=5, max_column_width=5, display_index=True, display_float_precision=4, display_max_rows=4)
    output_repr = sdm.__repr__()
    baseline_repr = """
    ┌───┬───────┬───────┬───────┐
    │   │     1 │     2 │     3 │
    ├───┼───────┼───────┼───────┤
    │ 0 │     X │     X │     X │
    │ 1 │     X │     X │     X │
    │ 2 │     X │     X │     X │
    └───┴───────┴───────┴───────┘
    [3 rows x 3 columns]
    """
    output_repr_lines = [x.strip() for x in output_repr.strip().splitlines()]
    baseline_repr_lines = [x.strip() for x in baseline_repr.strip().splitlines()]
    for i in range(len(baseline_repr_lines)):
        assert output_repr_lines[i] == baseline_repr_lines[i]              

@pytest.mark.core
def test_to_from_csv():
    ### test from csv file ###
    input_data = [('c1','c2','c3'),('r0-c1', 'r0-c2', 'r0-c3'),('r1-c1', 'r1-c2', 'r1-c3'),('r2-c1', 'r2-c2', 'r2-c3')]
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
            csv_file = temp_file.name
            csvwriter = csv.writer(temp_file)
            csvwriter.writerows(input_data)            
            temp_file.close()
            sdm = SQLDataModel.from_csv(csv_file)
    finally:
        os.unlink(csv_file)
    output_data = sdm.data(include_headers=True)
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i] 
    ### test from delimited raw csv literal ###
    delimited_csv_literal = """c1,c2,c3\nr0-c1, r0-c2, r0-c3\nr1-c1, r1-c2, r1-c3\nr2-c1, r2-c2, r2-c3"""
    sdm = SQLDataModel.from_csv(delimited_csv_literal)
    input_data = [tuple(x.split(',')) for x in delimited_csv_literal.split('\n')]
    output_data = sdm.data(include_headers=True)
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]     

@pytest.mark.core
def test_dtypes():
    input_data, input_headers = [['abcdefg', 12_345, b'bytes', 3.14159, datetime.date(1992,11,22), datetime.datetime(1978,1,3,7,23,59)]], ['strings','integers','binary','floats','date','datetime']
    input_dtypes = {input_headers[i]:type(input_data[0][i]).__name__ for i in range(len(input_headers))}
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    output_dtypes = sdm.get_column_dtypes()
    assert input_dtypes == output_dtypes
    input_row = input_data[0]
    output_row = sdm[0,:].data()
    for i in range(len(input_row)):
        assert type(input_row[i]) == type(output_row[i])

def test_to_from_pandas():
    input_headers, input_data = ['A','B','C','D'], [(1, 'foo', 4.5, datetime.date(1999, 11, 9)),(2, 'bar', 6.7, datetime.date(2024, 8, 24)),(3, 'baz', 8.9, datetime.date(1985, 1, 13))]
    df_in = pd.DataFrame(data=input_data,columns=input_headers)
    df_out = SQLDataModel.from_pandas(df_in).to_pandas()
    for i in range(len(df_in.index)):
        assert df_in.iloc[i].tolist() == df_out.iloc[i].tolist()     

def test_to_from_numpy():
    input_data = [('1', 'foo', '4.5', '1999-11-09'),('2', 'bar', '6.7', '2024-08-24'),('3', 'baz', '8.9', '1985-01-13')]
    sdm = SQLDataModel(input_data)
    output_data = SQLDataModel.from_numpy(sdm.to_numpy()).data()
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]    