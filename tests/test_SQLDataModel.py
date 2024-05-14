import datetime, os, tempfile, csv, sqlite3, random
import pytest
import pandas as pd
import polars as pl
import numpy as np
from .random_data_generator import data_generator
from src.SQLDataModel.SQLDataModel import SQLDataModel

@pytest.fixture
def sample_data() -> tuple[list[list], list[str]]:
    """Returns sample data in format `(data, headers)` to use for testing."""
    return data_generator(num_rows=120, float_precision=4, num_columns=8, seed=42, return_header_type='list', return_row_type='tuple', return_format='combined')

@pytest.fixture
def sample_data_parameterized() -> tuple[list[list], list[str]]:
    """Returns sample data from data_generator function with pass through parameters for customizing output for testing."""
    def _data_generator(num_rows=12, float_precision=2, num_columns=8, seed=42, return_header_type='list', return_row_type='tuple', return_format='separate', exclude_nonetype=True):
        return data_generator(num_rows=num_rows, float_precision=float_precision, num_columns=num_columns, seed=seed, return_header_type=return_header_type, return_row_type=return_row_type, return_format=return_format, exclude_nonetype=exclude_nonetype)
    return _data_generator

@pytest.mark.core
def test_init(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    input_rows, input_cols = len(input_data), len(input_data[0])
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    assert sdm.row_count == input_rows and sdm.column_count == input_cols
    assert sdm.headers == input_headers

@pytest.mark.core
def test_init_dict(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    ### dict with rows orientation ###
    input_rows_dict = {i:list(row) for i, row in enumerate(input_data)}
    sdm = SQLDataModel(data=input_rows_dict, headers=input_headers)
    output_data, output_headers = sdm.data(), sdm.headers
    assert output_headers == input_headers
    assert output_data == input_data
    ### dict with columns orientation ###
    input_cols_dict = {f'{col}': [row[idx] for row in input_data] for idx,col in enumerate(input_headers)}
    sdm = SQLDataModel(data=input_cols_dict, headers=input_headers)
    output_data, output_headers = sdm.data(), sdm.headers
    assert output_headers == input_headers
    assert output_data == input_data    
    ### json like list of dicts ###
    input_list_dict = [{col:row[i] for i,col in enumerate(input_headers)} for row in input_data]    
    sdm = SQLDataModel(data=input_list_dict, headers=input_headers)
    output_data, output_headers = sdm.data(), sdm.headers
    assert output_headers == input_headers
    assert output_data == input_data        

@pytest.mark.core
def test_data(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    out_data = sdm.data()
    assert all([input_data[i][j] == out_data[i][j] for j in range(len(input_data[0])) for i in range(len(input_data))])

@pytest.mark.core
def test_getitem():
    input_data, input_headers = [tuple([f"{x},{y}" for y in range(10)]) for x in range(10)], ['0','1','2','3','4','5','6','7','8','9']
    sdm = SQLDataModel(input_data, input_headers)
    rand_row, rand_col = random.randint(0,9), random.randint(0,9)
    rand_row_slice, rand_col_slice = slice(random.randint(0,4),random.randint(6,9)), slice(random.randint(0,4),random.randint(6,9))
    rand_row_set = set([random.randint(0,9) for i in range(random.randint(2,9))])
    rand_col_list = list(set([random.randint(0,9) for i in range(random.randint(2,9))]))
    ### row indexing ###
    assert sdm[rand_row].data() == input_data[rand_row] # row index by int
    assert sdm[rand_row_slice].data() == input_data[rand_row_slice] # row index by slice
    assert sdm[rand_row_set].data() == [input_data[i] for i in sorted(rand_row_set)] # discontiguous row indicies by set
    ### column indexing ###
    assert sdm[str(rand_col)].data() == [(row[rand_col],) for row in input_data] # single column str index
    assert sdm[str(rand_col)].data() == [(row[rand_col],) for row in input_data] # multiple column str indicies
    assert sdm[[str(x) for x in rand_col_list]].data() == [tuple([row[i] for i in rand_col_list]) for row in input_data] # discontiguous row indicies by set
    ### row and column indexing ###
    assert sdm[rand_row,rand_col].data() == input_data[rand_row][rand_col] # single row, single col
    assert sdm[rand_row_slice,str(rand_col)].data() == [(input_data[i][rand_col],) for i in range(rand_row_slice.start,rand_row_slice.stop)] # row slice, single column
    assert sdm[rand_row,rand_col_slice].data() == tuple([input_data[rand_row][i] for i in range(rand_col_slice.start,rand_col_slice.stop)]) # single row, column slice

@pytest.mark.core
def test_setitem():
    """Tests the `__setitem__()` method by using all possible type combinations `row, column` indexing and confirms the expected output."""
    grid_size = 10 # creates grid as (grid_size x grid_size)
    sdm = SQLDataModel([[f"F" for _ in range(grid_size)] for _ in range(grid_size)]) # create the grid canvas
    fill_char = 'X' # values to fill interior with
    test_corners = [('top left', 'top right'),('bottom left', 'bottom right')] # corner values
    test_indicies_and_values = [
        [(0), tuple([f'0,{i}' for i in range(sdm.column_count)])] # rowwise updates
        ,['0', [(f'{i},0',) for i in range(sdm.row_count)]] # columnwise updates
        ,[(grid_size), tuple([f'{grid_size},{i}' for i in range(sdm.column_count)])] # new row
        ,[f'{grid_size}', [(f'{i},{grid_size}',) for i in range(sdm.row_count + 1)]] # new column
        ,[(slice(1,-1),slice(1,-1)), [tuple([fill_char for _ in range(sdm.column_count-1)]) for _ in range(sdm.row_count-1)]] # new interior values
        ,[(0,0), test_corners[0][0]] # top left
        ,[(0,-1), test_corners[0][-1]] # top right
        ,[(-1,0), test_corners[-1][0]] # bottom left
        ,[(-1,-1),test_corners[-1][-1]] # bottom right
    ]
    for t_pair in test_indicies_and_values:
        t_idx, t_val = t_pair
        sdm[t_idx] = t_val
        assert t_val == sdm[t_idx].data()
    num_checked, output_data = 0, sdm.data()
    for x in range(grid_size + 1):
        for y in range(grid_size + 1):
            s_val = output_data[x][y]
            if grid_size > x > 0 and 0 < y < grid_size:
                assert s_val == fill_char
                num_checked += 1
                continue
            if x in (0,grid_size) and y in (0,grid_size):
                floor_it = lambda x: x if x < 1 else 1
                x, y = floor_it(x), floor_it(y)
                assert s_val == test_corners[x][y]
                num_checked += 1
            else:
                assert s_val == f"{x},{y}"
                num_checked += 1
    assert num_checked == (sdm.row_count * sdm.column_count)

@pytest.mark.core
def test_addition():
    data = [(f"{i}", i, i*1.0) for i in range(1,11)]
    expected_output = [(f"{row[0]}x", row[1]+1, row[2]+0.1, row[1]+1 + row[2]+0.1) for row in data]
    headers = ['str', 'int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['str concat'] = sdm['str'] + 'x'
    sdm['int scalar'] = sdm['int'] + 1
    sdm['float scalar'] = sdm['float'] + 0.1
    sdm['vector'] = sdm['int scalar'] + sdm['float scalar']
    model_output = sdm[:,[3,4,5,6]].data()
    assert model_output == expected_output

@pytest.mark.core
def test_subtraction():
    data = [(i, i*1.0) for i in range(1,11)]
    expected_output = [(row[0]-1, row[1]-0.1, (row[1]-0.1) - (row[0]-1)) for row in data]
    headers = ['int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['int scalar'] = sdm['int'] - 1
    sdm['float scalar'] = sdm['float'] - 0.1
    sdm['vector'] = sdm['float scalar'] - sdm['int scalar']
    model_output = sdm[:,[2,3,4]].data()
    assert model_output == expected_output    

@pytest.mark.core
def test_multiplication():
    data = [(i, i*1.0) for i in range(1,11)]
    expected_output = [(row[0] * 2, row[1]*3.0, (row[1]*3.0) * (row[0] * 2)) for row in data]
    headers = ['int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['int scalar'] = sdm['int'] * 2
    sdm['float scalar'] = sdm['float']*3.0
    sdm['vector'] = sdm['float scalar'] * sdm['int scalar']
    model_output = sdm[:,[2,3,4]].data()
    assert model_output == expected_output    

@pytest.mark.core
def test_division():
    data = [(i, i*1.0) for i in range(1,11)]
    expected_output = [(row[0] / 2, row[1]/3.0, (row[1]/3.0) / (row[0] / 2)) for row in data]
    headers = ['int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['int scalar'] = sdm['int'] / 2
    sdm['float scalar'] = sdm['float']/3.0
    sdm['vector'] = sdm['float scalar'] / sdm['int scalar']
    model_output = sdm[:,[2,3,4]].data()
    assert model_output == expected_output    

@pytest.mark.core
def test_floor_division():
    data = [(i*2+10, i*1.0) for i in range(1,11)]
    expected_output = [(row[0] // 3, row[1]//2.0, (row[0]//row[1])) for row in data]
    headers = ['int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['int scalar'] = sdm['int'] // 3
    sdm['float scalar'] = sdm['float']//2.0
    sdm['vector'] = sdm['int'] // sdm['float']
    model_output = sdm[:,[2,3,4]].data()
    assert model_output == expected_output  

@pytest.mark.core
def test_exponentiation():
    data = [(i, i*1.0) for i in range(1,11)]
    expected_output = [(row[0] **2, row[1]**1.5, (row[0]**row[1])) for row in data]
    headers = ['int', 'float']
    sdm = SQLDataModel(data, headers, display_index=False)
    sdm['int scalar'] = sdm['int'] **2
    sdm['float scalar'] = sdm['float']**1.5
    sdm['vector'] = sdm['int'] ** sdm['float']
    model_output = sdm[:,[2,3,4]].data()
    assert model_output == expected_output 

@pytest.mark.core
def test_headers(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
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
def test_py_sql_dtypes():
    input_data, input_headers = [['abcdefg', 12_345, b'bytes', 3.14159, datetime.date(1992,11,22), datetime.datetime(1978,1,3,7,23,59)]], ['strings','integers','binary','floats','date','datetime']
    input_dtypes = {input_headers[i]:type(input_data[0][i]).__name__ for i in range(len(input_headers))}
    sdm = SQLDataModel(data=input_data, headers=input_headers)
    output_dtypes = sdm.get_column_dtypes()
    assert input_dtypes == output_dtypes
    input_row = input_data[0]
    output_row = sdm[0,:].data()
    for i in range(len(input_row)):
        assert type(input_row[i]) == type(output_row[i])

@pytest.mark.core
def test_set_display_properties(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
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
def test_strip():
    input_data = [[f" A_{i}", f" B_{i}", f" C_{i}"] for i in range(1,4)]
    expected_output = [tuple(x.strip() for x in row) for row in input_data]
    # Test stripping whitespace and returning new model
    sdm = SQLDataModel(input_data)
    output_data = sdm.strip(characters=None,inplace=False).data()
    # Test stripping whitespace inplace
    assert output_data == expected_output
    sdm.strip(characters=None,inplace=True)
    output_data = sdm.data()
    assert output_data == expected_output
    # Test stripping different characters and returning as new (now modified from first test)
    expected_output = [tuple(x.strip().strip('_2') for x in row) for row in input_data]
    output_data = sdm.strip('_2', inplace=False).data()
    assert output_data == expected_output
    # Test stripping different characters inplace
    sdm.strip('_2', inplace=True)
    output_data = sdm.data()
    assert output_data == expected_output    

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
def test_set_column_dtypes():
    correct_types = ['str','int','float','bool','date','bytes','None','datetime']
    output_types = []
    data, headers = [[1,'2','3.3','True','1989-11-09',"b'blob-like'",'None','2022-01-01 12:34:56']], ['str','int','float','bool','date','bytes','None','datetime'] # use headers to signify which type columns should be set to
    sdm = SQLDataModel(data,headers,infer_types=False)
    for i, name_type in enumerate(headers):
        sdm.set_column_dtypes(i, name_type)
        val = sdm[0,name_type].data() if name_type != 'bool' else bool(sdm[0,name_type].data())
        output_types.append(type(val).__name__ if type(val).__name__ != 'NoneType' else 'None')
    assert output_types == correct_types

@pytest.mark.core
def test_infer_types_from_data(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    input_types = [type(x).__name__ if type(x).__name__ != 'NoneType' else 'None' for x in input_data[0]]
    stringified_data = [[str(x) for x in row] for row in input_data]
    inferred_types = SQLDataModel.infer_types_from_data(stringified_data)
    assert inferred_types == input_types


@pytest.mark.core
def test_infer_types_init(sample_data):
    typed_input, headers_input = sample_data[1:], tuple(sample_data[0])
    string_input = [[str(x) for x in row] for row in typed_input]
    sdm = SQLDataModel(string_input,headers_input,infer_types=True)
    inferred_output = sdm.data(include_headers=True)
    inferred_output, headers_output = inferred_output[1:], inferred_output[0]
    assert headers_input == headers_output
    for i in range(len(typed_input)):
        assert typed_input[i] == inferred_output[i]

@pytest.mark.core
def test_infer_types_post_init(sample_data):
    typed_input, headers_input = sample_data[1:], tuple(sample_data[0])
    string_input = [[str(x) for x in row] for row in typed_input]
    sdm = SQLDataModel(string_input,headers_input,infer_types=False)
    sdm.infer_dtypes(n_samples=16)
    inferred_output = sdm.data(include_headers=True)
    inferred_output, headers_output = inferred_output[1:], inferred_output[0]
    assert headers_input == headers_output
    for i in range(len(typed_input)):
        assert typed_input[i] == inferred_output[i]

@pytest.mark.core
def test_from_shape(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    row_lower, row_upper = 1, len(input_data)
    col_lower, col_upper = 1, len(input_headers)
    rand_row = random.randint(row_lower, row_upper)
    rand_col = random.randint(col_lower, col_upper)
    input_shape = (rand_row, rand_col)
    sdm = SQLDataModel.from_shape(shape=input_shape, fill=None, dtype=None)
    output_shape = sdm.shape
    assert output_shape == input_shape
    test_fills = (b'bytes', None, 'strings', 12, 3.14, datetime.date(1999,12,31), datetime.datetime(1999, 12, 31, 23, 59, 59), True)
    for input_fill in test_fills:
        expected_output = [tuple([input_fill for _ in range(rand_col)]) for _ in range(rand_row)]
        output_data = SQLDataModel.from_shape(shape=input_shape, fill=input_fill, display_float_precision=2).data(strict_2d=True)
        assert output_data == expected_output

@pytest.mark.core
def test_to_from_dict():
    input_dict = {0: ('hTpigTHKcfoK', 285, -497.17176, 0, datetime.date(1914, 6, 27), b'zPx3Bp', None, datetime.datetime(1985, 11, 10, 14, 20, 59)), 1: ('mNHnKXaXQv', -673, 106.451792, 1, datetime.date(2003, 5, 8), b'vKo', None, datetime.datetime(1996, 2, 1, 14, 39, 36)), 2: ('nVgx', 622, 884.861723, 1, datetime.date(1907, 4, 19), b'aRdeb', None, datetime.datetime(1912, 2, 18, 6, 32, 16)), 3: ('0LXSG8x', 393, 360.566821, 0, datetime.date(2021, 2, 3), b'eoPni5I', None, datetime.datetime(1916, 6, 3, 7, 23, 18)), 4: ('sM2wmeOV90', -136, -770.896514, 1, datetime.date(1993, 8, 27), b'pCzfOwz1d', None, datetime.datetime(1920, 8, 27, 17, 45, 19)), 5: ('xBZ', 221, 769.5769, 1, datetime.date(1908, 9, 25), b'9gzv1plB_rp5', None, datetime.datetime(1978, 11, 17, 0, 42, 52)), 6: ('xnq6', -870, 501.755599, 0, datetime.date(1916, 3, 22), b'X0gOHafUo', None, datetime.datetime(1970, 5, 22, 3, 56, 8)), 7: ('eNGpCr5QnuVd', -212, 537.197465, 1, datetime.date(1960, 9, 6), b'Dnzdx9qW', None, datetime.datetime(1933, 2, 4, 23, 35, 9)), 8: ('Xz', -219, -319.649054, 1, datetime.date(1933, 9, 28), b'rQTWODlnd', None, datetime.datetime(1934, 5, 20, 6, 45, 21)), 9: ('n62', 220, -412.999743, 0, datetime.date(1977, 7, 7), b'dtdD4GdE0e', None, datetime.datetime(1926, 11, 21, 8, 32, 31)), 10: ('nE2NjiT', -42, -683.684491, 0, datetime.date(2018, 9, 25), b'Poh', None, datetime.datetime(1932, 1, 3, 20, 27, 53)), 11: ('qJ3zn9Ffcg', 83, -993.509368, 1, datetime.date(1993, 12, 7), b'pKM_JF5mSpy', None, datetime.datetime(1935, 1, 1, 10, 49, 8))}
    output_dict = SQLDataModel.from_dict(input_dict).to_dict()
    for key in input_dict.keys():
        assert input_dict[key] == output_dict[key]

@pytest.mark.core
def test_to_from_json():
    input_json = '''[{"idx": 0, "strings": "hTpigTHKcfoK", "ints": "285", "floats": "-497.17176", "bools": "0", "dates": "1914-06-27", "bytes": "zPx3Bp", "nones": null, "datetimes": "1985-11-10 14:20:59"}, {"idx": 1, "strings": "mNHnKXaXQv", "ints": "-673", "floats": "106.451792", "bools": "1", "dates": "2003-05-08", "bytes": "vKo", "nones": null, "datetimes": "1996-02-01 14:39:36"}, {"idx": 2, "strings": "nVgx", "ints": "622", "floats": "884.861723", "bools": "1", "dates": "1907-04-19", "bytes": "aRdeb", "nones": null, "datetimes": "1912-02-18 06:32:16"}, {"idx": 3, "strings": "0LXSG8x", "ints": "393", "floats": "360.566821", "bools": "0", "dates": "2021-02-03", "bytes": "eoPni5I", "nones": null, "datetimes": "1916-06-03 07:23:18"}, {"idx": 4, "strings": "sM2wmeOV90", "ints": "-136", "floats": "-770.896514", "bools": "1", "dates": "1993-08-27", "bytes": "pCzfOwz1d", "nones": null, "datetimes": "1920-08-27 17:45:19"}, {"idx": 5, "strings": "xBZ", "ints": "221", "floats": "769.5769", "bools": "1", "dates": "1908-09-25", "bytes": "9gzv1plB_rp5", "nones": null, "datetimes": "1978-11-17 00:42:52"}, {"idx": 6, "strings": "xnq6", "ints": "-870", "floats": "501.755599", "bools": "0", "dates": "1916-03-22", "bytes": "X0gOHafUo", "nones": null, "datetimes": "1970-05-22 03:56:08"}, {"idx": 7, "strings": "eNGpCr5QnuVd", "ints": "-212", "floats": "537.197465", "bools": "1", "dates": "1960-09-06", "bytes": "Dnzdx9qW", "nones": null, "datetimes": "1933-02-04 23:35:09"}, {"idx": 8, "strings": "Xz", "ints": "-219", "floats": "-319.649054", "bools": "1", "dates": "1933-09-28", "bytes": "rQTWODlnd", "nones": null, "datetimes": "1934-05-20 06:45:21"}, {"idx": 9, "strings": "n62", "ints": "220", "floats": "-412.999743", "bools": "0", "dates": "1977-07-07", "bytes": "dtdD4GdE0e", "nones": null, "datetimes": "1926-11-21 08:32:31"}, {"idx": 10, "strings": "nE2NjiT", "ints": "-42", "floats": "-683.684491", "bools": "0", "dates": "2018-09-25", "bytes": "Poh", "nones": null, "datetimes": "1932-01-03 20:27:53"}, {"idx": 11, "strings": "qJ3zn9Ffcg", "ints": "83", "floats": "-993.509368", "bools": "1", "dates": "1993-12-07", "bytes": "pKM_JF5mSpy", "nones": null, "datetimes": "1935-01-01 10:49:08"}]'''
    output_json = SQLDataModel.from_json(input_json).to_json()
    for i in range(len(input_json)):
        assert input_json[i] == output_json[i]

@pytest.mark.ext
def test_to_from_excel(sample_data):
    input_data, input_headers = [tuple(str(x) if x is not None else None for x in row) for row in sample_data[1:]], tuple(sample_data[0])
    sdm = SQLDataModel(input_data,input_headers)
    excel_file = f'{os.getcwd()}\\tmp.xlsx'
    sdm.to_excel(excel_file)
    sdm = SQLDataModel.from_excel(excel_file)
    os.remove(excel_file)
    output_data = sdm.data(include_headers=True)
    output_data, output_headers = output_data[1:], output_data[0]
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]

@pytest.mark.ext
def test_to_from_pyarrow(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel.from_pyarrow(SQLDataModel(input_data, input_headers).to_pyarrow())
    output_data, output_headers = sdm.data(), sdm.get_headers()
    assert output_headers == input_headers
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i]    

@pytest.mark.ext
def test_to_from_parquet(sample_data):
    input_data, input_headers = sample_data[1:], tuple(sample_data[0])
    sdm = SQLDataModel(input_data,input_headers)
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            par_file = temp_file.name
            sdm.to_parquet(par_file)
            output_data = SQLDataModel.from_parquet(par_file).data(include_headers=True)
    finally:
        os.unlink(par_file)
    output_data, output_headers = output_data[1:], output_data[0]
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i]

@pytest.mark.ext
def test_to_from_pandas():
    input_headers, input_data = ['A','B','C','D'], [(1, 'foo', 4.5, datetime.date(1999, 11, 9)),(2, 'bar', 6.7, datetime.date(2024, 8, 24)),(3, 'baz', 8.9, datetime.date(1985, 1, 13))]
    df_in = pd.DataFrame(data=input_data,columns=input_headers)
    df_out = SQLDataModel.from_pandas(df_in).to_pandas()
    for i in range(len(df_in.index)):
        assert df_out.iloc[i].tolist() == df_in.iloc[i].tolist()     

@pytest.mark.ext
def test_to_from_polars():
    input_headers, input_data = ['A','B','C','D'], [(1, 'foo', 4.5, datetime.date(1999, 11, 9)),(2, 'bar', 6.7, datetime.date(2024, 8, 24)),(3, 'baz', 8.9, datetime.date(1985, 1, 13))]
    df_in = pl.DataFrame(data=input_data,schema=input_headers)
    df_out = SQLDataModel.from_polars(df_in).to_polars()
    output_data, output_headers = df_out.rows(), df_out.columns
    assert output_headers == input_headers
    assert output_data == input_data

@pytest.mark.ext
def test_to_from_numpy():
    input_data = [('1', 'foo', '4.5', '1999-11-09'),('2', 'bar', '6.7', '2024-08-24'),('3', 'baz', '8.9', '1985-01-13')]
    sdm = SQLDataModel(input_data)
    output_data = SQLDataModel.from_numpy(sdm.to_numpy()).data()
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i]
        
@pytest.mark.core
def test_to_from_pickle(sample_data):
    input_data, input_headers = sample_data[1:], tuple(sample_data[0])
    sdm = SQLDataModel(input_data, input_headers)
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            pkl_file = temp_file.name
            sdm.to_pickle(pkl_file)
            sdm = SQLDataModel.from_pickle(pkl_file)
            output_data = sdm.data(include_headers=True)
    finally:
        os.unlink(pkl_file)
    output_data, output_headers = output_data[1:], output_data[0]
    assert output_headers == input_headers
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i]

@pytest.mark.core
def test_to_from_text(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, input_headers, display_float_precision=4)
    output_data, output_headers = SQLDataModel.from_text(sdm.to_text(),infer_types=True).data(), sdm.get_headers()
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]

@pytest.mark.core
def test_to_from_markdown(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, input_headers, display_float_precision=4)
    output_data, output_headers = SQLDataModel.from_markdown(sdm.to_markdown(),infer_types=True).data(), sdm.get_headers()
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]

@pytest.mark.core
def test_to_from_latex(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, input_headers, display_float_precision=4)
    output_data, output_headers = SQLDataModel.from_latex(sdm.to_latex(),infer_types=True).data(), sdm.get_headers()
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]

@pytest.mark.core
def test_to_from_html(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, input_headers)
    output_data, output_headers = SQLDataModel.from_html(sdm.to_html(),infer_types=True).data(), sdm.get_headers()
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]

@pytest.mark.core
def test_to_from_sql(sample_data):
    input_data, input_headers = [tuple([int(j) if isinstance(j,bool) else j for j in i]) for i in sample_data[1:]], sample_data[0]
    tmp_db_conn = sqlite3.connect(":memory:")
    sdm = SQLDataModel(input_data, input_headers)
    sdm.to_sql('t2', tmp_db_conn)
    output_data = SQLDataModel.from_sql('t2', tmp_db_conn).data(include_headers=True)
    output_data, output_headers = output_data[1:], list(output_data[0])
    assert input_headers == output_headers
    for i in range(len(input_data)):
        assert input_data[i] == output_data[i]    

@pytest.mark.core
def test_to_from_csv(sample_data):
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
    input_literal = SQLDataModel(sample_data[1:], sample_data[0]).to_csv()
    output_literal = SQLDataModel.from_csv(input_literal).to_csv()
    assert input_literal == output_literal

@pytest.mark.core
def test_to_from_csv_delimiters(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data,input_headers)
    input_data = sdm.data(include_headers=True)
    valid_delimiters = (",","\t",";","|",":"," ")
    for delimiter in valid_delimiters:
        dsv = sdm.to_csv(delimiter=delimiter)
        sdm = SQLDataModel.from_csv(dsv, delimiter=delimiter)
        output_data = sdm.data(include_headers=True)
        assert input_data == output_data

@pytest.mark.core
def test_to_from_delimited_source(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data,input_headers)
    input_data = sdm.data(include_headers=True)
    valid_delimiters = (",","\t",";","|",":"," ")
    for delimiter in valid_delimiters:
        dsv = sdm.to_csv(delimiter=delimiter)
        sdm = SQLDataModel.from_delimited(dsv)
        output_data = sdm.data(include_headers=True)
        assert input_data == output_data

@pytest.mark.core
def test_append_row():
    num_rows = 12
    data_store = [tuple([f"{i}", i, float(i)]) for i in range(num_rows)] # str, int, float
    for rid in range(len(data_store)):
        input_data = data_store[:rid]
        sdm = SQLDataModel(headers=['A', 'B', 'C'], dtypes={'A':'str','B':'int','C':'float'})
        for input_row in input_data:
            sdm.append_row(input_row)
        output_data = sdm.data(strict_2d=True)
        assert output_data == input_data
    # test null append
    num_null_rows = 4
    null_row = (None, None, None)
    sdm = SQLDataModel(data_store, display_float_precision=1)
    for _ in range(num_null_rows):
        data_store.append(null_row)
        sdm.append_row(null_row)
        output_data = sdm.data(strict_2d=True)
        assert output_data == data_store 

@pytest.mark.core
def test_concat(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    mid = len(input_data) // 2
    base_data, other_data = input_data[:mid], input_data[mid:]
    base_sdm = SQLDataModel(base_data, headers=input_headers)    
    other_sdm = SQLDataModel(other_data, headers=input_headers)    
    # Concat other sdm returning new using inplace=False
    sdm = base_sdm.concat(other=other_sdm, inplace=False)
    output_data, output_headers = sdm.data(), sdm.get_headers()
    assert output_headers == input_headers
    assert output_data == input_data
    # Concat other sdm in place using inplace=True
    sdm = SQLDataModel(base_sdm.data(), headers=base_sdm.get_headers()) # clone to avoid contaminating base for later tests
    sdm.concat(other=other_sdm, inplace=True)
    output_data, output_headers = sdm.data(), sdm.get_headers()
    assert output_headers == input_headers
    assert output_data == input_data    
    # Concat list returning new using inplace=False
    other_list = tuple(None for _ in range(base_sdm.column_count))
    expected_data, expected_headers = [*base_data, other_list], input_headers
    sdm = base_sdm.concat(other=other_list, inplace=False)
    output_data, output_headers = sdm.data(), sdm.get_headers()
    assert output_headers == expected_headers
    assert output_data == expected_data
    # Concat list in place using inplace=True
    sdm = SQLDataModel(base_sdm.data(), headers=base_sdm.get_headers()) # clone to avoid contaminating base for later tests
    sdm.concat(other=other_list, inplace=True)
    output_data, output_headers = sdm.data(), sdm.get_headers()    
    assert output_headers == expected_headers
    assert output_data == expected_data

@pytest.mark.core
def test_copy(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm_original = SQLDataModel(input_data, headers=input_headers, display_float_precision=3, display_max_rows=7, display_index=False, column_alignment='right')
    # Test full copy
    sdm_copy_full = sdm_original.copy(data_only=False)  
    assert sdm_original.data(index=True,include_headers=True) == sdm_copy_full.data(index=True,include_headers=True)
    assert sdm_original._get_display_args() == sdm_copy_full._get_display_args()
    # Test data only copy
    sdm_copy_data = sdm_original.copy(data_only=True)  
    assert sdm_original.data(index=True,include_headers=True) == sdm_copy_data.data(index=True,include_headers=True)
    assert sdm_original._get_display_args() != sdm_copy_data._get_display_args()

@pytest.mark.core    
def test_head(sample_data):
    n_value = 24
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, headers=input_headers)
    expected_output = input_data[:n_value]
    output_data = sdm.head(n_rows=n_value).data()
    assert output_data == expected_output

@pytest.mark.core    
def test_insert_row():
    num_rows = 16
    data = [[f"{i}", i, float(i)] for i in range(num_rows)] # str, int, float
    sdm = SQLDataModel(data, display_float_precision=1)
    # test insert on_conflict = 'replace'
    rand_row = random.randint(0, (num_rows-2)) # leave room to add +2 to rand_row for ignore test and top and bottom insert new row tests
    input_data = tuple([f"{10+rand_row}", 10+rand_row, float(10+rand_row)])
    sdm.insert_row(rand_row, input_data, on_conflict='replace')
    output_data = sdm[rand_row].data()
    assert output_data == input_data
    # test insert on_conflict = 'ignore'
    rand_row = rand_row + 1
    input_fail = tuple([None, None, None])
    sdm.insert_row(rand_row, input_fail, on_conflict='ignore')
    output_data = sdm[rand_row].data()
    assert output_data != input_fail
    # test insert at top of model
    top_row = -1
    input_data = tuple([f"{top_row}", top_row, float(top_row)])
    sdm.insert_row(top_row, input_data, on_conflict='replace')
    output_data = sdm[0].data()
    assert output_data == input_data
    # test insert at bottom of model
    bottom_row = num_rows
    input_data = tuple([f"{bottom_row}", bottom_row, float(bottom_row)])
    sdm.insert_row(bottom_row, input_data, on_conflict='replace')
    output_data = sdm[-1].data()
    assert output_data == input_data
    # test metadata like shape and indicies
    expected_shape = (num_rows+2, 3)
    output_shape = sdm.shape
    print(expected_shape, output_shape)
    assert output_shape == expected_shape
    expected_indicies = tuple(range(-1, num_rows+1))
    output_indicies = sdm.indicies
    assert output_indicies == expected_indicies

@pytest.mark.core
def test_tail(sample_data):
    n_value = 24
    input_data, input_headers = sample_data[1:], sample_data[0]
    sdm = SQLDataModel(input_data, headers=input_headers)
    expected_output = input_data[-n_value:]
    output_data = sdm.tail(n_rows=n_value).data()
    assert output_data == expected_output    

@pytest.mark.core
def test_to_list(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    input_data = [list(row) for row in input_data]
    sdm = SQLDataModel(input_data, input_headers)
    ### all data: index=False, include_headers=False ###
    output_data = sdm.to_list(include_headers=False, index=False)
    assert output_data == input_data
    ### all data: index=False, include_headers=True ###
    output_data = sdm.to_list(include_headers=True, index=False)
    output_headers, output_data = output_data[0], output_data[1:]
    assert output_headers == input_headers
    assert output_data == input_data
    ### test each row ###
    for rid, row in enumerate(input_data):
        expected_output = [rid, *row]
        output_data = sdm[rid].to_list(index=True) # IMPORTANT: __getitem__ must be set to retain row index for this to work
        assert output_data == expected_output
    ### test each column ###
    for cid in range(len(input_headers)):
        expected_output = [row[cid] for row in input_data]
        output_data = sdm[:,cid].to_list(include_headers=False, index=False)
        assert output_data == expected_output    

@pytest.mark.core
def test_merge():
    left_headers = ["Name", "Age", "ID"]
    left_data = [        
        ["Bob", 35, 1],
        ["Alice", 30, 5],
        ["David", 40, None],
        ["Charlie", 25, 2]
    ]
    right_headers = ["ID", "Country"]
    right_data = [
        [1, "USA"],
        [2, "Germany"],
        [3, "France"],
        [4, "Latvia"]
    ]
    sdm_left = SQLDataModel(left_data, left_headers)
    sdm_right = SQLDataModel(right_data, right_headers)

    ### left join ###
    sdm_joined = sdm_left.merge(merge_with=sdm_right, how="left", left_on="ID", right_on="ID", include_join_column=False)
    joined_output = sdm_joined.data(include_headers=True)
    expected_output = [('Name', 'Age', 'ID', 'Country'), ('Bob', 35, 1, 'USA'), ('Alice', 30, 5, None), ('David', 40, None, None), ('Charlie', 25, 2, 'Germany')]
    assert joined_output == expected_output

    ### right join ###
    sdm_joined = sdm_left.merge(merge_with=sdm_right, how="right", left_on="ID", right_on="ID", include_join_column=False)
    joined_output = sdm_joined.data(include_headers=True)
    expected_output = [('Name', 'Age', 'ID', 'Country'), ('Bob', 35, 1, 'USA'), ('Charlie', 25, 2, 'Germany'), (None, None, None, 'France'), (None, None, None, 'Latvia')]      
    assert joined_output == expected_output

    ### inner join ###
    sdm_joined = sdm_left.merge(merge_with=sdm_right, how="inner", left_on="ID", right_on="ID", include_join_column=False)
    joined_output = sdm_joined.data(include_headers=True)
    expected_output = [('Name', 'Age', 'ID', 'Country'), ('Bob', 35, 1, 'USA'), ('Charlie', 25, 2, 'Germany')]
    assert joined_output == expected_output

    ### full outer join ###
    sdm_joined = sdm_left.merge(merge_with=sdm_right, how="full outer", left_on="ID", right_on="ID", include_join_column=False)
    joined_output = sdm_joined.data(include_headers=True)
    expected_output = [('Name', 'Age', 'ID', 'Country'), ('Bob', 35, 1, 'USA'), ('Alice', 30, 5, None), ('David', 40, None, None), ('Charlie', 25, 2, 'Germany'), (None, None, None, 'France'), (None, None, None, 'Latvia')]
    assert joined_output == expected_output

    ### cross join ###
    sdm_joined = sdm_left.merge(merge_with=sdm_right, how="cross", left_on="ID", right_on="ID", include_join_column=False)
    joined_output = sdm_joined.data(include_headers=True)
    expected_output = [('Name', 'Age', 'ID', 'Country'), ('Bob', 35, 1, 'USA'), ('Bob', 35, 1, 'Germany'), ('Bob', 35, 1, 'France'), ('Bob', 35, 1, 'Latvia'), ('Alice', 30, 5, 'USA'), ('Alice', 30, 5, 'Germany'), ('Alice', 30, 5, 'France'), ('Alice', 30, 5, 'Latvia'), ('David', 40, None, 'USA'), ('David', 40, None, 'Germany'), ('David', 40, None, 'France'), ('David', 40, None, 'Latvia'), ('Charlie', 25, 2, 'USA'), ('Charlie', 25, 2, 'Germany'), ('Charlie', 25, 2, 'France'), ('Charlie', 25, 2, 'Latvia')]    
    assert joined_output == expected_output

@pytest.mark.core
def test_sort(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]
    none_col_skip_idx = 6 # skip nonetype column for sort test
    sdm = SQLDataModel(input_data, headers=input_headers)
    ### test ascending and descending sort order
    sort_ordering = (True, False)
    for sort_order in sort_ordering:
        for j in range(len(input_data[0])):
            if j == none_col_skip_idx:
                continue
            expected_output = sorted([row[j] for row in input_data], reverse=(not sort_order))
            output_data = sdm.sort(by=j, asc=sort_order)[:,j].to_list()
            assert output_data == expected_output

@pytest.mark.core
def test_astype():
    sdm = SQLDataModel([['1111-11-11']], headers=['Value'])
    astype_dict = {'bool':int, 'bytes':bytes,'date':datetime.date, 'datetime':datetime.datetime, 'float':float, 'int':int, 'None':str, 'str':str}
    for type_name, expected_type in astype_dict.items():
        astype = sdm.astype(type_name).data()
        output_type = type(astype)
        assert output_type == expected_type

@pytest.mark.core
def test_drop_column(sample_data):
    input_data, input_headers = sample_data[1:], sample_data[0]    
    drop_column_args = [
            # Format: (column_arg, [validation data indicies], inplace_arg)
            ('date', [4], True),
            (['string','bool'], [0,3], True),
            (-1, [7], True),
            ([0,2,-3], [0,2,5], True),
            ('date', [4], False),
            (['string','bool'], [0,3], False),
            (-1, [7], False),
            ([0,2,-3], [0,2,5], False)  
        ]
    for drop_column, val_idx, inplace in drop_column_args:
        validation_data = [tuple([cell for i,cell in enumerate(row) if i not in val_idx]) for row in input_data]
        validation_headers = [col for j, col in enumerate(input_headers) if j not in val_idx]
        sdm = SQLDataModel(input_data,input_headers)
        if inplace:
            sdm.drop_column(drop_column, inplace=inplace)
        else:
            sdm = sdm.drop_column(drop_column, inplace=inplace)
        output_headers, output_data = sdm.get_headers(), sdm.data()
        assert output_headers == validation_headers
        assert output_data == validation_data

@pytest.mark.core
def test_count_min_max():
    headers = ['string', 'int', 'float', 'bool', 'date', 'bytes', 'nonetype', 'datetime']
    data = [
         ['hTpigTHKcfoK', -356, -470.239667, None, datetime.date(1967, 5, 18), b'sNLjVGWGaub5', None, datetime.datetime(1954, 3, 9, 14, 15, 55)]
        ,['mNHnKXaXQv', -565, -506.744985, True, datetime.date(2010, 1, 22), b'ppIeTY', None, datetime.datetime(2018, 2, 15, 17, 6, 3)]
        ,['nVgx', 342, None, True, datetime.date(1992, 2, 22), b'ViCHq1_nGS', None, datetime.datetime(1983, 9, 27, 0, 5, 59)]
        ,[None, 22, None, True, datetime.date(2013, 9, 25), b'RzPx3', None, None]
        ,['sM2wmeOV90', -190, 169.17198, True, datetime.date(1934, 11, 11), None, None, datetime.datetime(1927, 7, 2, 5, 24)]
        ,['xBZ', 811, None, False, datetime.date(1914, 5, 14), b'1e', None, datetime.datetime(1949, 5, 26, 14, 18, 27)]
        ,['xnq6', None, -201.19899, True, None, b'rDGS2', None, datetime.datetime(1989, 12, 26, 17, 42, 45)]
        ,['eNGpCr5QnuVd', 316, -561.358482, None, datetime.date(2022, 12, 24), b'CWXlgA_CSP9', None, datetime.datetime(1962, 3, 7, 9, 13, 3)]
        ,['Xz', -61, 995.075213, False, datetime.date(1933, 9, 25), None, None, datetime.datetime(1974, 12, 18, 1, 47, 20)]
        ,['n62', None, 19.052587, False, datetime.date(1922, 9, 4), b'plB', None, datetime.datetime(1907, 1, 19, 15, 32, 58)]
    ]
    expected_count = (9, 8, 7, 8, 9, 8, 0, 9)
    expected_min = ('Xz', -565, -561.358482, '0', datetime.date(1914, 5, 14), b'1e', None, datetime.datetime(1907, 1, 19, 15, 32, 58))
    expected_max = ('xnq6', 811, 995.075213, '1', datetime.date(2022, 12, 24), b'sNLjVGWGaub5', None, datetime.datetime(2018, 2, 15, 17, 6, 3))    
    sdm = SQLDataModel(data, headers)
    output_count = sdm.count().data()
    output_min = sdm.min().data()
    output_max = sdm.max().data()
    assert output_count == expected_count
    assert output_min == expected_min
    assert output_max == expected_max

@pytest.mark.core
def test_hstack():
    data_0, headers_0 = [('U_qF', -36, 62.04), ('4Z7w', 36, 80.43), ('ErUL', 80, -37.97)], ['string', 'int', 'float']
    data_1, headers_1 = [('IXdh', 84, 89.05), ('CxnD', -42, 80.29), ('AVaB', 51, -93.88)], ['string_2', 'int_2', 'float_2']
    data_2, headers_2 = [('fwkX', -5, 0.41), ('pncP', 39, 80.24), ('NX1F', 13, 74.21)], ['string_3', 'int_3', 'float_3']
    data_3, headers_3 = [('Ki4C', 63, 26.97), ('L4CH', -62, 73.61), ('9CZD', -41, 4.64)], ['string_4', 'int_4', 'float_4']
    sdm_0 = SQLDataModel(data=data_0, headers=headers_0, display_float_precision=2)
    sdm_1 = SQLDataModel(data=data_1, headers=headers_1, display_float_precision=2)
    sdm_2 = SQLDataModel(data=data_2, headers=headers_2, display_float_precision=2)
    sdm_3 = SQLDataModel(data=data_3, headers=headers_3, display_float_precision=2)
    ### single stack test returned as new model ###
    single_stack_test = sdm_0.hstack(sdm_1, inplace=False)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    expected_data, expected_headers = [(t1 + t2) for t1, t2 in zip(data_0, data_1)], [*headers_0, *headers_1]
    assert output_headers == expected_headers
    assert output_data == expected_data
    ### multiple stack test returned as new model ###
    multi_stack_test = sdm_0.hstack([sdm_1, sdm_2, sdm_3], inplace=False)
    output_data, output_headers = multi_stack_test.data(), multi_stack_test.get_headers()
    expected_data, expected_headers = [(t1 + t2 + t3 + t4) for t1, t2, t3, t4 in zip(data_0, data_1, data_2, data_3)], [*headers_0, *headers_1, *headers_2, *headers_3]
    assert output_headers == expected_headers
    assert output_data == expected_data    
    ### single stack inplace ###
    single_stack_test = SQLDataModel(sdm_0.data(), dtypes=sdm_0.get_column_dtypes()) # copy into new sdm to avoid tarnishing remaining tests
    single_stack_test.hstack(sdm_1, inplace=True)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    expected_data, expected_headers = [(t1 + t2) for t1, t2 in zip(data_0, data_1)], [*headers_0, *headers_1]
    assert output_headers == expected_headers
    assert output_data == expected_data
    ### multi stack inplace ###
    mult_stack_test = SQLDataModel(sdm_0.data(), dtypes=sdm_0.get_column_dtypes()) # copy into new sdm to avoid tarnishing remaining tests
    mult_stack_test.hstack([sdm_1, sdm_2, sdm_3], inplace=True)
    output_data, output_headers = multi_stack_test.data(), multi_stack_test.get_headers()
    expected_data, expected_headers = [(t1 + t2 + t3 + t4) for t1, t2, t3, t4 in zip(data_0, data_1, data_2, data_3)], [*headers_0, *headers_1, *headers_2, *headers_3]
    assert output_headers == expected_headers
    assert output_data == expected_data   
    ### test dimension coercion ###
    sdm_0[sdm_0.row_count] = ['new', 99, 3.1415]
    single_stack_test = sdm_0.hstack(sdm_1, inplace=False)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    data_0 += [('new', 99, 3.1415)] # simulate required padding for dim coercion
    data_1 += [(None, None, None)]
    expected_data, expected_headers = [(t1 + t2) for t1, t2 in zip(data_0, data_1)], [*headers_0, *headers_1]
    assert output_headers == expected_headers
    assert output_data == expected_data
    
@pytest.mark.core
def test_vstack():
    data_0, headers_0 = [('U_qF', -36, 62.04), ('4Z7w', 36, 80.43), ('ErUL', 80, -37.97)], ['string', 'int', 'float']
    data_1, headers_1 = [('IXdh', 84, 89.05), ('CxnD', -42, 80.29), ('AVaB', 51, -93.88)], ['string_2', 'int_2', 'float_2']
    data_2, headers_2 = [('fwkX', -5, 0.41), ('pncP', 39, 80.24), ('NX1F', 13, 74.21)], ['string_3', 'int_3', 'float_3']
    data_3, headers_3 = [('Ki4C', 63, 26.97), ('L4CH', -62, 73.61), ('9CZD', -41, 4.64)], ['string_4', 'int_4', 'float_4']
    sdm_0 = SQLDataModel(data=data_0, headers=headers_0, display_float_precision=2)
    sdm_1 = SQLDataModel(data=data_1, headers=headers_1, display_float_precision=2)
    sdm_2 = SQLDataModel(data=data_2, headers=headers_2, display_float_precision=2)
    sdm_3 = SQLDataModel(data=data_3, headers=headers_3, display_float_precision=2)
    ### single stack test returned as new model ###
    single_stack_test = sdm_0.vstack(sdm_1, inplace=False)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    expected_data, expected_headers = data_0 + data_1, headers_0
    assert output_headers == expected_headers
    assert output_data == expected_data
    ### multiple stack test returned as new model ###
    multi_stack_test = sdm_0.vstack([sdm_1, sdm_2, sdm_3], inplace=False)
    output_data, output_headers = multi_stack_test.data(), multi_stack_test.get_headers()
    expected_data, expected_headers = data_0 + data_1 + data_2 + data_3, headers_0
    assert output_headers == expected_headers
    assert output_data == expected_data  
    ### single stack inplace ###
    single_stack_test = SQLDataModel(sdm_0.data(), dtypes=sdm_0.get_column_dtypes()) # copy into new sdm to avoid tarnishing remaining tests
    single_stack_test.vstack(sdm_1, inplace=True)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    expected_data, expected_headers = data_0 + data_1, headers_0
    assert output_headers == expected_headers
    assert output_data == expected_data
    ### multi stack inplace ###
    mult_stack_test = SQLDataModel(sdm_0.data(), dtypes=sdm_0.get_column_dtypes()) # copy into new sdm to avoid tarnishing remaining tests
    mult_stack_test.vstack([sdm_1, sdm_2, sdm_3], inplace=True)
    output_data, output_headers = multi_stack_test.data(), multi_stack_test.get_headers()
    expected_data, expected_headers = data_0 + data_1 + data_2 + data_3, headers_0
    assert output_headers == expected_headers
    assert output_data == expected_data   
    ### test dimension coercion ###
    sdm_0['bytes'] = b'pad test'
    single_stack_test = sdm_0.vstack(sdm_1, inplace=False)
    output_data, output_headers = single_stack_test.data(), single_stack_test.get_headers()
    data_0 = [tuple([*row, b'pad test']) for row in data_0]
    data_1 = [tuple([*row, None]) for row in data_1]
    expected_data, expected_headers = data_0 + data_1, [*headers_0, 'bytes']
    assert output_headers == expected_headers
    assert output_data == expected_data    

@pytest.mark.core
def test_transpose():
    num_rows, num_cols = 10, 5
    input_data = [tuple(f"{i},{j}" for j in range(num_cols)) for i in range(num_rows)]
    output_data = SQLDataModel(input_data).transpose(infer_types=True,include_headers=False).transpose(infer_types=True,include_headers=False).data()
    assert output_data == input_data

@pytest.mark.core
def test_min(sample_data_parameterized):
    data, headers = sample_data_parameterized(exclude_nonetype=True)
    expected_data = tuple(min([row[j] for row in data]) for j in range(len(data[0])))
    output_data = SQLDataModel(data, headers).min().data()
    assert output_data == expected_data

@pytest.mark.core
def test_max(sample_data_parameterized):
    data, headers = sample_data_parameterized(exclude_nonetype=True)
    expected_data = tuple(max([row[j] for row in data]) for j in range(len(data[0])))
    output_data = SQLDataModel(data, headers).max().data()
    assert output_data == expected_data    

@pytest.mark.core
def test_mean():
    data = [('QNkrwWcsJnL', -633, 996.8432, 1, datetime.date(1962, 6, 7), b'3S7Z8ZcCeSc', datetime.datetime(1948, 10, 13, 9, 42, 47), 'waRB660S'), ('C5', -380, -585.4195, 1, datetime.date(1944, 9, 6), b'VceF78VXuWI', datetime.datetime(2020, 4, 23, 20, 14, 56), 'rABpuXdI'), ('FfeWNVq', -378, 575.8778, 0, datetime.date(1998, 3, 3), b'YyPoUKrPK', datetime.datetime(1958, 4, 18, 3, 11, 46), 'j2lb'), ('Xx', -744, 972.1331, 1, datetime.date(1994, 1, 19), b'PJyDr', datetime.datetime(2022, 12, 17, 14, 49, 10), 'KpWirYfv'), ('SuzgO', -627, 332.6614, 1, datetime.date(2011, 9, 14), b'b9', datetime.datetime(1922, 2, 25, 23, 23, 58), 'WM1VHoJ25zf'), ('69UEuh0', -954, -906.1676, 1, datetime.date(1980, 7, 26), b'PYesip__', datetime.datetime(1954, 8, 5, 4, 38, 33), 'jUWRBAyJXV')]    
    expected_data = ('NaN', -619.3333333333334, 230.98806666666667, 0.8333333333333334, datetime.date(1981, 12, 7), 'NaN', datetime.datetime(1971, 3, 8, 16, 40, 11), 'NaN')
    output_data = SQLDataModel(data).mean().data()
    assert output_data == expected_data

@pytest.mark.core
def test_table_styles():
    style_output_dict = {
'ascii':"""
+---+--------+-----+-------+------------+
|   | string | int | float | date       |
+---+--------+-----+-------+------------+
| 0 | text 1 |   1 |  1.10 | 2001-01-11 |
| 1 | text 2 |   2 |  2.20 | 2002-02-12 |
| 2 | text 3 |   3 |  3.30 | 2003-03-13 |
| 3 | text 4 |   4 |  4.40 | 2004-04-14 |
+---+--------+-----+-------+------------+
[4 rows x 4 columns]
""",
'bare':"""
   string  int  float  date      
---------------------------------
0  text 1    1   1.10  2001-01-11
1  text 2    2   2.20  2002-02-12
2  text 3    3   3.30  2003-03-13
3  text 4    4   4.40  2004-04-14
[4 rows x 4 columns]
""",
'dash':"""
┌───┬────────┬─────┬───────┬────────────┐
│   ╎ string ╎ int ╎ float ╎ date       │
├╴╴╴┼╴╴╴╴╴╴╴╴┼╴╴╴╴╴┼╴╴╴╴╴╴╴┼╴╴╴╴╴╴╴╴╴╴╴╴┤
│ 0 ╎ text 1 ╎   1 ╎  1.10 ╎ 2001-01-11 │
│ 1 ╎ text 2 ╎   2 ╎  2.20 ╎ 2002-02-12 │
│ 2 ╎ text 3 ╎   3 ╎  3.30 ╎ 2003-03-13 │
│ 3 ╎ text 4 ╎   4 ╎  4.40 ╎ 2004-04-14 │
└───┴────────┴─────┴───────┴────────────┘
[4 rows x 4 columns]
""",
'default':"""
┌───┬────────┬─────┬───────┬────────────┐
│   │ string │ int │ float │ date       │
├───┼────────┼─────┼───────┼────────────┤
│ 0 │ text 1 │   1 │  1.10 │ 2001-01-11 │
│ 1 │ text 2 │   2 │  2.20 │ 2002-02-12 │
│ 2 │ text 3 │   3 │  3.30 │ 2003-03-13 │
│ 3 │ text 4 │   4 │  4.40 │ 2004-04-14 │
└───┴────────┴─────┴───────┴────────────┘
[4 rows x 4 columns]
""",
'double':"""
╔═══╦════════╦═════╦═══════╦════════════╗
║   ║ string ║ int ║ float ║ date       ║
╠═══╬════════╬═════╬═══════╬════════════╣
║ 0 ║ text 1 ║   1 ║  1.10 ║ 2001-01-11 ║
║ 1 ║ text 2 ║   2 ║  2.20 ║ 2002-02-12 ║
║ 2 ║ text 3 ║   3 ║  3.30 ║ 2003-03-13 ║
║ 3 ║ text 4 ║   4 ║  4.40 ║ 2004-04-14 ║
╚═══╩════════╩═════╩═══════╩════════════╝
[4 rows x 4 columns]
""",
'list':"""
   string  int  float  date      
-  ------  ---  -----  ----------
0  text 1    1   1.10  2001-01-11
1  text 2    2   2.20  2002-02-12
2  text 3    3   3.30  2003-03-13
3  text 4    4   4.40  2004-04-14
[4 rows x 4 columns]
""",
'markdown':"""
|   | string | int | float | date       |
|---|--------|-----|-------|------------|
| 0 | text 1 |   1 |  1.10 | 2001-01-11 |
| 1 | text 2 |   2 |  2.20 | 2002-02-12 |
| 2 | text 3 |   3 |  3.30 | 2003-03-13 |
| 3 | text 4 |   4 |  4.40 | 2004-04-14 |
[4 rows x 4 columns]
""",
'outline':"""
┌───────────────────────────────────┐
│    string  int  float  date       │
├───────────────────────────────────┤
│ 0  text 1    1   1.10  2001-01-11 │
│ 1  text 2    2   2.20  2002-02-12 │
│ 2  text 3    3   3.30  2003-03-13 │
│ 3  text 4    4   4.40  2004-04-14 │
└───────────────────────────────────┘
[4 rows x 4 columns]
""",
'pandas':"""
   string  int  float  date      
0  text 1    1   1.10  2001-01-11
1  text 2    2   2.20  2002-02-12
2  text 3    3   3.30  2003-03-13
3  text 4    4   4.40  2004-04-14
[4 rows x 4 columns]
""",

'polars':"""
┌───┬────────┬─────┬───────┬────────────┐
│   ┆ string ┆ int ┆ float ┆ date       │
╞═══╪════════╪═════╪═══════╪════════════╡
│ 0 ┆ text 1 ┆   1 ┆  1.10 ┆ 2001-01-11 │
│ 1 ┆ text 2 ┆   2 ┆  2.20 ┆ 2002-02-12 │
│ 2 ┆ text 3 ┆   3 ┆  3.30 ┆ 2003-03-13 │
│ 3 ┆ text 4 ┆   4 ┆  4.40 ┆ 2004-04-14 │
└───┴────────┴─────┴───────┴────────────┘
[4 rows x 4 columns]
""",
'postgresql':"""
  | string | int | float | date      
--+--------+-----+-------+-----------
0 | text 1 |   1 |  1.10 | 2001-01-11
1 | text 2 |   2 |  2.20 | 2002-02-12
2 | text 3 |   3 |  3.30 | 2003-03-13
3 | text 4 |   4 |  4.40 | 2004-04-14
[4 rows x 4 columns]
""",
'round':"""
╭───┬────────┬─────┬───────┬────────────╮
│   │ string │ int │ float │ date       │
├───┼────────┼─────┼───────┼────────────┤
│ 0 │ text 1 │   1 │  1.10 │ 2001-01-11 │
│ 1 │ text 2 │   2 │  2.20 │ 2002-02-12 │
│ 2 │ text 3 │   3 │  3.30 │ 2003-03-13 │
│ 3 │ text 4 │   4 │  4.40 │ 2004-04-14 │
╰───┴────────┴─────┴───────┴────────────╯
[4 rows x 4 columns]
"""}
    headers = ['string', 'int', 'float', 'date']
    data = [
         ('text 1',  1, 1.1, datetime.date(2001, 1, 11))
        ,('text 2', 2, 2.2, datetime.date(2002, 2, 12))
        ,('text 3', 3, 3.3, datetime.date(2003, 3, 13))
        ,('text 4', 4, 4.4, datetime.date(2004, 4, 14))
    ]
    sdm = SQLDataModel(data,headers, min_column_width=3, max_column_width=38, display_index=True, display_float_precision=2)
    repr_styles = ['ascii','bare','dash','default','double','list','markdown','outline','pandas','polars','postgresql','round']   
    for style in repr_styles:
        sdm.set_table_style(style=style)
        expected_repr = style_output_dict[style].strip('\n')
        output_repr = sdm.__repr__()
        assert output_repr == expected_repr