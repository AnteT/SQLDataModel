import importlib.util
from .exceptions import ErrorFormat

try:
    _has_dateutil = importlib.util.find_spec("dateutil.parser") is not None
except:
    _has_dateutil = False
    
try:
    _has_np = importlib.util.find_spec("numpy") is not None
except:
    _has_np = False
    
try:
    _has_pd = importlib.util.find_spec("pandas") is not None
except:
    _has_pd = False
    
try:
    _has_pl = importlib.util.find_spec("polars") is not None
except:
    _has_pl = False
    
try:
    _has_pa = importlib.util.find_spec("pyarrow") is not None and importlib.util.find_spec("pyarrow.parquet") is not None
except:
    _has_pa = False
    
try:
    _has_xl = importlib.util.find_spec("openpyxl") is not None
except:
    _has_xl = False
    

def _get_dateparser():
    if not _has_dateutil:
        msg = ErrorFormat(f"ModuleNotFoundError: required package not found, python-dateutil must be installed in order to use advanced datetime parsing. Install directly or using optional flag `pip install sqldatamodel[dateutil]`")
        raise ModuleNotFoundError(msg) 
    from dateutil.parser import parse as dateparser
    return dateparser

def _get_np():
    if not _has_np:
        msg = ErrorFormat(f"""ModuleNotFoundError: required package not found, numpy must be installed in order to use the `from_numpy()` method. Install directly or using optional flag `pip install sqldatamodel[numpy]`""")
        raise ModuleNotFoundError(msg)
    import numpy as np
    return np

def _get_pd():
    if not _has_pd:
        msg = ErrorFormat(f"""ModuleNotFoundError: required package not found, pandas must be installed in order to use `.to_pandas()` method. Install directly or using optional flag `pip install sqldatamodel[pandas]`""")
        raise ModuleNotFoundError(msg)
    import pandas as pd
    return pd

def _get_pl():
    if not _has_pl:
        msg = ErrorFormat(f"""ModuleNotFoundError: required package not found, polars must be installed in order to use `.to_polars()` method. Install directly or using optional flag `pip install sqldatamodel[polars]`""")
        raise ModuleNotFoundError(msg)
    import polars as pl
    return pl

def _get_pa_pq():
    """Return (pyarrow, pyarrow.parquet). Raises an exception if either is missing."""
    if not _has_pa:
        msg = ErrorFormat(f"ModuleNotFoundError: required package not found, pyarrow must be installed in order to use `.to_parquet()` method. Install directly or using optional flag `pip install sqldatamodel[pyarrow]`")
        raise ModuleNotFoundError(msg)        
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq

def _get_xl():
    if not _has_xl:
        msg = ErrorFormat(f"ModuleNotFoundError: required package not found, `openpyxl` must be installed in order to use `from_excel()` method. Install directly or using optional flag `pip install sqldatamodel[openpyxl]`")
        raise ModuleNotFoundError(msg)
    import openpyxl as xl
    return xl
