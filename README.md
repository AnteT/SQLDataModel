<img src="figs/sdm_banner.PNG" src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_banner.PNG?raw=true" alt="SQLDataModel" style="width:100vw; border-radius: .6vw" />

---

### SQLDataModel: fast & lightweight source agnostic data model

![SQLDataModel Home](https://img.shields.io/badge/SQLDataModel-Home-blue?style=flat&logo=github&link=https%3A%2F%2Fgithub.com%2FAnteT%2FSQLDataModel&link=https%3A%2F%2Fgithub.com%2FAnteT%2FSQLDataModel)
![PyPI License](https://img.shields.io/pypi/l/sqldatamodel)
![PyPI Version](https://img.shields.io/pypi/v/sqldatamodel)
[![Docs Status](https://readthedocs.org/projects/sqldatamodel/badge/?version=latest)](https://sqldatamodel.readthedocs.io/en/latest/?badge=latest)

SQLDataModel is a fast & lightweight data model with no additional dependencies for quickly fetching and storing your tabular data to and from the most commonly used databases & data sources in a couple lines of code. It's as easy as ETL:

```python
from SQLDataModel import SQLDataModel

# Do the E part:
my_table = SQLDataModel.from_sql("your_table", cx_Oracle.Connection)

# Take care of your T business:
for row in my_table.iter_rows():
    print(row)

# Finish the L and be done:
my_table.to_sql("new_table", psycopg2.Connection)
```

Made for those times when you just want to use raw SQL on your dataframe, or need to move data around but the full Pandas, Numpy, SQLAlchemy installation is just overkill. SQLDataModel includes all the most commonly used features, including additional ones like pretty printing your table, at _1/1000_ the size, 0.03MB vs 30MB

---

### Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SQLDataModel.

```bash
$ pip install SQLDataModel
```
Then import the main class `SQLDataModel` into your local project, see [usage](#usage) below or go straight to the [project docs](https://sqldatamodel.readthedocs.io).

---

### Quick Example
A `SQLDataModel` can be created from any number of [sources](#data-sources), as a quick demo lets create one using a Wikipedia page:

```python
>>> from SQLDataModel import SQLDataModel
>>> 
>>> url = 'https://en.wikipedia.org/wiki/1998_FIFA_World_Cup'
>>> 
>>> sdm = SQLDataModel.from_html(url, table_identifier=94)   
>>> 
>>> sdm[:4, ['R', 'Team', 'W', 'Pts.']]
┌──────┬─────────────┬──────┬──────┐
│ R    │ Team        │ W    │ Pts. │
├──────┼─────────────┼──────┼──────┤
│ 1    │ France      │ 6    │ 19   │
│ 2    │ Brazil      │ 4    │ 13   │
│ 3    │ Croatia     │ 5    │ 15   │
│ 4    │ Netherlands │ 3    │ 12   │
└──────┴─────────────┴──────┴──────┘
[4 rows x 4 columns]
```

SQLDataModel provides a quick and easy way to import, view, transform and export your data in multiple [formats](#data-sources) and sources, providing the full power of executing raw SQL against your model in the process. 

---

### Usage

```python
from SQLDataModel import SQLDataModel

# Create a SQLDataModel object from any valid source, whether csv:
sdm = SQLDataModel.from_csv('region_data.csv')

# Any DB-API 2.0 connection like psycopg2, cx-oracle, pyodbc, sqlite3:
sdm = SQLDataModel.from_sql('region_data', psycopg2.Connection) 

# Python objects like dicts, lists, tuples, iterables:
sdm = SQLDataModel.from_dict(data=region_data)

# Slice it by rows and columns
sdm_country = sdm[2:7, ['country','total']]

# Transform and filter it
sdm = sdm[sdm['total'] < 3200]

# View it
print(sdm)
```
<img src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_purple.PNG?raw=true" alt="sdm_colorful_table" style="width:100vw; border-radius: .6vw" />

```python
# Group by single or multiple columns:
sdm_group = sdm.group_by(['region','check'])

# View output.
print(sdm_group)
```
<img src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_group.PNG?raw=true" alt="sdm_grouped_table" style="width:100vw; border-radius: .6vw" />

```python
# Loop through it.
for row in sdm.iter_rows():
    print(row)

# Save it for later as csv.
sdm.to_csv('region_data.csv')

# Or SQL databases like PostgreSQL, SQL Server, SQLite.
sdm.to_sql('table_data', sqlite3.Connection)

# Get it back again from any number of sources.
sdm_new = SQLDataModel.from_sql('table_data', sqlite3.Connection)
```

---

### Data Sources

SQLDataModel supports various data formats and sources, including:
- HTML files or websites
- SQL database connections (PostgreSQL, SQLite, Oracle DB, SQL Server, TeraData)
- CSV or delimited text files
- JSON files or objects
- LaTeX files or formatted strings
- Markdown files or formatted strings
- Numpy arrays
- Pandas dataframes
- Parquet files
- Python objects
- Pickle files

Note that `SQLDataModel` does not install any additional dependencies by default. This is done to keep the package as light-weight and small as possible. This means that to use package dependent methods like `to_parquet()` or the inverse `from_parquet()` the `pyarrow` package is required. The same goes for other package dependent methods like those converting to and from `pandas` and `numpy` objects.

---

### Documentation
SQLDataModel's documentation can be found at https://sqldatamodel.readthedocs.io containing detailed descriptions for the key modules in the package. These are listed below as links to their respective sections in the docs:

  * [`ANSIColor`](https://sqldatamodel.readthedocs.io/en/latest/SQLDataModel.html#SQLDataModel.ANSIColor.ANSIColor) for terminal styling.
  * [`HTMLParser`](https://sqldatamodel.readthedocs.io/en/latest/SQLDataModel.html#SQLDataModel.HTMLParser.HTMLParser) for parsing tabular data from the web.
  * [`JSONEncoder`](https://sqldatamodel.readthedocs.io/en/latest/SQLDataModel.html#SQLDataModel.JSONEncoder.DataTypesEncoder) for type casting and encoding JSON data.
  * [`SQLDataModel`](https://sqldatamodel.readthedocs.io/en/latest/SQLDataModel.html#SQLDataModel.SQLDataModel.SQLDataModel) for wrapping it all up.


However, to skip over the less relevant modules and jump straight to the meat of the package, the ``SQLDataModel`` module, click [here](https://sqldatamodel.readthedocs.io/en/latest/SQLDataModel.html#SQLDataModel.SQLDataModel.SQLDataModel). 

---

### Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

---

### License

[MIT](https://choosealicense.com/licenses/mit/)


Thank you!  
Ante Tonkovic-Capin