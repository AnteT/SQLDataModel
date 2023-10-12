# SQLDataModel

SQLDataModel is a speedy & lightweight data model with no external dependencies for quickly fetching and storing your tabular data

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SQLDataModel.

```bash
python -m pip install SQLDataModel
```

## Usage

```python
from SQLDataModel import SQLDataModel

# create a SQLDataModel object from any valid source:
sdm = SQLDataModel.from_csv('data.csv')

# manipulate it
sdm_sliced = sdm[2:7]

# view it as a table
print(sdm)
```

|     | country | region    | check | total | report date         |
|-----|---------|-----------|-------|-------|---------------------|
| 1   | US      | West      | Yes   | 2016  | 2023-08-23 13:11:43 |
| 2   | US      | West      | No    | 1996  | 2023-08-23 13:11:43 |
| 3   | US      | West      | Yes   | 1296  | 2023-08-23 13:11:43 |
| 4   | US      | West      | No    | 2392  | 2023-08-23 13:11:43 |
| 5   | US      | Northeast | Yes   | 1233  | 2023-08-23 13:11:43 |
| 6   | US      | Northeast | No    | 3177  | 2023-08-23 13:11:43 |
| 7   | US      | Midwest   | Yes   | 1200  | 2023-08-23 13:11:43 |
| 8   | US      | Midwest   | No    | 2749  | 2023-08-23 13:11:43 |
| 9   | US      | Midwest   | Yes   | 1551  | 2023-08-23 13:11:43 |

```python
# group by columns:
sdm_group = sdm.group_by('region','check')
print(sdm_group)
```

| idx | region    | check | count |
|-----|-----------|-------|-------|
| 1   | Midwest   | Yes   | 2     |
| 2   | West      | No    | 2     |
| 3   | West      | Yes   | 2     |
| 4   | Midwest   | No    | 1     |
| 5   | Northeast | No    | 1     |
| 6   | Northeast | Yes   | 1     |

```python
# loop through it:
for row in sdm.iter_rows():
    print(row)

# or save it for later as csv:
sdm.to_csv('data.csv')

# or to sqlite database:
sdm.to_sql('data', 'sqlite.db')

# and get it back again as a new model:
sdm_new = SQLDataModel.from_sql('select * from data', 'sqlite.db')

```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
