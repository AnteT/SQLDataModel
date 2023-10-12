# SQLDataModel

SQLDataModel is a speedy & lightweight data model with no external dependencies for quickly fetching and storing your tabular data

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package when its uploaded.

```bash
pip install SQLDataModel
```

## Usage

```python
import SQLDataModel

# create a SQLDataModel object from any valid source:
sdm = SQLDataModel.from_csv('world-cup-2022.csv')

# manipulate it
sdm_sliced = sdm.get_rows_at_index_range(1,4)

# loop through it:
for row in sdm_full.iter_rows():
    print(row)

# view it as a table
print(sdm)
```

|    | team        | rank | federation |
|----|-------------|------|------------|
| 1  | Argentina   | 3    | CONMEBOL   |
| 2  | Brazil      | 1    | CONMEBOL   |
| 3  | Ecuador     | 44   | CONMEBOL   |
| 4  | Uruguay     | 14   | CONMEBOL   |
| 5  | Belgium     | 2    | UEFA       |
| 6  | Croatia     | 12   | UEFA       |
| 7  | Denmark     | 10   | UEFA       |
| 8  | England     | 5    | UEFA       |
| 9  | France      | 4    | UEFA       |
| 10 | Germany     | 11   | UEFA       |
| 11 | Netherlands | 8    | UEFA       |
| 12 | Poland      | 26   | UEFA       |
| 13 | Portugal    | 9    | UEFA       |
| 14 | Serbia      | 21   | UEFA       |
| 15 | Spain       | 7    | UEFA       |
| 16 | Switzerland | 15   | UEFA       |
| 17 | Wales       | 19   | UEFA       |

```python
# group by columns:
print(sdm.group_by('federation'))
```

|   | federation | count |
|---|------------|-------|
| 1 | UEFA       | 13    |
| 2 | CONMEBOL   | 4     |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
|   |            |       |
```python
# group by columns:
print(sdm.group_by('federation'))
```

|   | federation | count |
|---|------------|-------|
| 1 | UEFA       | 13    |
| 2 | CONMEBOL   | 4     |

```python
# or save it for later as csv:
sdm.to_csv('world_cup_22.csv')

# or to sqlite database:
sdm.to_sql('world_cup_22', 'sqlite.db')

# and get it back again as a new model:
sdm_new = SQLDataModel.from_sql('select * from world_cup_22', 'sqlite.db')

```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)