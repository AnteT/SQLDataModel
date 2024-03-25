<img src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_banner_cyan.PNG?raw=true" alt="SQLDataModel Logo" style="width:100vw; border-radius: .6vw" />

---

### SQLDataModel: fast & lightweight source agnostic data model

[![SQLDataModel Home](https://img.shields.io/badge/SQLDataModel-Home-blue?style=flat&logo=github)](https://github.com/AnteT/SQLDataModel)
[![PyPI License](https://img.shields.io/pypi/l/sqldatamodel)](https://pypi.org/project/SQLDataModel/)
[![PyPI Version](https://img.shields.io/pypi/v/sqldatamodel)](https://pypi.org/project/SQLDataModel/)
[![Docs Status](https://readthedocs.org/projects/sqldatamodel/badge/?version=latest)](https://sqldatamodel.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/sqldatamodel)](https://pepy.tech/project/sqldatamodel)

SQLDataModel is a fast & lightweight data model with no additional dependencies for quickly fetching and storing your tabular data to and from the most commonly used databases & data sources in a couple lines of code. It's as easy as ETL:

```python
from SQLDataModel import SQLDataModel

# Extract your data:
sdm = SQLDataModel.from_sql("your_table", cx_Oracle.Connection)

# Transform it:
for row in sdm.iter_rows():
    print(row)

# Load it wherever you need to!
sdm.to_sql("new_table", psycopg2.Connection)
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
A `SQLDataModel` can be created from any number of [sources](#data-formats), as a quick demo lets create one using a Wikipedia page:

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

SQLDataModel provides a quick and easy way to import, view, transform and export your data in multiple [formats](#data-formats) and sources, providing the full power of executing raw SQL against your model in the process. 

---

### Usage

`SQLDataModel` is a versatile data model leveraging the mighty power of in-memory `sqlite3` to perform fast and light-weight transformations allowing you to easily move and manipulate data from source to destination regardless of where, or in what format, the data is. If you need to extract, transform, or simply transfer data across [various formats](#data-formats), `SQLDataModel` can make your life easier. Here's a few examples how: 

1. [From Website to Markdown](#from-website-to-markdown)
2. [SQL on your Pandas DataFrame](#sql-on-your-pandas-dataframe)
3. [From SQL to HTML Table](#from-sql-to-html-table)
4. [Pretty PyTorch Training Results to JSON](#pretty-pytorch-training-results-to-json)
5. [From PostgreSQL to SQLite](#from-postgresql-to-sqlite)

---

#### From Website to Markdown
Say we find some cool data online, perhaps some planetary data, and we want to go and get it for our own purposes:

```python
from SQLDataModel import SQLDataModel

# Target url with some planetary data we can extract
url = 'https://antet.github.io/sdm-planets'

# Create a model from the first table element found on the web page
sdm = SQLDataModel.from_html(url, table_identifier=1)

# Add some color to it
sdm.set_display_color('#A6D7E8')

# Lets see!
print(sdm)
```
`SQLDataModel`'s default output is pretty printed and formatted to fit within the current terminal's width and height, since we added some color here's the result:

<img src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_planets.PNG?raw=true" alt="sdm_planets" style="width:100vw; border-radius: .6vw" />

Now that we have our data as a `SQLDataModel`, we can do any number of things with it using the provided methods or using your own SQL and returning the query results as a new model! Lets find out, all extreme temperatures and pressure aside, if it would be easier to fly given the planet's gravity relative to Earth:

```python
# Extract: Slice by rows and columns
sdm = sdm[:,['Planet','Gravity']] # or sdm[:,:2]

# Transform: Create a new column based on existing values
sdm['Flyable?'] = sdm['Gravity'].apply(lambda x: str(x < 1.0))

# Filter: Keep only the 'Flyable' planets
sdm = sdm[sdm['Flyable?'] == 'True']

# Load: Let's turn our data into markdown!
sdm.to_markdown('Planet-Flying.MD')
```
Here's the raw markdown of the new file we created `Planet-Flying.MD`:

```markdown
| Planet  | Gravity | Flyable? |
|:--------|--------:|:---------|
| Mercury |    0.38 | True     |
| Venus   |    0.91 | True     |
| Mars    |    0.38 | True     |
| Saturn  |    0.92 | True     |
| Uranus  |    0.89 | True     |
```

Notice that the output from the `to_markdown()` method also aligned our columns based on the data type, and padded the values so that even the raw markdown is pretty printed! While we used the `from_html()` and `to_markdown()` methods for this demo, we could just as easily have created the same table in any number of formats. Once we have our data as a `SQLDataModel`, it would've been just as easy to have the data in LaTeX:

```python
# Let's convert it to a LaTeX table instead
sdm_latex = sdm.to_latex()

# Here's the output
print(sdm_latex)
```
As with our markdown output, the columns are correctly aligned and pretty printed:

```latex
\begin{tabular}{|l|r|l|}
\hline
    Planet  & Gravity & Flyable? \\
\hline
    Mercury &    0.38 & True     \\
    Venus   &    0.91 & True     \\
    Mars    &    0.38 & True     \\
    Saturn  &    0.92 & True     \\
    Uranus  &    0.89 & True     \\
\hline
\end{tabular}
```
In fact, using the `to_html()` method is how the table from the beginning of this demo was created! Click [here](https://antet.github.io/sdm-planets) for an example of how the styling and formatting applied to `SQLDataModel` gets exported along with it when using `to_html()`. 

---

#### SQL on your Pandas DataFrame
I can't tell you how many times I've found myself searching for information on how to do this or that operation in `pandas` and wished I could just quickly do it in SQL instead. Enter `SQLDataModel`: 

```python
import pandas as pd
from SQLDataModel import SQLDataModel

# Titanic dataset
df = pd.read_csv('titanic.csv')

# Transformations you don't want to do in pandas if you already know SQL
sql_query = """
select 
    Pclass, Sex, count(*) as 'Survived' 
from sdm where 
    Survived = 1 
group by 
    Pclass, Sex 
order by 
    count(*) desc
"""

# Extract: Create SQLDataModel from the df
sdm = SQLDataModel.from_pandas(df)

# Transform: Do them in SQLDataModel
sdm = sdm.execute_fetch(sql_query)

# Load: Then hand it back to pandas!
df = sdm.to_pandas()
```
Here we're using `SQLDataModel` to avoid performing the complex pandas operations required for aggregation if we already know SQL. Here's the output of the `sdm` we used to do the operations in and the `df`:

```text
SQLDataModel:                         pandas:
┌───┬────────┬────────┬──────────┐    
│   │ Pclass │ Sex    │ Survived │       Pclass     Sex  Survived
├───┼────────┼────────┼──────────┤    0       1  female        91
│ 0 │      1 │ female │       91 │    1       3  female        72
│ 1 │      3 │ female │       72 │    2       2  female        70
│ 2 │      2 │ female │       70 │    3       3    male        47
│ 3 │      3 │ male   │       47 │    4       1    male        45
│ 4 │      1 │ male   │       45 │    5       2    male        17
│ 5 │      2 │ male   │       17 │    
└───┴────────┴────────┴──────────┘    
[6 rows x 3 columns]               
```
In this example our source and destination formats were both `pd.DataFrame` objects, however `pandas` is not required to use, nor is it a dependency of, `SQLDataModel`. It is only required if you're using the `from_pandas()` or `to_pandas()` methods.

---

#### From SQL to HTML Table

Say we have a table located on a remote PostgreSQL server that we want to put on our website, normally we could pip install `SQLAlchemy`, `psycopg2`, `pandas` including whatever dependencies they come with, like `numpy`. This time all we need is `psycopg2` for the PostgreSQL driver:

```python
import psycopg2
import datetime
from SQLDataModel import SQLDataModel

# Setup the connection
psql_conn = psycopg2.connect(...)

# Grab a table with missions to Saturn
sdm = SQLDataModel.from_sql('saturn_missions', psql_conn) # or SQL statement

# Filter to only 'Future' missions
sdm = sdm[sdm['Status'] == 'Future']

# Create new column with today's date so it ages better!
sdm['Updated'] = datetime.date.today()

# Send our table to a new html file, this time without the index
saturn_html = sdm.to_html('Future-Saturn.html', include_index=False)
```
Here's a snippet from the `Future-Saturn.html` file generated by the `to_html()` method:
```html
<!-- Metadata Removed -->
<table>
    <tr>
        <th>Mission</th>
        <th>Status</th>
        <th>Launch</th>
        <th>Destination</th>
        <th>Updated</th>
    </tr>
    <tr>
        <td>New Frontiers 4</td>
        <td>Future</td>
        <td>2028</td>
        <td>Surface of Titan</td>
        <td>2024-03-21</td>
    </tr>
    <tr>
        <td>Enceladus Orbilander</td>
        <td>Future</td>
        <td>2038</td>
        <td>Surface of Enceladus</td>
        <td>2024-03-21</td>
    </tr>
</table>
<!-- SQLDataModel css styles removed -->
```
The Surface of Titan in 2028, how cool is that!

---

#### Pretty PyTorch Training Results to JSON
Let's say we're training a `pytorch` model and we want to keep the results from training. If you've ever trained one or followed a tutorial, I'm sure you'll be familiar with this pattern:

```python
# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Train the model and get the metrics
    train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate(model, criterion, val_loader, device)

    # Print training results to terminal
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```
This ubiquitous setup, where results from each epoch are output to the terminal, has become the `Hello, World!` of ML training loops. While this is great for monitoring the progress of training, it's miserable for actually keeping and analyzing the data afterwards. Here's how we can use `SQLDataModel` to help:

```python
from SQLDataModel import SQLDataModel

# Initialize an empty model with headers for the data we want to store
sdm = SQLDataModel(headers=['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

# ... Training code ...
```
We create an empty `SQLDataModel` that we can use with `headers` corresponding to the metrics we want to store. Now in our training loop we can fill it with our results:

```python
# Initialize an empty model for training
sdm = SQLDataModel(headers=['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Train the model and get the metrics
    train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate(model, criterion, val_loader, device)

    # Let's store the results first
    sdm.insert_row([epoch+1, train_loss, train_acc, val_loss, val_acc])

    # Print training results to terminal
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```
We use the `insert_row()` method to store the results from training, inserting a new row for each epoch. Now that we're storing the data instead of just printing it to the terminal, we can track our results and easily export them into whichever format we need. For this example, let's store them as a `.json` file. All we need is to add one more line at the end of our training loop:

```python
# ... Training loop ...
sdm.to_json('Training-Metrics.json')
```
This will export all the training data stored in our model to a new JSON file `Training-Metrics.json`, trimmed for brevity:

```json
[
    {
      "Epoch": 1,
      "Train Loss": 0.6504,
      "Train Acc": 0.6576,
      "Val Loss": 0.6126,
      "Val Acc": 0.6608
    },
    {
      "Epoch": 2,
      "Train Loss": 0.5733,
      "Train Acc": 0.6989,
      "Val Loss": 0.5484,
      "Val Acc": 0.7268
    },
    {
      "Epoch": 3,
      "Train Loss": 0.5222,
      "Train Acc": 0.7239,
      "Val Loss": 0.5294,
      "Val Acc": 0.7332
    }
  ]
```
In this example we kept the original print statements intact during training, but we could've removed them altogether and done something like this instead:

```python
# Initialize an empty model for training
sdm = SQLDataModel(headers=['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Train the model and get the metrics
    train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate(model, criterion, val_loader, device)

    # Let's store the results first
    sdm.insert_row([epoch+1, train_loss, train_acc, val_loss, val_acc])

    # Pretty print last row of training results to terminal
    print(sdm[-1])

# Print all results when training finishes
print(sdm)

# Export them as a json file
sdm.to_json('Training-Metrics.json')
```
Here we've removed the previous and rather verbose print statement and replaced it with a simple `print(sdm[-1])`, this will print the last row of data in the model. This results in the following terminal output for each epoch during training:

```shell
┌───────┬────────────┬───────────┬──────────┬─────────┐
│ Epoch │ Train Loss │ Train Acc │ Val Loss │ Val Acc │
├───────┼────────────┼───────────┼──────────┼─────────┤
│    10 │     0.2586 │    0.8063 │   0.5391 │  0.7728 │
└───────┴────────────┴───────────┴──────────┴─────────┘
```
Which, besides having the added benefit of storing our data, is far less code than the prior print statement, and is _in my humble opinion_ much nicer to look at while we wait for training to complete. We've also added one more line, a final `print(sdm)` to output all the training results before saving them to `Training-Metrics.json`. That way, for those of us who still like to see them all in one place, we can have our saved json cake and pretty print it too:

```shell
┌───────┬────────────┬───────────┬──────────┬─────────┐
│ Epoch │ Train Loss │ Train Acc │ Val Loss │ Val Acc │
├───────┼────────────┼───────────┼──────────┼─────────┤
│     1 │     0.6504 │    0.6576 │   0.6126 │  0.6608 │
│     2 │     0.5733 │    0.6989 │   0.5484 │  0.7268 │
│     3 │     0.5222 │    0.7239 │   0.5294 │  0.7332 │
│     4 │     0.4749 │    0.7288 │   0.5461 │  0.7238 │
│     5 │     0.4417 │    0.7301 │   0.5862 │  0.7148 │
│     6 │     0.4065 │    0.7544 │   0.4882 │  0.7674 │
│     7 │     0.3635 │    0.7679 │   0.4793 │  0.7690 │
│     8 │     0.3229 │    0.7884 │   0.4833 │  0.7732 │
│     9 │     0.2860 │    0.7921 │   0.5041 │  0.7746 │
│    10 │     0.2586 │    0.8063 │   0.5391 │  0.7728 │
└───────┴────────────┴───────────┴──────────┴─────────┘
[10 rows x 5 columns]
```
Whelp, clearly it's overfitting...

---

#### From PostgreSQL to SQLite
This example will showcase one of the original motivations of this package, moving data from one SQL database to another. It'll be a short one, just as `SQLDataModel` was designed to be! Let's say we need to move data from a PostgreSQL server into a local SQLite connection:

```python
import sqlite3
import psycopg2
from SQLDataModel import SQLDataModel

# Setup our postgresql connection
pg_conn = psycopg2.connect(...)

# Setup our local sqlite3 connection
sqlite_conn = sqlite3.connect(...)

# Create our model from a query on our postgresql connection
sdm = SQLDataModel.from_sql("SELECT * FROM pg_table", pg_conn)

# Import it into our sqlite3 connection, appending to existing data
sdm.to_sql("local_table", sqlite_conn, replace_existing=False)
```
Thats it! What once required installing packages like `pandas`, `numpy`, and `SQLAlchemy`, plus all of their dependencies, just to able to use a nice DataFrame API when interacting with SQL data, is now just a single package, `SQLDataModel`. While I **love** all three of those packages, 99% of what I consistently use them for can be done with far less "package baggage".

---

#### The Workflow

For all the `SQLDataModel` examples, the same basic workflow and pattern is present:

```python
from SQLDataModel import SQLDataModel

# Extract: Create the model from a source
sdm = SQLDataModel.from_data(...)

# Transform: Manipulate the data if needed
sdm['New Column'] = sdm['Column A'].apply(func)

# Load: Move it to a destination format
sdm.to_text('table.txt') # to_csv, to_json, to_latex, to_markdown, to_html, ..., etc.
```
We can take a step back and illustrate the broader  **source, transform, destination** flow with `SQLDataModel` like this:

<img src="https://github.com/AnteT/SQLDataModel/raw/master/figs/sdm_graph.PNG?raw=true" alt="sdm_graph" style="width:100vw; border-radius: .6vw" />

Regardless of where the data originated or where it ends up, `SQLDataModel`'s best use-case is to be the light-weight intermediary that's agnostic to the original source, or the final destination, of the data.

---

### Data Formats
`SQLDataModel` seamlessly interacts with a wide range of data formats providing a versatile platform for data extraction, conversion, and writing. Supported formats include:

- **CSV**: Extract from and write to comma separated value, `.csv`, files.
- **HTML**: Extract from web and write to and from `.html` files including formatted string literals.
- **JSON**: Extract from and write to `.json` files, JSON-like objects, or JSON formatted sring literals.
- **LaTeX**: Extract from and write to `.tex` files, LaTeX formatted string literals.
- **Markdown**: Extract from and write to `.MD` files, Markdown formatted string literals.
- **Numpy**: Convert to and from `numpy.ndarray` objects, `numpy` required.
- **Pandas**: Convert to and from `pandas.DataFrame` objects, `pandas` required.
- **Parquet**: Extract from and write to `.parquet` files, `pyarrow` required.
- **Pickle**: Extract from and write to `.pkl` files, package uses `.sdm` extension when pickling for `SQLDataModel` metadata.
- **SQL**: Extract from and write to the following popular SQL databases:
    - **SQLite**: Using the built-in `sqlite3` module.
    - **PostgreSQL**: Using the `psycopg2` package.
    - **SQL Server**: Using the `pyodbc` package.
    - **Oracle**: Using the `cx_Oracle` package.
    - **Teradata**: Using the `teradatasql` package.
- **Text**: Write to and from `.txt` files including other `SQLDataModel` string representations.
- **TSV** or delimited: Write to and from files delimited by:
    - **`\t`**: Tab separated values or `.tsv` files.
    - **`\s`**: Single space or whitespace separated values.
    - **`;`**: Semicolon separated values.
    - **`|`**: Pipe separated values.
    - **`:`**: Colon separated values.
    - **`,`**: Comma separated values or `.csv` files.
- **Python** objects:
    - **dictionaries**: Convert to and from collections of python `dict` objects.
    - **lists**: Convert to and from collections of python `list` objects.
    - **tuples**: Convert to and from collections of python `tuple` objects.
    - **namedtuples**: Convert to and from collections of `namedtuples` objects.

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

### Motivation

While there are packages/dependencies out there that can accomplish some of the same tasks as `SQLDataModel`, they're either missing key features or end up being overkill for common tasks like grabbing and converting tables from source A to destination B, or they don't quite handle the full process and require additional dependencies to make it all work. When you find yourself doing the same thing over and over again, eventually you sit down and write a package to do it for you.

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