import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
     name='SQLDataModel',  
     version='0.12.1',
     scripts=['src/SQLDataModel/SQLDataModel.py'] ,
     author='Ante Tonkovic-Capin',
     author_email='antetc@icloud.com',
     description='SQLDataModel is a lightweight dataframe library designed for efficient data extraction, transformation, and loading (ETL) across various sources and destinations, providing an efficient alternative to common setups like pandas, numpy, and sqlalchemy while also providing additional features without the overhead of external dependencies.',
     keywords=['SQL','ETL','dataframe','terminal-tables','pretty-print-tables','sql2sql','data-analysis','data-science','datamodel','extract','transform','load','web-scraping-tables','data-mining','html','html-table-parsing','apache-arrow','pyarrow','pyarrow-conversion','pyarrow-to-table','pyarrow-to-sql','pyarrow-to-csv','parquet-file-parsing','csv','csv-parsing','markdown','markdown-table-parsing','latex','latex-table-parsing','csv2latex','csv2tex','csvtolatex','delimited','delimited-data-parsing','file-conversion','format-conversion','terminal-styling','table-styling','from-sqlite','to-sqlite','from-postgresql','to-postgresql','sql-to-sql','excel','xlsx-file','excel-to-sql','DataFrames','polars2pandas','pandas2polars','csv2rst','rst-table','sphinx-table','md2rst'],
     license_file='LICENSE',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/AnteT/SQLDataModel',
     project_urls = {
        'Documentation':'https://sqldatamodel.readthedocs.io/en/latest/',
        'Source': 'https://github.com/AnteT/SQLDataModel.git'
    },
     package_dir = {'': 'src'},
     packages = setuptools.find_packages(where='src'),
     classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',         
        'Programming Language :: Python :: 3.11',         
        'Programming Language :: Python :: 3.12',         
        'Topic :: Scientific/Engineering',
     ],
     python_requires = '>=3.9'
 )
