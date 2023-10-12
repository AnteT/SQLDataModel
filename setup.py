import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='SQLDataModel',  
     version='0.1',
     scripts=['SQLDataModel.py'] ,
     author="Ante Tonkovic-Capin",
     author_email="antetc@icloud.com",
     description="A data model based on in-memory sqlite",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/AnteT/sql-model",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )