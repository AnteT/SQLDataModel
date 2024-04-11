import sys, os

project_root = os.path.abspath('../../src')
sys.path.append(project_root)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SQLDataModel'
copyright = '2024, Ante Tonkovic-Capin'
author = 'Ante Tonkovic-Capin'
release = '0.3.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Enable autodoc extension
    'sphinx.ext.napoleon',  # Enable Napoleon extension
    'sphinx.ext.mathjax', #  Enable math formulas
    'myst_parser', #  Enable parsing of Markdown files
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True

# napoleon_custom_sections = [('Metrics', 'params_style')]

templates_path = ['_templates']
modindex_common_prefix = ['SQLDataModel'] # doesnt seem to do anything
add_modules_names = False # doesnt seem to make a difference

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = ''
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "sdm_banner_cyan.PNG"
html_theme_options = {
    'logo_only': True,
    'display_version': False
}

def process_docstring(app, what, name, obj, options, lines:list[str]):
    i = 0
    while i < len(lines):
        lines[i] = lines[i].replace('```python', '.. code-block:: python').replace('```shell', '.. code-block:: console').replace('```text', '.. code-block:: text').replace('```','')
        if '.. code-block:: ' in lines[i]:
            lines.insert(i + 1, '') # insert blank line to accomodate removal of extra line preceding directives
            i += 1 
        i += 1

def setup(app):
    app.add_css_file('custom.css')  # custom css file has to be placed in ./docs/source/_static/custom.css
    app.connect('autodoc-process-docstring', process_docstring)