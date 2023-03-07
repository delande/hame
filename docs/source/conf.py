# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Hame (Hydrogen atom matrix element)'
copyright = '2023, Dominique Delande'
author = 'Dominique Delande'
release = '1.0'
version = '1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]


templates_path = ['_templates']

# -- Options for HTML output


# -- Options for EPUB output
epub_show_urls = 'footnote'

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
