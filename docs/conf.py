# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# [print(i) for i in sys.path]

# -- Project information -----------------------------------------------------

project = 'MSUtils'
copyright = '2024, DAE'
author = 'DAE'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              "sphinx.ext.napoleon",
              "sphinx.ext.githubpages",
              "nbsphinx",
              "sphinx.ext.mathjax",
]

# napoleon for google-style docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
# sphinx.ext.githubpages for creating .nojekyll file
# sphinxcontrib.mermaid for mermaid-support: https://github.com/mgaitan/sphinxcontrib-mermaid

autosummary_generate = True  # Turn on sphinx.ext.autosummary
# moves typehints into docstring, but does not process them correctly:
# from https://stackoverflow.com/a/66295922
autodoc_typehints = "description"
# Napoleon settings
# napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']