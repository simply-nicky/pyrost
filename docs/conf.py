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
import pyrost
# sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'pyrost'
copyright = '2020, Nikolay Ivanov'
author = 'Nikolay Ivanov'

# The full version, including alpha/beta/rc tags
release = '0.7.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx_autodoc_typehints', 'sphinx.ext.autodoc',
              'sphinx.ext.intersphinx', 'sphinx.ext.doctest']
intersphinx_mapping = {'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'python': ('https://docs.python.org/3.10', None),
                       'h5py': ('https://docs.h5py.org/en/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None)}

# Autodoc settings
autodoc_docstring_signature = False
autodoc_typehints = 'description'
autoclass_content = 'class'
autodoc_class_signature = 'separated'

# Sphinx_autodoc_typehints settings
always_document_param_types = False

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_preprocess_types = False

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
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options = {
    "light_logo": "pyrost_logo.png",
    "dark_logo": "pyrost_logo_dark.png",
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/simply-nicky/pyrost",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ]
}