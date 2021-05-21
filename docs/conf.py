#!/usr/bin/env python3

import os
import shutil

if not os.path.exists("notebooks"):
    #    os.mkdir('notebooks')
    shutil.copytree(os.path.abspath("../notebooks"), "notebooks")

# Select nbsphinx and, if needed, other Sphinx extensions:
extensions = [
    'nbsphinx',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
    'sphinx.ext.mathjax',  # for math equations
    'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
]

autodoc_mock_imports = ['mpl_toolkits',  'cartopy']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
# Default language for syntax highlighting in reST and Markdown cells:
highlight_language = 'python3'

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ''


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


#def setup(app):
#    """Sphinx setup function."""
#    app.add_css_file('theme_override.css')


# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]


# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']


# Environment variables to be passed to the kernel:
# os.environ['MY_DUMMY_VARIABLE'] = 'Hello from conf.py!'

# nbsphinx_thumbnails = {
#    'gallery/thumbnail-from-conf-py': 'gallery/a-local-file.png',
#    'gallery/*-rst': '_static/copy-button.svg',
# }

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/FESOM/pyfesom2/blob/{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/FESOM/pyfesom2/{{ env.config.release|e }}?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

autodoc_mock_imports = ['mpl_toolkits', 'cartopy']

# Additional files needed for generating LaTeX/PDF output:
# latex_additional_files = ['references.bib']

# Support for notebook formats other than .ipynb
# nbsphinx_custom_formats = {
#    '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
# }

# -- The settings below this line are not specific to nbsphinx ------------

master_doc = 'index'

# General information about the project.
project = u'pyfesom2'
copyright = u"2020, FESOM team"
author = u"FESOM team"

linkcheck_ignore = [
    r'http://localhost:\d+/',
    'https://github.com/spatialaudio/nbsphinx/compare/',
]

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.

# -- Get version/release information and date from Git ----------------------------

try:
    from subprocess import check_output

    release = check_output(['git', 'describe', '--tags', '--always'])
    release = release.decode().strip()
    today = check_output(['git', 'show', '-s', '--format=%ad', '--date=short'])
    today = today.decode().strip()
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfesom2"])
    import pyfesom2

    version = pyfesom2.__version__
    release = version
    today = '<unknown date>'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns = ['*.txt', '*.md', '_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------------------------------------

html_title = project + ' version ' + release

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'sticky_navigation': False}

# -- Options for LaTeX output ---------------------------------------------

# See https://www.sphinx-doc.org/en/master/latex.html
latex_elements = {
    'papersize': 'a4paper',
    'printindex': '',
    'sphinxsetup': r"""
        %verbatimwithframe=false,
        %verbatimwrapslines=false,
        %verbatimhintsturnover=false,
        VerbatimColor={HTML}{F5F5F5},
        VerbatimBorderColor={HTML}{E0E0E0},
        noteBorderColor={HTML}{E0E0E0},
        noteborder=1.5pt,
        warningBorderColor={HTML}{E0E0E0},
        warningborder=1.5pt,
        warningBgColor={HTML}{FBFBFB},
    """,
    'preamble': r"""
\usepackage[sc,osf]{mathpazo}
\linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
\renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
\IfFileExists{zlmtt.sty}{
    \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
}{
    \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
}
\usepackage{booktabs}  % for Pandas dataframes
""",
}

latex_documents = [
    (master_doc, 'nbsphinx.tex', project, author, 'howto'),
]

latex_show_urls = 'footnote'
latex_show_pagerefs = True

# -- Options for EPUB output ----------------------------------------------

# These are just defined to avoid Sphinx warnings related to EPUB:
version = release
suppress_warnings = ['epub.unknown_project_files']
