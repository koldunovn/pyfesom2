pyfesom2
========

pyfesom2 is a Python library and collection of command-line tools for working with `FESOM2 <http://www.fesom.de/>`_  ocean model data.
FESOM2 uses a unstructured and varying-resolution triangular grid (and oftenly for very high-resolution simulations), this makes commonly used libraries for ananlsys and  visualization
is commonly used plots and calculations.
As such pyfesom2 library provides interface from FESOM2 data to these commonly used librroes and provides examples for frequently used calculations in ocean-modeling.
for analysis and visulto enable use of commonly used tools, for commonly used
enables

:ref:`tools` are python scripts with command line interfaces that are used for quick actions with FESOM2 model output. For example::

    pfplot /path/to/mesh/ /path/to/output/ salt

will produce a map with global spatial distribution of salinity on the surface at the first time step.

Library is a python library that contains functions for working with FESOM2 mesh and data. For example loading FESOM mesh can be done as simple as::

    import pyfesom2 as pf
    meshpath  ='/path/to/mesh/'
    mesh = pf.load_mesh(meshpath)

Examples of tools are :ref:`pfplot` for quick visualization of FESOM data and :ref:`pfinterp` for interpolation to regular lon/lat grid.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   examples
   tools
   library
   api
   contributing
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
