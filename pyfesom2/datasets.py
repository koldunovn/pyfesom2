import warnings
from typing import Sequence
from typing import Tuple, Optional

import xarray as xr

from . import load_mesh
from .ut import get_no_cyclic

cmip6_grids = {
    'AWI-CM-LR': {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-lr",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-lr"},
    'AWI-CM-MR': {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-mr",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-mr"},
    'AWI-CM-HR': {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-hr",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/cmip6-grids/zarr/awicm-hr"},
}

frontier_datasets = {
    "ROSSBY4.2"        : {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42"},
    "ROSSBY4.2_level"  : {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42_level_aceess",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42_level_aceess"},
    "ROSSBY4.2_spatial": {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42_spatial_aceess",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/frontier/rossby42_spatial_aceess"},
    "A01"              : {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_02942825-0cab-44f3-ad37-80fd5d2e37e3/FESOM2_data/A01",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_02942825-0cab-44f3-ad37-80fd5d2e37e3/FESOM2_data/A01"}
}

all_datasets = {
    "CORE"   : {
        "path_url"   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/tutorial/core2",
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/tutorial/core2",
        "group"      : "variables"},
    "pi-grid": {
        'path_url'   : "https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/tutorial/pi-grid",
        "var_list"   : ['a_ice', 'm_ice', 'temp', 'u', 'v', 'w', 'mesh'],
        "Dataset URL": "https://swiftbrowser.dkrz.de/public/dkrz_035d8f6ff058403bb42f8302e6badfbc/pyfesom2/tutorial/pi-grid"},
    **cmip6_grids,
    **frontier_datasets
}


class RemoteZarrDataset:
    """Fetches a remote Zarr dataset.

    Dataset is only loaded on .load() for dataset contaning more variables
    """

    def __init__(self, path_url: str, var_list: Optional[Sequence] = None, group: Optional[str] = None,
                 consolidated: bool = True, **kwargs):
        """Initializes a remote zarr dataset.

        Dataset is loaded only on .load() method.
        Parameters
        ----------
        path_url: str
            Remote http(s) url for a dataset.
        var_list: optional, list
            These variable names are suffixed to path_url for retrieving data variables and each retrieved variable is
            merged into a Xarray dataset. This is not necessary if there is only one variable in remote dataset.
        consolidated: bool
            Read remote dataset as a consolidated Zarr dataset, applicable to datasets stored as consolidated.
        kwargs: optional
            These kwargs are added to dataset as attributes.
        """
        self.path_url = path_url  # can also be local path remove fsspec in that case
        self.var_list = var_list
        self.group = group
        self.is_consolidated = consolidated
        self.dset_attrs = kwargs
        self._ds = None

    @property
    def merged_dataset(self):
        """Merges data variables from remote url

        Returns
        -------
            xr.Dataset
        """
        import fsspec
        if self._ds is None:
            if self.var_list is not None:
                urls = [self.path_url + "/" + var for var in self.var_list]
                dataset_list = [
                    xr.open_zarr(fsspec.get_mapper(url), group=self.group, consolidated=self.is_consolidated) for url in
                    urls]
                self._ds = xr.merge(dataset_list)
            else:
                self._ds = xr.open_zarr(self.path_url, group=self.group, consolidated=self.is_consolidated)

            self._ds.attrs.update(self.dset_attrs)
        return self._ds

    def load(self):
        return self.merged_dataset


core = RemoteZarrDataset(**all_datasets['CORE'])
tutorial_dataset = RemoteZarrDataset(**all_datasets['pi-grid'])

arctic_1km = RemoteZarrDataset(**all_datasets['A01'])
rossby42 = RemoteZarrDataset(**all_datasets['ROSSBY4.2'])
rossby42_spatial = RemoteZarrDataset(**all_datasets['ROSSBY4.2_spatial'])
rossby42_level = RemoteZarrDataset(**all_datasets['ROSSBY4.2_level'])

cmip6_lr = RemoteZarrDataset(**all_datasets['AWI-CM-LR'])
cmip6_mr = RemoteZarrDataset(**all_datasets['AWI-CM-MR'])
cmip6_hr = RemoteZarrDataset(**all_datasets['AWI-CM-HR'])


def fesom_mesh_to_xr(path: str, alpha: int = 0, beta: int = 0, gamma: int = 0) -> xr.Dataset:
    """Returns Xarray coordinate dataset from FESOM mesh files.

    Note
    ----
    1. The coordinate dataset does not contain cyclic points in its faces.
    2. pyfesom2.load_mesh used here is eager to read mesh data into memory, hence repeated use
    of this on large datasets may crash.

    Parameters
    ----------
    path: str
        Local path to directory containing FESOM mesh files.

    alpha: int
        Mesh rotation alpha.
    beta: int
        Mesh rotation beta.
    gamma: int
        Mesh rotation gamma.

    Returns
    -------
    xr.Dataset

    """
    mesh = load_mesh(path, abg=[alpha, beta, gamma])
    #ncyclic_inds = get_no_cyclic(mesh, mesh.elem)
    #triangles = mesh.elem[ncyclic_inds]
    triangles = mesh.elem
    if any(mesh.zlev < 0.):
        nz_values = mesh.zlev * -1.0
    else:
        nz_values = mesh.zlev
    coords_dataset = xr.Dataset(coords={'lon'  : ('nod2', mesh.x2),
                                        'lat'  : ('nod2', mesh.y2),
                                        'faces': (('nelem', 'three'), triangles.astype('uint32')),
                                        'nz'   : nz_values,
                                        'nz1'  : (nz_values[:-1] + nz_values[1:]) / 2.0})

    coords_dataset.coords['lon'].attrs['long_name'] = 'longitude'
    coords_dataset.coords['lon'].attrs['units'] = 'degrees_east'
    coords_dataset.coords['lat'].attrs['long_name'] = 'latitude'
    coords_dataset.coords['lat'].attrs['units'] = 'degrees_north'
    coords_dataset.coords['faces']['long_name'] = 'Triangulation Faces containing indices'

    coords_dataset.coords['nz1'].attrs['long_name'] = 'depth at half level'
    coords_dataset.coords['nz1'].attrs['units'] = 'm'
    coords_dataset.coords['nz1'].attrs['positive'] = 'down'
    coords_dataset.coords['nz1'].attrs['axis'] = 'Z'

    coords_dataset.coords['nz'].attrs['long_name'] = 'depth'
    coords_dataset.coords['nz'].attrs['units'] = 'm'
    coords_dataset.coords['nz1'].attrs['positive'] = 'down'
    coords_dataset.coords['nz1'].attrs['axis'] = 'Z'
    coords_dataset.attrs['Conventions'] = 'CF-1.7'
    return coords_dataset


def open_dataset(path_or_pattern: str, mesh_path: str, abg: Sequence = (0, 0, 0),
                 parallel=True, **kwargs) -> xr.Dataset:
    """Returns a merged Xarray dataset from multi-file FESOM model output.

    Parameters
    ----------
    path_or_pattern: str
        File path referring to a single file or posix style pattern to merge multiple files to single dataset.
    mesh_path: str
        File system path to mesh directory.
    abg: list
        A list or tuple of length 3. These values represent alpha, beta, gamma of rotated mesh.
    parallel: bool
        Read files in parallel, only relevant in case of argument path_or_pattern is a pattern.
    kwargs: optional
        These key words are passed to Xarray's open_mfdataset. For instance, this may be used to control chunks:
        chunks={'time': 1, 'nod2': 100000, .....}.

    Returns
    -------
    xr.Dataset
    """
    combine = kwargs.pop('combine', 'by_coords')
    da = xr.open_mfdataset(path_or_pattern, parallel=parallel, combine=combine, **kwargs)
    mesh = fesom_mesh_to_xr(mesh_path, *abg)
    return xr.merge([da, mesh])


def fesom_like(spatial_size: int, spatial_extent: Tuple[float, float, float, float] = (-180, -90, 180, 90),

               times: Optional[int] = None, levels: Optional[int] = None, holes: Optional[int] = None,
               var_name: str = 'fesom_var') -> xr.Dataset:
    """Returns a FESOM-like Xarray dataset for quick prototyping, development and benchmarking.

    This method is most useful for developers, it provides a FESOM-like Xarray dataset for specified spatial_size. This
    is very useful for creating FESOM-like datasets of various spatial sizes that might not be available in a pyfesom2
    developer's system. The nodes(lon, lat) are uniformly distributed in ranges supplied by parameter spacial_extent.
    Additional dimensions time and levels can be added with parameters times, levels. When dask is
    available it also returns a dummy data variable uniformly distributed in range (0,1) if not  returns just a
    coordinate dataset.

    Parameters
    ----------
    spatial_size: int
        spatial size of returned dataset.
    spatial_extent: (min_lon, min_lat, max_lon, max_lat)
        Sequence that defines the boundary of
    times: int, array-like, optional
        If provided as integer, times of this size are added as dimensions to dataset with monthly data starting in
        2010 as values, a array of numpy datetime objects can be passed instead to control the time values.
    levels: int, array-like, optional
        If provided as integer, these many levels are added to the dataset. levels are linearly spaced from 0 to -6000.
        A 1-d  array-like values of levels can be passed to control level values.
    holes: int, optional
        If provided, it randomly removes these number of faces to mimic mimic internal regions e.g,, land regions.
    var_name: str
        Name of data variable in returned dataset. This data variable is only returned if dask is present.

    Returns
    -------
    xr.Dataset

    Examples
    --------
    >>> fesom_like(100)
    <xarray.Dataset>
    Dimensions:    (nelem: 184, nod2: 100, three: 3)
    Coordinates:
        lon        (nod2) float64 4.858 -11.65 -69.51 40.91 ... 170.7 -101.2 108.0
        lat        (nod2) float64 -28.72 -13.71 13.67 -76.21 ... -12.08 10.64 -22.7
        faces      (nelem, three) int32 62 47 63 49 47 62 74 ... 46 36 56 56 57 46
    Dimensions without coordinates: nelem, nod2, three
    Data variables:
        fesom_var  (nod2) float64 dask.array<chunksize=(100,), meta=np.ndarray>

    To add time (or level) as dimension:

    >>> fesom_like(100, times=10)
    <xarray.Dataset>
    Dimensions:    (nelem: 186, nod2: 100, three: 3, time: 10)
    Coordinates:
      * time       (time) datetime64[ns] 2010-01-31 2010-02-28 ... 2010-10-31
        lon        (nod2) float64 -98.46 1.187 -27.48 93.83 ... 110.6 -108.8 -83.98
        lat        (nod2) float64 59.19 47.07 -70.14 -41.33 ... -47.75 25.72 11.9
        faces      (nelem, three) int32 67 21 14 12 43 45 91 ... 48 48 97 38 38 4 48
    Dimensions without coordinates: nelem, nod2, three
    Data variables:
        fesom_var  (time, nod2) float64 dask.array<chunksize=(10, 100), meta=np.ndarray>

    """
    from matplotlib.tri import Triangulation
    import pandas as pd
    import numpy as np
    try:
        import dask.array as da
        random_function = da.random.uniform
    except ImportError:
        random_function = None
        warnings.warn("Dask is not available, only coordinate dataset is returned to save memory. \n"
                      "Data variable can still be assigned to coordinate dataset as: \n"
                      "dataset[var_name]= (('time', 'nz1', 'nod2'), dataarray[ntimes, nlevels, spatial_size])")

    minx, miny, maxx, maxy = spatial_extent
    lons = np.random.uniform(minx, maxx, spatial_size)
    lats = np.random.uniform(miny, maxy, spatial_size)
    tri_obj = Triangulation(lons, lats)
    tri_arr = tri_obj.triangles

    if holes is not None:
        if isinstance(holes, int):
            tri_arr = tri_arr[:-1 * holes]
        else:
            raise ValueError('holes needs be None or integer')

    other_dims = {}
    if times is not None:
        if isinstance(times, int):
            other_dims['time'] = pd.date_range('2010-01-01', freq='M', periods=times)
        else:
            other_dims['time'] = times

    if levels is not None:
        if isinstance(levels, int):
            other_dims['nz1'] = np.linspace(0, -6000, levels)
        else:
            other_dims['nz1'] = levels

    dataset = xr.Dataset(coords={**other_dims,
                                 'lon'  : ('nod2', lons),
                                 'lat'  : ('nod2', lats),
                                 'faces': (('nelem', 'three'), tri_arr),
                                 })

    if random_function is not None:
        dims_sizes = [(dim, len(dataset[dim])) for dim in dataset.dims if dim not in ['three', 'nelem']]
        # Sort order of dims in data variable by size.
        # Technically use OrderedDict but dict preserves order mostly.
        dims_sizes_dict = dict(sorted(dims_sizes, key=lambda kv: kv[1]))
        dims = list(dims_sizes_dict.keys())
        dim_sizes = list(dims_sizes_dict.values())
        dataset[var_name] = (dims, random_function(0., 1., dim_sizes))
    return dataset
