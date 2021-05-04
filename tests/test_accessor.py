import numpy as np
import pytest
from shapely.geometry import box, Polygon

from pyfesom2.datasets import open_dataset


# TODO: push this to test config conftest.py so that it is available globally
@pytest.fixture(scope='module')
def local_dataset(request):
    import os.path
    cur_dir = os.path.dirname(request.fspath)
    data_path = os.path.join(cur_dir, "data", "pi-results", "temp.fesom.*.nc")
    mesh_path = os.path.join(cur_dir, "data", "pi-grid")
    da = open_dataset(data_path, mesh_path=mesh_path)
    yield da


# use scope=function if we want to change grid size dynamically for each test case,
# but merged dataset then needs to be changed appropriately to take the param.
@pytest.fixture(scope='module', params=[36 * 18, 360 * 180])  # same pts as 10deg, 1deg reg grid
def random_spatial_dataset(request):
    from matplotlib.tri import Triangulation
    import xarray as xr

    if request.param is not None:
        spatial_size = request.param
    else:
        spatial_size = 36 * 18  # default, it covers more edge cases

    lons = np.random.uniform(-180., 180., spatial_size)
    lats = np.random.uniform(-90., 90., spatial_size)
    tris = Triangulation(lons, lats)
    dataset = xr.Dataset(coords={'lon'  : ('nod2', lons),
                                 'lat'  : ('nod2', lats),
                                 'faces': (('nelem', 'three'), tris.triangles)})
    dataset['dummy_2d_var'] = ('nod2', np.random.uniform(0., 1., spatial_size))

    yield dataset


@pytest.fixture
def random_nd_dataset(random_spatial_dataset):
    """time(12), level(40), parametrized nod2 dataset"""
    import pandas as pd
    times = pd.date_range('2019-01-01', freq='m', periods=12)
    nz1 = np.linspace(0, -5000, 40)
    dataset = random_spatial_dataset.assign_coords({'time': times, 'nz1': nz1})
    dummy_nd_var = np.random.uniform(0., 1., (len(times), len(random_spatial_dataset.nod2), len(nz1)))
    dataset['dummy_nd_var'] = (('time', 'nod2', 'nz1'), dummy_nd_var)
    yield dataset


# merged dataset fixture using just 2 for now, in future add remote too
@pytest.fixture(params=["local_dataset", "random_spatial_dataset"])
def dataset(local_dataset, random_spatial_dataset, request):
    yield request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def five_point_dataset():
    """
    A five point dataset defined at four corners of -180,-90 -> 180,-90 -> 180, 90 -> 0,0 -> -180,90,
    four corners of global map and center. This gives a very controlled fesom-like-triangular mesh with predictable
    triangulation (in ccw): [[0,1,3],[1,2,3], [2,4,3], [4,0,3]]. This makes testing of point selection, region
    selections on both nodes and faces easy. This can also be used to test interpolations.
    in practice use nodes slightly inside domain adjusted by 1 deg, to accommodate shapely's contains.
    """
    import xarray as xr
    lons = [-180., 180., 180., 0., -180.]
    lats = [-90., -90., 90., 0., 90.]
    faces = np.array([[0, 1, 3],
                      [1, 2, 3],
                      [2, 4, 3],
                      [4, 0, 3]])  # no of faces = 2*nodes - 2*boundary nodes -2
    dataset = xr.Dataset(coords={'lon'  : ('nod2', lons),
                                 'lat'  : ('nod2', lats),
                                 'faces': (('nelem', 'three'), faces)}
                         )
    dataset['dummy_2d_var'] = ('nod2', np.random.uniform(0., 1., len(lons)))
    yield dataset


# TODO: skipping because remote datasets need to be fixed for plotting and optimized for remote transfer:
#     current unsorted points leads to too much data transfer, atleast when remote dataset contains faces,
#     integrate this in dataset fixture as a param and mark it slow
# @pytest.fixture(scope='module', params=[LCORE, A01])
# def lcore_dataset(request):
#     da = request.param.load()
#     yield da

# Test selection utils

def test_normalize_distances():
    from pyfesom2.accessor import normalize_distance

    dists_lt_1km = np.linspace(0, 999, 10)
    dists_gt_1km = np.linspace(1000, 10000, 2)
    units, dist = normalize_distance(np.hstack([dists_lt_1km, dists_gt_1km]))
    assert units == 'm'
    assert ~np.any(np.isnan(dist)), 'NaNs in dists should not be possible.'

    dists_lt_1km = np.linspace(0, 999, 2)
    dists_gt_1km = np.linspace(1000, 10000, 10)
    units, dist = normalize_distance(np.hstack([dists_lt_1km, dists_gt_1km]))
    assert units == 'km'
    assert ~np.any(np.isnan(dist)), 'NaNs in dists should not be possible.'


# Test selection methods

bbox_tests = [
    (-10, 50, 60, 80),  # LL -> UR (minx, miny, maxx, maxy)
    # use bad values and check
    # bounds smaller then grid size
    # invalid bounds.
]


@pytest.mark.parametrize("bbox", bbox_tests)
def test_select_bbox(dataset, bbox):
    from pyfesom2.accessor import select_bbox

    # selection on datasets
    sda = select_bbox(dataset, bbox)
    slon_min, slat_min, slon_max, slat_max = (sda.lon.min(), sda.lat.min(),
                                              sda.lon.max(), sda.lon.max())
    # check within bbox's bounds
    # because pyfesom2.ut's cut_region uses closed bounds and sel is only when all nodes
    # of triangle are in bounds.

    assert slon_min >= bbox[0] and slon_max <= bbox[2]
    assert slat_min >= bbox[1] and slat_max <= bbox[3]

    # selection on data arrays
    data_var = list(dataset.data_vars.keys())[0]
    with pytest.raises(ValueError):
        select_bbox(dataset[data_var], bbox)

    sda = select_bbox(dataset[data_var], bbox, faces=dataset.faces)
    slon_min, slat_min, slon_max, slat_max = (sda.lon.min(), sda.lat.min(),
                                              sda.lon.max(), sda.lon.max())
    assert slon_min >= bbox[0] and slon_max <= bbox[2]
    assert slat_min >= bbox[1] and slat_max <= bbox[3]


region_tests = [
    *bbox_tests,  # bbox tests should be valid inputs for select_region
    *[box(*bbox) for bbox in bbox_tests],  # bbox tests cst s shapely's box
    Polygon([(-70, 30), (-10, 0), (-10, 60)]),  # a triangle in atlantic
    # in future add complex region
    # polygons from pyfesom's ut
]


@pytest.mark.parametrize("region", region_tests)
def test_select_region(dataset, region):
    from typing import Sequence
    import warnings
    from shapely.geometry import MultiPoint
    from pyfesom2.accessor import select_region
    import xarray as xr
    #     # convex hull of returned dataset
    if isinstance(region, Sequence):
        outer_polygon = box(*region)
    else:
        outer_polygon = region

    sda = select_region(dataset, region)

    if len(sda.lon) == 0:
        with pytest.warns(Warning):
            warnings.warn("No points in domain are within region, returning original data.", Warning)
    else:
        mp = MultiPoint(np.vstack((sda.lon, sda.lat)).T)
        assert outer_polygon.contains(mp.convex_hull)

    # check passing dataarray
    test_data_var_name = list(dataset.data_vars.keys())[0]
    test_data_array = dataset[test_data_var_name]
    with pytest.raises(ValueError):
        # region selection is not done when faces are not present
        # dataarrays cannot contain faces so ValueError
        sda = select_region(test_data_array, region)

    sda = select_region(test_data_array, region, faces=dataset.faces)
    # returned region selection on data array is a dataset
    assert isinstance(sda, xr.Dataset)


@pytest.mark.parametrize("npoints", [10])
def test_select_points(dataset, npoints, request):
    from pyfesom2.accessor import select_points
    # dataset = local_dataset # makes it easy to change fixture
    # select random points from datapoints
    random_pts = np.random.choice(len(dataset.nod2), npoints)  # replace=True by default
    lats = dataset.lat[random_pts]
    lons = dataset.lon[random_pts]

    # pass lons and lats as numpy array
    sda = select_points(dataset, lons.values, lats.values)
    assert np.array_equal(sda.lon.values, lons.values) & np.array_equal(sda.lat.values, lats.values)

    # test passing lon and lat as scalars
    slon, slat = np.array(lons.values[0], ndmin=1), np.array(lats.values[0], ndmin=1)  # to match shapes
    sda = select_points(dataset, float(slon), float(slat))
    assert np.array_equal(sda.lon.values, slon) & np.array_equal(sda.lat.values, slat)

    # selection on data arrays
    data_var = list(dataset.data_vars.keys())[0]
    sda = select_points(dataset[data_var], lons, lats)
    assert np.array_equal(sda.lon.values, lons) & np.array_equal(sda.lat.values, lats)

    # pass lons and lats as xarray data array
    sda = select_points(dataset, lons, lats)
    assert np.array_equal(sda.lon.values, lons) & np.array_equal(sda.lat.values, lats)

    # check other attrs
    assert 'nod2' in sda.dims, 'point selection will always have points in dimensions'
    assert len(sda.nod2) == npoints, 'length of selected points should be same as lats, lons'
    assert 'faces' not in sda, 'faces are not returned by select points'

    assert 'distance' in sda.coords, 'by default select points returns distance as coordinate'
    assert np.isclose(sda.distance[0], 0.0), ' distance to first point is always 0.'

    sda = select_points(dataset, lons, lats, return_distance=False)
    assert 'distance' not in sda, ' point selection with return_distance=False does not have distance'

    # test against points not necessarily in dataset
    npoints = 20
    lons = np.linspace(-180, 180, npoints)
    lats = np.linspace(-90, 90, npoints)
    sda = select_points(dataset, lons, lats)
    assert len(sda.nod2) == npoints


@pytest.mark.parametrize("npoints", [10])
def test_select_points_advanced(random_nd_dataset, npoints):
    """Test trajectory like selection on time, level dimensions"""
    import pandas as pd
    from pyfesom2.accessor import select_points

    dataset = random_nd_dataset
    random_pts = np.random.choice(len(dataset.nod2), npoints)  # replace=True by default
    lats = dataset.lat[random_pts].values
    lons = dataset.lon[random_pts].values
    nz1 = np.random.choice(dataset.nz1.values, npoints)
    # not linearly increasing times are awkward for trajectory
    linear_times = pd.date_range(dataset.time.min().values, dataset.time.max().values, periods=npoints)

    # checking if dimension name of selection can be changed
    sda = select_points(dataset, lon=lons, lat=lats, time=linear_times, nz1=nz1, selection_dim_name='points')
    assert "points" in sda.dims

    assert len(sda.points) == npoints
    assert len(sda.lon) == len(sda.lat) == len(sda.nz1) == len(sda.time)
    assert all([coord in sda.coords for coord in ['lon', 'lat', 'time', 'nz1']])
    assert not all([dim in sda.dims for dim in ('time', 'nz1')])


def test_selection_of_faces(five_point_dataset):
    from shapely.geometry import box
    from pyfesom2.accessor import select_region
    dataset = five_point_dataset
    region = [0., -90., 180., 90.]  # right half
    sh_region = box(*region).buffer(1e-6)
    sel_da = select_region(dataset, region=sh_region)

    assert len(sel_da.nod2) == 3  # 3 points
    assert sel_da.faces.shape == (1, 3)  # 1 triangle
    assert np.all(np.isin([0., 180.], sel_da.lon))
    assert np.all(np.isin([-90., 90., 0.], sel_da.lat))


def test_select(random_nd_dataset):
    dataset = random_nd_dataset
    npoints = 20
    lons = np.linspace(-180, 180, npoints)
    lats = np.linspace(-90, 90, npoints)
    # test passing slices of other dims
    sda = dataset.pyfesom2.select(lon=lons, lat=lats, time=slice('2019-05-01', '2019-12-31'), nz1=slice(0, -1000))
    assert "time" in sda.dims
    assert "nz1" in sda.dims

    # check exceptions
    region = region_tests[0]
    with pytest.raises(ValueError):
        sda = dataset.pyfesom2.select(lon=lons, lat=lats, region=region)

    with pytest.raises(NotImplementedError):
        sda = dataset.pyfesom2.select(lon=lons, lat=lats, method='linear')

    with pytest.raises(ValueError):
        sda = dataset.pyfesom2.select(lon=lons)

    # path cannot have more then 2 tuples
    with pytest.raises(ValueError):
        sda = dataset.pyfesom2.select(path=(lons, lats, dataset.nz1.values))
    with pytest.raises(ValueError):
        sda = dataset.pyfesom2.select(path=lons)


# Test methods on accessors

def test_dataset_accessor_attrs(dataset):
    assert hasattr(dataset, "pyfesom2")

    dataset_methods = ["select", "select_points", "__repr__", "_repr_html_"]  # , "plot", "triplot"]
    for method in dataset_methods:
        assert hasattr(dataset.pyfesom2, method)
    # check if reprs are same as xarray dataset
    assert dataset.__repr__() == dataset.pyfesom2.__repr__()
    assert dataset.pyfesom2._repr_html_() is not None


def test_dataarray_accessor_attrs(dataset):
    """Test dataarray methods on a data variable"""
    test_dataarray = list(dataset.data_vars.keys())[0]
    dataarray_methods = ["select", "select_points", "__repr__", "_repr_html_"]  # , "plot", "triplot"]

    for method in dataarray_methods:
        assert hasattr(dataset.pyfesom2, method)

    # check if reprs exist
    assert getattr(dataset.pyfesom2, test_dataarray).__repr__().startswith("Wrapped")
    assert getattr(dataset.pyfesom2, test_dataarray)._repr_html_() is not None


def test_dataset_accessor_methods(random_nd_dataset):
    import xarray as xr
    dataset = random_nd_dataset
    region = region_tests[0]
    sda = dataset.pyfesom2.select(region=region)
    assert isinstance(sda, xr.Dataset)
    # test accessor to select points
    npoints = 10
    lons = np.linspace(-180, 180, npoints)
    lats = np.linspace(-90, 90, npoints)
    sda = dataset.pyfesom2.select(lon=lons, lat=lats)
    assert isinstance(sda, xr.Dataset)

    sda = dataset.pyfesom2.select_points(lon=lons, lat=lats)
    assert isinstance(sda, xr.Dataset)


def test_dataarray_accessor_methods(dataset):
    from shapely.geometry import LineString
    for data_var in dataset.data_vars.keys():
        assert hasattr(dataset.pyfesom2, data_var)

        data_var = list(dataset.data_vars.keys())[0]

        # TODO: do this better, not extensive
        # test select region
        for region in region_tests:
            assert getattr(dataset.pyfesom2, data_var).select(region=region) is not None

        # test select points
        npoints = 10
        lons = np.linspace(-180, 180, npoints)
        lats = np.linspace(-90, 90, npoints)

        # TODO: do this better, assert the returned objects correspond to that of selection methods
        assert getattr(dataset.pyfesom2, data_var).select(
            path=(lons, lats)) is not None, 'select cannot take lon,lat as sequence'
        shapely_path = LineString(np.column_stack([lons, lats]))
        assert getattr(dataset.pyfesom2, data_var).select(
            path=shapely_path) is not None, 'select cannot take lon,lat as LineString'
        dict_path = {'lon': lons, 'lat': lats}
        assert getattr(dataset.pyfesom2, data_var).select(
            path=dict_path) is not None, 'select cannot take lon,lat as dictionary'
        assert getattr(dataset.pyfesom2, data_var).select_points(lon=lons, lat=lats) is not None


def test_accessor_mesh_plotting(dataset):
    """Basic tests to check if plot_mesh exists on dataset and dataarray accessor exists
    and return right type."""

    from matplotlib.lines import Line2D

    assert hasattr(dataset.pyfesom2, 'plot_mesh')
    mesh_plot = dataset.pyfesom2.plot_mesh()
    assert isinstance(mesh_plot, list)
    assert isinstance(mesh_plot[0], Line2D)

    for data_var in dataset.data_vars.keys():
        mesh_plot = getattr(dataset.pyfesom2, data_var).plot_mesh()
        assert isinstance(mesh_plot[0], Line2D)


def test_accessor_contour_plotting(dataset):
    """Basic tests to check if contour plotting methods exist on accessor and return right type."""
    from matplotlib.tri.tricontour import TriContourSet
    from matplotlib.contour import QuadContourSet
    from matplotlib.collections import PolyCollection

    # reduce all other dims then nod2 to simplify plotting
    sel_da = dataset.isel(**{dim: 1 for dim in list(dataset.dims) if not dim in ['nod2', 'elem', 'nelem', 'three']})

    for data_var in sel_da.data_vars.keys():
        # contours
        contour_plot = getattr(sel_da.pyfesom2, data_var).contour()
        assert isinstance(contour_plot, TriContourSet)
        # filled contours
        contour_plot = getattr(sel_da.pyfesom2, data_var).contourf()
        assert isinstance(contour_plot, TriContourSet)
        # raster plot
        raster_plot = getattr(sel_da.pyfesom2, data_var).pcolor()
        assert isinstance(raster_plot, PolyCollection)
