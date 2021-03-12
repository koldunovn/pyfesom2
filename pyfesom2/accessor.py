import warnings
from typing import Optional, Sequence, Union, MutableMapping, Tuple

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon, LineString

# New Types
BoundingBox = Sequence[float]
Region = Union[BoundingBox, Polygon]
MultiRegion = Union[Sequence[Polygon], MultiPolygon]
ArrayLike = Union[Sequence[float], np.ndarray, xr.DataArray]
Path = Union[LineString, Tuple[ArrayLike, ArrayLike]]


# Selection

# ---Utilities for selection

def distance_along_trajectory(lons: ArrayLike, lats: ArrayLike) -> ArrayLike:
    """Returns geodesic distance along a trajectory of lons and lons.

    Computes cumulative distance from starting lon, lat till end of array.

    Parameters
    ----------
    lons
        Array-like longitude values.
    lats
        Array-like latitude values.

    Returns
    -------
    ArrayLike
        Returns array containing distances in meters
    """
    from cartopy.geodesic import Geodesic
    geod = Geodesic()
    lons, lats = np.array(lons, ndmin=1, copy=False), np.array(lats, ndmin=1, copy=False)

    if np.ndim(lons) > 2:
        raise NotImplementedError('More then 2 dims in lons are currenty not supported')

    dists = np.zeros(lons.shape)

    if np.ndim(lons) == 1:
        points = np.c_[lons, lats]
        temp_dist = geod.inverse(points[0:-1], points[1:])[:, 0]
        dists[1:] = np.cumsum(temp_dist)
    else:
        for i, (_lons, _lats) in enumerate(zip(lons, lats)):
            points = np.c_[_lons, _lats]
            temp_dist = geod.inverse(points[0:-1], points[1:])[:, 0]
            dists[i, 1:] = np.cumsum(temp_dist)

    return dists


def normalize_distance(distance_array_in_m: ArrayLike) -> Tuple[str, ArrayLike]:
    """Returns best representation for distances in m or km and values.

    Parameters
    ----------
    distance_array_in_m

    Returns
    -------
    tuple
        Returns tuple containing best units in m or km and transformed values.
    """
    distance_array_in_km = distance_array_in_m / 1000.0
    len_array = distance_array_in_m.shape[0]
    # if more then 1/3 of points are best suited to be expressed in m else in km
    if np.count_nonzero(distance_array_in_km < 1) > len_array // 3:
        return "m", distance_array_in_m
    else:
        return "km", distance_array_in_km


class SimpleMesh:
    """Wrapper that fakes pyfesom's mesh object for purposes of this module"""

    def __init__(self, lon: ArrayLike, lat: ArrayLike, faces: ArrayLike):
        self.x2 = lon
        self.y2 = lat
        self.elem = faces


# ---Selection functions

def select_bbox(xr_obj: Union[xr.DataArray, xr.Dataset],
                bbox: BoundingBox,
                faces: Optional[ArrayLike] = None) -> xr.Dataset:
    """Returns subset Dataset or DataArray for bounding box.

    This method uses triangulation indices in faces (as argument or as coordinate in a dataset) to select nodes
    belonging to faces in bounding box. Hence, nodes that belong to faces entirely contained in bounding box are
    returned. A Xarray dataset is returned regardless of input type to retain face coordinate information in the subset.
    Returned values of faces in returned subset correspond to triangulation using new indices of nodes.
    As this method uses triangulation information, it is somewhat true to underlying grid unlike other methods which
    start with node information.

    Parameters
    ----------
    xr_obj
        Xarray's Dataset or DaraArray, for DataArrays faces argument is necessary.
    bbox
        Bounding box can be specified as as sequence of size 4 (lists or tuple or array) containing bounds
        from lower-left to upper-right of longitudes and latitudes. For instance: (xmin, ymin, xmax, ymax).
    faces
        For Datasets containing faces as coordinate information the argument is not necessary.
        For DataArrays faces argument, defining indices of faces defining triangles, is necessary.

    Returns
    -------

    """
    from .ut import cut_region
    faces = getattr(xr_obj, "faces", faces)
    if faces is None:
        raise ValueError(f"When passing a dataset it needs have faces in coords, or "
                         f"faces need to be passed explicitly.\n"
                         f"When passing a data array, argument faces can't be None,"
                         f"faces must be indices[nelem,3] that define triangles.")

    mesh = SimpleMesh(xr_obj.lon, xr_obj.lat, faces)
    bbox = np.asarray(bbox)
    # cut region takes xmin, xmax, ymin, ymax
    cut_faces, _ = cut_region(mesh, [bbox[0], bbox[2], bbox[1], bbox[3]])
    cut_faces = np.asarray(cut_faces)
    uniq, inv_index = np.unique(cut_faces.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(cut_faces.shape)
    ret = xr_obj.isel(nod2=uniq)
    if isinstance(xr_obj, xr.DataArray):
        ret = ret.to_dataset()
    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select_region(xr_obj: Union[xr.DataArray, xr.Dataset], region: Region,
                  faces: Optional[ArrayLike] = None) -> xr.Dataset:
    from shapely.geometry import box, Polygon
    from shapely.prepared import prep
    from shapely.vectorized import contains as vectorized_contains

    if isinstance(region, Sequence) and len(region) == 4:
        region = box(*region)
    elif isinstance(region, Polygon):
        region = region
    else:
        raise ValueError(f"Supplied region data can be a sequence of (minlon, minlat, maxlon, maxlat) or "
                         f"a Shapely's Polygon. This {region} is not supported.")

    faces = getattr(xr_obj, "faces", faces)
    if faces is None:
        raise ValueError(f"When passing a dataset it needs have faces in coords, or"
                         f"faces need to be passed explicitly.\n"
                         f"When passing a data array, argument faces can't be None,"
                         f"faces must be indices[nelem,3] that define triangles.")
    faces = np.asarray(faces)

    # buffer is necessry to facilitte floating point comparisions
    # buffer can be thought as tolerance around region in degrees
    # its value should be at least precision of data type of lats, lons (np.finfo)
    region = region.buffer(1e-6)
    prep_region = prep(region)
    selection = vectorized_contains(prep_region, np.asarray(xr_obj.lon), np.asarray(xr_obj.lat))
    if np.count_nonzero(selection) == 0:
        warnings.warn('No points in domain are within region, returning original data.')
        return xr_obj

    selection = selection[faces]
    face_mask = np.all(selection, axis=1)
    cut_faces = faces[face_mask]
    cut_faces = np.array(cut_faces, ndmin=1)
    uniq, inv_index = np.unique(cut_faces.ravel(), return_inverse=True)
    new_faces = inv_index.reshape(cut_faces.shape)
    ret = xr_obj.isel(nod2=uniq)

    if 'faces' in ret.coords:
        ret = ret.drop_vars('faces')

    if len(uniq) == 0:
        warnings.warn("No found points for the region are contained in dataset's triangulation (faces), "
                      "returning object without faces.")
        return ret  # no faces in coords

    if isinstance(xr_obj, xr.DataArray):
        ret = ret.to_dataset()

    ret = ret.assign_coords({'faces': (('nelem', 'three'), new_faces)})
    return ret


def select_points(xrobj: Union[xr.Dataset, xr.DataArray],
                  lon: ArrayLike, lat: ArrayLike, method: str = 'nearest', tolerance: Optional[float] = None,
                  tree: Optional[object] = None, return_distance: Optional[bool] = True,
                  selection_dim_name: Optional[str] = "nod2", **other_dims) -> Union[xr.Dataset, xr.DataArray]:
    """

    TODO: check id all dims are of same length.
    """
    from cartopy.crs import Geocentric, Geodetic
    from scipy.spatial import cKDTree
    src_lons, src_lats = np.asarray(xrobj.lon), np.asarray(xrobj.lat)

    set_len_dims = {np.size(lon), np.size(lat), *[np.size(val) for val in other_dims.values()]}

    if len(set_len_dims) > 1:
        raise ValueError('For point selection length of all supplied dims args should be same.')

    if not method == 'nearest':
        raise NotImplementedError("Spatial selection currently supports only nearest neighbor lookup")
    geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
    if tree is None:
        src_pts = geocentric_crs.transform_points(geodetic_crs, src_lons, src_lats)
        tree = cKDTree(src_pts, leafsize=32, compact_nodes=False, balanced_tree=False)

    if isinstance(lon, xr.DataArray) and isinstance(lat, xr.DataArray):
        sel_dim = tuple(lon.dims)
    else:
        sel_dim = selection_dim_name

    dst_pts = geocentric_crs.transform_points(geodetic_crs, np.asarray(lon), np.asarray(lat))

    if tolerance is None:
        _, ind = tree.query(dst_pts)
    else:
        raise NotImplementedError('tolerance is currently not supported.')

    other_dims = {k: xr.DataArray(np.array(v, ndmin=1), dims=sel_dim) for k, v in other_dims.items()}
    ret_obj = xrobj.isel(nod2=xr.DataArray(ind, dims=sel_dim)).sel(**other_dims, method=method)

    # from faces, which will not be useful in returned dataset
    # unless we reindex them, but is there a use case for that?
    if 'faces' in ret_obj.coords:
        ret_obj = ret_obj.drop_vars('faces')
    if return_distance:
        dist = distance_along_trajectory(lon, lat)
        dist_units, dist = normalize_distance(dist)
        ret_obj = ret_obj.assign_coords({'distance': (sel_dim, dist)})
        ret_obj.distance.attrs['units'] = dist_units
        ret_obj.distance.attrs['long_name'] = f"distance along trajectory"
    return ret_obj


def select(xr_obj: Union[xr.Dataset, xr.DataArray], method: str = 'nearest',
           tolerance: float = None, region: Optional[Region] = None,
           path: Optional[Union[Path, MutableMapping]] = None, tree: Optional[object] = None,
           **indexers) -> Union[xr.Dataset, xr.DataArray]:
    """
    Higher level interface that does different kinds of selection emulates xarray's sel method.
    """
    lat = indexers.pop('lat', None)
    lon = indexers.pop('lon', None)
    lat_indexer = True if lat is not None else False
    lon_indexer = True if lon is not None else False

    if (lat_indexer or lon_indexer) and (region is not None or path is not None):
        # TODO: do this combinations better, doesn't check if path and region are both given
        raise ValueError("Only one option: lat, lon as indexer or path or region is supported")

    ret_arr = xr_obj

    if lat_indexer or lon_indexer:
        if lat_indexer and lon_indexer:
            if method == 'nearest':
                ret_arr = select_points(xr_obj, lon, lat, method=method, tolerance=tolerance, tree=tree,
                                        return_distance=False)
            else:
                raise NotImplementedError("Only method='nearest' is currently supported.")
        else:
            raise ValueError("Both lat, lon are needed as indexers, else use path, region arguments or "
                             ".select_points(lon=..., lat=...) method.")
    elif region is not None:
        ret_arr = select_region(xr_obj, region)
    elif path is not None:
        if isinstance(path, Sequence) or isinstance(path, LineString):
            if isinstance(path, LineString):
                path = np.asarray(path).T
            else:
                path = np.asarray(path)

            if not np.ndim(path) == 2:
                raise ValueError('Path of more then 2 columns (lons, lats) is ambiguous, use dictionary instead')
            else:
                lon, lat = path
                ret_arr = select_points(xr_obj, lon, lat, method=method, tolerance=tolerance, tree=tree)
        elif isinstance(path, dict):
            ret_arr = select_points(xr_obj, method=method, tolerance=tolerance, tree=tree, **path)
        else:
            raise ValueError('Invalid path argument it can only be sequence of (lons, lats), shapely 2D LineString or'
                             'dictionary containing coords.')

    # xarray doesn't support slice indexer when method argument is passed.
    # allow mixing indexers with values and slices.
    slice_indexers = {dim: dim_val for dim, dim_val in indexers.items() if isinstance(dim_val, slice)}

    if slice_indexers:
        ret_arr = ret_arr.sel(**slice_indexers)
        # remove slice indexers from indexers
        for dim in slice_indexers.keys():
            indexers.pop(dim)

    return ret_arr.sel(**indexers, method=method)


# Accessors

# Dataset accessor

@xr.register_dataset_accessor("pyfesom2")
class FESOMDataset:
    def __init__(self, xr_dataset: xr.Dataset):
        self._xrobj = xr_obj = xr_dataset
        # TODO: check valid fesom data? otherwise accessor is available on all xarray datasets
        self._tree_obj = None
        for datavar in xr_obj.data_vars.keys():
            setattr(self, str(datavar), FESOMDataArray(xr_obj[datavar], xr_obj))

    def select(self, method: str = 'nearest', tolerance: Optional[float] = None, region: Optional[Region] = None,
               path: Optional[Path] = None, **indexers):
        sel_obj = select(self._xrobj, method=method, tolerance=tolerance, region=region, path=path, **indexers)
        return sel_obj

    def select_points(self, lon: ArrayLike, lat: ArrayLike, method: str = 'nearest',
                      tolerance: Optional[float] = None, **other_dims):
        tree = self._tree
        return select_points(self._xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree, return_distance=True,
                             **other_dims)

    def _build_tree(self):
        from cartopy.crs import Geocentric, Geodetic
        from scipy.spatial import cKDTree
        geocentric_crs, geodetic_crs = Geocentric(), Geodetic()
        src_pts = geocentric_crs.transform_points(geodetic_crs, np.asarray(self._xrobj.lon),
                                                  np.asarray(self._xrobj.lat))
        self._tree_obj = cKDTree(src_pts, leafsize=32, compact_nodes=False, balanced_tree=False)
        return self._tree_obj

    @property
    def _tree(self):
        """Property to regulate tree access, _tree to hide from jupyter notebook"""
        if self._tree_obj is not None:
            return self._tree_obj
        return self._build_tree()

    def __repr__(self):
        return self._xrobj.__repr__()

    def _repr_html_(self):
        return self._xrobj._repr_html_()


class FESOMDataArray:
    """ A wrapper around Dataarray, that passes dataset context around"""

    def __init__(self, xr_dataarray: xr.DataArray, context_dataset: Optional[xr.Dataset] = None):
        self._xrobj = xr_dataarray
        self._context_dataset = context_dataset
        self._native_projection = ccrs.PlateCarree()

    def select(self, method: str = 'nearest', tolerance: float = None, region: Optional[Region] = None,
               path: Optional[Path] = None, **indexers):
        sel_obj = self._xrobj.to_dataset()
        sel_obj = sel_obj.assign_coords({'faces': (self._context_dataset.faces.dims,
                                                   self._context_dataset.faces.values)})
        tree = self._context_dataset.pyfesom2._tree
        sel_obj = select(sel_obj, method=method, tolerance=tolerance, region=region, path=path, tree=tree,
                         **indexers)
        return sel_obj

    def select_points(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], method: str = 'nearest',
                      tolerance: Optional[float] = None, **other_dims):
        tree = self._context_dataset.pyfesom2._tree
        return select_points(self._xrobj, lon, lat, method=method, tolerance=tolerance, tree=tree, return_distance=True,
                             **other_dims)

    def __repr__(self):
        return f"Wrapped {self._xrobj.__repr__()}\n{super().__repr__()}"

    def _repr_html_(self):
        return self._xrobj._repr_html_()
