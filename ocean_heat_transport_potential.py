import os
import datetime
import calendar
import inspect
import pickle

import netCDF4
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

import cartopy
import cartopy.util
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cmocean.cm

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

data_dir_path = 'D:\\data\\'


def distance(lat1, lon1, lat2, lon2):
    R = 6371.228e3  # average radius of the earth [m]

    # Calculate the distance between two points on the Earth (lat1, lon1) and (lat2, lon2) using the haversine formula.
    # See: http://www.movable-type.co.uk/scripts/latlong.html

    lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


def plot_scalar_field(lats, lons, field):  # , cmap, vmin, vmax):
    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    field, lons = cartopy.util.add_cyclic_point(field, coord=lons)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    # im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.contourf(lons, lats, field/1e12, transform=vector_crs, cmap=cm.get_cmap('viridis', 15), vmin=-8, vmax=8)

    m = plt.cm.ScalarMappable(cmap=cm.get_cmap('viridis', 15))
    m.set_array(field)
    m.set_clim(-8, 8)

    clb = fig.colorbar(m, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'$\phi_o$ GW')
    # clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)

    plt.show()
    plt.close()


def plot_vector_field(lats, lons, field, u, v):  # , cmap, vmin, vmax):
    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    u, lons_cyclic = cartopy.util.add_cyclic_point(u, coord=lons)
    v, lons_cyclic = cartopy.util.add_cyclic_point(v, coord=lons)
    field, lons_cyclic = cartopy.util.add_cyclic_point(field, coord=lons)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    # im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.contourf(lons_cyclic, lats, field/1e6, transform=vector_crs, cmap=cm.get_cmap('seismic', 15), vmin=-1, vmax=1)

    Q1 = ax.quiver(lons_cyclic[::3], lats[::3], u[::3, ::3]/1e6, v[::3, ::3]/1e6,
                   pivot='middle', transform=vector_crs, units='width', width=0.002, scale=50)

    m = plt.cm.ScalarMappable(cmap=cm.get_cmap('seismic', 15))
    m.set_array(u)
    m.set_clim(-1, 1)

    clb = fig.colorbar(m, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'MW/m')

    plt.show()
    plt.close(fig)
