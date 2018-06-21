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
import coloredlogs, logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

data_dir_path = 'D:\\data\\'

# Load land sea mask (land=0, sea=1)
lsmask_filepath = os.path.join(data_dir_path, 'lsmask.oisst.v2.nc')
logger.info('Loading dataset: {}'.format(lsmask_filepath))
lsmask_dataset = netCDF4.Dataset(lsmask_filepath)

lats_lsmask = np.array(lsmask_dataset.variables['lat'])
lons_lsmask = np.array(lsmask_dataset.variables['lon'])
land_sea_mask = np.array(lsmask_dataset.variables['lsmask'])[0]


def is_land(lat, lon):
    idx_lat = np.abs(lats_lsmask - lat).argmin()
    idx_lon = np.abs(lons_lsmask - lon).argmin()
    return False if land_sea_mask[idx_lat][idx_lon] else True


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


def plot_scalar_field(lats, lons, field, cmap, vmin, vmax):
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
    # ax.add_feature(land_50m)
    # ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)

    plt.show()
    plt.close()


def plot_vector_field(lats, lons, field, u, v, cmap, vmin, vmax):
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

    im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)

    Q = ax.quiver(lons_cyclic[::3], lats[::3], u[::3, ::3]/1e6, v[::3, ::3]/1e6,
                  pivot='middle', transform=vector_crs, units='width', width=0.002, scale=50)

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'MW/m')

    plt.show()
    plt.close(fig)


def solve_for_ocean_heat_transport_potential_spherical():
    # Load net upward heat flux dataset.
    nuhf_filepath = os.path.join(data_dir_path, 'net_upward_heatflux.nc')
    logger.info('Loading dataset: {}'.format(nuhf_filepath))
    nuhf_dataset = netCDF4.Dataset(nuhf_filepath)

    lats_nuhf = np.array(nuhf_dataset.variables['Y'])  # TODO: But this includes 90 N and 90 S!
    lons_nuhf = np.array(nuhf_dataset.variables['X'])
    net_upward_heat_flux = np.array(nuhf_dataset.variables['asum'])
    net_upward_heat_flux, lons_nuhf = cartopy.util.add_cyclic_point(net_upward_heat_flux, coord=lons_nuhf)

    # Load land sea mask (land=0, sea=1)
    lsmask_filepath = os.path.join(data_dir_path, 'lsmask.oisst.v2.nc')
    logger.info('Loading dataset: {}'.format(lsmask_filepath))
    lsmask_dataset = netCDF4.Dataset(lsmask_filepath)

    lats_lsmask = np.array(lsmask_dataset.variables['lat'])
    lons_lsmask = np.array(lsmask_dataset.variables['lon'])
    land_sea_mask = np.array(lsmask_dataset.variables['lsmask'])[0]

    # plot_scalar_field(lats_nuhf, lons_nuhf, net_upward_heat_flux, 'BrBG', -200, 200)
    # plot_scalar_field(lats_lsmask, lons_lsmask, land_sea_mask, 'RdBu', 0, 1)

    # Setting up the linear system A*chi = f for the discretized Poisson equation with Dirchlet boundary conditions on
    # a sphere.
    lats, lons = np.deg2rad(lats_nuhf), np.deg2rad(lons_nuhf)
    m, n = lats.size, lons.size
    delta_phi, delta_lambda = np.pi/m, 2*np.pi/n

    beta = 4 / (n * delta_phi**2)

    logger.info('m = {:d}, n = {:d}, delta_phi = {:f}, delta_lambda = {:f}'.format(m, n, delta_phi, delta_lambda))

    A = sparse.lil_matrix((m*n+2, m*n+2))
    chi = sparse.lil_matrix((m*n+2, 1))
    f = sparse.lil_matrix((m*n+2, 1))

    # Applying Dirchlet boundary condition that chi=0 over land.
    for i in np.arange(0, n):
        lat = lats_nuhf[i]
        for j in np.arange(0, m):
            lon = lons_nuhf[j]
            if is_land(lat, lon):
                chi[i*n + j*m] = 0

    # North pole condition
    A[0, 0] = -beta*n
    for j in np.arange(1, n+1):
        A[0, j*m] = beta

    f[0] = np.mean(net_upward_heat_flux[0, :])  # Use average value at 90 N to get f_N.

    # South pole condition
    A[m*n+1, m*n+1] = -beta*n
    for j in np.arange(0, n):
        A[m*n+1, j*m] = beta

    f[m*n+1] = np.mean(net_upward_heat_flux[-1, :])  # Use average value at 90 S to get f_S.

    # Interior points
    for i in np.arange(0, n):
        phi_i = i * delta_phi
        phi_imh = (i - 0.5) * delta_phi
        phi_iph = (i + 0.5) * delta_phi

        a_i = np.sin(phi_imh) / (delta_phi**2 * np.sin(phi_i))
        b_i = np.sin(phi_iph) / (delta_phi**2 * np.sin(phi_i))
        d_i = 1 / (delta_lambda**2 * np.sin(phi_i)**2)

        for j in np.arange(1, m+1):
            A[i*n + j*m, i*n + j*m - 1] = a_i               # Coefficient of chi(i-1,j)
            A[i*n + j*m, i*n + j*m + 1] = b_i               # Coefficient of chi(i+1,j)
            A[i*n + j*m, i*n + j*m - m] = d_i               # Coefficient of chi(i,j-1)
            A[i*n + j*m, i*n + j*m + m] = d_i               # Coefficient of chi(i,j+1)
            A[i*n + j*m, i*n + j*m] = -(a_i + b_i + 2*d_i)  # Coefficient of chi(i,j)

            # Construct the RHS vector source term
            f[i*n + j*m] = net_upward_heat_flux[i, j]

    # 4. Use conjugate gradient iterative method from SciPy to solve this system.


def solve_for_ocean_heat_transport_potential_cartesian():
    # Load net upward heat flux dataset.
    nuhf_filepath = os.path.join(data_dir_path, 'net_upward_heatflux.nc')
    logger.info('Loading dataset: {}'.format(nuhf_filepath))
    nuhf_dataset = netCDF4.Dataset(nuhf_filepath)

    lats_nuhf = np.array(nuhf_dataset.variables['Y'])
    lons_nuhf = np.array(nuhf_dataset.variables['X'])
    net_upward_heat_flux = np.array(nuhf_dataset.variables['asum'])

    # Setting net upward flux to zero over land.
    for i in range(len(lats_nuhf)):
        for j in range(len(lons_nuhf)):
            lat, lon = lats_nuhf[i], lons_nuhf[j]
            if is_land(lat, lon):
                net_upward_heat_flux[i, j] = 0

    # Normalize array to integrate to zero, to satisfy the compatibility condition.
    logger.info('Before normalization: sum={:f}, mean={:f}'
                .format(np.sum(net_upward_heat_flux), np.mean(net_upward_heat_flux)))

    net_upward_heat_flux = net_upward_heat_flux - np.mean(net_upward_heat_flux)

    logger.info('After normalization: sum={:f}, mean={:f}'
                .format(np.sum(net_upward_heat_flux), np.mean(net_upward_heat_flux)))

    plot_scalar_field(lats_nuhf, lons_nuhf, net_upward_heat_flux, cmap=cmocean.cm.balance, vmin=-100, vmax=100)

    # Setting up the linear system A*u = f for the discretized Poisson equation.
    m, n = lons_nuhf.size, lats_nuhf.size

    # A = sparse.lil_matrix((m*n, m*n))
    A = np.zeros((m * n, m * n))
    f = np.zeros((m * n, 1))

    # Using the 90S and 90N rows as ghost points.
    # lats_nuhf = lats_nuhf[1:-1]

    for j in np.arange(1, n - 1):
        for i in np.arange(m):
            # Taking modulus of i-1 and i+1 to get the correct index in the special cases of
            #  * i=0 (180 W) and need to use the value from i=m (180 E)
            #  * i=m (180 E) and need to use the value from i=0 (180 W)
            im1 = (i - 1) % m
            ip1 = (i + 1) % m

            dx_j = distance(lats_nuhf[j], lons_nuhf[0], lats_nuhf[j + 1], lons_nuhf[0])
            dy = distance(lats_nuhf[j], lons_nuhf[0], lats_nuhf[j], lons_nuhf[1])

            beside_land = False

            # Imposing Neumann boundary conditions at continental boundaries.
            if is_land(lats_nuhf[j-1], lons_nuhf[i]):
                A[j*m + i, j*m + i - m] = 1
                beside_land = True
            if is_land(lats_nuhf[j+1], lons_nuhf[i]):
                A[j*m + i, j*m + i + m] = 1
                beside_land = True
            if is_land(lats_nuhf[j], lons_nuhf[im1]):
                A[j*m + i, j*m + i - 1] = 1
                beside_land = True
            if is_land(lats_nuhf[j], lons_nuhf[ip1]):
                A[j*m + i, j*m + i + 1] = 1
                beside_land = True

            if not beside_land:
                A[j * m + i, j * m + i - 1] = 1 / dx_j ** 2  # Coefficient of u(i-1,j)
                A[j * m + i, j * m + i + 1] = 1 / dx_j ** 2  # Coefficient of u(i+1,j)
                A[j * m + i, j * m + i - m] = 1 / dy ** 2  # Coefficient of u(i,j-1)
                A[j * m + i, j * m + i + m] = 1 / dy ** 2  # Coefficient of u(i,j+1)
                A[j * m + i, j * m + i] = -2 * (dx_j ** 2 + dy ** 2) / (dx_j ** 2 * dy ** 2)  # Coefficient of u(i,j)

            f[j * m + i] = net_upward_heat_flux[j, i]

    logger.info('net_upward_heat_flux.shape={:}'.format(net_upward_heat_flux.shape))
    logger.info('A.shape={:}, f.shape={:}'.format(A.shape, f.shape))
    logger.info('m={:d}, n={:d}'.format(m, n))

    def report(xk):
        frame = inspect.currentframe().f_back
        print('iter={:d} resid={:f} info={:} ndx1={:} ndx2={:} sclr1={:} sclr2={:} ijob={:}'
              .format(frame.f_locals['iter_'], frame.f_locals['resid'], frame.f_locals['info'], frame.f_locals['ndx1'],
                      frame.f_locals['ndx2'], frame.f_locals['sclr1'], frame.f_locals['sclr2'], frame.f_locals['ijob']))

    u, _ = sparse_linalg.cgs(A, f, callback=report)
    # u, _ = sparse_linalg.cg(A, f, callback=report)

    pickle_filepath = 'D:\\output\\u_cg.pickle'

    # Create directory if it does not exist already.
    pickle_dir = os.path.dirname(pickle_filepath)
    if not os.path.exists(pickle_dir):
        logger.info('Creating directory: {:s}'.format(pickle_dir))
        os.makedirs(pickle_dir)

    with open(pickle_filepath, 'wb') as f:
        pickle.dump(u, f, pickle.HIGHEST_PROTOCOL)

    with open(pickle_filepath, 'rb') as f:
        u = pickle.load(f)

    phi = np.reshape(u, (n, m))

    phi[:, 0] = phi[:, 2]
    phi[:, -1] = phi[:, -3]

    phi_x = np.zeros(phi.shape)
    phi_y = np.zeros(phi.shape)

    for j in np.arange(1, n - 1):
        for i in np.arange(m):
            # Taking modulus of i-1 and i+1 to get the correct index in the special cases of
            #  * i=0 (180 W) and need to use the value from i=m (180 E)
            #  * i=m (180 E) and need to use the value from i=0 (180 W)
            im1 = (i - 1) % m
            ip1 = (i + 1) % m

            dx_j = distance(lats_nuhf[j], lons_nuhf[0], lats_nuhf[j + 1], lons_nuhf[0])
            dy = distance(lats_nuhf[j], lons_nuhf[0], lats_nuhf[j], lons_nuhf[1])

            phi_x[j, i] = (phi[j, ip1] - phi[j, im1]) / (2 * dx_j)
            phi_y[j, i] = (phi[j + 1, i] - phi[j - 1, i]) / (2 * dx_j)

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    phi, lons_cyclic = cartopy.util.add_cyclic_point(phi, coord=lons_nuhf)
    phi_x, lons_cyclic = cartopy.util.add_cyclic_point(phi_x, coord=lons_nuhf)
    phi_y, lons_cyclic = cartopy.util.add_cyclic_point(phi_y, coord=lons_nuhf)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # ax.add_feature(land_50m)
    # ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.8, linestyle='--')
    LON_TICKS = [-180, -90, 0, 90, 180]
    LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(LON_TICKS)
    gl.ylocator = mticker.FixedLocator(LAT_TICKS)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    im = ax.contourf(lons_cyclic, lats_nuhf, phi, transform=vector_crs, cmap=cmocean.cm.balance)

    # m = plt.cm.ScalarMappable(cmap=cm.get_cmap('viridis', 15))
    # m.set_array(phi)
    # m.set_clim(-8, 8)
    # clb = fig.colorbar(m, extend='both', fraction=0.046, pad=0.1)
    # clb.ax.set_title(r'$\phi_o$ GW')

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)

    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # ax.add_feature(land_50m)
    # ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.8, linestyle='--')
    LON_TICKS = [-180, -90, 0, 90, 180]
    LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(LON_TICKS)
    gl.ylocator = mticker.FixedLocator(LAT_TICKS)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(lons_cyclic, lats_nuhf, phi_x, transform=vector_crs, cmap=cmocean.cm.balance)

    # Q1 = ax.quiver(lons_cyclic[::3], lats_nuhf[::3], phi_x[::3, ::3] / 1e6, phi_y[::3, ::3] / 1e6,
    #                pivot='middle', transform=vector_crs, units='width', width=0.002, scale=50)

    # m = plt.cm.ScalarMappable(cmap=cm.get_cmap('seismic', 15))
    # m.set_array(u)
    # m.set_clim(-1, 1)
    # clb = fig.colorbar(m, extend='both', fraction=0.046, pad=0.1)
    # clb.ax.set_title(r'MW/m')

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)

    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # ax.add_feature(land_50m)
    # ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.8, linestyle='--')
    LON_TICKS = [-180, -90, 0, 90, 180]
    LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(LON_TICKS)
    gl.ylocator = mticker.FixedLocator(LAT_TICKS)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # im = ax.pcolormesh(lons, lats, field, transform=vector_crs, cmap=cmap, vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(lons_cyclic, lats_nuhf, phi_y, transform=vector_crs, cmap=cmocean.cm.balance)

    # Q1 = ax.quiver(lons_cyclic[::3], lats_nuhf[::3], phi_x[::3, ::3] / 1e6, phi_y[::3, ::3] / 1e6,
    #                pivot='middle', transform=vector_crs, units='width', width=0.002, scale=50)
    #
    # m = plt.cm.ScalarMappable(cmap=cm.get_cmap('seismic', 15))
    # m.set_array(u)
    # m.set_clim(-1, 1)
    # clb = fig.colorbar(m, extend='both', fraction=0.046, pad=0.1)
    # clb.ax.set_title(r'MW/m')

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    solve_for_ocean_heat_transport_potential_cartesian()