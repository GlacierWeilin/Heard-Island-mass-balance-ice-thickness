import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'legend.fontsize': 8})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth': 0.5})
plt.rcParams.update({'axes.titlepad': 3})
plt.rcParams.update({'axes.titlesize': 9})
plt.rcParams.update({'axes.labelpad': 2})
plt.rcParams.update({'xtick.major.pad': 2})
plt.rcParams.update({'ytick.major.pad': 2})
plt.rcParams.update({'xtick.major.width': 0.5})
plt.rcParams.update({'ytick.major.width': 0.5})
plt.rcParams.update({'xtick.major.size': 1.5})
plt.rcParams.update({'ytick.major.size': 1.5})

# %% add data

root = Path(__file__).resolve().parents[1]
datapath = root / 'data'
outpath = root / 'figure'
outpath.mkdir(parents=True, exist_ok=True)

hugonnet_file = datapath / 'Hugonnet_2000-01-01_2020-01-01_dhdt_100m.tif'
study_file = datapath / 'NASADEM_Pleiades_dhdt_20m.tif'

island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
rgi = gpd.read_file(
    datapath / 'RGI60_Heard_Island.shp'
).to_crs('EPSG:32743')
study_glaciers = gpd.read_file(
    datapath / '2019_Heard_SPEIADES.shp'
).to_crs('EPSG:32743')

# %% read Hugonnet dh/dt on its original 100 m grid

with rasterio.open(hugonnet_file) as source:
    hugonnet = source.read(1).astype('float64')
    hugonnet_transform = source.transform
    hugonnet_crs = source.crs
    hugonnet_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        hugonnet[hugonnet == source.nodata] = np.nan

hugonnet_outside_island = geometry_mask(
    island.geometry,
    out_shape=hugonnet.shape,
    transform=hugonnet_transform,
    invert=False
)
hugonnet[hugonnet_outside_island] = np.nan

# %% read this study and resample Hugonnet to the 20 m reference grid

with rasterio.open(study_file) as source:
    study = source.read(1).astype('float64')
    study_transform = source.transform
    study_crs = source.crs
    study_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        study[study == source.nodata] = np.nan

hugonnet_20m = np.full(study.shape, np.nan, dtype='float64')
reproject(
    source=hugonnet,
    destination=hugonnet_20m,
    src_transform=hugonnet_transform,
    src_crs=hugonnet_crs,
    src_nodata=np.nan,
    dst_transform=study_transform,
    dst_crs=study_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear
)

study_outside_island = geometry_mask(
    island.geometry,
    out_shape=study.shape,
    transform=study_transform,
    invert=False
)
study[study_outside_island] = np.nan
hugonnet_20m[study_outside_island] = np.nan

difference = study - hugonnet_20m
difference[
    ~np.isfinite(study) |
    ~np.isfinite(hugonnet_20m)
] = np.nan

# %% setup figure

fig_width_inch = 14 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 2.5), dpi=600)

grid = fig.add_gridspec(
    1, 2,
    left=0.015,
    right=0.995,
    bottom=0.16,
    top=0.99,
    wspace=0.04
)

dhdt_norm = Normalize(
    vmin=-5.0,
    vmax=3.0
)
difference_norm = TwoSlopeNorm(
    vmin=-3.0,
    vcenter=0,
    vmax=3.0
)
dhdt_cmap = LinearSegmentedColormap.from_list(
    'dhdt_shifted',
    [
        (0.000, '#67001f'),
        (0.250, '#d6604d'),
        (0.500, '#f4a582'),
        (0.625, '#f7f7f7'),
        (0.750, '#92c5de'),
        (0.875, '#4393c3'),
        (1.000, '#053061')
    ]
)
difference_cmap = plt.get_cmap('RdBu')

map_minimum_x, map_minimum_y, map_maximum_x, map_maximum_y = (
    island.total_bounds
)
map_padding_x = (map_maximum_x - map_minimum_x) * 0.005
map_padding_y = (map_maximum_y - map_minimum_y) * 0.01

############################################################################ Figure_S5A
ax_a = fig.add_subplot(grid[0, 0])

island.plot(
    ax=ax_a,
    facecolor='#eeeeee',
    edgecolor='0.35',
    linewidth=0.5
)
hugonnet_image = ax_a.imshow(
    hugonnet,
    extent=hugonnet_extent,
    origin='upper',
    cmap=dhdt_cmap,
    norm=dhdt_norm,
    interpolation='nearest',
    zorder=2
)
rgi.boundary.plot(
    ax=ax_a,
    color='black',
    linewidth=0.25,
    zorder=3
)

ax_a.set_xlim(
    map_minimum_x - map_padding_x,
    map_maximum_x + map_padding_x
)
ax_a.set_ylim(
    map_minimum_y - map_padding_y,
    map_maximum_y + map_padding_y
)
ax_a.set_aspect('equal')
ax_a.set_axis_off()
ax_a.text(
    0.99, 0.99, '(A)',
    fontsize=10,
    fontweight='bold',
    ha='right',
    va='top',
    transform=ax_a.transAxes,
    zorder=10
)

############################################################################ Figure_S5B
ax_b = fig.add_subplot(grid[0, 1])

island.plot(
    ax=ax_b,
    facecolor='#eeeeee',
    edgecolor='0.35',
    linewidth=0.5
)
difference_image = ax_b.imshow(
    difference,
    extent=study_extent,
    origin='upper',
    cmap=difference_cmap,
    norm=difference_norm,
    interpolation='nearest',
    zorder=2
)
study_glaciers.boundary.plot(
    ax=ax_b,
    color='black',
    linewidth=0.25,
    zorder=3
)

ax_b.set_xlim(
    map_minimum_x - map_padding_x,
    map_maximum_x + map_padding_x
)
ax_b.set_ylim(
    map_minimum_y - map_padding_y,
    map_maximum_y + map_padding_y
)
ax_b.set_aspect('equal')
ax_b.set_axis_off()
ax_b.text(
    0.99, 0.99, '(B)',
    fontsize=10,
    fontweight='bold',
    ha='right',
    va='top',
    transform=ax_b.transAxes,
    zorder=10
)

hugonnet_colorbar = fig.colorbar(
    hugonnet_image,
    ax=ax_a,
    orientation='horizontal',
    fraction=0.055,
    pad=0.025,
    ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3],
    extend='both'
)
hugonnet_colorbar.set_label(
    'Elevation-change rate (m yr$^{-1}$)',
    fontsize=8
)
hugonnet_colorbar.ax.tick_params(labelsize=7)
hugonnet_colorbar.outline.set_linewidth(0.5)

difference_colorbar = fig.colorbar(
    difference_image,
    ax=ax_b,
    orientation='horizontal',
    fraction=0.055,
    pad=0.025,
    ticks=[-3, -2, -1, 0, 1, 2, 3],
    extend='both'
)
difference_colorbar.set_label(
    'Difference (this study $-$ Hugonnet; m yr$^{-1}$)',
    fontsize=8
)
difference_colorbar.ax.tick_params(labelsize=7)
difference_colorbar.outline.set_linewidth(0.5)

out_png = outpath / 'Figure_S3.png'
out_pdf = outpath / 'Figure_S3.pdf'

fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out_pdf, dpi=600, bbox_inches='tight', pad_inches=0.02)
