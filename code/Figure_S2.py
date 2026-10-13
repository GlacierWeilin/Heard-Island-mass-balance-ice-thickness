import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon, Polygon

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

nasadem_file = datapath / 'NASADEM_2000_ASTER_filled_buffered_20m.tif'
pleiades_mosaic_file = (
    datapath / 'Pleiades_SGM_COP30_filled_buffered_20m.tif'
)

island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')

# %% read gap-filled DEMs

with rasterio.open(nasadem_file) as source:
    nasadem = source.read(1).astype('float64')
    nasadem_transform = source.transform
    nasadem_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        nasadem[nasadem == source.nodata] = np.nan

with rasterio.open(pleiades_mosaic_file) as source:
    pleiades_mosaic = source.read(1).astype('float64')
    pleiades_mosaic_transform = source.transform
    pleiades_mosaic_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        pleiades_mosaic[pleiades_mosaic == source.nodata] = np.nan

island_geometry = island.geometry.union_all()
if island_geometry.geom_type == 'Polygon':
    island_geometry = Polygon(island_geometry.exterior)
elif island_geometry.geom_type == 'MultiPolygon':
    island_geometry = MultiPolygon([
        Polygon(polygon.exterior)
        for polygon in island_geometry.geoms
    ])

outside_island = geometry_mask(
    [island_geometry],
    out_shape=nasadem.shape,
    transform=nasadem_transform,
    invert=False
)
nasadem[outside_island] = np.nan

outside_island = geometry_mask(
    [island_geometry],
    out_shape=pleiades_mosaic.shape,
    transform=pleiades_mosaic_transform,
    invert=False
)
pleiades_mosaic[outside_island] = np.nan

# %% setup map coordinates

map_minimum_x, map_minimum_y, map_maximum_x, map_maximum_y = (
    island.total_bounds
)
map_padding_x = (map_maximum_x - map_minimum_x) * 0.035
map_padding_y = (map_maximum_y - map_minimum_y) * 0.045

elevation_norm = Normalize(vmin=0, vmax=2500)
elevation_cmap = plt.get_cmap('terrain').copy()
elevation_cmap.set_bad('white')

# 
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

############################################################################ Figure_S2A
ax_a = fig.add_subplot(grid[0, 0])

nasadem_image = ax_a.imshow(
    nasadem,
    extent=nasadem_extent,
    origin='upper',
    cmap=elevation_cmap,
    norm=elevation_norm,
    interpolation='nearest'
)
island.boundary.plot(
    ax=ax_a,
    color='black',
    linewidth=0.35,
    zorder=3
)
ax_a.text(
    0.98, 0.98, '(A)',
    fontsize=10,
    fontweight='bold',
    ha='right',
    va='top',
    transform=ax_a.transAxes,
    zorder=6
)

############################################################################ Figure_S2B
ax_b = fig.add_subplot(grid[0, 1], sharex=ax_a, sharey=ax_a)

ax_b.imshow(
    pleiades_mosaic,
    extent=pleiades_mosaic_extent,
    origin='upper',
    cmap=elevation_cmap,
    norm=elevation_norm,
    interpolation='nearest'
)
island.boundary.plot(
    ax=ax_b,
    color='black',
    linewidth=0.35,
    zorder=3
)
ax_b.text(
    0.98, 0.98, '(B)',
    fontsize=10,
    fontweight='bold',
    ha='right',
    va='top',
    transform=ax_b.transAxes,
    zorder=6
)

# %% format axes and colorbar

for axis in [ax_a, ax_b]:
    axis.set_xlim(
        map_minimum_x - map_padding_x,
        map_maximum_x + map_padding_x
    )
    axis.set_ylim(
        map_minimum_y - map_padding_y,
        map_maximum_y + map_padding_y
    )
    axis.set_aspect('equal')
    axis.set_axis_off()

colorbar_axis = fig.add_axes([0.25, 0.132, 0.50, 0.025])

elevation_colorbar = fig.colorbar(
    nasadem_image,
    cax=colorbar_axis,
    orientation='horizontal',
    ticks=[0, 500, 1000, 1500, 2000, 2500],
    extend='max'
)
elevation_colorbar.set_label('Elevation (m a.s.l.)', fontsize=8)
elevation_colorbar.ax.tick_params(labelsize=7)
elevation_colorbar.outline.set_linewidth(0.5)

fig.savefig(
    outpath / 'Figure_S2.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    outpath / 'Figure_S2.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
