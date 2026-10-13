'''Plot global ice-thickness products for Heard Island.'''

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
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


#%% Paths and data

root = Path(__file__).resolve().parents[1]
datapath = root / 'data'
outpath = root / 'figure'
outpath.mkdir(parents=True, exist_ok=True)

thickness_files = [
    datapath / 'Thickness_Farinotti2019_RGI60_native25m.tif',
    datapath / 'Thickness_Millan2022_native50m.tif',
    datapath / 'Thickness_IceBoost_v2_RGI70_native100m.tif',
    datapath / 'Thickness_Frank_Heard_Island_native100m.tif',
]
panel_titles = [
    '(A) Farinotti et al. (2019)',
    '(B) Millan et al. (2022)',
    '(C) Maffezzoli et al. (2026)',
    '(D) Frank et al. (2026)',
]
island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
rgi6 = gpd.read_file(
    datapath / 'RGI60_Heard_Island.shp'
).to_crs('EPSG:32743')
rgi7 = gpd.read_file(
    datapath / 'RGI70_Heard_Island.shp'
).to_crs('EPSG:32743')
glacier_outlines = [rgi6, rgi6, rgi7, rgi6]

thickness_grids = []
raster_extents = []
thickness_statistics = []

for thickness_file in thickness_files:
    with rasterio.open(thickness_file) as source:
        thickness = source.read(1).astype('float64')
        if source.nodata is not None:
            thickness[thickness == source.nodata] = np.nan
        thickness[~np.isfinite(thickness) | (thickness <= 0)] = np.nan
        thickness_grids.append(thickness)
        valid_thickness = thickness[np.isfinite(thickness)]
        thickness_statistics.append({
            'mean': np.mean(valid_thickness),
            'maximum': np.max(valid_thickness),
        })
        raster_extents.append([
            source.bounds.left,
            source.bounds.right,
            source.bounds.bottom,
            source.bounds.top,
        ])


# Use exactly the same colour scale as Figure 5. Its upper limit is the
# 99th percentile of the reconstructed 1947 ice-thickness distribution.
with xr.open_dataset(
    datapath / 'Heard_Island_1947_ice_thickness.nc'
) as dataset:
    reference_thickness = dataset['ice_thickness'].sel(
        shape_factor=0.8,
        basal_shear_stress_factor=1.0,
    ).values.astype('float64')
    reference_thickness[
        ~np.isfinite(reference_thickness)
    ] = np.nan

colourbar_maximum = np.nanpercentile(reference_thickness, 99)


#%% Plot Figure S5

ice_cmap = plt.get_cmap('viridis').copy()
ice_cmap.set_bad((1, 1, 1, 0))

fig_width_inch = 14 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 5.0), dpi=600)
grid = fig.add_gridspec(
    2,
    2,
    left=0.015,
    right=0.995,
    bottom=0.105,
    top=0.99,
    wspace=0.04,
    hspace=0.12,
)

minimum_x, minimum_y, maximum_x, maximum_y = island.total_bounds
colour_norm = Normalize(vmin=0.0, vmax=colourbar_maximum)
map_axes = []

for index in range(4):
    row = index // 2
    column = index % 2
    axis = fig.add_subplot(grid[row, column])
    map_axes.append(axis)

    island.plot(
        ax=axis,
        facecolor='#eeeeee',
        edgecolor='0.35',
        linewidth=0.45,
        zorder=1,
    )
    thickness_image = axis.imshow(
        thickness_grids[index],
        extent=raster_extents[index],
        origin='upper',
        cmap=ice_cmap,
        norm=colour_norm,
        interpolation='nearest',
        zorder=2,
    )
    glacier_outlines[index].boundary.plot(
        ax=axis,
        color='black',
        linewidth=0.20,
        zorder=3,
    )

    axis.set_xlim(minimum_x, maximum_x)
    axis.set_ylim(minimum_y, maximum_y)
    axis.set_aspect('equal')
    axis.set_axis_off()
    axis.set_title(
        panel_titles[index],
        fontsize=10,
        fontweight='bold'
    )
    axis.text(
        0.03,
        0.015,
        (
            'Ice thickness\n'
            f'Mean: {thickness_statistics[index]["mean"]:.1f} m\n'
            f'Maximum: {thickness_statistics[index]["maximum"]:.1f} m'
        ),
        transform=axis.transAxes,
        ha='left',
        va='bottom',
        fontsize=8,
        color='black',
        zorder=4,
    )

ticks = np.array([0.0, 100.0, 200.0, colourbar_maximum])
colourbar_axis = fig.add_axes([
    0.317,
    0.075,
    0.376,
    0.023,
])
colourbar = fig.colorbar(
    thickness_image,
    cax=colourbar_axis,
    orientation='horizontal',
    ticks=ticks,
    extend='max',
)
colourbar.set_label('Ice thickness (m)', fontsize=8)
colourbar.ax.tick_params(labelsize=7)
colourbar.outline.set_linewidth(0.5)
colourbar.ax.set_xticklabels([f'{tick:.0f}' for tick in ticks])

png_path = outpath / 'Figure_S5.png'
pdf_path = outpath / 'Figure_S5.pdf'

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    pdf_path,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02,
)
plt.close(fig)
