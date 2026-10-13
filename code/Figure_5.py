from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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
    datapath / 'Heard_Island_1947_ice_thickness.nc',
    datapath / 'Heard_Island_1988_ice_thickness.nc',
    datapath / 'Eastern_glaciers_2001_ice_thickness.nc',
    datapath / 'Heard_Island_2019_ice_thickness.nc',
]
panel_titles = ['(A) 1947', '(B) 1988', '(C) 2001', '(D) 2019']

island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
glaciers_1947 = gpd.read_file(
    datapath / '1947_Heard_TOPO.shp'
).to_crs('EPSG:32743')
glaciers_1988 = gpd.read_file(
    datapath / '1988_Heard_SPOT.shp'
).to_crs('EPSG:32743')
glaciers_2001 = gpd.read_file(
    datapath / 'Eastern_glaciers_2001-2019.shp'
).to_crs('EPSG:32743')
glaciers_2001['Date'] = glaciers_2001['Date'].astype(str).str[:4].astype(int)
glaciers_2001 = glaciers_2001[glaciers_2001['Date'] == 2001].copy()
glaciers_2019 = gpd.read_file(
    datapath / '2019_Heard_SPEIADES.shp'
).to_crs('EPSG:32743')
glacier_outlines = [
    glaciers_1947,
    glaciers_1988,
    glaciers_2001,
    glaciers_2019,
]

thickness_grids = []
raster_extents = []
thickness_statistics = []

for thickness_file in thickness_files:
    with xr.open_dataset(thickness_file) as dataset:
        thickness = dataset['ice_thickness'].sel(
            shape_factor=0.8,
            basal_shear_stress_factor=1.0,
        ).values.astype('float64')
        x = dataset['x'].values
        y = dataset['y'].values
        thickness_grids.append(thickness)
        valid_thickness = thickness[
            np.isfinite(thickness) & (thickness > 0)
        ]
        thickness_statistics.append({
            'mean': np.mean(valid_thickness),
            'maximum': np.max(valid_thickness),
        })
        raster_extents.append(
            [
                x.min() - np.abs(np.diff(x)).mean() / 2,
                x.max() + np.abs(np.diff(x)).mean() / 2,
                y.min() - np.abs(np.diff(y)).mean() / 2,
                y.max() + np.abs(np.diff(y)).mean() / 2,
            ]
        )

# Use the 99th percentile of the 1947 reconstruction (panel A) so isolated
# extreme values do not compress the colour range in all four panels.
colourbar_maximum = np.nanpercentile(thickness_grids[0], 99)


# %% setup figure

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
map_axes = []
colour_norm = Normalize(vmin=0.0, vmax=colourbar_maximum)

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
        fontweight='bold',
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

png_path = outpath / 'Figure_5.png'
pdf_path = outpath / 'Figure_5.pdf'

fig.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig(pdf_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
