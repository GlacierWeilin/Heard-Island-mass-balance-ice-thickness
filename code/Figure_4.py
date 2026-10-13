from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

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

study_balance = pd.read_csv(
    datapath / 'glacier_geodetic_mass_balance.csv'
)
study_balance['specific_balance_mwe_yr'] = pd.to_numeric(
    study_balance['specific_balance_mwe_yr']
)

island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
glacier_labels = gpd.read_file(
    datapath / 'Table_S1_glacier_inventory.shp'
).to_crs('EPSG:32743')
glacier_labels['Fig1_ID'] = pd.to_numeric(
    glacier_labels['Fig1_ID']
).astype(int)
glacier_label_points = gpd.GeoDataFrame(
    glacier_labels[['Fig1_ID']].copy(),
    geometry=gpd.points_from_xy(
        glacier_labels['Longitude'],
        glacier_labels['Latitude']
    ),
    crs='EPSG:4326'
).to_crs('EPSG:32743')

# %% assemble the original 2019 polygons without dissolving split glaciers

study_outline_list = []
for outline_file in (datapath / 'glaciers').glob('glacier_*.shp'):
    glacier_outlines = gpd.read_file(outline_file).to_crs('EPSG:32743')
    glacier_2019 = glacier_outlines[
        pd.to_numeric(glacier_outlines['Date']) == 2019
    ].copy()
    if len(glacier_2019) == 0:
        continue
    glacier_2019['numeric_id'] = pd.to_numeric(
        glacier_2019['ID']
    ).astype(int)
    study_outline_list.append(
        glacier_2019[['numeric_id', 'geometry']]
    )

study_glaciers = pd.concat(
    study_outline_list,
    ignore_index=True
)
study_glaciers = gpd.GeoDataFrame(
    study_glaciers,
    geometry='geometry',
    crs='EPSG:32743'
)
study_glaciers = study_glaciers.merge(
    study_balance[[
        'numeric_id',
        'specific_balance_mwe_yr'
    ]],
    on='numeric_id',
    how='inner'
)

# %% read the complete dh/dt raster

with rasterio.open(
    datapath / 'NASADEM_Pleiades_dhdt_filled_W1988_E2001_20m.tif'
) as source:
    dhdt = source.read(1).astype('float64')
    dhdt_transform = source.transform
    dhdt_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        dhdt[dhdt == source.nodata] = np.nan

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

dhdt_minimum = -4.1
dhdt_maximum = 1.2
dhdt_zero_position = (
    -dhdt_minimum / (dhdt_maximum - dhdt_minimum)
)
dhdt_norm = Normalize(
    vmin=dhdt_minimum,
    vmax=dhdt_maximum
)
mass_balance_norm = Normalize(
    vmin=-1.2,
    vmax=0.3
)
original_balance_cmap = plt.get_cmap('RdBu')
balance_color_points = []
for color_position in np.linspace(0, 1, 257):
    if color_position <= dhdt_zero_position:
        original_color_position = (
            color_position / dhdt_zero_position * 0.5
        )
    else:
        original_color_position = 0.5 + (
            (color_position - dhdt_zero_position)
            / (1 - dhdt_zero_position)
            * 0.5
        )
    balance_color_points.append(
        (
            color_position,
            original_balance_cmap(original_color_position)
        )
    )
balance_cmap = LinearSegmentedColormap.from_list(
    'elevation_change',
    balance_color_points
)
mass_balance_cmap = LinearSegmentedColormap.from_list(
    'mass_balance',
    [
        (0.00, '#9e0142'),
        (0.40, '#f46d43'),
        (0.65, '#fdae61'),
        (0.80, '#f7f7f7'),
        (0.90, '#67a9cf'),
        (1.00, '#2166ac')
    ]
)

map_minimum_x, map_minimum_y, map_maximum_x, map_maximum_y = (
    island.total_bounds
)
map_padding_x = (map_maximum_x - map_minimum_x) * 0.005
map_padding_y = (map_maximum_y - map_minimum_y) * 0.01

############################################################################ Figure_4A
ax_a = fig.add_subplot(grid[0, 0])

island.plot(
    ax=ax_a,
    facecolor='#eeeeee',
    edgecolor='0.35',
    linewidth=0.5
)
dhdt_image = ax_a.imshow(
    dhdt,
    extent=dhdt_extent,
    origin='upper',
    cmap=balance_cmap,
    norm=dhdt_norm,
    interpolation='nearest',
    zorder=2
)
study_glaciers.boundary.plot(
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

############################################################################ Figure_4B
ax_b = fig.add_subplot(grid[0, 1])

island.plot(
    ax=ax_b,
    facecolor='#eeeeee',
    edgecolor='0.35',
    linewidth=0.5
)
study_glaciers.plot(
    ax=ax_b,
    column='specific_balance_mwe_yr',
    cmap=mass_balance_cmap,
    norm=mass_balance_norm,
    edgecolor='0.50',
    linewidth=0.25
)

for _, glacier_label in glacier_label_points[
    glacier_label_points['Fig1_ID'].le(22)
].iterrows():
    label_point = glacier_label.geometry
    label_offset = {
        1: (100, 0),
        22: (-100, 100)
    }.get(glacier_label['Fig1_ID'], (0, 0))
    ax_b.text(
        label_point.x + label_offset[0],
        label_point.y + label_offset[1],
        f'{glacier_label["Fig1_ID"]}',
        color='black',
        fontsize=5,
        ha='center',
        va='center',
        zorder=6
    )

small_glacier_geometry = glacier_labels[
    glacier_labels['Fig1_ID'].between(23, 35)
].geometry.union_all()
small_glacier_label_point = small_glacier_geometry.centroid
ax_b.text(
    small_glacier_label_point.x,
    small_glacier_label_point.y + 1600,
    '35–23',
    color='black',
    fontsize=5,
    ha='center',
    va='center',
    zorder=6
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

dhdt_colorbar = fig.colorbar(
    dhdt_image,
    ax=ax_a,
    orientation='horizontal',
    fraction=0.055,
    pad=0.025,
    ticks=[-4.1, -3, -2, -1, 0, 1.2],
    extend='both'
)
dhdt_colorbar.set_label(
    'Elevation-change rate (m yr$^{-1}$)',
    fontsize=8
)
dhdt_colorbar.ax.tick_params(labelsize=7)
dhdt_colorbar.outline.set_linewidth(0.5)

mass_balance_image = plt.cm.ScalarMappable(
    norm=mass_balance_norm,
    cmap=mass_balance_cmap
)
mass_balance_colorbar = fig.colorbar(
    mass_balance_image,
    ax=ax_b,
    orientation='horizontal',
    fraction=0.055,
    pad=0.025,
    ticks=[-1.2, -0.9, -0.6, -0.3, 0, 0.3]
)
mass_balance_colorbar.set_label(
    'Specific mass balance (m w.e. yr$^{-1}$)',
    fontsize=8
)
mass_balance_colorbar.ax.tick_params(labelsize=7)
mass_balance_colorbar.outline.set_linewidth(0.5)

out_png = outpath / 'Figure_4.png'
out_pdf = outpath / 'Figure_4.pdf'

fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out_pdf, dpi=600, bbox_inches='tight', pad_inches=0.02)
