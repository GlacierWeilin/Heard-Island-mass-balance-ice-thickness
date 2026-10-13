import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.features import geometry_mask
from matplotlib.patches import Rectangle
from shapely.geometry import MultiPolygon, Polygon

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

nasadem_file = datapath / 'NASADEM_2000_20m.tif'
pleiades_mosaic_file = datapath / 'Pleiades_SGM_20m.tif'
dhdt_file = datapath / 'NASADEM_Pleiades_dhdt_20m.tif'
pleiades_files = [
    datapath / '2018-04-22_0444140_Heard_ANT_1B_DEM_SGM_20m.tif',
    datapath / '2019-04-07_0451575_Heard_ANT_1B_DEM_SGM_20m.tif',
    datapath / '2019-04-07_0452069_Heard_ANT_1B_DEM_SGM_20m.tif',
    datapath / '2019-11-21_0448386_Heard_ANT_1A_DEM_SGM_20m.tif'
]
pleiades_names = [
    'P1 · 0444140\n22 Apr 2018',
    'P2 · 0451575\n7 Apr 2019',
    'P3 · 0452069\n7 Apr 2019',
    'P4 · 0448386\n21 Nov 2019'
]
pleiades_colors = ['#7b3294', '#008837', '#c51b7d', '#2166ac']

island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
study_glaciers = gpd.read_file(
    datapath / '2019_Heard_SPEIADES.shp'
).to_crs('EPSG:32743')

# %% read original DEMs

with rasterio.open(nasadem_file) as source:
    nasadem = source.read(1).astype('float64')
    nasadem_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        nasadem[nasadem == source.nodata] = np.nan

pleiades_extents = []
for pleiades_file in pleiades_files:
    with rasterio.open(pleiades_file) as source:
        pleiades_extents.append([
            source.bounds.left,
            source.bounds.right,
            source.bounds.bottom,
            source.bounds.top
        ])

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
    out_shape=pleiades_mosaic.shape,
    transform=pleiades_mosaic_transform,
    invert=False
)
pleiades_mosaic[outside_island] = np.nan

with rasterio.open(dhdt_file) as source:
    dhdt = source.read(1).astype('float64')
    dhdt_extent = [
        source.bounds.left,
        source.bounds.right,
        source.bounds.bottom,
        source.bounds.top
    ]
    if source.nodata is not None:
        dhdt[dhdt == source.nodata] = np.nan

# %% setup map coordinates

map_minimum_x, map_minimum_y, map_maximum_x, map_maximum_y = (
    island.total_bounds
)
map_padding_x = 0
map_padding_y = 0

elevation_norm = Normalize(vmin=0, vmax=2500)
elevation_cmap = plt.get_cmap('terrain').copy()
elevation_cmap.set_bad('white')

dhdt_minimum = -3.0
dhdt_maximum = 1.3
dhdt_zero_position = (
    -dhdt_minimum / (dhdt_maximum - dhdt_minimum)
)
dhdt_norm = Normalize(vmin=dhdt_minimum, vmax=dhdt_maximum)
original_dhdt_cmap = plt.get_cmap('RdBu')
dhdt_color_points = []
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
    dhdt_color_points.append(
        (
            color_position,
            original_dhdt_cmap(original_color_position)
        )
    )
dhdt_cmap = LinearSegmentedColormap.from_list(
    'elevation_change',
    dhdt_color_points
)


fig_width_inch = 16.69 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 2.44), dpi=600)
grid = fig.add_gridspec(
    1, 3,
    left=0.002,
    right=0.998,
    bottom=0.22,
    top=0.94,
    wspace=-0.075
)

############################################################################ Figure_S1A
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
ax_a.set_title(
    '(A) NASADEM (16 Feb 2000)',
    fontsize=8,
    fontweight='bold',
    y=1.0
)

############################################################################ Figure_S1B
ax_b = fig.add_subplot(grid[0, 1])

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

for extent, name, color in zip(
    pleiades_extents,
    pleiades_names,
    pleiades_colors
):
    ax_b.add_patch(Rectangle(
        (extent[0], extent[2]),
        extent[1] - extent[0],
        extent[3] - extent[2],
        fill=False,
        edgecolor=color,
        linewidth=0.9,
        zorder=4
    ))

label_positions = [
    (-0.01, 0.64),
    (0.92, 0.70),
    (0.32, -0.08),
    (0.50, 0.92)
]
label_alignments = [
    ('left', 'center'),
    ('right', 'center'),
    ('center', 'bottom'),
    ('center', 'center')
]
for name, color, position, alignment in zip(
    pleiades_names,
    pleiades_colors,
    label_positions,
    label_alignments
):
    ax_b.text(
        position[0], position[1], name,
        color=color,
        fontsize=5.0,
        ha=alignment[0],
        va=alignment[1],
        transform=ax_b.transAxes,
        bbox={
            'facecolor': 'white',
            'edgecolor': color,
            'linewidth': 0.4,
            'pad': 0.8
        },
        zorder=5
    )

ax_b.set_title(
    '(B) Pléiades DEM mosaic (2018–2019)',
    fontsize=8,
    fontweight='bold',
    y=1.0
)

############################################################################ Figure_S1C
ax_c = fig.add_subplot(grid[0, 2])

island.plot(
    ax=ax_c,
    facecolor='#eeeeee',
    edgecolor='0.35',
    linewidth=0.4,
    zorder=1
)
dhdt_image = ax_c.imshow(
    dhdt,
    extent=dhdt_extent,
    origin='upper',
    cmap=dhdt_cmap,
    norm=dhdt_norm,
    interpolation='nearest',
    zorder=2
)
study_glaciers.boundary.plot(
    ax=ax_c,
    color='black',
    linewidth=0.20,
    zorder=3
)
ax_c.set_title(
    r'(C) Original $\mathbfit{dh/dt}$ (2000–2019)',
    fontsize=8,
    fontweight='bold',
    y=1.0
)

# %% format axes and colorbar

for axis in [ax_a, ax_b, ax_c]:
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

colorbar_axis = fig.add_axes([0.08, 0.105, 0.52, 0.035])

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

dhdt_colorbar_axis = fig.add_axes([0.70, 0.105, 0.25, 0.035])
dhdt_colorbar = fig.colorbar(
    dhdt_image,
    cax=dhdt_colorbar_axis,
    orientation='horizontal',
    ticks=[-3, -2, -1, 0, 1.3],
    extend='both'
)
dhdt_colorbar.set_label(
    'Elevation change rate (m yr$^{-1}$)',
    fontsize=8
)
dhdt_colorbar.ax.tick_params(labelsize=7)
dhdt_colorbar.outline.set_linewidth(0.5)

fig.savefig(
    outpath / 'Figure_S1.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    outpath / 'Figure_S1.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
