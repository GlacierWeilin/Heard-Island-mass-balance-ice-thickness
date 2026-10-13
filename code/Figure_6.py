from pathlib import Path

import geopandas as gpd
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


#%% Paths and data

root = Path(__file__).resolve().parents[1]
datapath = root / 'data'
outpath = root / 'figure'
outpath.mkdir(parents=True, exist_ok=True)

rgi = gpd.read_file(
    datapath / 'RGI60_Heard_Island.shp'
).to_crs('EPSG:32743')
island = gpd.read_file(
    datapath / 'Island_boundary_polygon.shp'
).to_crs('EPSG:32743')
glacier_labels = gpd.read_file(
    datapath / 'Table_S1_glacier_inventory.shp'
).to_crs('EPSG:32743')
glacier_labels['Fig1_ID'] = glacier_labels['Fig1_ID'].astype(int)
glacier_label_points = gpd.GeoDataFrame(
    glacier_labels[['Fig1_ID']].copy(),
    geometry=gpd.points_from_xy(
        glacier_labels['Longitude'],
        glacier_labels['Latitude']
    ),
    crs='EPSG:4326'
).to_crs('EPSG:32743')
small_glacier_label_point = glacier_labels[
    glacier_labels['Fig1_ID'].between(23, 35)
].geometry.union_all().centroid

product_files = {
    'Hugonnet (2021)': (
        datapath / 'Hugonnet_2021_Heard_Island_2000-2019.gpkg'
    ),
    'The WGMS Network (2026)': (
        datapath / 'WGMS_2026_Heard_Island_2000-2019.gpkg'
    )
}

product_maps = {}
for product, product_file in product_files.items():
    product_data = gpd.read_file(product_file)
    product_maps[product] = rgi.merge(
        product_data[['RGIId', 'mean_mwe_yr']],
        on='RGIId',
        how='left',
        validate='one_to_one'
    )


#%% Colour scale

mass_balance_norm = Normalize(vmin=-1.2, vmax=0.3)
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


#%% Set figure

fig_width_inch = 14 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 2.5), dpi=600)
grid = fig.add_gridspec(
    1,
    2,
    left=0.015,
    right=0.995,
    bottom=0.16,
    top=0.99,
    wspace=0.04
)
ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])

map_minimum_x, map_minimum_y, map_maximum_x, map_maximum_y = (
    island.total_bounds
)
map_padding_x = (map_maximum_x - map_minimum_x) * 0.005
map_padding_y = (map_maximum_y - map_minimum_y) * 0.01

map_axes = [ax_a, ax_b]
map_products = ['Hugonnet (2021)', 'The WGMS Network (2026)']
map_titles = [
    '(A) Hugonnet et al. (2021)',
    '(B) WGMS (2026)'
]

for axis, product, title in zip(
    map_axes,
    map_products,
    map_titles
):
    island.plot(
        ax=axis,
        facecolor='#eeeeee',
        edgecolor='0.35',
        linewidth=0.4,
        zorder=1
    )
    product_maps[product].plot(
        ax=axis,
        column='mean_mwe_yr',
        cmap=mass_balance_cmap,
        norm=mass_balance_norm,
        edgecolor='0.40',
        linewidth=0.2,
        missing_kwds={
            'color': '#bdbdbd',
            'edgecolor': '0.40'
        },
        zorder=2
    )
    for _, glacier_label in glacier_label_points[
        glacier_label_points['Fig1_ID'].le(22)
    ].iterrows():
        label_offset = {
            1: (300, 200),
            22: (-300, 250)
        }.get(glacier_label['Fig1_ID'], (0, 0))
        axis.text(
            glacier_label.geometry.x + label_offset[0],
            glacier_label.geometry.y + label_offset[1],
            f'{glacier_label["Fig1_ID"]}',
            color='black',
            fontsize=5.5,
            ha='center',
            va='center',
            zorder=6
        )
    axis.text(
        small_glacier_label_point.x -1000,
        small_glacier_label_point.y + 1600,
        '35–23',
        color='black',
        fontsize=5.5,
        ha='center',
        va='center',
        zorder=6
    )
    axis.set_xlim(
        map_minimum_x - map_padding_x,
        map_maximum_x + map_padding_x
    )
    axis.set_ylim(
        map_minimum_y - map_padding_y,
        map_maximum_y + map_padding_y
    )
    axis.set_aspect('equal', adjustable='box')
    axis.set_axis_off()
    axis.set_title(
        title,
        fontsize=10,
        fontweight='bold',
        pad=3
    )

mass_balance_image = plt.cm.ScalarMappable(
    norm=mass_balance_norm,
    cmap=mass_balance_cmap
)
mass_balance_colorbar = fig.colorbar(
    mass_balance_image,
    ax=map_axes,
    orientation='horizontal',
    fraction=0.055,
    pad=0.025,
    ticks=[-1.2, -0.9, -0.6, -0.3, 0, 0.3],
    extend='both'
)
mass_balance_colorbar.set_label(
    'Mass balance (m w.e. yr$^{-1}$)',
    fontsize=8
)
mass_balance_colorbar.ax.tick_params(labelsize=7)
mass_balance_colorbar.outline.set_linewidth(0.5)

fig.savefig(
    outpath / 'Figure_6.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02,
)
fig.savefig(
    outpath / 'Figure_6.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
plt.close(fig)
