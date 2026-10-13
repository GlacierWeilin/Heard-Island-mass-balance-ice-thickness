from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling

import matplotlib.pyplot as plt

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


# %% paths and data

root = Path(__file__).resolve().parents[1]
datapath = root / 'data'
outpath = root / 'figure'
outpath.mkdir(parents=True, exist_ok=True)

study_file = datapath / 'NASADEM_Pleiades_dhdt_20m.tif'
nasadem_file = datapath / 'NASADEM_2000_coregistered_20m.tif'
hugonnet_file = datapath / 'Hugonnet_2000-01-01_2020-01-01_dhdt_100m.tif'

study_glaciers = gpd.read_file(
    datapath / '2019_Heard_SPEIADES.shp'
)
rgi_glaciers = gpd.read_file(
    datapath / 'RGI60_Heard_Island.shp'
)


# %% calculate the elevation-bin means for this study

with rasterio.open(study_file) as source:
    study_dhdt = source.read(1).astype('float64')
    study_transform = source.transform
    study_crs = source.crs
    study_shape = source.shape
    if source.nodata is not None:
        study_dhdt[study_dhdt == source.nodata] = np.nan

with rasterio.open(nasadem_file) as source:
    study_elevation = source.read(1).astype('float64')
    if source.nodata is not None:
        study_elevation[study_elevation == source.nodata] = np.nan

study_glaciers = study_glaciers.to_crs(study_crs)
study_inside_glacier = geometry_mask(
    study_glaciers.geometry,
    out_shape=study_shape,
    transform=study_transform,
    invert=True
)

study_area_valid = (
    study_inside_glacier
    & np.isfinite(study_elevation)
    & (study_elevation >= 0)
)
study_valid = study_area_valid & np.isfinite(study_dhdt)
study_pixel_area = abs(
    study_transform.a * study_transform.e
    - study_transform.b * study_transform.d
)


# %% calculate the elevation-bin means for Hugonnet et al. (2021)

with rasterio.open(hugonnet_file) as source:
    hugonnet_dhdt = source.read(1).astype('float64')
    hugonnet_transform = source.transform
    hugonnet_crs = source.crs
    hugonnet_shape = source.shape
    if source.nodata is not None:
        hugonnet_dhdt[hugonnet_dhdt == source.nodata] = np.nan

hugonnet_elevation = np.full(
    hugonnet_shape,
    np.nan,
    dtype='float64'
)
with rasterio.open(nasadem_file) as source:
    reproject(
        source=rasterio.band(source, 1),
        destination=hugonnet_elevation,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=hugonnet_transform,
        dst_crs=hugonnet_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

rgi_glaciers = rgi_glaciers.to_crs(hugonnet_crs)
hugonnet_inside_glacier = geometry_mask(
    rgi_glaciers.geometry,
    out_shape=hugonnet_shape,
    transform=hugonnet_transform,
    invert=True
)

hugonnet_area_valid = (
    hugonnet_inside_glacier
    & np.isfinite(hugonnet_elevation)
    & (hugonnet_elevation >= 0)
)
hugonnet_valid = hugonnet_area_valid & np.isfinite(hugonnet_dhdt)
hugonnet_pixel_area = abs(
    hugonnet_transform.a * hugonnet_transform.e
    - hugonnet_transform.b * hugonnet_transform.d
)


# %% aggregate the raw dh/dt values into 100 m NASADEM elevation bins

bin_size = 100
maximum_elevation = max(
    np.nanmax(study_elevation[study_valid]),
    np.nanmax(hugonnet_elevation[hugonnet_valid])
)
bin_edges = np.arange(
    0,
    np.ceil(maximum_elevation / bin_size) * bin_size + bin_size,
    bin_size
)

bin_rows = []
for dataset_name, elevation, dhdt, valid, area_valid, pixel_area in [
    (
        'This study',
        study_elevation,
        study_dhdt,
        study_valid,
        study_area_valid,
        study_pixel_area
    ),
    (
        'Hugonnet et al. (2021)',
        hugonnet_elevation,
        hugonnet_dhdt,
        hugonnet_valid,
        hugonnet_area_valid,
        hugonnet_pixel_area
    )
]:
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        selected = (
            valid
            & (elevation >= lower)
            & (elevation < upper)
        )
        pixel_count = int(np.count_nonzero(selected))
        area_selected = (
            area_valid
            & (elevation >= lower)
            & (elevation < upper)
        )
        area_pixel_count = int(np.count_nonzero(area_selected))
        if area_pixel_count == 0:
            continue

        bin_rows.append({
            'Dataset': dataset_name,
            'Elevation bin lower (m a.s.l.)': lower,
            'Elevation bin upper (m a.s.l.)': upper,
            'Elevation bin centre (m a.s.l.)': (lower + upper) / 2,
            'Glacier area (km2)': area_pixel_count * pixel_area / 1e6,
            'Mean dhdt (m yr-1)': (
                np.nanmean(dhdt[selected])
                if pixel_count >= 5 else np.nan
            ),
            'Standard deviation dhdt (m yr-1)': (
                np.nanstd(dhdt[selected], ddof=1)
                if pixel_count >= 5 else np.nan
            ),
            'Glacier pixel count': area_pixel_count,
            'Valid dhdt pixel count': pixel_count,
            'Valid dhdt coverage (%)': (
                pixel_count / area_pixel_count * 100
            )
        })

bin_results = pd.DataFrame(bin_rows)
# %% plot Figure S7

fig_width_inch = 14 / 2.54
fig_height_inch = 6.5 / 2.54
fig, axes = plt.subplots(
    1,
    2,
    figsize=(fig_width_inch, fig_height_inch),
    dpi=600,
    sharey=True
)

fig.subplots_adjust(
    left=0.12,
    right=0.99,
    bottom=0.16,
    top=0.88,
    wspace=0.11
)

line_values = bin_results['Mean dhdt (m yr-1)'].dropna()
line_minimum = np.floor(min(line_values.min(), 0) * 2) / 2
line_maximum = np.ceil(max(line_values.max(), 0) * 2) / 2

for ax, dataset_name, panel_label in zip(
    axes,
    ['This study', 'Hugonnet et al. (2021)'],
    ['(A)', '(B)']
):
    selected = bin_results[
        bin_results['Dataset'].eq(dataset_name)
    ]
    elevation_centres = selected[
        'Elevation bin centre (m a.s.l.)'
    ]

    mean_dhdt = selected['Mean dhdt (m yr-1)'].to_numpy()
    bar_colors = np.where(mean_dhdt < 0, '#d73027', '#4575b4')
    ax.barh(
        elevation_centres,
        mean_dhdt,
        height=92,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.35,
        zorder=3
    )
    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.7,
        zorder=2
    )
    ax.set_title(
        panel_label + ' ' + dataset_name,
        loc='center',
        fontweight='bold'
    )
    ax.set_xlabel('Mean elevation-change rate (m yr$^{-1}$)')
    ax.set_ylabel('NASADEM elevation bin (m a.s.l.)')
    ax.set_xlim(line_minimum, line_maximum)
    ax.set_ylim(0, bin_edges[-1])
    ax.set_yticks(np.arange(0, bin_edges[-1] + 1, 500))

fig.savefig(
    outpath / 'Figure_S4.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    outpath / 'Figure_S4.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
plt.close(fig)
