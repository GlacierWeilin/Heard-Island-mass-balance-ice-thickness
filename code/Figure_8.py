from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.lines import Line2D
from rasterio.mask import mask

plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'legend.fontsize': 6})
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

outline_dir = datapath / 'glaciers'
study_1988_file = datapath / 'Heard_Island_1988_ice_thickness.nc'
product_names = [
    'Farinotti et al. (2019)',
    'Millan et al. (2022)',
    'Frank et al. (2026)',
    'Maffezzoli et al. (2026)'
]
product_files = [
    (
        datapath / 'Thickness_Farinotti2019_RGI60_native25m.tif'
    ),
    (
        datapath / 'Thickness_Millan2022_native50m.tif'
    ),
    (
        datapath / 'Thickness_Frank_Heard_Island_native100m.tif'
    ),
    (
        datapath / 'Thickness_IceBoost_v2_RGI70_native100m.tif'
    )
]
product_uncertainty_files = [
    None,
    (
        datapath / 'Thickness_error_Millan2022_native50m.tif'
    ),
    None,
    (
        datapath / 'Thickness_error_IceBoost_v2_RGI70_native100m.tif'
    )
]
product_relative_uncertainties = [12.06 / 46.47, None, 8.34 / 50.57, None]
product_colors = ['#d95f02', '#7570b3', '#1b9e77', '#e7298a']
product_markers = ['s', '^', 'P', 'D']


#%% Read the 1988 GlabTop2 volume ensemble

with xr.open_dataset(study_1988_file) as dataset_1988:
    study_1988_ids = dataset_1988['glacier_id'].values.astype(int)
    study_1988_volume = dataset_1988['glacier_volume'].load()
    study_1988_ensemble_size = (
        study_1988_volume.sizes['shape_factor']
        * study_1988_volume.sizes['basal_shear_stress_factor']
    )
    if study_1988_ensemble_size != 9:
        raise ValueError('The 1988 volume ensemble must contain nine runs.')
    study_1988_main = study_1988_volume.sel(
        shape_factor=0.8,
        basal_shear_stress_factor=1.0
    ).values
    study_1988_uncertainty = study_1988_volume.std(
        dim=['shape_factor', 'basal_shear_stress_factor'],
        ddof=1
    ).values

study_1988_main_by_id = dict(zip(
    study_1988_ids,
    study_1988_main
))
study_1988_uncertainty_by_id = dict(zip(
    study_1988_ids,
    study_1988_uncertainty
))


#%% Select glacier units represented by one unique RGI 6.0 identifier

glacier_records = []
for outline_file in sorted(
    outline_dir.glob('glacier_*.shp'),
    key=lambda path: int(path.stem.removeprefix('glacier_'))
):
    outlines = gpd.read_file(outline_file)
    outlines['Date'] = outlines['Date'].astype(int)
    outline_1988 = outlines[outlines['Date'] == 1988]

    if len(outline_1988) == 0:
        continue

    glacier_records.append({
        'glacier_id': int(outline_file.stem.removeprefix('glacier_')),
        'RGIId': outline_1988['RGIId'].iloc[0],
        'location': outline_1988['Location'].iloc[0],
        'geometry': outline_1988.geometry.union_all(),
        'crs': outline_1988.crs
    })

rgi_id_counts = pd.Series([
    glacier['RGIId'] for glacier in glacier_records
]).value_counts()

single_rgi_glaciers = [
    glacier for glacier in glacier_records
    if rgi_id_counts[glacier['RGIId']] == 1
]


#%% Figure 8A glacier-volume comparison

volume_comparison_rows = []
for glacier in single_rgi_glaciers:
    glacier_id = glacier['glacier_id']
    comparison_row = {
        'glacier_id': glacier_id,
        'RGIId': glacier['RGIId'],
        'This study': study_1988_main_by_id[glacier_id],
        'This study uncertainty (9-member SD)': (
            study_1988_uncertainty_by_id[glacier_id]
        )
    }

    for (
        product_name,
        product_file,
        uncertainty_file,
        relative_uncertainty
    ) in zip(
        product_names,
        product_files,
        product_uncertainty_files,
        product_relative_uncertainties
    ):
        with rasterio.open(product_file) as source:
            glacier_geometry = gpd.GeoSeries(
                [glacier['geometry']],
                crs=glacier['crs']
            ).to_crs(source.crs)

            clipped_thickness, _ = mask(
                source,
                glacier_geometry.geometry,
                crop=True,
                filled=False,
                all_touched=False
            )
            valid_thickness = clipped_thickness[0].compressed()
            valid_thickness = valid_thickness.astype('float64')
            valid_thickness = valid_thickness[
                np.isfinite(valid_thickness)
                & (valid_thickness > 0)
            ]
            pixel_area = abs(
                source.transform.a * source.transform.e
                - source.transform.b * source.transform.d
            )
            comparison_row[product_name] = (
                np.sum(valid_thickness) * pixel_area / 1e9
            )

        if relative_uncertainty is not None:
            comparison_row[product_name + ' uncertainty'] = (
                comparison_row[product_name]
                * relative_uncertainty
            )
        elif uncertainty_file is not None:
            with rasterio.open(uncertainty_file) as uncertainty_source:
                uncertainty_geometry = gpd.GeoSeries(
                    [glacier['geometry']],
                    crs=glacier['crs']
                ).to_crs(uncertainty_source.crs)
                clipped_uncertainty, _ = mask(
                    uncertainty_source,
                    uncertainty_geometry.geometry,
                    crop=True,
                    filled=False,
                    all_touched=False
                )
                valid_uncertainty = (
                    clipped_uncertainty[0].compressed().astype('float64')
                )
                valid_uncertainty = valid_uncertainty[
                    np.isfinite(valid_uncertainty)
                    & (valid_uncertainty >= 0)
                ]
                uncertainty_pixel_area = abs(
                    uncertainty_source.transform.a
                    * uncertainty_source.transform.e
                    - uncertainty_source.transform.b
                    * uncertainty_source.transform.d
                )
                comparison_row[product_name + ' uncertainty'] = (
                    np.sum(valid_uncertainty)
                    * uncertainty_pixel_area
                    / 1e9
                )
        else:
            comparison_row[product_name + ' uncertainty'] = np.nan

    volume_comparison_rows.append(comparison_row)

volume_comparison = pd.DataFrame(volume_comparison_rows)
#%% Figure 8A linear regressions

volume_regressions = []
for product_name in product_names:
    x_values = volume_comparison['This study'].to_numpy()
    y_values = volume_comparison[product_name].to_numpy()
    valid = np.isfinite(x_values) & np.isfinite(y_values)

    slope, intercept = np.polyfit(
        x_values[valid],
        y_values[valid],
        1
    )
    fitted_values = slope * x_values[valid] + intercept
    r_squared = 1 - np.sum(
        (y_values[valid] - fitted_values) ** 2
    ) / np.sum(
        (y_values[valid] - np.mean(y_values[valid])) ** 2
    )
    volume_regressions.append({
        'product': product_name,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'number_of_points': np.count_nonzero(valid)
    })


#%% Figure 8B regional ice-volume comparison

location_by_id = dict(zip(
    [glacier['glacier_id'] for glacier in glacier_records],
    [glacier['location'] for glacier in glacier_records]
))
study_1988_index = {
    glacier_id: index
    for index, glacier_id in enumerate(study_1988_ids)
}

region_ids = {
    'Heard Island': study_1988_ids.tolist(),
    'Eastern': [
        glacier_id
        for glacier_id in study_1988_ids
        if location_by_id[glacier_id] == 'E'
    ],
    'Western': [
        glacier_id
        for glacier_id in study_1988_ids
        if location_by_id[glacier_id] == 'W'
    ]
}

regional_volume_rows = []
for region_name, selected_ids in region_ids.items():
    selected_indices = [
        study_1988_index[glacier_id]
        for glacier_id in selected_ids
    ]
    regional_study_ensemble = study_1988_volume.values[
        :, :, selected_indices
    ].sum(axis=2)
    regional_study_volume = np.sum([
        study_1988_main_by_id[glacier_id]
        for glacier_id in selected_ids
    ])
    regional_study_uncertainty = np.std(
        regional_study_ensemble,
        ddof=1
    )

    selected_geometries = [
        glacier['geometry']
        for glacier in glacier_records
        if glacier['glacier_id'] in selected_ids
    ]
    regional_geometry = gpd.GeoSeries(
        selected_geometries,
        crs=glacier_records[0]['crs']
    ).union_all()

    for (
        product_name,
        product_file,
        uncertainty_file,
        relative_uncertainty
    ) in zip(
        product_names,
        product_files,
        product_uncertainty_files,
        product_relative_uncertainties
    ):
        with rasterio.open(product_file) as source:
            product_geometry = gpd.GeoSeries(
                [regional_geometry],
                crs=glacier_records[0]['crs']
            ).to_crs(source.crs)
            clipped_thickness, _ = mask(
                source,
                product_geometry.geometry,
                crop=True,
                filled=False,
                all_touched=False
            )
            valid_thickness = (
                clipped_thickness[0].compressed().astype('float64')
            )
            valid_thickness = valid_thickness[
                np.isfinite(valid_thickness)
                & (valid_thickness > 0)
            ]
            pixel_area = abs(
                source.transform.a * source.transform.e
                - source.transform.b * source.transform.d
            )
            product_volume = (
                np.sum(valid_thickness) * pixel_area / 1e9
            )

        if relative_uncertainty is not None:
            product_uncertainty = (
                product_volume * relative_uncertainty
            )
        elif uncertainty_file is not None:
            with rasterio.open(uncertainty_file) as uncertainty_source:
                uncertainty_geometry = gpd.GeoSeries(
                    [regional_geometry],
                    crs=glacier_records[0]['crs']
                ).to_crs(uncertainty_source.crs)
                clipped_uncertainty, _ = mask(
                    uncertainty_source,
                    uncertainty_geometry.geometry,
                    crop=True,
                    filled=False,
                    all_touched=False
                )
                valid_uncertainty = (
                    clipped_uncertainty[0]
                    .compressed()
                    .astype('float64')
                )
                valid_uncertainty = valid_uncertainty[
                    np.isfinite(valid_uncertainty)
                    & (valid_uncertainty >= 0)
                ]
                uncertainty_pixel_area = abs(
                    uncertainty_source.transform.a
                    * uncertainty_source.transform.e
                    - uncertainty_source.transform.b
                    * uncertainty_source.transform.d
                )
                product_uncertainty = (
                    np.sum(valid_uncertainty)
                    * uncertainty_pixel_area
                    / 1e9
                )
        else:
            product_uncertainty = np.nan

        regional_volume_rows.append({
            'region': region_name,
            'product': product_name,
            'This study': regional_study_volume,
            'This study uncertainty (9-member SD)': (
                regional_study_uncertainty
            ),
            'Global product': product_volume,
            'Global product uncertainty': product_uncertainty
        })

regional_volume_comparison = pd.DataFrame(regional_volume_rows)
#%% Determine equal x and y limits

volume_maximum = np.nanmax(
    volume_comparison[product_names].to_numpy()
)
volume_maximum = max(
    volume_maximum,
    np.nanmax(
        volume_comparison['This study']
        + volume_comparison['This study uncertainty (9-member SD)']
    )
)
for product_name in product_names:
    product_upper = (
        volume_comparison[product_name]
        + volume_comparison[product_name + ' uncertainty'].fillna(0)
    )
    volume_maximum = max(volume_maximum, np.nanmax(product_upper))
volume_limit = np.ceil(volume_maximum * 2) / 2
volume_ticks = np.arange(0, volume_limit + 0.25, 0.5)

regional_volume_maximum = np.nanmax(np.concatenate([
    (
        regional_volume_comparison['This study']
        + regional_volume_comparison[
            'This study uncertainty (9-member SD)'
        ]
    ).to_numpy(),
    (
        regional_volume_comparison['Global product']
        + regional_volume_comparison['Global product uncertainty']
    ).to_numpy()
]))
regional_tick_step = next(
    step for step in [1.0, 2.0, 5.0, 10.0, 20.0]
    if step >= regional_volume_maximum / 6
)
regional_volume_limit = (
    np.ceil(regional_volume_maximum / regional_tick_step)
    * regional_tick_step
)
regional_volume_ticks = np.arange(
    0,
    regional_volume_limit + regional_tick_step / 2,
    regional_tick_step
)


#%% Plot

fig_width_inch = 14 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 2.85), dpi=600)
grid = fig.add_gridspec(
    1,
    2,
    left=0.09,
    right=0.995,
    bottom=0.13,
    top=0.98,
    wspace=0.28
)
ax_b = fig.add_subplot(grid[0, 0])
ax_a = fig.add_subplot(grid[0, 1])


ax_a.plot(
    [0, volume_limit],
    [0, volume_limit],
    color='0.25',
    linewidth=0.8,
    linestyle='--',
    zorder=1
)
fit_x = np.linspace(0, volume_limit, 200)

for product_index, regression in enumerate(volume_regressions):
    product_name = regression['product']
    x_values = volume_comparison['This study'].to_numpy()
    x_error = volume_comparison[
        'This study uncertainty (9-member SD)'
    ].to_numpy()
    y_values = volume_comparison[product_name].to_numpy()
    y_error = volume_comparison[
        product_name + ' uncertainty'
    ].to_numpy()
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    plotted_y_error = y_error[valid]
    if not np.any(np.isfinite(plotted_y_error)):
        plotted_y_error = None

    ax_a.errorbar(
        x_values[valid],
        y_values[valid],
        xerr=x_error[valid],
        yerr=plotted_y_error,
        fmt=product_markers[product_index],
        markersize=3.6,
        markerfacecolor=product_colors[product_index],
        markeredgecolor='white',
        markeredgewidth=0.25,
        ecolor=product_colors[product_index],
        elinewidth=0.45,
        capsize=0,
        alpha=0.78,
        zorder=3
    )
    ax_a.plot(
        fit_x,
        regression['slope'] * fit_x + regression['intercept'],
        color=product_colors[product_index],
        linewidth=1.0,
        label=(
            product_name
            + ':'
            + '\ny = '
            + f'{regression["slope"]:.2f}x '
            + ('+ ' if regression['intercept'] >= 0 else '- ')
            + f'{abs(regression["intercept"]):.2f}; '
            + f'R$^2$ = {regression["r_squared"]:.2f}'
        ),
        zorder=2
    )

ax_a.set_xlim(0, volume_limit)
ax_a.set_ylim(0, volume_limit)
ax_a.set_xticks(volume_ticks)
ax_a.set_yticks(volume_ticks)
ax_a.set_aspect('equal', adjustable='box')
ax_a.set_xlabel('This study ice volume in 1988 (km$^3$)')
ax_a.set_ylabel('Global-product ice volume (km$^3$)')
ax_a.grid(color='0.88', linewidth=0.4, zorder=0)
ax_a.tick_params(labelsize=7)
ax_a.text(
    0.02,
    0.98,
    '(B)',
    fontsize=10,
    fontweight='bold',
    ha='left',
    va='top',
    transform=ax_a.transAxes
)
legend_a = ax_a.legend(
    loc='lower right',
    bbox_to_anchor=(0.995, 0.005),
    borderaxespad=0,
    fontsize=5.2,
    frameon=True,
    framealpha=0.82,
    borderpad=0.28,
    labelspacing=0.24,
    handlelength=1.2,
    handletextpad=0.35
)
legend_a.get_frame().set_linewidth(0.4)


######

ax_b.plot(
    [0, regional_volume_limit],
    [0, regional_volume_limit],
    color='0.25',
    linewidth=0.8,
    linestyle='--',
    zorder=1
)

region_markers = {
    'Heard Island': 'o',
    'Eastern': '^',
    'Western': 's'
}
for product_index, product_name in enumerate(product_names):
    selected_product = regional_volume_comparison[
        regional_volume_comparison['product'].eq(product_name)
    ]
    for region_name in region_markers:
        selected = selected_product[
            selected_product['region'].eq(region_name)
        ].iloc[0]
        product_uncertainty = selected['Global product uncertainty']
        if not np.isfinite(product_uncertainty):
            product_uncertainty = None
        ax_b.errorbar(
            selected['This study'],
            selected['Global product'],
            xerr=selected['This study uncertainty (9-member SD)'],
            yerr=product_uncertainty,
            fmt=region_markers[region_name],
            markersize=4.2,
            markerfacecolor=product_colors[product_index],
            markeredgecolor='white',
            markeredgewidth=0.25,
            ecolor=product_colors[product_index],
            elinewidth=0.45,
            capsize=2.0,
            capthick=0.45,
            alpha=0.82,
            zorder=3
        )

ax_b.set_xlim(0, regional_volume_limit)
ax_b.set_ylim(0, regional_volume_limit)
ax_b.set_xticks(regional_volume_ticks)
ax_b.set_yticks(regional_volume_ticks)
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xlabel('This study ice volume in 1988 (km$^3$)')
ax_b.set_ylabel('Global-product ice volume (km$^3$)')
ax_b.grid(color='0.88', linewidth=0.4, zorder=0)
ax_b.tick_params(labelsize=7)
ax_b.text(
    0.02,
    0.98,
    '(A)',
    fontsize=10,
    fontweight='bold',
    ha='left',
    va='top',
    transform=ax_b.transAxes
)
product_handles = [
    Line2D(
        [0],
        [0],
        marker='o',
        linestyle='none',
        markersize=4.2,
        markerfacecolor=product_colors[index],
        markeredgecolor='white',
        markeredgewidth=0.25,
        label=product_name
    )
    for index, product_name in enumerate(product_names)
]
region_handles = [
    Line2D(
        [0],
        [0],
        marker=marker,
        linestyle='none',
        markersize=4.2,
        markerfacecolor='0.45',
        markeredgecolor='white',
        markeredgewidth=0.25,
        label={
            'Heard Island': 'Heard Island',
            'Eastern': 'Eastern Sector',
            'Western': 'Western Sector'
        }[region_name]
    )
    for region_name, marker in region_markers.items()
]
legend_b = ax_b.legend(
    handles=product_handles + region_handles,
    loc='lower right',
    bbox_to_anchor=(0.995, 0.005),
    borderaxespad=0,
    ncol=1,
    fontsize=5.2,
    frameon=True,
    framealpha=0.82,
    borderpad=0.28,
    labelspacing=0.24,
    handlelength=1.2,
    handletextpad=0.35
)
legend_b.get_frame().set_linewidth(0.4)

png_path = outpath / 'Figure_8.png'
pdf_path = outpath / 'Figure_8.pdf'

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02,
    pil_kwargs={'compress_level': 9}
)
fig.savefig(
    pdf_path,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
plt.close(fig)
