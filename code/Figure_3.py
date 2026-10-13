import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

glaciers_1947 = gpd.read_file(
    datapath / '1947_Heard_TOPO.shp'
).to_crs('EPSG:32743')
glaciers_1988 = gpd.read_file(
    datapath / '1988_Heard_SPOT.shp'
).to_crs('EPSG:32743')
glaciers_2019 = gpd.read_file(
    datapath / '2019_Heard_SPEIADES.shp'
).to_crs('EPSG:32743')
eastern_glaciers = gpd.read_file(
    datapath / 'Eastern_glaciers_2001-2019.shp'
).to_crs('EPSG:32743')

glaciers_1947['area_km2'] = glaciers_1947.geometry.area / 1e6
glaciers_1988['area_km2'] = glaciers_1988.geometry.area / 1e6
glaciers_2019['area_km2'] = glaciers_2019.geometry.area / 1e6
eastern_glaciers['area_km2'] = eastern_glaciers.geometry.area / 1e6

eastern_glaciers['Date'] = pd.to_numeric(eastern_glaciers['Date'])

# %% All glaciers, eastern and western glacier area

years_a = np.array([1947, 1988, 2019])

island_area = np.array([
    glaciers_1947['area_km2'].sum(),
    glaciers_1988['area_km2'].sum(),
    glaciers_2019['area_km2'].sum()
])

eastern_area = np.array([
    glaciers_1947.loc[
        glaciers_1947['Location'] == 'E',
        'area_km2'
    ].sum(),
    glaciers_1988.loc[
        glaciers_1988['Location'] == 'E',
        'area_km2'
    ].sum(),
    glaciers_2019.loc[
        glaciers_2019['Location'] == 'E',
        'area_km2'
    ].sum()
])

western_area = np.array([
    glaciers_1947.loc[
        glaciers_1947['Location'] == 'W',
        'area_km2'
    ].sum(),
    glaciers_1988.loc[
        glaciers_1988['Location'] == 'W',
        'area_km2'
    ].sum(),
    glaciers_2019.loc[
        glaciers_2019['Location'] == 'W',
        'area_km2'
    ].sum()
])

island_remaining = island_area / island_area[0] * 100
eastern_remaining = eastern_area / eastern_area[0] * 100
western_remaining = western_area / western_area[0] * 100

# %% area uncertainty from the outline buffers

area_uncertainty_rows_a = []
for year, glacier_inventory, buffer_m in [
    (1947, glaciers_1947, 30.0),
    (1988, glaciers_1988, 20.0),
    (2019, glaciers_2019, 10.0)
]:
    for region, location in [
        ('All glaciers', None),
        ('Eastern glaciers', 'E'),
        ('Western glaciers', 'W')
    ]:
        selected = glacier_inventory.copy()
        if location is not None:
            selected = selected[selected['Location'] == location]
        area = selected['area_km2'].sum()
        geometry = selected.geometry.union_all().buffer(0)
        area_uncertainty = (
            geometry.buffer(buffer_m).area
            - geometry.buffer(-buffer_m).area
        ) / 2e6
        mapping_uncertainty = area_uncertainty / area
        area_uncertainty_rows_a.append({
            'panel': 'A',
            'series': region,
            'year': year,
            'outline_buffer_m': buffer_m,
            'mapping_relative_uncertainty_1sigma_percent': (
                mapping_uncertainty * 100
            ),
            'area_uncertainty_1sigma_km2': area_uncertainty
        })

area_uncertainty_a = pd.DataFrame(area_uncertainty_rows_a)

island_area_uncertainty = area_uncertainty_a.loc[
    area_uncertainty_a['series'] == 'All glaciers',
    'area_uncertainty_1sigma_km2'
].to_numpy()
eastern_area_uncertainty = area_uncertainty_a.loc[
    area_uncertainty_a['series'] == 'Eastern glaciers',
    'area_uncertainty_1sigma_km2'
].to_numpy()
western_area_uncertainty = area_uncertainty_a.loc[
    area_uncertainty_a['series'] == 'Western glaciers',
    'area_uncertainty_1sigma_km2'
].to_numpy()

island_area_change = island_area - island_area[0]
eastern_area_change_a = eastern_area - eastern_area[0]
western_area_change = western_area - western_area[0]

island_area_change_uncertainty = (
    np.sqrt(
        island_area_uncertainty ** 2
        + island_area_uncertainty[0] ** 2
    )
)
eastern_area_change_uncertainty = (
    np.sqrt(
        eastern_area_uncertainty ** 2
        + eastern_area_uncertainty[0] ** 2
    )
)
western_area_change_uncertainty = (
    np.sqrt(
        western_area_uncertainty ** 2
        + western_area_uncertainty[0] ** 2
    )
)
island_area_change_uncertainty[0] = 0
eastern_area_change_uncertainty[0] = 0
western_area_change_uncertainty[0] = 0

island_remaining_uncertainty = (
    island_remaining * np.sqrt(
        (island_area_uncertainty / island_area) ** 2
        + (island_area_uncertainty[0] / island_area[0]) ** 2
    )
)
eastern_remaining_uncertainty = (
    eastern_remaining * np.sqrt(
        (eastern_area_uncertainty / eastern_area) ** 2
        + (eastern_area_uncertainty[0] / eastern_area[0]) ** 2
    )
)
western_remaining_uncertainty = (
    western_remaining * np.sqrt(
        (western_area_uncertainty / western_area) ** 2
        + (western_area_uncertainty[0] / western_area[0]) ** 2
    )
)
island_remaining_uncertainty[0] = 0
eastern_remaining_uncertainty[0] = 0
western_remaining_uncertainty[0] = 0

# %% eastern glacier area from 2001 to 2019

eastern_glaciers['Name'] = eastern_glaciers['Name'].replace({
    'G07362': 'G073625E53121S'
})

eastern_area_change = eastern_glaciers.groupby(
    ['Name', 'Date'], as_index=False
)['area_km2'].sum()

eastern_area_change = eastern_area_change.sort_values(
    ['Name', 'Date']
)

area_uncertainty_rows_b = []
for (glacier_name, year), selected in eastern_glaciers.groupby(
    ['Name', 'Date']
):
    geometry = selected.geometry.union_all().buffer(0)
    buffer_m = 10.0 if int(year) in [2017, 2019] else 15.0
    area_uncertainty = (
        geometry.buffer(buffer_m).area
        - geometry.buffer(-buffer_m).area
    ) / 2e6
    area_uncertainty_rows_b.append({
        'Name': glacier_name,
        'Date': int(year),
        'outline_buffer_m': buffer_m,
        'area_uncertainty_1sigma_km2': area_uncertainty
    })

area_uncertainty_b = pd.DataFrame(area_uncertainty_rows_b)
eastern_area_change = eastern_area_change.merge(
    area_uncertainty_b,
    on=['Name', 'Date'],
    how='left'
)

eastern_area_change = eastern_area_change[
    eastern_area_change['Name'].isin(['Brown', 'Stephenson']) |
    eastern_area_change['Date'].isin([2001, 2005, 2010, 2019])
]
eastern_area_change['remaining'] = (
    eastern_area_change['area_km2'] /
    eastern_area_change.groupby('Name')['area_km2'].transform('first') *
    100
)
eastern_area_change['baseline_area_km2'] = (
    eastern_area_change.groupby('Name')['area_km2'].transform('first')
)
eastern_area_change['baseline_area_uncertainty_1sigma_km2'] = (
    eastern_area_change.groupby('Name')[
        'area_uncertainty_1sigma_km2'
    ].transform('first')
)
eastern_area_change['mapping_relative_uncertainty_1sigma_percent'] = (
    eastern_area_change['area_uncertainty_1sigma_km2']
    / eastern_area_change['area_km2']
    * 100
)
eastern_area_change['area_change_km2'] = (
    eastern_area_change['area_km2']
    - eastern_area_change['baseline_area_km2']
)
eastern_area_change['area_change_uncertainty_1sigma_km2'] = (
    np.sqrt(
        eastern_area_change['area_uncertainty_1sigma_km2'] ** 2
        + eastern_area_change[
            'baseline_area_uncertainty_1sigma_km2'
        ] ** 2
    )
)
eastern_area_change['remaining_uncertainty_1sigma_percent'] = (
    eastern_area_change['remaining']
    * np.sqrt(
        (
            eastern_area_change['area_uncertainty_1sigma_km2']
            / eastern_area_change['area_km2']
        ) ** 2
        + (
            eastern_area_change[
                'baseline_area_uncertainty_1sigma_km2'
            ]
            / eastern_area_change['baseline_area_km2']
        ) ** 2
    )
)
baseline_rows = eastern_area_change['Date'] == 2001
eastern_area_change.loc[
    baseline_rows,
    'area_change_uncertainty_1sigma_km2'
] = 0
eastern_area_change.loc[
    baseline_rows,
    'remaining_uncertainty_1sigma_percent'
] = 0

# %% save the numerical data used in the figure

panel_a_data = pd.DataFrame({
    'panel': ['A'] * 9,
    'series': np.repeat([
        'All glaciers', 'Eastern glaciers', 'Western glaciers'
    ], 3),
    'year': np.tile(years_a, 3),
    'area_km2': np.concatenate([
        island_area, eastern_area, western_area
    ]),
    'area_uncertainty_1sigma_km2': np.concatenate([
        island_area_uncertainty,
        eastern_area_uncertainty,
        western_area_uncertainty
    ]),
    'mapping_relative_uncertainty_1sigma_percent': np.concatenate([
        area_uncertainty_a.loc[
            area_uncertainty_a['series'] == 'All glaciers',
            'mapping_relative_uncertainty_1sigma_percent'
        ].to_numpy(),
        area_uncertainty_a.loc[
            area_uncertainty_a['series'] == 'Eastern glaciers',
            'mapping_relative_uncertainty_1sigma_percent'
        ].to_numpy(),
        area_uncertainty_a.loc[
            area_uncertainty_a['series'] == 'Western glaciers',
            'mapping_relative_uncertainty_1sigma_percent'
        ].to_numpy()
    ]),
    'area_change_km2': np.concatenate([
        island_area_change,
        eastern_area_change_a,
        western_area_change
    ]),
    'area_change_uncertainty_1sigma_km2': np.concatenate([
        island_area_change_uncertainty,
        eastern_area_change_uncertainty,
        western_area_change_uncertainty
    ]),
    'remaining_area_percent': np.concatenate([
        island_remaining, eastern_remaining, western_remaining
    ]),
    'remaining_area_uncertainty_1sigma_percent': np.concatenate([
        island_remaining_uncertainty,
        eastern_remaining_uncertainty,
        western_remaining_uncertainty
    ]),
    'outline_buffer_m': np.tile([30.0, 20.0, 10.0], 3)
})

eastern_intermediate_rows = []
for year in [2001, 2005, 2010]:
    selected = eastern_glaciers[eastern_glaciers['Date'] == year]
    geometry = selected.geometry.union_all().buffer(0)
    area = selected['area_km2'].sum()
    area_uncertainty = (
        geometry.buffer(15.0).area
        - geometry.buffer(-15.0).area
    ) / 2e6
    area_change_uncertainty = np.sqrt(
        area_uncertainty ** 2
        + eastern_area_uncertainty[0] ** 2
    )
    remaining = area / eastern_area[0] * 100
    remaining_uncertainty = remaining * np.sqrt(
        (area_uncertainty / area) ** 2
        + (eastern_area_uncertainty[0] / eastern_area[0]) ** 2
    )
    eastern_intermediate_rows.append({
        'panel': 'A',
        'series': 'Eastern glaciers',
        'year': year,
        'area_km2': area,
        'area_uncertainty_1sigma_km2': area_uncertainty,
        'mapping_relative_uncertainty_1sigma_percent': (
            area_uncertainty / area * 100
        ),
        'area_change_km2': area - eastern_area[0],
        'area_change_uncertainty_1sigma_km2': area_change_uncertainty,
        'remaining_area_percent': remaining,
        'remaining_area_uncertainty_1sigma_percent': remaining_uncertainty,
        'outline_buffer_m': 15.0
    })

panel_a_data = pd.concat(
    [panel_a_data, pd.DataFrame(eastern_intermediate_rows)],
    ignore_index=True
).sort_values(['series', 'year']).reset_index(drop=True)
panel_a_data['area_uncertainty_95_km2'] = (
    panel_a_data['area_uncertainty_1sigma_km2'] * 1.96
)
panel_a_data['area_change_uncertainty_95_km2'] = (
    panel_a_data['area_change_uncertainty_1sigma_km2'] * 1.96
)
panel_a_data['remaining_area_uncertainty_95_percent'] = (
    panel_a_data['remaining_area_uncertainty_1sigma_percent'] * 1.96
)

panel_b_data = eastern_area_change.rename(columns={
    'Name': 'series',
    'Date': 'year',
    'remaining': 'remaining_area_percent'
}).copy()
panel_b_data['panel'] = 'B'
panel_b_data['area_uncertainty_95_km2'] = (
    panel_b_data['area_uncertainty_1sigma_km2'] * 1.96
)
panel_b_data['area_change_uncertainty_95_km2'] = (
    panel_b_data['area_change_uncertainty_1sigma_km2'] * 1.96
)
panel_b_data['remaining_area_uncertainty_95_percent'] = (
    panel_b_data['remaining_uncertainty_1sigma_percent'] * 1.96
)
panel_b_data = panel_b_data[[
    'panel',
    'series',
    'year',
    'area_km2',
    'area_uncertainty_1sigma_km2',
    'area_uncertainty_95_km2',
    'mapping_relative_uncertainty_1sigma_percent',
    'area_change_km2',
    'area_change_uncertainty_1sigma_km2',
    'area_change_uncertainty_95_km2',
    'remaining_area_percent',
    'remaining_uncertainty_1sigma_percent',
    'remaining_area_uncertainty_95_percent',
    'outline_buffer_m'
]]
panel_b_data = panel_b_data.rename(columns={
    'remaining_uncertainty_1sigma_percent': (
        'remaining_area_uncertainty_1sigma_percent'
    )
})

# %% Setup figure

# 14 cm figure width.
fig_width_inch = 14 / 2.54
fig = plt.figure(figsize=(fig_width_inch, 2.0), dpi=600)

grid = fig.add_gridspec(
    1, 2,
    left=0.09,
    right=0.975,
    bottom=0.15,
    top=0.98,
    wspace=0.24
)

island_color = '#333333'
eastern_color = '#d95f02'
western_color = '#2166ac'

############################################################################ Figure_2A
ax_a = fig.add_subplot(grid[0, 0])

island_plot = panel_a_data[
    panel_a_data['series'] == 'All glaciers'
].sort_values('year')
eastern_plot = panel_a_data[
    panel_a_data['series'] == 'Eastern glaciers'
].sort_values('year')
western_plot = panel_a_data[
    panel_a_data['series'] == 'Western glaciers'
].sort_values('year')

ax_a.errorbar(
    island_plot['year'],
    island_plot['remaining_area_percent'] - 100,
    color=island_color,
    marker='o',
    markersize=3,
    linewidth=1.0,
    yerr=island_plot['remaining_area_uncertainty_1sigma_percent'],
    capsize=1.8,
    elinewidth=0.6,
    capthick=0.6,
    label='All glaciers'
)
ax_a.errorbar(
    eastern_plot['year'],
    eastern_plot['remaining_area_percent'] - 100,
    color=eastern_color,
    marker='s',
    markersize=3,
    linewidth=1.0,
    yerr=eastern_plot['remaining_area_uncertainty_1sigma_percent'],
    capsize=1.8,
    elinewidth=0.6,
    capthick=0.6,
    label='Eastern glaciers'
)
ax_a.errorbar(
    western_plot['year'],
    western_plot['remaining_area_percent'] - 100,
    color=western_color,
    marker='^',
    markersize=3.2,
    linewidth=1.0,
    yerr=western_plot['remaining_area_uncertainty_1sigma_percent'],
    capsize=1.8,
    elinewidth=0.6,
    capthick=0.6,
    label='Western glaciers'
)

ax_a.set_xlim(1940, 2025)
ax_a.set_ylim(-52, 12)
ax_a.set_xticks(np.arange(1940, 2031, 20))
ax_a.set_yticks(np.arange(-50, 11, 10))
ax_a.set_xlabel('Year')
ax_a.set_ylabel('Glacier area changes\n(%; rel. to 1947)')
ax_a.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax_a.grid(axis='y', color='0.85', linewidth=0.4)
ax_a.text(
    0.99, 0.99, '(A)', fontsize=10, fontweight='bold',
    ha='right', va='top', transform=ax_a.transAxes
)
ax_a.legend(
    loc='lower left',
    ncol=1,
    frameon=False,
    fontsize=7,
    labelspacing=0.3,
    handlelength=1.8
)

############################################################################ Figure_2B
ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)

glacier_colors = {
    'Winston': '#66a61e',
    'G073625E53121S': '#7570b3',
    'Stephenson': '#e7298a',
    'Brown': '#1b9e77',
    'Compton': '#d95f02'
}
glacier_markers = {
    'Winston': 'v',
    'G073625E53121S': '^',
    'Stephenson': 'D',
    'Brown': 'o',
    'Compton': 's'
}

for glacier_name in [
    'Compton', 'Brown', 'Stephenson', 'G073625E53121S', 'Winston'
]:
    glacier_data = eastern_area_change[
        eastern_area_change['Name'] == glacier_name
    ]
    ax_b.errorbar(
        glacier_data['Date'],
        glacier_data['remaining'] - 100,
        color=glacier_colors[glacier_name],
        marker=glacier_markers[glacier_name],
        markersize=2.8,
        linewidth=1.0,
        yerr=glacier_data[
            'remaining_uncertainty_1sigma_percent'
        ],
        capsize=1.6,
        elinewidth=0.55,
        capthick=0.55,
        label=glacier_name
    )

ax_b.set_xlim(2000, 2020)
ax_b.set_ylim(-52, 12)
ax_b.set_xticks(np.arange(2000, 2021, 10))
ax_b.set_yticks(np.arange(-50, 11, 10))
ax_b.set_xlabel('Year')
ax_b.set_ylabel('Glacier area changes\n(%; rel. to 2001)')
ax_b.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax_b.tick_params(axis='y', labelleft=True)
ax_b.grid(axis='y', color='0.85', linewidth=0.4)
ax_b.text(
    0.99, 0.99, '(B)', fontsize=10, fontweight='bold',
    ha='right', va='top', transform=ax_b.transAxes
)
ax_b.legend(
    loc='lower left',
    ncol=1,
    frameon=False,
    fontsize=7,
    labelspacing=0.3,
    handlelength=1.8
)

out_png = outpath / 'figure_3.png'
out_pdf = outpath / 'figure_3.pdf'

plt.savefig(out_png, dpi=600, facecolor='white')
plt.savefig(out_pdf, dpi=600, facecolor='white')
