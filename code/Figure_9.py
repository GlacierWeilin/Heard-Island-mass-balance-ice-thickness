from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress


plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'legend.fontsize': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})
plt.rcParams.update({'axes.linewidth': 0.5})
plt.rcParams.update({'axes.titlepad': 3})
plt.rcParams.update({'axes.titlesize': 10})
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
figurepath = root / 'figure'

daily_files = {
    'ERA5': datapath / 'ERA5_Heard_Island_daily_1947_2019.nc',
    'ERA5-Land': datapath / 'ERA5_Land_Heard_Island_daily_1950_2019.nc'
}


#%% Calculate annual climate from the daily files

years = np.arange(1947, 2020)
annual_climate = pd.DataFrame(index=years)
annual_climate.index.name = 'year'

for product_name, daily_file in daily_files.items():
    with xr.open_dataset(daily_file) as source:
        daily = source.load()

    weights = (
        daily['island_intersection_area']
        / daily['island_intersection_area'].sum()
    )
    temperature_daily = daily['t2m'].weighted(weights).mean('cell') - 273.15
    precipitation_daily = daily['tp'].weighted(weights).mean('cell') * 1000
    snowfall_daily = daily['sf'].weighted(weights).mean('cell') * 1000

    temperature_annual = temperature_daily.resample(time='YS').mean()
    precipitation_annual = precipitation_daily.resample(time='YS').sum()
    snowfall_annual = snowfall_daily.resample(time='YS').sum()
    positive_days_annual = (
        temperature_daily > 0
    ).resample(time='YS').sum()
    snowfall_fraction_annual = (
        snowfall_annual / precipitation_annual * 100
    )

    product_years = temperature_annual['time'].dt.year.values
    baseline_mask = (product_years >= 1995) & (product_years <= 2005)
    temperature_baseline = float(
        temperature_annual.values[baseline_mask].mean()
    )
    precipitation_baseline = float(
        precipitation_annual.values[baseline_mask].mean()
    )
    annual_climate.loc[
        product_years,
        product_name + ' temperature (degC)'
    ] = temperature_annual.values
    annual_climate.loc[
        product_years,
        product_name + ' temperature change (degC)'
    ] = temperature_annual.values - temperature_baseline
    annual_climate.loc[
        product_years,
        product_name + ' precipitation (mm yr-1)'
    ] = precipitation_annual.values
    annual_climate.loc[
        product_years,
        product_name + ' precipitation change (%)'
    ] = (
        precipitation_annual.values / precipitation_baseline - 1
    ) * 100
    annual_climate.loc[
        product_years,
        product_name + ' days above 0 degC (days yr-1)'
    ] = positive_days_annual.values
    annual_climate.loc[
        product_years,
        product_name + ' snowfall (mm w.e. yr-1)'
    ] = snowfall_annual.values
    annual_climate.loc[
        product_years,
        product_name + ' snowfall fraction (%)'
    ] = snowfall_fraction_annual.values

#%% Calculate trends

periods = {
    '1947-2000': (1947, 2000),
    '2000-2019': (2000, 2019)
}
variables = {
    'temperature': 'temperature change (degC)',
    'precipitation': 'precipitation change (%)',
    'positive_days': 'days above 0 degC (days yr-1)',
    'snowfall_fraction': 'snowfall fraction (%)'
}
trends = {}

for product_name in daily_files:
    for period_name, (start_year, end_year) in periods.items():
        for variable_name, column_suffix in variables.items():
            values = annual_climate.loc[
                start_year:end_year,
                product_name + ' ' + column_suffix
            ].dropna()
            trend = linregress(values.index, values.values)
            trends[(product_name, period_name, variable_name)] = trend


#%% Plot Figure 9

product_colors = {
    'ERA5': '#e41a1c',
    'ERA5-Land': '#377eb8'
}
panel_settings = [
    {
        'variable': 'temperature',
        'suffix': 'temperature change (degC)',
        'title': '(a) Annual temperature change',
        'ylabel': 'Temperature change (°C)\nrel. to 2000 ± 5',
        'label_offset': {'ERA5': 0.46, 'ERA5-Land': -0.46},
        'label_y': {
            ('ERA5', '1947-2000'): 0.45,
            ('ERA5', '2000-2019'): 0.78,
            ('ERA5-Land', '1947-2000'): -0.80,
            ('ERA5-Land', '2000-2019'): -0.72
        },
        'zero_line': True
    },
    {
        'variable': 'precipitation',
        'suffix': 'precipitation change (%)',
        'title': '(b) Annual precipitation change',
        'ylabel': 'Precipitation change (%)\nrel. to 2000 ± 5',
        'label_offset': {'ERA5': 10.0, 'ERA5-Land': -10.0},
        'label_y': {
            ('ERA5', '1947-2000'): 9,
            ('ERA5', '2000-2019'): 16,
            ('ERA5-Land', '1947-2000'): -24,
            ('ERA5-Land', '2000-2019'): -10
        },
        'zero_line': True
    },
    {
        'variable': 'positive_days',
        'suffix': 'days above 0 degC (days yr-1)',
        'title': '(c) Days above 0 °C',
        'ylabel': 'Days > 0 °C (days yr$^{-1}$)',
        'label_offset': {'ERA5': 20.0, 'ERA5-Land': -15.0},
        'label_y': {
            ('ERA5', '1947-2000'): 255,
            ('ERA5', '2000-2019'): 255,
            ('ERA5-Land', '1947-2000'): 210,
            ('ERA5-Land', '2000-2019'): 220
        },
        'zero_line': False
    },
    {
        'variable': 'snowfall_fraction',
        'suffix': 'snowfall fraction (%)',
        'title': '(d) Solid precipitation fraction',
        'ylabel': 'Snowfall / total precipitation (%)',
        'label_offset': {'ERA5': 5.0, 'ERA5-Land': -5.0},
        'label_y': {
            ('ERA5', '1947-2000'): 57,
            ('ERA5', '2000-2019'): 52,
            ('ERA5-Land', '1947-2000'): 33,
            ('ERA5-Land', '2000-2019'): 34
        },
        'zero_line': False
    }
]

fig_width_inch = 14 / 2.54
fig, axes = plt.subplots(
    2,
    2,
    figsize=(fig_width_inch, 4.55),
    dpi=600,
    sharex=True
)
axes = axes.ravel()

fig.subplots_adjust(
    left=0.105,
    right=0.995,
    bottom=0.105,
    top=0.955,
    wspace=0.23,
    hspace=0.18
)

for panel_index, settings in enumerate(panel_settings):
    ax = axes[panel_index]
    for product_name, color in product_colors.items():
        values = annual_climate[
            product_name + ' ' + settings['suffix']
        ].dropna()
        annual_linewidth = 0.7
        annual_zorder = 2
        ax.plot(
            values.index,
            values.values,
            color=color,
            linewidth=annual_linewidth,
            label=product_name,
            zorder=annual_zorder
        )

        for period_name, (start_year, end_year) in periods.items():
            period_values = values.loc[start_year:end_year]
            trend = trends[
                (product_name, period_name, settings['variable'])
            ]
            ax.plot(
                period_values.index,
                trend.intercept + trend.slope * period_values.index,
                color=color,
                linewidth=1.0,
                linestyle='--',
                zorder=3
            )
            label_year = float(np.mean(period_values.index.to_numpy()))
            label_y = (
                trend.intercept
                + trend.slope * label_year
                + settings['label_offset'][product_name]
            )
            if 'label_y' in settings:
                label_y = settings['label_y'][
                    (product_name, period_name)
                ]
            ax.text(
                label_year,
                label_y,
                f'$k$ = {trend.slope:+.3f}',
                color=color,
                fontsize=6.5,
                ha='center',
                va='center',
                zorder=5
            )

    ax.set_title(settings['title'], fontweight='bold')
    ax.set_ylabel(settings['ylabel'])
    ax.set_xlim(1946.5, 2019.5)
    ax.set_xticks([1947, 1960, 1980, 2000, 2019])
    ax.grid(color='0.88', linewidth=0.4, zorder=0)
    ax.tick_params(labelsize=7)
    ax.axvline(
        2000,
        color='0.55',
        linewidth=0.5,
        linestyle=':',
        zorder=1
    )
    if settings['zero_line']:
        ax.axhline(0, color='0.45', linewidth=0.5, zorder=1)

axes[2].set_xlabel('Year')
axes[3].set_xlabel('Year')

legend = axes[1].legend(
    loc='best',
    ncol=2,
    fontsize=6.5,
    frameon=True,
    borderpad=0.3,
    labelspacing=0.25,
    handlelength=1.2,
    handletextpad=0.35,
    columnspacing=0.7
)
legend.get_frame().set_linewidth(0.4)
legend.get_frame().set_facecolor('white')

fig.savefig(
    figurepath / 'Figure_9.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    figurepath / 'Figure_9.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
plt.close(fig)
