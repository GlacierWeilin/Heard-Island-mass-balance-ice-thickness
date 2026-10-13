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
outpath = root / 'figure'
outpath.mkdir(parents=True, exist_ok=True)

era5 = xr.open_dataset(
    datapath / 'ERA5_Heard_Island_monthly_1940_2025.nc'
)
era5_land = xr.open_dataset(
    datapath / 'ERA5_Land_Heard_Island_monthly_1950_2026.nc'
)


#%% Calculate island-wide annual climate

era5_weights = (
    era5['island_intersection_area']
    / era5['island_intersection_area'].sum()
)
era5_land_weights = (
    era5_land['island_intersection_area']
    / era5_land['island_intersection_area'].sum()
)

era5_temperature_monthly = (
    era5['t2m'].weighted(era5_weights).mean('cell') - 273.15
)
era5_precipitation_monthly = (
    era5['tp'].weighted(era5_weights).mean('cell')
    * era5['time'].dt.days_in_month
    * 1000
)

era5_land_temperature_monthly = (
    era5_land['t2m'].weighted(era5_land_weights).mean('cell') - 273.15
)
era5_land_precipitation_monthly = (
    era5_land['tp'].weighted(era5_land_weights).mean('cell')
    * era5_land['time'].dt.days_in_month
    * 1000
)

era5_temperature_annual = era5_temperature_monthly.sel(
    time=slice('1947-01-01', '2019-12-31')
).resample(time='YS').mean()
era5_precipitation_annual = era5_precipitation_monthly.sel(
    time=slice('1947-01-01', '2019-12-31')
).resample(time='YS').sum()

era5_land_temperature_annual = era5_land_temperature_monthly.sel(
    time=slice('1947-01-01', '2019-12-31')
).resample(time='YS').mean()
era5_land_precipitation_annual = era5_land_precipitation_monthly.sel(
    time=slice('1947-01-01', '2019-12-31')
).resample(time='YS').sum()

years = np.arange(1947, 2020)
annual_climate = pd.DataFrame(index=years)
annual_climate.index.name = 'year'
temperature_baselines = {}
precipitation_baselines = {}

for product_name, temperature, precipitation in [
    (
        'ERA5',
        era5_temperature_annual,
        era5_precipitation_annual
    ),
    (
        'ERA5-Land',
        era5_land_temperature_annual,
        era5_land_precipitation_annual
    )
]:
    product_years = temperature['time'].dt.year.values
    annual_climate.loc[
        product_years,
        product_name + ' temperature (degC)'
    ] = temperature.values
    baseline_mask = (product_years >= 1995) & (product_years <= 2005)
    temperature_baseline = float(
        temperature.values[baseline_mask].mean()
    )
    temperature_baselines[product_name] = temperature_baseline
    annual_climate.loc[
        product_years,
        product_name + ' temperature change (degC)'
    ] = temperature.values - temperature_baseline
    annual_climate.loc[
        product_years,
        product_name + ' precipitation (mm yr-1)'
    ] = precipitation.values
    precipitation_baseline = float(
        precipitation.values[baseline_mask].mean()
    )
    precipitation_baselines[product_name] = precipitation_baseline
    annual_climate.loc[
        product_years,
        product_name + ' precipitation change (%)'
    ] = (
        precipitation.values / precipitation_baseline - 1
    ) * 100

#%% Calculate linear trends

trend_rows = []
temperature_trends = {}
precipitation_trends = {}

periods = {
    '1947-2000': (1947, 2000),
    '2000-2019': (2000, 2019)
}

for product_name in ['ERA5', 'ERA5-Land']:
    for period_name, (start_year, end_year) in periods.items():
        period_mask = (
            (annual_climate.index >= start_year)
            & (annual_climate.index <= end_year)
        )
        temperature_values = annual_climate.loc[
            period_mask,
            product_name + ' temperature change (degC)'
        ]
        precipitation_values = annual_climate.loc[
            period_mask,
            product_name + ' precipitation change (%)'
        ]

        temperature_valid = temperature_values.notna()
        precipitation_valid = precipitation_values.notna()
        temperature_trend = linregress(
            temperature_values.index[temperature_valid],
            temperature_values[temperature_valid]
        )
        precipitation_trend = linregress(
            precipitation_values.index[precipitation_valid],
            precipitation_values[precipitation_valid]
        )
        temperature_trends[(product_name, period_name)] = temperature_trend
        precipitation_trends[(product_name, period_name)] = precipitation_trend

        trend_rows.append({
            'product': product_name,
            'period': period_name,
            'temperature_start_year': temperature_values.index[
                temperature_valid
            ].min(),
            'temperature_end_year': temperature_values.index[
                temperature_valid
            ].max(),
            'temperature_baseline_1995_2005_degC': (
                temperature_baselines[product_name]
            ),
            'temperature_trend_degC_per_year': temperature_trend.slope,
            'temperature_trend_degC_per_decade': temperature_trend.slope * 10,
            'temperature_p_value': temperature_trend.pvalue,
            'precipitation_start_year': precipitation_values.index[
                precipitation_valid
            ].min(),
            'precipitation_end_year': precipitation_values.index[
                precipitation_valid
            ].max(),
            'precipitation_baseline_1995_2005_mm_per_year': (
                precipitation_baselines[product_name]
            ),
            'precipitation_trend_percent_per_year': precipitation_trend.slope,
            'precipitation_trend_percent_per_decade': (
                precipitation_trend.slope * 10
            ),
            'precipitation_p_value': precipitation_trend.pvalue
        })

#%% Plot Figure 9

product_colors = {
    'ERA5': '#e41a1c',
    'ERA5-Land': '#377eb8'
}

fig_width_inch = 14 / 2.54
fig, axes = plt.subplots(
    1,
    2,
    figsize=(fig_width_inch, 2.40),
    dpi=600
)

fig.subplots_adjust(
    left=0.09,
    right=0.995,
    bottom=0.16,
    top=0.93,
    wspace=0.20
)

for product_name in product_colors:
    temperature_values = annual_climate[
        product_name + ' temperature change (degC)'
    ]
    precipitation_values = annual_climate[
        product_name + ' precipitation change (%)'
    ]
    temperature_valid = temperature_values.notna()
    precipitation_valid = precipitation_values.notna()

    axes[0].plot(
        annual_climate.index[temperature_valid],
        temperature_values[temperature_valid],
        color=product_colors[product_name],
        linewidth=0.7,
        label=product_name,
        zorder=2
    )
    axes[1].plot(
        annual_climate.index[precipitation_valid],
        precipitation_values[precipitation_valid],
        color=product_colors[product_name],
        linewidth=0.7,
        zorder=2
    )
    for period_name, (start_year, end_year) in periods.items():
        temperature_period = temperature_values.loc[start_year:end_year].dropna()
        precipitation_period = precipitation_values.loc[start_year:end_year].dropna()
        temperature_trend = temperature_trends[(product_name, period_name)]
        precipitation_trend = precipitation_trends[(product_name, period_name)]

        axes[0].plot(
            temperature_period.index,
            (
                temperature_trend.intercept
                + temperature_trend.slope * temperature_period.index
            ),
            color=product_colors[product_name],
            linewidth=1.05,
            linestyle='--',
            zorder=3
        )
        axes[1].plot(
            precipitation_period.index,
            (
                precipitation_trend.intercept
                + precipitation_trend.slope * precipitation_period.index
            ),
            color=product_colors[product_name],
            linewidth=1.05,
            linestyle='--',
            zorder=3
        )

        label_year = (start_year + end_year) / 2
        temperature_label_y = (
            temperature_trend.intercept
            + temperature_trend.slope * label_year
        )
        precipitation_label_y = (
            precipitation_trend.intercept
            + precipitation_trend.slope * label_year
        )
        if product_name == 'ERA5':
            temperature_label_y += 0.6
            precipitation_label_y += 13.0
        else:
            temperature_label_y -= 0.6
            precipitation_label_y -= 15.0

        axes[0].text(
            label_year,
            temperature_label_y,
            f'$k$ = {temperature_trend.slope:+.2f}',
            color=product_colors[product_name],
            fontsize=6.5,
            ha='center',
            va='center',
            zorder=5
        )
        axes[1].text(
            label_year,
            precipitation_label_y,
            f'$k$ = {precipitation_trend.slope:+.2f}',
            color=product_colors[product_name],
            fontsize=6.5,
            ha='center',
            va='center',
            zorder=5
        )

axes[0].set_title(
    '(A) Annual temperature change',
    loc='center',
    fontweight='bold'
)
axes[1].set_title(
    '(B) Annual precipitation change',
    loc='center',
    fontweight='bold'
)
axes[0].set_ylabel('Temperature change (°C) rel. to 2000 ± 5')
axes[1].set_ylabel('Precipitation change (%) rel. to 2000 ± 5')
precipitation_minimum = annual_climate[
    [
        'ERA5 precipitation change (%)',
        'ERA5-Land precipitation change (%)'
    ]
].min().min()
precipitation_maximum = annual_climate[
    [
        'ERA5 precipitation change (%)',
        'ERA5-Land precipitation change (%)'
    ]
].max().max()
precipitation_lower = np.floor(precipitation_minimum / 10) * 10 - 5
precipitation_upper = np.ceil(precipitation_maximum / 10) * 10 + 5
axes[1].set_ylim(precipitation_lower, precipitation_upper)
axes[1].axhline(0, color='0.45', linewidth=0.5, zorder=1)

for ax in axes:
    ax.set_xlim(1946.5, 2019.5)
    ax.set_xticks([1947, 1960, 1980, 2000, 2019])
    ax.set_xlabel('Year')
    ax.grid(color='0.88', linewidth=0.4, zorder=0)
    ax.tick_params(labelsize=7)
    ax.axvline(2000, color='0.55', linewidth=0.5, linestyle=':', zorder=1)

legend = axes[1].legend(
    handles=axes[0].get_legend_handles_labels()[0],
    labels=axes[0].get_legend_handles_labels()[1],
    loc='upper center',
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
    outpath / 'Figure_9.png',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
fig.savefig(
    outpath / 'Figure_9.pdf',
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.02
)
plt.close(fig)

era5.close()
era5_land.close()
