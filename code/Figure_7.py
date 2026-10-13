from pathlib import Path

import numpy as np
import pandas as pd

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

glacier_results = pd.read_csv(
    datapath / 'eastern_glacier_mass_balance_by_outline.csv'
)
regional_results = pd.read_csv(
    datapath / 'eastern_regional_mass_balance_5_outlines.csv'
)
weighted_results = pd.read_csv(
    datapath / 'eastern_glacier_mass_balance_5_outline_weighted.csv'
)

outline_years = np.array([1988, 2001, 2005, 2010, 2019])
glacier_names = [
    'Winston',
    'G073625E53121S',
    'Stephenson',
    'Brown',
    'Compton'
]

plot_rows = []
for glacier_name in glacier_names:
    for year in outline_years:
        selected = glacier_results[
            glacier_results['glacier_id'].eq(glacier_name)
            & glacier_results['outline_year'].eq(year)
        ].iloc[0]

        plot_rows.append({
            'Glacier': glacier_name,
            'Outline year': year,
            'Mass balance': selected['specific_balance_mwe_yr'],
            'Mass balance uncertainty': (
                selected['specific_balance_uncertainty_1sigma_mwe_yr']
            )
        })

for year in outline_years:
    selected = regional_results[
        regional_results['result_period'].astype(str).eq(str(year))
    ].iloc[0]
    plot_rows.append({
        'Glacier': 'Eastern sector',
        'Outline year': year,
        'Mass balance': selected['specific_balance_mwe_yr'],
        'Mass balance uncertainty': (
            selected['specific_balance_uncertainty_1sigma_mwe_yr']
        )
    })

plot_data = pd.DataFrame(plot_rows)
weighted_rows = []
for glacier_name in glacier_names:
    selected = weighted_results[
        weighted_results['glacier_id'].eq(glacier_name)
    ].iloc[0]
    weighted_rows.append({
        'Glacier': glacier_name,
        'Mass balance': selected['specific_balance_mwe_yr'],
        'Mass balance uncertainty': (
            selected['specific_balance_uncertainty_1sigma_mwe_yr']
        )
    })

selected = regional_results[
    regional_results['result_period'].eq('Weighted 2000-2019')
].iloc[0]
weighted_rows.append({
    'Glacier': 'Eastern sector',
    'Mass balance': selected['specific_balance_mwe_yr'],
    'Mass balance uncertainty': (
        selected['specific_balance_uncertainty_1sigma_mwe_yr']
    )
})
weighted_plot_data = pd.DataFrame(weighted_rows)

# %% plot

series_order = [
    'Eastern sector',
    'Winston',
    'G073625E53121S',
    'Stephenson',
    'Brown',
    'Compton'
]
series_colors = {
    'Eastern sector': 'black',
    'Winston': '#66a61e',
    'G073625E53121S': '#7570b3',
    'Stephenson': '#e7298a',
    'Brown': '#1b9e77',
    'Compton': '#d95f02'
}
series_markers = {
    'Eastern sector': 'h',
    'Winston': 'v',
    'G073625E53121S': '^',
    'Stephenson': 'D',
    'Brown': 'o',
    'Compton': 's'
}

# %% 
fig_width_inch = 8 / 2.54
fig_height_inch = 6 / 2.54
fig, (ax, ax_b) = plt.subplots(
    1,
    2,
    figsize=(fig_width_inch, fig_height_inch),
    dpi=600,
    sharey=True,
    gridspec_kw={'width_ratios': [5, 0.8]}
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.27,
    top=0.97,
    wspace=0.035
)

for glacier_name in series_order:
    selected = plot_data[
        plot_data['Glacier'].eq(glacier_name)
    ].sort_values('Outline year')
    ax.errorbar(
        selected['Outline year'],
        selected['Mass balance'],
        yerr=selected['Mass balance uncertainty'],
        color=series_colors[glacier_name],
        marker=series_markers[glacier_name],
        markersize=2.0,
        linestyle='none',
        elinewidth=0.55,
        capsize=1.6,
        capthick=0.55,
        label=glacier_name,
        zorder=4 if glacier_name == 'Eastern sector' else 3
    )

ax.set_xlim(1987, 2020)
ax.set_xticks(outline_years)
ax.set_ylim(-1.4, 0)
ax.set_yticks(np.arange(-1.4, 0.01, 0.2))
ax.set_xlabel('Glacier outline year')
ax.set_ylabel('Glacier mass balance (m w.e. yr$^{-1}$)')
ax.grid(axis='y', color='0.85', linewidth=0.4, zorder=0)
ax.tick_params(labelsize=8)
ax.text(
    0.98,
    0.02,
    '(A)',
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=10,
    fontweight='bold'
)

for glacier_name in series_order:
    selected = weighted_plot_data[
        weighted_plot_data['Glacier'].eq(glacier_name)
    ].iloc[0]
    ax_b.errorbar(
        0,
        selected['Mass balance'],
        yerr=selected['Mass balance uncertainty'],
        color=series_colors[glacier_name],
        marker=series_markers[glacier_name],
        markersize=2.0,
        linestyle='none',
        elinewidth=0.55,
        capsize=1.6,
        capthick=0.55,
        zorder=4 if glacier_name == 'Eastern sector' else 3
    )

ax_b.set_xlim(-0.5, 0.5)
ax_b.set_xticks([0])
ax_b.set_xticklabels(['Multi-\ninventory'])
ax_b.grid(axis='y', color='0.85', linewidth=0.4, zorder=0)
ax_b.tick_params(axis='x', labelsize=8)
ax_b.tick_params(axis='y', left=False, labelleft=False)
ax_b.text(
    0.90,
    0.02,
    '(B)',
    transform=ax_b.transAxes,
    ha='right',
    va='bottom',
    fontsize=10,
    fontweight='bold'
)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.56, 0.00),
    ncol=3,
    frameon=True,
    framealpha=1,
    fontsize=7,
    borderpad=0.3,
    columnspacing=0.9,
    labelspacing=0.3,
    handlelength=1.4,
    handletextpad=0.35
)
legend.get_frame().set_linewidth(0.4)
legend.get_frame().set_facecolor('white')

fig.savefig(
    outpath / 'Figure_7.png',
    dpi=600,
)
fig.savefig(
    outpath / 'Figure_7.pdf',
    dpi=600,
)
plt.close(fig)
