'''Calculate glacier and regional geodetic mass balance with xDEM errors.'''

from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt

#%% Define functions

def fill_empty_bands(values, centres, glacier_id, variable):
    '''Interpolate and extrapolate elevation bands without observations.'''

    values = np.asarray(values, dtype='float64')
    measured = np.isfinite(values)
    if measured.sum() == 1:
        values[:] = values[measured][0]
    elif measured.sum() > 1:
        values = np.interp(centres, centres[measured], values[measured])
    else:
        raise ValueError(f'No valid {variable} bands for glacier {glacier_id}')
    return values


def prepare_reference_glacier(geometry, glacier_id, reference_year):
    '''Filter and fill dh/dt within one glacier reference outline.'''

    mask = geometry_mask([geometry], shape, transform, invert=True)
    lower = np.floor(np.nanmin(binning_elevation[mask]) / bin_size) * bin_size
    upper = (
        np.floor(np.nanmax(binning_elevation[mask]) / bin_size) * bin_size
        + bin_size
    )
    edges = np.arange(lower, upper + bin_size, bin_size)
    centres = edges[:-1] + bin_size / 2
    means, error_medians, retained_masks = [], [], []
    statistics = []

    for band_lower in edges[:-1]:
        band = (
            mask
            & (binning_elevation >= band_lower)
            & (binning_elevation < band_lower + bin_size)
        )
        band_valid = band & np.isfinite(dh) & np.isfinite(dhdt)
        change_values = dh[band_valid]
        median_change = np.nanmedian(change_values) if change_values.size else np.nan
        nmad_change = (
            1.4826 * np.nanmedian(np.abs(change_values - median_change))
            if change_values.size
            else np.nan
        )
        if np.isfinite(nmad_change) and nmad_change > 0:
            retained = band_valid & (
                np.abs(dh - median_change) <= outlier_nmad_factor * nmad_change
            )
        else:
            retained = band_valid

        retained_values = dhdt[retained]
        error_values = dhdt_error[band & np.isfinite(dhdt_error)]
        means.append(np.nanmean(retained_values) if retained_values.size else np.nan)
        error_medians.append(
            np.nanmedian(error_values) if error_values.size else np.nan
        )
        retained_masks.append(retained)
        statistics.append(
            (
                np.count_nonzero(band),
                change_values.size,
                retained_values.size,
                median_change,
                nmad_change,
            )
        )

    means = fill_empty_bands(means, centres, glacier_id, 'dh/dt')
    error_medians = fill_empty_bands(
        error_medians, centres, glacier_id, 'dh/dt uncertainty'
    )

    for band_index, band_lower in enumerate(edges[:-1]):
        band = (
            mask
            & (binning_elevation >= band_lower)
            & (binning_elevation < band_lower + bin_size)
        )
        retained = retained_masks[band_index]
        observed_error = band & np.isfinite(dhdt_error)
        filled_dhdt_reference[retained] = dhdt[retained]
        filled_dhdt_reference[band & ~retained] = means[band_index]
        filled_error_reference[observed_error] = dhdt_error[observed_error]
        filled_error_reference[band & ~observed_error] = error_medians[band_index]
        retained_mask_reference[retained] = True
        rejected_mask_reference[band & np.isfinite(dhdt) & ~retained] = True
        filled_mask_reference[band & ~retained] = True

        pixel_count, input_count, retained_count, median_change, nmad_change = (
            statistics[band_index]
        )
        bin_rows.append(
            {
                'glacier_id': glacier_id,
                'outline_year': reference_year,
                'elevation_bin_centre_m': centres[band_index],
                'mean_dhdt_m_yr': means[band_index],
                'bin_area_km2': pixel_count * pixel_area_m2 / 1e6,
                'input_pixels': input_count,
                'retained_pixels': retained_count,
                'rejected_pixels': input_count - retained_count,
                'missing_pixels': pixel_count - input_count,
                'filled_pixels': pixel_count - retained_count,
                'median_dh_m': median_change,
                'nmad_dh_m': nmad_change,
            }
        )


def calculate_outline(geometry, glacier_id, year):
    '''Integrate the preprocessed dh/dt field within one dated outline.'''

    mask = geometry_mask([geometry], shape, transform, invert=True)
    calculation = mask & np.isfinite(filled_dhdt_reference)
    if not np.any(calculation):
        raise ValueError(f'No valid dh/dt pixels for glacier {glacier_id}, {year}')

    original = mask & np.isfinite(dh) & np.isfinite(dhdt)
    calculation_pixels = np.count_nonzero(calculation)
    retained_pixels = np.count_nonzero(mask & retained_mask_reference)
    filled_pixels = np.count_nonzero(mask & filled_mask_reference)
    return {
        'volume_rate_km3_yr': (
            np.sum(filled_dhdt_reference[calculation]) * pixel_area_m2 / 1e9
        ),
        'analysis_area_km2': geometry.area / 1e6,
        'calculation_area_km2': calculation_pixels * pixel_area_m2 / 1e6,
        'mean_elapsed_years': np.nanmean(elapsed[original]),
        'valid_fraction': retained_pixels / calculation_pixels,
        'filled_fraction': filled_pixels / calculation_pixels,
        'input_pixels': np.count_nonzero(original),
        'retained_pixels': retained_pixels,
        'rejected_pixels': np.count_nonzero(mask & rejected_mask_reference),
        'filled_pixels': filled_pixels,
    }


#%% Set paths and calculation parameters

root = Path(__file__).resolve().parents[1]
glacier_dir = root / 'data' / 'glaciers'
dem_dir = root / 'data'
output_dir = root / 'data'
output_dir.mkdir(parents=True, exist_ok=True)

density = 850.0             # Volume-to-mass factor, kg m-3
density_sigma = 60.0        # 1-sigma conversion-factor uncertainty, kg m-3
pixel_area_m2 = 400.0       # Area of one 20 m pixel
bin_size = 100.0            # Hypsometric elevation-band interval, m
outlier_nmad_factor = 5.0   # Within-band elevation-change filter

#%% Read elevation-change and uncertainty data

# Sign convention: dh = Pléiades - NASADEM, so negative dh/dt indicates
# surface lowering and negative mass balance.
with rasterio.open(dem_dir / 'NASADEM_Pleiades_dh_20m.tif') as src:
    dh = src.read(1).astype('float64')
    dh[dh == src.nodata] = np.nan
with rasterio.open(dem_dir / 'NASADEM_Pleiades_dhdt_20m.tif') as src:
    dhdt = src.read(1).astype('float64')
    dhdt[dhdt == src.nodata] = np.nan
    transform, crs, shape = src.transform, src.crs, src.shape
    dhdt_profile = src.profile.copy()
with rasterio.open(dem_dir / 'NASADEM_2000_coregistered_20m.tif') as src:
    elevation = src.read(1).astype('float64')
    elevation[elevation == src.nodata] = np.nan

# Complete only the missing coordinates needed for band assignment using the
# nearest valid co-registered NASADEM elevation. No other DEM is introduced.
binning_elevation = elevation.copy()
missing = ~np.isfinite(binning_elevation)
nearest = distance_transform_edt(
    missing,
    return_distances=False,
    return_indices=True,
)
binning_elevation[missing] = binning_elevation[
    nearest[0][missing], nearest[1][missing]
]
del missing, nearest

with rasterio.open(dem_dir / 'DEM_elapsed_years.tif') as src:
    elapsed = src.read(1).astype('float64')
    elapsed[elapsed == src.nodata] = np.nan

# Add the spatially variable annual 1-sigma dh/dt uncertainty raster.
with rasterio.open(dem_dir / 'xDEM_dhdt_uncertainty_1sigma_20m.tif') as src:
    dhdt_error = src.read(1).astype('float64')
    dhdt_error[dhdt_error == src.nodata] = np.nan

# Variogram parameters describe spatial correlation in the standardized errors.
variogram = pd.read_csv(dem_dir / 'xDEM_variogram_parameters_20m.csv')
regional_only = os.environ.get('HEARD_REGIONAL_ONLY', '0') == '1'
final_results_file = output_dir / 'glacier_geodetic_mass_balance.csv'
if regional_only:
    rows = pd.read_csv(final_results_file).to_dict('records')
else:
    rows = []
bin_rows = []
analysis_geometries = []

# Filter and fill one common dh/dt field using NASADEM hypsometry. Western
# glaciers use their 1988 outlines; eastern glaciers use their 2001 outlines.
# All dated outlines later extract from this same preprocessed field.
filled_dhdt_reference = np.full(shape, np.nan, dtype='float32')
filled_error_reference = np.full(shape, np.nan, dtype='float32')
retained_mask_reference = np.zeros(shape, dtype=bool)
rejected_mask_reference = np.zeros(shape, dtype=bool)
filled_mask_reference = np.zeros(shape, dtype=bool)

for path in sorted(
    glacier_dir.glob('glacier_*.shp'),
    key=lambda item: int(item.stem.removeprefix('glacier_')),
):
    glacier_id = int(path.stem.removeprefix('glacier_'))
    inventory = gpd.read_file(path).to_crs(crs)
    inventory['Date'] = pd.to_numeric(
        inventory['Date'], errors='raise'
    ).astype(int)
    locations = inventory['Location'].astype(str).str.upper().str.strip().unique()
    if len(locations) != 1 or locations[0] not in {'E', 'W'}:
        raise ValueError(f'Invalid Location in {path.name}')
    reference_year = 2001 if locations[0] == 'E' else 1988
    reference_geometry = (
        inventory[inventory['Date'] == reference_year]
        .geometry.union_all()
        .buffer(0)
    )
    prepare_reference_glacier(reference_geometry, glacier_id, reference_year)

dhdt_error_processed = xdem.DEM.from_array(
    np.ma.masked_invalid(filled_error_reference),
    transform=transform,
    crs=crs,
    nodata=-9999.0,
)


#%% Calculate glacier-specific mass balance

for path in sorted(
    glacier_dir.glob('glacier_*.shp'),
    key=lambda item: int(item.stem.removeprefix('glacier_')),
):
    
    glacier_id = int(path.stem.removeprefix('glacier_'))
    outlines = gpd.read_file(path).to_crs(crs)
    outlines['Date'] = pd.to_numeric(outlines['Date'], errors='raise').astype(int)
    locations = outlines['Location'].astype(str).str.upper().str.strip().unique()
    if len(locations) != 1 or locations[0] not in {'E', 'W'}:
        raise ValueError(f'Invalid Location in {path.name}')
    location = locations[0]
    outlines = outlines.dissolve(by='Date', as_index=False)

    # Eastern glaciers use every available 2001-2019 outline;
    # Western glaciers use the 1988 and 2019 outlines.
    if location == 'E':
        outlines = outlines[outlines['Date'].between(2001, 2019)].sort_values('Date')
        if (
            outlines.empty
            or outlines.iloc[0].Date != 2001
            or outlines.iloc[-1].Date != 2019
        ):
            continue
    else:
        outlines = outlines[outlines['Date'].isin([1988, 2019])].sort_values('Date')
        if set(outlines['Date']) != {1988, 2019}:
            continue

    analysis_geometry = outlines.geometry.union_all().buffer(0)
    analysis_geometries.append(
        {
            'glacier_id': glacier_id,
            'location': location,
            'geometry': analysis_geometry,
        }
    )
    if regional_only:
        continue

    years = outlines['Date'].to_numpy(dtype=float)
    areas = outlines.geometry.area.to_numpy() / 1e6
    if location == 'E':
        # Trapezoidal temporal weights provide the representative area and
        # combine the outline-specific volume-change estimates.
        interval = years[-1] - years[0]
        representative_area = np.trapezoid(areas, years) / interval
        weights = np.zeros(len(years))
        weights[0] = (years[1] - years[0]) / (2 * interval)
        weights[-1] = (years[-1] - years[-2]) / (2 * interval)
        if len(years) > 2:
            weights[1:-1] = (years[2:] - years[:-2]) / (2 * interval)
    else:
        representative_area = areas.mean()
        weights = np.array([0.5, 0.5])

    area_sigmas, central_rates, minimum_rates, maximum_rates = [], [], [], []
    dem_volume_sigmas, valid_fractions, filled_fractions = [], [], []
    calculation_areas, elapsed_means = [], []
    input_pixel_counts, retained_pixel_counts = [], []
    rejected_pixel_counts, filled_pixel_counts = [], []
    for _, outline in outlines.iterrows():
        year = int(outline.Date)
        geometry = outline.geometry.buffer(0)

        # Mapping buffers for each year follow Levan et al. (2025).
        if year == 1947:
            buffer_m = 30.0
        elif year == 1988:
            buffer_m = 20.0
        elif 2001 <= year <= 2010:
            buffer_m = 15.0
        elif year in {2017, 2019}:
            buffer_m = 10.0
        else:
            raise ValueError(f'No mapping buffer defined for {year}')
        
        minimum_geometry = geometry.buffer(-buffer_m).buffer(0)
        maximum_geometry = geometry.buffer(buffer_m).buffer(0)
        area_sigmas.append((maximum_geometry.area - minimum_geometry.area) / 2e6)

        minimum = calculate_outline(minimum_geometry, glacier_id, year)
        central = calculate_outline(geometry, glacier_id, year)
        maximum = calculate_outline(maximum_geometry, glacier_id, year)
        minimum_rates.append(minimum['volume_rate_km3_yr'])
        central_rates.append(central['volume_rate_km3_yr'])
        maximum_rates.append(maximum['volume_rate_km3_yr'])
        valid_fractions.append(central['valid_fraction'])
        filled_fractions.append(central['filled_fraction'])
        calculation_areas.append(central['calculation_area_km2'])
        elapsed_means.append(central['mean_elapsed_years'])
        input_pixel_counts.append(central['input_pixels'])
        retained_pixel_counts.append(central['retained_pixels'])
        rejected_pixel_counts.append(central['rejected_pixels'])
        filled_pixel_counts.append(central['filled_pixels'])

        # Integrate the reference-outline-preprocessed annual pixel errors and
        # their covariance over the dated outline used for volume calculation.
        vector = gpd.GeoDataFrame(geometry=[geometry], crs=crs).to_crs(
            dhdt_error_processed.crs
        )
        error = dhdt_error_processed.crop(
            vector.total_bounds.tolist(), mode='match_pixel'
        )
        mean_dhdt_sigma = xdem.spatialstats.spatial_error_propagation(
            areas=[vector],
            errors=error,
            params_variogram_model=variogram,
            subsample=1000,
            vectorized=True,
            random_state=42,
        )[0]
        dem_volume_sigmas.append(
            mean_dhdt_sigma * central['calculation_area_km2'] / 1000
        )

    area_sigmas = np.asarray(area_sigmas)
    central_rates = np.asarray(central_rates)
    minimum_rates = np.asarray(minimum_rates)
    maximum_rates = np.asarray(maximum_rates)
    dem_volume_sigmas = np.asarray(dem_volume_sigmas)
    calculation_areas = np.asarray(calculation_areas)
    representative_area_sigma = np.sqrt(np.sum((weights * area_sigmas) ** 2))
    calculation_area = np.sum(weights * calculation_areas)
    volume_rate = np.sum(weights * central_rates)
    dem_volume_sigma = np.sqrt(np.sum((weights * dem_volume_sigmas) ** 2))
    buffered_mapping_sigma = np.sqrt(
        np.sum((weights * np.abs(maximum_rates - minimum_rates) / 2) ** 2)
    )
    
    if location == 'W':
        # Half the 1988-2019 difference represents western geometry sensitivity.
        geometry_sensitivity_sigma = abs(central_rates[1] - central_rates[0]) / 2
        mapping_volume_sigma = np.hypot(
            buffered_mapping_sigma, geometry_sensitivity_sigma
        )
    else:
        geometry_sensitivity_sigma = 0.0
        mapping_volume_sigma = buffered_mapping_sigma

    # DEM and outline uncertainties are treated as independent 1-sigma terms.
    volume_sigma = np.hypot(dem_volume_sigma, mapping_volume_sigma)

    # Convert km3 yr-1 to Gt yr-1, then normalize by representative area.
    mass_rate = volume_rate * density / 1000
    mass_sigma = np.hypot(
        volume_sigma * density / 1000, abs(volume_rate) * density_sigma / 1000
    )
    specific_balance = mass_rate / representative_area * 1000
    specific_sigma = np.hypot(
        mass_sigma / representative_area * 1000,
        abs(mass_rate) * representative_area_sigma / representative_area**2 * 1000,
    )
    rows.append(
        {
            'glacier_id': f'Heard_Island_{glacier_id:02d}',
            'numeric_id': glacier_id,
            'location': location,
            'dhdt_reference_outline_year': 2001 if location == 'E' else 1988,
            'outline_years': ', '.join(map(str, years.astype(int))),
            'analysis_area_km2': analysis_geometry.area / 1e6,
            'calculation_area_km2': calculation_area,
            'representative_area_km2': representative_area,
            'representative_area_uncertainty_1sigma_km2': representative_area_sigma,
            'valid_dh_fraction': np.sum(weights * np.asarray(valid_fractions)),
            'filled_dhdt_fraction': np.sum(
                weights * np.asarray(filled_fractions)
            ),
            'outlier_filter_nmad_factor': outlier_nmad_factor,
            'weighted_input_pixels': np.sum(weights * np.asarray(input_pixel_counts)),
            'weighted_retained_pixels': np.sum(
                weights * np.asarray(retained_pixel_counts)
            ),
            'weighted_rejected_pixels': np.sum(
                weights * np.asarray(rejected_pixel_counts)
            ),
            'weighted_filled_pixels': np.sum(
                weights * np.asarray(filled_pixel_counts)
            ),
            'mean_elapsed_years': np.sum(weights * np.asarray(elapsed_means)),
            'mean_dhdt_uncertainty_1sigma_m_yr': dem_volume_sigma
            / calculation_area
            * 1000,
            'volume_rate_km3_yr': volume_rate,
            'dem_volume_uncertainty_1sigma_km3_yr': dem_volume_sigma,
            'buffered_mapping_uncertainty_1sigma_km3_yr': buffered_mapping_sigma,
            'western_geometry_sensitivity_1sigma_km3_yr': geometry_sensitivity_sigma,
            'mapping_volume_uncertainty_1sigma_km3_yr': mapping_volume_sigma,
            'volume_uncertainty_1sigma_km3_yr': volume_sigma,
            'mass_rate_gt_yr': mass_rate,
            'mass_uncertainty_1sigma_gt_yr': mass_sigma,
            'specific_balance_mwe_yr': specific_balance,
            'specific_balance_uncertainty_1sigma_mwe_yr': specific_sigma,
        }
    )
    pd.DataFrame(rows).to_csv(
        output_dir / 'glacier_geodetic_mass_balance_progress.csv', index=False
    )
    print(f'Completed Heard_Island_{glacier_id:02d}')

#%% Calculate eastern, western and island-wide mass balance

results = pd.DataFrame(rows).sort_values('numeric_id')
geometry_frame = gpd.GeoDataFrame(analysis_geometries, crs=crs)
regional_rows = []
regional_groups = [
    ('Eastern', results[results.location == 'E']),
    ('Western', results[results.location == 'W']),
    ('Heard Island', results),
    (
        'Eastern excluding Stephenson',
        results[
            (results.location == 'E')
            & (results.numeric_id != 15)
        ],
    ),
    (
        'Heard Island excluding Stephenson',
        results[results.numeric_id != 15],
    ),
]
if regional_only:
    regional_groups = regional_groups[-2:]

for region_name, selected in regional_groups:
    # Propagate DEM uncertainty over the regional union with xDEM's native
    # implementation so that covariance between glaciers is retained.
    ids = set(selected.numeric_id)
    region_geometry = geometry_frame[
        geometry_frame.glacier_id.isin(ids)
    ].geometry.union_all()
    vector = gpd.GeoDataFrame(geometry=[region_geometry], crs=crs).to_crs(
        dhdt_error_processed.crs
    )
    error = dhdt_error_processed.crop(
        vector.total_bounds.tolist(), mode='match_pixel'
    )
    regional_dhdt_sigma = xdem.spatialstats.spatial_error_propagation(
        areas=[vector],
        errors=error,
        params_variogram_model=variogram,
        subsample=1000,
        vectorized=True,
        random_state=42,
    )[0]
    analysis_area = selected.analysis_area_km2.sum()
    calculation_area = selected.calculation_area_km2.sum()
    representative_area = selected.representative_area_km2.sum()
    volume_rate = selected.volume_rate_km3_yr.sum()
    dem_volume_sigma = regional_dhdt_sigma * calculation_area / 1000
    
    # Mapping and representative-area uncertainties are combined in quadrature across glaciers.
    mapping_volume_sigma = np.sqrt(
        np.sum(selected.mapping_volume_uncertainty_1sigma_km3_yr**2)
    )
    volume_sigma = np.hypot(dem_volume_sigma, mapping_volume_sigma)
    mass_rate = volume_rate * density / 1000
    mass_sigma = np.hypot(
        volume_sigma * density / 1000, abs(volume_rate) * density_sigma / 1000
    )
    area_sigma = np.sqrt(np.sum(selected.representative_area_uncertainty_1sigma_km2**2))
    specific_balance = mass_rate / representative_area * 1000
    specific_sigma = np.hypot(
        mass_sigma / representative_area * 1000,
        abs(mass_rate) * area_sigma / representative_area**2 * 1000,
    )
    regional_rows.append(
        {
            'region': region_name,
            'glaciers': len(selected),
            'analysis_area_km2': analysis_area,
            'calculation_area_km2': calculation_area,
            'representative_area_km2': representative_area,
            'representative_area_uncertainty_1sigma_km2': area_sigma,
            'regional_mean_dhdt_uncertainty_1sigma_m_yr': regional_dhdt_sigma,
            'volume_rate_km3_yr': volume_rate,
            'volume_uncertainty_1sigma_km3_yr': volume_sigma,
            'mass_rate_gt_yr': mass_rate,
            'mass_uncertainty_1sigma_gt_yr': mass_sigma,
            'specific_balance_mwe_yr': specific_balance,
            'specific_balance_uncertainty_1sigma_mwe_yr': specific_sigma,
        }
    )

#%% Save final tables

if regional_only:
    existing_regional = pd.read_csv(
        output_dir / 'regional_geodetic_mass_balance.csv'
    )
    existing_regional = existing_regional[
        ~existing_regional['region'].isin([
            'Eastern excluding Stephenson',
            'Heard Island excluding Stephenson',
        ])
    ]
    regional_rows = (
        existing_regional.to_dict('records') + regional_rows
    )

# Rates retain their physical signs; all uncertainty columns are positive 1-sigma.
results.to_csv(output_dir / 'glacier_geodetic_mass_balance.csv', index=False)
pd.DataFrame(bin_rows).to_csv(output_dir / 'elevation_bin_results.csv', index=False)
pd.DataFrame(regional_rows).to_csv(
    output_dir / 'regional_geodetic_mass_balance.csv', index=False
)

# Save dh/dt filtered and filled within the western 1988 and eastern 2001
# reference inventories.
dhdt_profile.update(dtype='float32', nodata=-9999.0, compress='deflate')
with rasterio.open(
    dem_dir / 'NASADEM_Pleiades_dhdt_filled_W1988_E2001_20m.tif',
    'w',
    **dhdt_profile,
) as destination:
    destination.write(
        np.where(
            np.isfinite(filled_dhdt_reference),
            filled_dhdt_reference,
            -9999.0,
        ),
        1,
    )
    destination.update_tags(
        processing=(
            'Original dh/dt retained where accepted; missing and 5xNMAD '
            'rejected pixels filled within 100 m elevation bands.'
        ),
        glacier_extent='western 1988 and eastern 2001 glacier outlines',
        elevation_reference='co-registered NASADEM',
        units='m yr-1',
    )
