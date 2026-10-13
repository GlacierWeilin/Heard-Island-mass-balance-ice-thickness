'''Co-register NASADEM and Pléiades DEMs and estimate xDEM errors.'''

from datetime import date
from pathlib import Path

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject


def calculate_residual_statistics(
    reference_dem,
    aligned_dem,
    stable_mask,
    source,
    acquisition_date,
    stage,
):
    valid = (
        stable_mask
        & np.isfinite(reference_dem)
        & np.isfinite(aligned_dem)
    )
    residual = reference_dem[valid] - aligned_dem[valid]
    median = np.nanmedian(residual)

    return {
        'source': source,
        'acquisition_date': acquisition_date,
        'stage': stage,
        'median_m': median,
        'nmad_m': 1.4826 * np.nanmedian(np.abs(residual - median)),
        'mean_m': np.nanmean(residual),
        'std_m': np.nanstd(residual),
        'count': np.count_nonzero(valid),
    }

#%%
# Input and output folders
root = Path(__file__).resolve().parents[1]
dem_dir = root / 'data'
pleiades_dir = root / 'data'
glacier_dir = root / 'data' / 'glaciers'
output_dir = root / 'data'
output_dir.mkdir(parents=True, exist_ok=True)

pixel_size = 20.0
nasadem_date = date(2000, 2, 16)

# NASADEM
with rasterio.open(dem_dir / 'NASADEM_2000_20m.tif') as source:
    nasadem = source.read(1).astype('float64')
    nasadem[nasadem == source.nodata] = np.nan
    profile = source.profile.copy()
    transform = source.transform
    crs = source.crs
    shape = source.shape

# Pleiades DEM
pleiades_sources = [
    (
        '2018-04-22_0444140_Heard_ANT_1B_DEM_SGM_20m.tif',
        date(2018, 4, 22),
    ),
    (
        '2019-04-07_0452069_Heard_ANT_1B_DEM_SGM_20m.tif',
        date(2019, 4, 7),
    ),
    (
        '2019-04-07_0451575_Heard_ANT_1B_DEM_SGM_20m.tif',
        date(2019, 4, 7),
    ),
    (
        '2019-11-21_0448386_Heard_ANT_1A_DEM_SGM_20m.tif',
        date(2019, 11, 21),
    ),
]

pleiades = np.full(shape, np.nan)
elapsed_years = np.full(shape, np.nan)
source_id = np.full(shape, -1, dtype='int16')

for source_number, (relative_path, acquisition_date) in enumerate(
    pleiades_sources
):
    aligned_dem = np.full(shape, np.nan)

    with rasterio.open(pleiades_dir / relative_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=aligned_dem,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(aligned_dem)
    pleiades[valid] = aligned_dem[valid]
    elapsed_years[valid] = (acquisition_date - nasadem_date).days / 365.2425
    source_id[valid] = source_number

#%%
# Define stable terrain inside the island and outside the 1988 glaciers
glacier_frames = [
    gpd.read_file(path).to_crs(crs) for path in glacier_dir.glob('*.shp')
]
outlines = gpd.GeoDataFrame(
    pd.concat(glacier_frames, ignore_index=True),
    crs=crs,
)
outlines['Date'] = pd.to_numeric(outlines['Date'], errors='raise').astype(int)
glaciers_1988 = outlines.loc[outlines['Date'] == 1988].geometry.union_all()

island_path = root / 'data' / 'Island_boundary_polygon.shp'
island = gpd.read_file(island_path).to_crs(crs)
island_mask = geometry_mask(
    island.geometry,
    shape,
    transform,
    invert=True,
)
glacier_mask = geometry_mask(
    [glaciers_1988],
    shape,
    transform,
    invert=True,
)

slope = np.asarray(xdem.terrain.slope(nasadem, resolution=pixel_size))
stable_nuth_kaab = (
    island_mask
    & ~glacier_mask
    & np.isfinite(nasadem)
    & np.isfinite(pleiades)
    & (slope >= 3)
    & (slope <= 45)
)
stable_statistics = island_mask & ~glacier_mask & np.isfinite(pleiades)

#%%
# Co-register NASADEM to the Pléiades mosaic using Nuth and Kääb
stage_statistics = [
    calculate_residual_statistics(
        pleiades,
        nasadem,
        stable_statistics,
        'Pleiades_mosaic',
        '2018-2019',
        'original',
    )
]

nuth_kaab = xdem.coreg.NuthKaab(
    max_iterations=10,
    subsample=200000,
)
nuth_kaab.fit(
    reference_elev=pleiades,
    to_be_aligned_elev=nasadem,
    inlier_mask=stable_nuth_kaab,
    transform=transform,
    crs=crs,
    random_state=42,
)
nasadem_coregistered, output_transform = nuth_kaab.apply(
    nasadem,
    transform=transform,
    crs=crs,
    resample=True,
)
nasadem_coregistered = np.asarray(
    nasadem_coregistered.filled(np.nan),
    dtype='float64',
)

if output_transform != transform:
    raise RuntimeError('Unexpected xDEM output grid')

stage_statistics.append(
    calculate_residual_statistics(
        pleiades,
        nasadem_coregistered,
        stable_statistics,
        'Pleiades_mosaic',
        '2018-2019',
        'nuth_kaab',
    )
)

#%%
# Remove the residual vertical offset from each Pléiades acquisition
tile_statistics = []

for source_number, (relative_path, acquisition_date) in enumerate(
    pleiades_sources
):
    stable_tile = (
        island_mask
        & ~glacier_mask
        & (source_id == source_number)
        & np.isfinite(pleiades)
        & np.isfinite(nasadem_coregistered)
    )
    residual = pleiades[stable_tile] - nasadem_coregistered[stable_tile]
    vertical_offset = np.nanmedian(residual)
    pleiades[source_id == source_number] -= vertical_offset

    tile_statistics.append(
        {
            'source': Path(relative_path).name,
            'acquisition_date': acquisition_date.isoformat(),
            'vertical_offset_m': vertical_offset,
            'stable_nmad_m': (
                1.4826 * np.nanmedian(np.abs(residual - vertical_offset))
            ),
            'stable_pixels': np.count_nonzero(stable_tile),
        }
    )

stage_statistics.append(
    calculate_residual_statistics(
        pleiades,
        nasadem_coregistered,
        stable_statistics,
        'Pleiades_mosaic',
        '2018-2019',
        'tile_vertical_correction',
    )
)

#%%
# Correct the curvature-dependent elevation bias
maximum_curvature = np.asarray(
    xdem.terrain.max_curvature(
        nasadem_coregistered,
        resolution=pixel_size,
    )
)
stable_curvature = (
    island_mask
    & ~glacier_mask
    & np.isfinite(maximum_curvature)
    & np.isfinite(pleiades)
)

curvature_correction = xdem.coreg.TerrainBias(
    terrain_attribute='max_curvature',
    fit_or_bin='bin_and_fit',
    fit_func='norder_polynomial',
    bin_sizes=30,
)
curvature_correction.fit(
    reference_elev=pleiades,
    to_be_aligned_elev=nasadem_coregistered,
    inlier_mask=stable_curvature,
    bias_vars={'max_curvature': maximum_curvature},
    transform=transform,
    crs=crs,
    random_state=42,
)
nasadem_coregistered, output_transform = curvature_correction.apply(
    nasadem_coregistered,
    bias_vars={'max_curvature': maximum_curvature},
    transform=transform,
    crs=crs,
)
nasadem_coregistered = np.asarray(
    nasadem_coregistered.filled(np.nan),
    dtype='float64',
)
stage_statistics.append(
    calculate_residual_statistics(
        pleiades,
        nasadem_coregistered,
        stable_statistics,
        'Pleiades_mosaic',
        '2018-2019',
        'curvature_correction',
    )
)

#%%
# Remove the final median vertical residual and calculate dh and dh/dt
stable_final = (
    island_mask
    & ~glacier_mask
    & np.isfinite(pleiades)
    & np.isfinite(nasadem_coregistered)
)
final_vertical_offset = np.nanmedian(
    pleiades[stable_final] - nasadem_coregistered[stable_final]
)
nasadem_coregistered += final_vertical_offset
stage_statistics.append(
    calculate_residual_statistics(
        pleiades,
        nasadem_coregistered,
        stable_statistics,
        'Pleiades_mosaic',
        '2018-2019',
        'final_vertical_correction',
    )
)

dh = pleiades - nasadem_coregistered
dhdt = dh / elapsed_years

stable_residual = dh[stable_final]
stable_median = np.nanmedian(stable_residual)
stable_nmad = 1.4826 * np.nanmedian(
    np.abs(stable_residual - stable_median)
)

#%%
# Save the DEM, elevation-change and observation-interval rasters
profile.update(
    dtype='float32',
    nodata=-9999.0,
    compress='deflate',
    tiled=True,
)

output_rasters = {
    'NASADEM_2000_original_20m.tif': nasadem,
    'NASADEM_2000_coregistered_20m.tif': nasadem_coregistered,
    'Pleiades_SGM_20m.tif': pleiades,
    'NASADEM_Pleiades_dh_20m.tif': dh,
    'NASADEM_Pleiades_dhdt_20m.tif': dhdt,
    'DEM_elapsed_years.tif': elapsed_years,
}

for name, array in output_rasters.items():
    with rasterio.open(output_dir / name, 'w', **profile) as destination:
        output = np.where(np.isfinite(array), array, -9999.0).astype('float32')
        destination.write(output, 1)

#%%
# Estimate spatially variable elevation uncertainty using xDEM H2022
early_dem = xdem.DEM(output_dir / 'NASADEM_2000_coregistered_20m.tif')
late_dem = xdem.DEM(output_dir / 'Pleiades_SGM_20m.tif').reproject(early_dem)

stable_geometry = island.geometry.union_all().difference(glaciers_1988)
stable_vector = gu.Vector(
    gpd.GeoDataFrame(
        geometry=[stable_geometry],
        crs=crs,
    )
)

elevation_difference = late_dem - early_dem
uncertainty_variables = [
    xdem.terrain.slope(early_dem),
    xdem.terrain.max_curvature(early_dem),
]

elevation_error, _, _ = (
    xdem.spatialstats.infer_heteroscedasticity_from_stable(
        dvalues=elevation_difference,
        list_var=uncertainty_variables,
        list_var_names=['slope', 'max_curvature'],
        stable_mask=stable_vector,
    )
)
elevation_error.to_file(
    output_dir / 'xDEM_dh_uncertainty_1sigma_20m.tif'
)

# Convert elevation-change uncertainty to annual dh/dt uncertainty
elapsed_years_dem = xdem.DEM(output_dir / 'DEM_elapsed_years.tif')
dhdt_error = elevation_error / elapsed_years_dem
dhdt_error.to_file(
    output_dir / 'xDEM_dhdt_uncertainty_1sigma_20m.tif'
)

variogram_results = xdem.spatialstats.infer_spatial_correlation_from_stable(
    dvalues=elevation_difference,
    list_models=['gaussian', 'spherical'],
    stable_mask=stable_vector,
    errors=elevation_error,
    subsample=5000,
    n_variograms=5,
    random_state=42,
)
empirical_variogram, variogram_parameters, spatial_correlation = (
    variogram_results
)
empirical_variogram.to_csv(
    output_dir / 'xDEM_empirical_variogram_20m.csv',
    index=False,
)
variogram_parameters.to_csv(
    output_dir / 'xDEM_variogram_parameters_20m.csv',
    index=False,
)

#%%
# Save co-registration statistics
pd.DataFrame(tile_statistics).to_csv(
    output_dir / 'Pleiades_tile_vertical_offsets.csv',
    index=False,
)
pd.DataFrame(stage_statistics).to_csv(
    output_dir / 'DEM_coregistration_stage_statistics.csv',
    index=False,
)

transformation_matrix = nuth_kaab.to_matrix()
summary = {
    'horizontal_shift_east_m': transformation_matrix[0, 3],
    'horizontal_shift_north_m': transformation_matrix[1, 3],
    'vertical_shift_m': transformation_matrix[2, 3],
    'final_vertical_residual_removed_m': final_vertical_offset,
    'stable_median_dh_m': stable_median,
    'stable_nmad_dh_m': stable_nmad,
    'stable_pixels': np.count_nonzero(stable_final),
    'nasadem_date_used': nasadem_date.isoformat(),
    'xdem_version': xdem.__version__,
}
pd.DataFrame([summary]).to_csv(
    output_dir / 'DEM_coregistration_summary.csv',
    index=False,
)
