'''Calculate island-wide ice thickness for 2019 with the Pleiades DEM.'''

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.mask import mask

from Ice_thickness_GlabTop2 import calculate_ice_thickness


#%% Set paths and model parameters

root = Path(__file__).resolve().parents[1]
outline_dir = root / 'data' / 'glaciers'
dem_path = root / 'data' / 'Pleiades_SGM_COP30_filled_buffered_20m.tif'
output_dir = root / 'data'
output_dir.mkdir(parents=True, exist_ok=True)
netcdf_path = output_dir / 'Heard_Island_2019_ice_thickness.nc'

year = 2019
shape_factors = np.array([0.7, 0.8, 0.9], dtype='float64')
basal_shear_stress_factors = np.array([0.8, 1.0, 1.2], dtype='float64')

glacier_ids = []
glacier_locations = []
for outline_path in sorted(
    outline_dir.glob('glacier_*.shp'),
    key=lambda path: int(path.stem.removeprefix('glacier_')),
):
    outlines = gpd.read_file(outline_path)
    dates = outlines['Date'].astype(int)
    if year in dates.values:
        glacier_ids.append(int(outline_path.stem.removeprefix('glacier_')))
        glacier_locations.append(
            outlines['Location'].astype(str).str.upper().str.strip().iloc[0]
        )


def run_shape_factor(shape_factor):
    '''Run all shear-stress factors for one shape factor.'''
    with rasterio.open(dem_path) as reference:
        grid_shape = reference.shape
        grid_transform = reference.transform

    ice_thickness = np.full(
        (len(basal_shear_stress_factors), *grid_shape),
        np.nan,
        dtype='float32',
    )
    glacier_volume = np.full(
        (len(basal_shear_stress_factors), len(glacier_ids)),
        np.nan,
        dtype='float64',
    )
    basal_shear_stress = np.full(
        (len(basal_shear_stress_factors), len(glacier_ids)),
        np.nan,
        dtype='float32',
    )

    for stress_index, stress_factor in enumerate(
        basal_shear_stress_factors
    ):
        with rasterio.open(dem_path) as source:
            for glacier_index, glacier_id in enumerate(glacier_ids):
                outlines = gpd.read_file(
                    outline_dir / f'glacier_{glacier_id}.shp'
                ).to_crs(source.crs)
                outlines['Date'] = outlines['Date'].astype(int)
                outline = (
                    outlines[outlines['Date'] == year]
                    .dissolve()
                    .reset_index(drop=True)
                )

                glacier_geometry = outline.geometry.iloc[0]
                bounds = glacier_geometry.bounds
                clipping_buffer = max(
                    bounds[2] - bounds[0],
                    bounds[3] - bounds[1],
                )
                clipped_values, clipped_transform = mask(
                    source,
                    [glacier_geometry.buffer(clipping_buffer)],
                    crop=True,
                    filled=False,
                )
                profile = source.profile.copy()
                profile.update(
                    height=clipped_values.shape[1],
                    width=clipped_values.shape[2],
                    transform=clipped_transform,
                )

                with MemoryFile() as memory_file:
                    with memory_file.open(**profile) as clipped_dem:
                        clipped_dem.write(
                            clipped_values.filled(profile['nodata'])
                        )
                        result = calculate_ice_thickness(
                            clipped_dem,
                            outline,
                            shape_factor=float(shape_factor),
                            basal_shear_stress_factor=float(stress_factor),
                        )

                glacier_thickness, volume_km3, tau_kpa = result
                glacier_volume[stress_index, glacier_index] = volume_km3
                basal_shear_stress[stress_index, glacier_index] = tau_kpa

                column_offset = int(
                    round(
                        (clipped_transform.c - grid_transform.c)
                        / grid_transform.a
                    )
                )
                row_offset = int(
                    round(
                        (grid_transform.f - clipped_transform.f)
                        / abs(grid_transform.e)
                    )
                )
                row_slice = slice(
                    row_offset,
                    row_offset + glacier_thickness.shape[0],
                )
                column_slice = slice(
                    column_offset,
                    column_offset + glacier_thickness.shape[1],
                )
                output_window = ice_thickness[
                    stress_index,
                    row_slice,
                    column_slice,
                ]
                valid = np.isfinite(glacier_thickness)
                output_window[valid] = glacier_thickness[valid]

    return ice_thickness, glacier_volume, basal_shear_stress


def main():
    '''Run the sensitivity analysis and save the outputs.'''
    with rasterio.open(dem_path) as reference:
        transform = reference.transform
        crs = reference.crs
        x = transform.c + (np.arange(reference.width) + 0.5) * transform.a
        y = transform.f + (np.arange(reference.height) + 0.5) * transform.e

    glacier_area = np.full(len(glacier_ids), np.nan, dtype='float64')
    source_polygon_count = np.zeros(len(glacier_ids), dtype='int16')

    for glacier_index, glacier_id in enumerate(glacier_ids):
        outlines = gpd.read_file(
            outline_dir / f'glacier_{glacier_id}.shp'
        ).to_crs(crs)
        outlines['Date'] = outlines['Date'].astype(int)
        dated_outlines = outlines[outlines['Date'] == year]
        source_polygon_count[glacier_index] = len(dated_outlines)
        dissolved = dated_outlines.dissolve()
        glacier_area[glacier_index] = dissolved.geometry.area.iloc[0] / 1e6

    workers = min(len(shape_factors), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_shape_factor, shape_factors))

    ice_thickness = np.stack([result[0] for result in results])
    glacier_volume = np.stack([result[1] for result in results])
    basal_shear_stress = results[0][2]
    regional_volume = np.sum(glacier_volume, axis=2)

    dataset = xr.Dataset(
        data_vars={
            'ice_thickness': (
                ('shape_factor', 'basal_shear_stress_factor', 'y', 'x'),
                ice_thickness,
                {
                    'long_name': 'GlabTop2 modelled ice thickness',
                    'units': 'm',
                },
            ),
            'glacier_volume': (
                ('shape_factor', 'basal_shear_stress_factor', 'glacier_id'),
                glacier_volume,
                {
                    'long_name': 'Modelled glacier volume',
                    'units': 'km3',
                },
            ),
            'regional_volume': (
                ('shape_factor', 'basal_shear_stress_factor'),
                regional_volume,
                {
                    'long_name': 'Total modelled Heard Island glacier volume',
                    'units': 'km3',
                },
            ),
            'basal_shear_stress': (
                ('basal_shear_stress_factor', 'glacier_id'),
                basal_shear_stress,
                {
                    'long_name': 'GlabTop2 basal shear stress after scaling',
                    'units': 'kPa',
                },
            ),
            'glacier_area': (
                ('glacier_id',),
                glacier_area,
                {'long_name': 'Glacier outline area', 'units': 'km2'},
            ),
            'source_polygon_count': (
                ('glacier_id',),
                source_polygon_count,
                {'long_name': 'Number of source polygons dissolved'},
            ),
            'location': (
                ('glacier_id',),
                np.asarray(glacier_locations, dtype=str),
                {'long_name': 'Eastern or western glacier location'},
            ),
        },
        coords={
            'shape_factor': (
                'shape_factor',
                shape_factors,
                {'long_name': 'valley-shape factor', 'units': '1'},
            ),
            'basal_shear_stress_factor': (
                'basal_shear_stress_factor',
                basal_shear_stress_factors,
                {
                    'long_name': (
                        'multiplier applied to the internally calculated '
                        'basal shear stress'
                    ),
                    'units': '1',
                },
            ),
            'glacier_id': (
                'glacier_id',
                np.asarray(glacier_ids, dtype='int32'),
            ),
            'x': ('x', x, {'units': 'm'}),
            'y': ('y', y, {'units': 'm'}),
        },
        attrs={
            'title': 'Heard Island glacier thickness in 2019',
            'model': 'GlabTop2-py v2.1.1',
            'year': year,
            'main_shape_factor': 0.8,
            'main_basal_shear_stress_factor': 1.0,
            'basal_shear_stress_method': (
                'GlabTop2 elevation-range parameterization'
            ),
            'dem': dem_path.name,
            'crs': crs.to_wkt(),
            'ice_density_kg_m3': 900.0,
            'gravity_m_s2': 9.81,
            'random_cell_fraction': 0.3,
            'number_of_runs': 3,
            'elevation_intervals': 20,
            'margin_ice_thickness_m': 0.0,
        },
    )

    dataset.to_netcdf(
        netcdf_path,
        encoding={
            'ice_thickness': {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32',
                'chunksizes': (1, 1, 256, 256),
            },
            'glacier_volume': {'dtype': 'float64'},
            'regional_volume': {'dtype': 'float64'},
            'basal_shear_stress': {'dtype': 'float32'},
            'glacier_area': {'dtype': 'float64'},
            'source_polygon_count': {'dtype': 'int16'},
        },
    )

if __name__ == '__main__':
    main()
