'''Run GlabTop2-py for calculating ice thickness'''

import configparser
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pcraster as pcr
import rasterio
from GlabTop2.main import GlabTop2
from rasterio.features import rasterize


def calculate_ice_thickness(
    dem,
    outline,
    shape_factor=0.8,
    basal_shear_stress_factor=1.0,
    fixed_basal_shear_stress_kpa=None,
    ice_density_kg_m3=900.0,
    gravity_m_s2=9.81,
    random_cell_fraction=0.3,
    number_of_runs=3,
    elevation_intervals=20,
    margin_ice_thickness_m=0.0,
):
    '''
    Calculate ice thickness and volume for one glacier.

    Parameters
    ----------
    dem : rasterio dataset
        An opened DEM cropped around the single glacier outline.
    outline : one-row GeoDataFrame
        Outline of one glacier.
    shape_factor : float
        default: 0.8
        Valley-shape factor used in the ice-thickness calculation.
    basal_shear_stress_factor : float
        default: 1.0
        Multiplier applied to the basal shear stress calculated internally by
        GlabTop2.
    fixed_basal_shear_stress_kpa : float or None
        Fixed basal shear stress in kPa. When supplied, this replaces the
        elevation-range parameterization and the stress factor is not used.
    Returns
    -------
    ice_thickness : numpy.ndarray
        Ice thickness in metres on the input DEM grid. Pixels outside the
        glacier are NaN.
    volume_km3 : float
        Total glacier volume in cubic kilometres.
    basal_shear_stress_kpa : float
        Basal shear stress after applying the requested multiplier, in kPa.
    '''
    transform = dem.transform
    crs = dem.crs
    dem_values = dem.read(1, masked=True).filled(np.nan).astype('float32')
    outline = outline.geometry.iloc[0]

    glacier_mask = rasterize(
        [(outline, 1)],
        out_shape=dem_values.shape,
        transform=transform,
        fill=0,
        dtype='uint8',
    ).astype(bool)
    with tempfile.TemporaryDirectory(prefix='glabtop2_') as temporary_name:
        temporary = Path(temporary_name)

        map_profile = {
            'driver': 'PCRaster',
            'height': dem_values.shape[0],
            'width': dem_values.shape[1],
            'count': 1,
            'crs': crs,
            'transform': transform,
        }
        with rasterio.open(
            temporary / 'surface.map',
            'w',
            dtype='float32',
            nodata=-9999.0,
            PCRASTER_VALUESCALE='VS_SCALAR',
            **map_profile,
        ) as destination:
            destination.write(
                np.where(np.isfinite(dem_values), dem_values, -9999.0).astype(
                    'float32'
                ),
                1,
            )

        glacier_id_grid = glacier_mask.astype('int32')
        with rasterio.open(
            temporary / 'glacier_id.map',
            'w',
            dtype='int32',
            nodata=0,
            PCRASTER_VALUESCALE='VS_NOMINAL',
            **map_profile,
        ) as destination:
            destination.write(glacier_id_grid, 1)

        config = configparser.RawConfigParser()
        config['OUTPUT_FILES'] = {
            'iceDepthMap': 'ice_thickness.map',
            'csvRGI': 'glacier_parameters.csv',
        }

        model = GlabTop2.__new__(GlabTop2)
        model.config = config
        model.mv = False
        model.demMV = -9999.0
        model.results_path = str(temporary)
        model.uIDs = np.array([1])
        model.rgi = pd.DataFrame(
            index=pd.Index([1], name='GLACID'),
            columns=['dH_dem', 'hmin', 'Tau'],
            dtype='float64',
        )
        
        # PCRaster keeps a global clone between model calls. Reset it so each
        # glacier uses the dimensions and georeferencing of its own DEM crop.
        pcr.setclone(str(temporary / 'surface.map'))
        model.dem = pcr.pcr2numpy(
            pcr.readmap(str(temporary / 'surface.map')),
            model.demMV,
        )
        model.glacID = pcr.pcr2numpy(
            pcr.readmap(str(temporary / 'glacier_id.map')),
            model.mv,
        )
        model.eIntervals = elevation_intervals
        model.rho = ice_density_kg_m3
        model.g = gravity_m_s2
        model.n = number_of_runs
        model.r = random_cell_fraction
        model.f = shape_factor
        model.hga = margin_ice_thickness_m

        if fixed_basal_shear_stress_kpa is None:
            standard_shear_stress = model.shearStress
            model.shearStress = lambda elevation_range: (
                standard_shear_stress(elevation_range)
                * basal_shear_stress_factor
            )
        else:
            model.shearStress = lambda elevation_range: (
                fixed_basal_shear_stress_kpa
            )

        model.run()
        basal_shear_stress_kpa = float(model.rgi.loc[1, 'Tau'])

        with rasterio.open(temporary / 'ice_thickness.map') as source:
            ice_thickness = source.read(1).astype('float32')
            ice_thickness[ice_thickness == source.nodata] = np.nan

    ice_thickness[~glacier_mask] = np.nan
    pixel_area_m2 = abs(transform.a * transform.e)
    volume_km3 = float(np.sum(ice_thickness[glacier_mask]) * pixel_area_m2 / 1e9)

    return ice_thickness, volume_km3, basal_shear_stress_kpa
