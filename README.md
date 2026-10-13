# Heard-Island-mass-balance-ice-thickness
This repository contains the Python code and analysis products associated with the manuscript “Glacier changes across Subantarctic Heard Island revealed by remote sensing and ice thickness inversions” by Weilin Yang, Levan G. Tielidze, Regine Hock, Bethan J. Davies, Duane D. McKenna, Sarah S. Thompson, and Andrew N. Mackintosh.

We quantify glacier area change from 1947 to 2019, glacier mass balance from 2000 to 2019, and changes in glacier ice thickness and volume across Subantarctic Heard Island using multi-temporal glacier inventories, NASADEM, Pléiades digital elevation models (DEMs), and ice-thickness inversion. The mass balance analysis derives from glacier elevation differences from the co-registered NASADEM and Pléiades DEMs. Glacier ice thickness and bed topography are reconstructed for 1947, 1988, 2001, and 2019. DEM co-registration and spatial uncertainty propagation are based on [xDEM v0.2.3](https://github.com/GlacioHack/xdem.git), developed by the xDEM community, and the ice-thickness reconstructions use [GlabTop2-py v2.1.1](https://github.com/WilcoTerink/GlabTop2-py.git), developed by Wilco Terink based on the GlabTop2 model of Frey et al. (2014).

## Overview of the code
### Mass balance

This part contains the DEM co-registration and geodetic mass-balance calculations. The two scripts must be run in the following order.

- `Coregistered_DEM.py` mosaics the four Pléiades DEMs, co-registers NASADEM to the Pléiades reference surface using xDEM, corrects the curvature-dependent elevation bias, calculates `dh` and `dh/dt`, and estimates the spatially variable DEM uncertainty.
- `Cal_geodetic_mass_balance.py` uses the outputs from `Coregistered_DEM.py` to filter and fill glacier `dh/dt`, calculate glacier and regional volume and mass changes, and propagate the corresponding uncertainties.

### Ice thickness

This part contains the GlabTop2 ice-thickness function and the calculations for the four glacier inventories.

- `Ice_thickness_GlabTop2.py` defines the core `calculate_ice_thickness()` function for a single glacier and is imported by the four year-specific scripts rather than run independently.
- `Cal_thk_1947.py` and `Cal_thk_1988.py` calculate island-wide ice thickness for the 1947 and 1988 glacier inventories, respectively.
- `Cal_thk_2001.py` calculates ice thickness for the eastern glaciers in the 2001 inventory.
- `Cal_thk_2019.py` calculates island-wide ice thickness for the 2019 glacier inventory.
- Each `Cal_thk_*.py` script runs nine combinations of the valley shape and basal shear stress factors and saves the ice thickness grids, glacier volumes and uncertainty information in a NetCDF file.

### Figures

This part contains the scripts used to reproduce the main and supplementary figures in the manuscript.

- `Figure_*.py`


## Overview of the data

The `data/` directory contains only the products that can be redistributed with this repository:

- `Heard_Island_1947_ice_thickness.nc`
- `Heard_Island_1988_ice_thickness.nc`
- `Eastern_glaciers_2001_ice_thickness.nc`
- `Heard_Island_2019_ice_thickness.nc`
- `glacier_mass_balance.csv`

Due to copyright and licensing restrictions, the existing datasets summarized in Table 1 of the manuscript are not redistributed in this repository and must be downloaded independently from their original sources. The glacier and lagoon outlines are available upon request from Weilin Yang or Levan G. Tielidze. 


## Citation

The manuscript is currently in preparation. Please cite the associated article after publication. The full citation and DOI will be added here when available.

## Contact

For questions about the code, please contact:

**Dr Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  
Email: weilin.yang@monash.edu and ywlcwc@gmail.com


