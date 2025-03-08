import rasterio
import numpy as np

dem_path = "../Data/Dorchester_DEM/dorc2015_m/"  # Adjust path as needed

with rasterio.open(dem_path) as dataset:
    print("DEM CRS:", dataset.crs)  # Print Coordinate Reference System
    print("DEM Metadata:", dataset.meta)

    # Read full dataset
    data = dataset.read(1)

    print("DEM Data Stats:")
    print("Min:", np.nanmin(data))
    print("Max:", np.nanmax(data))
    print("Mean:", np.nanmean(data))
