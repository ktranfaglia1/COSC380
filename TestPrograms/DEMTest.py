import rasterio
from rasterio.warp import transform_bounds

# Define Cambridge bounding box in Lat/Lon (EPSG:4326)
cambridge_bbox = [-76.10, 38.55, -76.05, 38.60]  # min_lon, min_lat, max_lon, max_lat

# Open DEM dataset
dem_path = "../Data/Dorchester_DEM/dorc2015_m/"
with rasterio.open(dem_path) as dataset:
    dem_crs = dataset.crs  # Get DEM projection

    # Transform bounding box from EPSG:4326 (lat/lon) to the DEM CRS
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dem_crs, *cambridge_bbox)

    # Print transformed bounding box
    print("Transformed Cambridge Bounds:", minx, miny, maxx, maxy)
    print("DEM Bounds:", dataset.bounds)

    # Check if the transformed bbox is within DEM bounds
    if minx > dataset.bounds.right or maxx < dataset.bounds.left or \
       miny > dataset.bounds.top or maxy < dataset.bounds.bottom:
        print("ERROR: Adjusted Cambridge bounds do not intersect with DEM.")
    else:
        print("SUCCESS: Cambridge bounds are inside DEM extent.")
