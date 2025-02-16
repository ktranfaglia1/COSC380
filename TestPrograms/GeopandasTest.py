#  Author: Kyle Tranfaglia
#  Title: Test Program for the City of Cambridge GIS Tool
#  Last updated: 02/15/25
#  Description: This program tests geopandas features to preform operations, calculate geometric attributes, and plot
#  static maps. It also explores shape plotting with shapely
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# Path to your shapefile (update the filename if different)
shapefile_path = "../Data/WorldCountries/world_countries.shp"

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Perform basic spatial operations
gdf["centroid"] = gdf.geometry.centroid  # Compute centroids
gdf["buffered"] = gdf.geometry.buffer(1)  # Buffer around geometries (1 unit)
gdf["bounding_box"] = gdf.geometry.envelope  # Bounding box (min/max coordinates)

# Calculate geometric attributes
gdf["area"] = gdf.geometry.area  # Area of each country
gdf["length"] = gdf.geometry.length  # Length of each geometry

# Geometric predicates
point = Point(0, 0)  # Example point
gdf["contains_point"] = gdf.geometry.contains(point)

# Plot buffer zones
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color="lightblue", edgecolor="black")
gdf["buffered"].plot(ax=ax, color="orange", alpha=0.5)
ax.set_title("Buffered Zones")
plt.legend(["Country Boundaries", "Buffered Zones"])
plt.show()

# Plot centroids
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color="lightgrey", edgecolor="black")
gdf["centroid"].plot(ax=ax, color="red", markersize=5)
ax.set_title("Centroids of Countries")
plt.legend(["Country Boundaries", "Centroids"])
plt.show()

# Plot bounding boxes
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color="lightgreen", edgecolor="black")
gdf["bounding_box"].plot(ax=ax, color="yellow", alpha=0.5)
ax.set_title("Bounding Boxes")
plt.legend(["Country Boundaries", "Bounding Boxes"])
plt.show()

# Create a new geometry
custom_point = Point(3, 7)
custom_polygon = Polygon([(0, 0), (3, 0), (7, 9), (0, 9)])

# Create a new GeoDataFrame with custom geometries
new_gdf = gpd.GeoDataFrame(geometry=[custom_point, custom_polygon])

# Plot custom geometries
new_gdf.plot(figsize=(10, 6), color=["blue", "green"], edgecolor="black")
plt.title("Custom Point and Polygon")
plt.show()
