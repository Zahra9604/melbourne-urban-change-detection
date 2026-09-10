import geopandas as gpd
from shapely.geometry import Polygon
import os

# Define bounding box coordinates
west, south = 144.62, -37.90
east, north = 144.70, -37.82

# Create polygon geometry
polygon = Polygon([
    (west, south),
    (east, south),
    (east, north),
    (west, north),
    (west, south)
])

# Create GeoDataFrame with WGS84 CRS (EPSG:4326)
gdf = gpd.GeoDataFrame(
    [{'id': 1, 'name': 'Bounding Box'}], 
    geometry=[polygon], 
    crs="EPSG:4326"
)

# 1. Define folder and file path separately
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'AOI_shp')
output_file = os.path.join(output_dir, 'bounding_box.shp')

# 2. Create the parent directory only
os.makedirs(output_dir, exist_ok=True)

# 3. Save to Shapefile
gdf.to_file(output_file)