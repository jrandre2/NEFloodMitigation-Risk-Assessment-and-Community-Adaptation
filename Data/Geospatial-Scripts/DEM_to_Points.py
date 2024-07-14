import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
import os
from tqdm import tqdm

def get_elevations(points, dem_src, chunk_size=1000, search_radius=3):
    """Get elevations for multiple points efficiently."""
    elevations = []
    total_points = len(points)
    
    for i in tqdm(range(0, total_points, chunk_size), desc="Processing elevations"):
        chunk = points[i:i+chunk_size]
        rows, cols = zip(*[dem_src.index(point.x, point.y) for point in chunk])
        
        # Determine the window for reading the DEM
        min_row, max_row = min(rows) - search_radius, max(rows) + search_radius + 1
        min_col, max_col = min(cols) - search_radius, max(cols) + search_radius + 1
        window = Window(min_col, min_row, max_col - min_col, max_row - min_row)
        
        # Read the DEM data for this window
        dem_data = dem_src.read(1, window=window)
        
        for j, (row, col) in enumerate(zip(rows, cols)):
            # Adjust coordinates to the window
            adj_row, adj_col = row - min_row, col - min_col
            
            # Search for valid elevation
            for r in range(search_radius + 1):
                for di in range(-r, r+1):
                    for dj in range(-r, r+1):
                        try:
                            elevation = dem_data[adj_row + di, adj_col + dj]
                            if elevation is not np.ma.masked:
                                elevations.append(round(float(elevation), 1))
                                raise StopIteration  # Break out of all loops
                        except IndexError:
                            continue
                        except StopIteration:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                elevations.append(None)  # No valid elevation found

    return elevations

def main():
    # File paths
    buildings_path = r"/Users/jesseandrews/Library/CloudStorage/OneDrive-UniversityofNebraskaatKearney/GIS Projects/FEMA_Classifed_Structure_Centriods/Buildings_with_Census_Tracts.shp"
    dem_path = r"/Users/jesseandrews/Library/CloudStorage/OneDrive-UniversityofNebraskaatKearney/GIS Projects/LiDAR/dem.tif"

    # Read buildings shapefile
    print("Reading buildings shapefile...")
    buildings_gdf = gpd.read_file(buildings_path)

    # Open DEM
    print("Processing elevations...")
    with rasterio.open(dem_path) as dem_src:
        # Check if CRS matches
        if buildings_gdf.crs != dem_src.crs:
            print("CRS mismatch detected. Reprojecting buildings to DEM CRS.")
            buildings_gdf = buildings_gdf.to_crs(dem_src.crs)

        # Get elevations
        elevations = get_elevations(buildings_gdf.geometry, dem_src)
        buildings_gdf['elevation'] = elevations

    # Check for any points without elevation data
    no_elevation = buildings_gdf[buildings_gdf['elevation'].isna()]
    if len(no_elevation) > 0:
        print(f"Warning: {len(no_elevation)} buildings could not be assigned an elevation.")
        print("This could be due to points far outside the DEM extent or large areas of NoData values in the DEM.")

    # Basic statistics on elevations
    elevations = buildings_gdf['elevation'].dropna()
    print(f"\nElevation Statistics:")
    print(f"Minimum: {elevations.min():.1f}")
    print(f"Maximum: {elevations.max():.1f}")
    print(f"Mean: {elevations.mean():.1f}")
    print(f"Median: {elevations.median():.1f}")

    # Save the result
    print("Saving results...")
    output_path = os.path.join(os.path.dirname(buildings_path), "Buildings_with_Census_Tracts_and_Elevation.shp")
    buildings_gdf.to_file(output_path)

    print(f"\nProcessing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()