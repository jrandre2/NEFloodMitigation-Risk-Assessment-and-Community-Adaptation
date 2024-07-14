import geopandas as gpd
import os

def main():
    # File paths
    buildings_path = r"/Users/jesseandrews/Library/CloudStorage/OneDrive-UniversityofNebraskaatKearney/GIS Projects/FEMA_Classifed_Structure_Centriods/FEMA_Classifed_Structure_Centriods.shp"
    census_tracts_path = r"/Users/jesseandrews/Library/CloudStorage/OneDrive-UniversityofNebraskaatKearney/GIS Projects/Study_Area_ACS/Nebraska_ACS_data.shp"

    # Read shapefiles
    buildings_gdf = gpd.read_file(buildings_path)
    census_tracts_gdf = gpd.read_file(census_tracts_path)

    print("Original CRS:")
    print(f"Buildings CRS: {buildings_gdf.crs}")
    print(f"Census Tracts CRS: {census_tracts_gdf.crs}")

    # Check and align CRS
    if buildings_gdf.crs != census_tracts_gdf.crs:
        print("CRS mismatch detected. Aligning to Census Tracts CRS.")
        buildings_gdf = buildings_gdf.to_crs(census_tracts_gdf.crs)

    print("\nAfter alignment:")
    print(f"Buildings CRS: {buildings_gdf.crs}")
    print(f"Census Tracts CRS: {census_tracts_gdf.crs}")

    # Clip buildings to census tracts extent
    census_tracts_bounds = census_tracts_gdf.total_bounds
    buildings_clipped = buildings_gdf.cx[census_tracts_bounds[0]:census_tracts_bounds[2],
                                         census_tracts_bounds[1]:census_tracts_bounds[3]]

    print(f"\nOriginal building count: {len(buildings_gdf)}")
    print(f"Clipped building count: {len(buildings_clipped)}")

    # Perform spatial join
    buildings_with_tracts = gpd.sjoin(buildings_clipped, census_tracts_gdf, how="left", predicate="within")

    # Check for unmatched buildings
    unmatched = buildings_with_tracts[buildings_with_tracts["index_right"].isna()]
    print(f"\nUnmatched buildings: {len(unmatched)}")

    if len(unmatched) > 0:
        print("Warning: Some buildings could not be matched to a census tract.")
        print("This could be due to buildings outside tract boundaries or precision issues.")

    # Remove unnecessary columns from the join operation
    columns_to_drop = [col for col in buildings_with_tracts.columns if col.endswith('_right') and col != 'index_right']
    buildings_with_tracts = buildings_with_tracts.drop(columns=columns_to_drop)

    # Save the result
    output_path = os.path.join(os.path.dirname(buildings_path), "Buildings_with_Census_Tracts.shp")
    buildings_with_tracts.to_file(output_path)

    print(f"\nProcessing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()