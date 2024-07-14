import pandas as pd
import geopandas as gpd
import libpysal
from spreg import ML_Lag, ML_Error
from statsmodels.tools import add_constant
from esda.moran import Moran

# Load the shapefile
points_path = "/Users/jesseandrews/Library/CloudStorage/OneDrive-UniversityofNebraskaatKearney/GIS Projects/Study_Area_ACS/Nebraska_ACS_data.shp"
points_gdf = gpd.read_file(points_path)

# Ensure the relevant columns are included, including FEMA code variables
points_gdf = points_gdf[['percent_in', 'Total_Pop', 'Male_Pop', 'Black_Pop', 'Hispanic_P', 'Poverty_St', 'Disability',
                         'Med_HH_Inc', 'Per_Cap_In', 'Vehicles_A', 'Broadband_', 'FZ_AE', 'FZ_X', 'FZ_A', 'FZ_AH', 
                         'FZ_AO', 'geometry']]

# Calculate centroids for polygon geometries
points_gdf['centroid'] = points_gdf.geometry.centroid
points_gdf['x'] = points_gdf.centroid.x
points_gdf['y'] = points_gdf.centroid.y

# Convert to DataFrame for further processing
data = pd.DataFrame(points_gdf.drop(columns=['geometry', 'centroid']))

# Convert necessary variables to percentages using Total_Pop
data['Male_Pop_perc'] = (data['Male_Pop'] / data['Total_Pop']) * 100
data['Black_Pop_perc'] = (data['Black_Pop'] / data['Total_Pop']) * 100
data['Hispanic_P_perc'] = (data['Hispanic_P'] / data['Total_Pop']) * 100
data['Poverty_St_perc'] = (data['Poverty_St'] / data['Total_Pop']) * 100
data['Disability_perc'] = (data['Disability'] / data['Total_Pop']) * 100

# Handle missing data as needed (imputation, filling with mean, etc.)
data['Med_HH_Inc'].fillna(data['Med_HH_Inc'].mean(), inplace=True)
data['Per_Cap_In'].fillna(data['Per_Cap_In'].mean(), inplace=True)
data['Vehicles_A'].fillna(data['Vehicles_A'].mean(), inplace=True)
data['Broadband_'].fillna(data['Broadband_'].mean(), inplace=True)
data['FZ_AE'].fillna(data['FZ_AE'].mean(), inplace=True)
data['FZ_X'].fillna(data['FZ_X'].mean(), inplace=True)
data['FZ_A'].fillna(data['FZ_A'].mean(), inplace=True)
data['FZ_AH'].fillna(data['FZ_AH'].mean(), inplace=True)
data['FZ_AO'].fillna(data['FZ_AO'].mean(), inplace=True)

# Define dependent and independent variables
y_spatial = data['percent_in'].values
X_spatial = data[['Male_Pop_perc', 'Black_Pop_perc', 'Hispanic_P_perc', 'Poverty_St_perc', 'Disability_perc',
                  'Med_HH_Inc', 'Per_Cap_In', 'Vehicles_A', 'Broadband_', 'FZ_AE', 'FZ_X', 'FZ_A', 'FZ_AH', 
                  'FZ_AO', 'x', 'y']].values

# Add a constant to the independent variables
X_spatial = add_constant(X_spatial)

# Create a spatial weights matrix using contiguity (Queen criterion)
w = libpysal.weights.Queen.from_dataframe(points_gdf, use_index=True)
w.transform = 'R'

# Fit a Spatial Lag Model (SLM)
lag_model = ML_Lag(y_spatial, X_spatial, w=w, name_y='percent_in', 
                   name_x=['const', 'Male_Pop_perc', 'Black_Pop_perc', 'Hispanic_P_perc', 'Poverty_St_perc', 'Disability_perc',
                           'Med_HH_Inc', 'Per_Cap_In', 'Vehicles_A', 'Broadband_', 'FZ_AE', 'FZ_X', 'FZ_A', 'FZ_AH', 
                           'FZ_AO', 'x', 'y'])

# Print the SLM model summary
print(lag_model.summary)

# Fit a Spatial Error Model (SEM)
error_model = ML_Error(y_spatial, X_spatial, w=w, name_y='percent_in', 
                       name_x=['const', 'Male_Pop_perc', 'Black_Pop_perc', 'Hispanic_P_perc', 'Poverty_St_perc', 'Disability_perc',
                               'Med_HH_Inc', 'Per_Cap_In', 'Vehicles_A', 'Broadband_', 'FZ_AE', 'FZ_X', 'FZ_A', 'FZ_AH', 
                               'FZ_AO', 'x', 'y'])

# Print the SEM model summary
print(error_model.summary)

# Perform diagnostics for spatial autocorrelation
moran_residuals = Moran(lag_model.u, w)

# Print the diagnostics
print("Moran's I for residuals:", moran_residuals.I)
print("Moran's I p-value:", moran_residuals.p_norm)
