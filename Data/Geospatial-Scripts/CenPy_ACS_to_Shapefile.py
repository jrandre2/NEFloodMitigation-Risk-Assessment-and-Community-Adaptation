# Import Libraries
from cenpy import products
import geopandas as gpd
import pandas as pd
import os

# List of counties
counties = ['Sarpy County, NE', 'Dodge County, NE', 'Douglas County, NE', 'Saunders County, NE']

# List of relevant variables = [
    'B01001_001E',  # Total population
    'B01001_002E',  # Male population
    'B01001_026E',  # Female population
    'B02001_003E',  # Black or African American alone
    'B03002_012E',  # Hispanic or Latino
    'B19013_001E',  # Median household income
    'B19301_001E',  # Per capita income
    'B17001_001E',  # Poverty status
    'B25034_001E',  # Year structure built
    'B25024_001E',  # Units in structure
    'B18101_001E',  # Total civilian noninstitutionalized population
    'B18101_002E',  # Total population with a disability
    'B25044_001E',  # Vehicles available
    'B28002_001E',  # Total households (for broadband)
    'B28002_004E'   # Households with a broadband internet subscription
]

# Fetch ACS data for the counties
acs = products.ACS(2017)

dataframes = []
for county in counties:
    df = acs.from_county(county, level='tract', variables=variables)
    dataframes.append(df)

# Combine the dataframes
combined_df = gpd.GeoDataFrame(pd.concat(dataframes, ignore_index=True))

# Drop rows with missing data (optional)
combined_df = combined_df.dropna()

# Calculate percentages for Black and Hispanic populations
combined_df['Black_Percent'] = (combined_df['B02001_003E'] / combined_df['B01001_001E']) * 100
combined_df['Hispanic_Percent'] = (combined_df['B03002_012E'] / combined_df['B01001_001E']) * 100
combined_df['Broadband_Percent'] = (combined_df['B28002_004E'] / combined_df['B28002_001E']) * 100

# Rename columns for better readability
combined_df = combined_df.rename(columns={
    'B01001_001E': 'Total_Pop',
    'B01001_002E': 'Male_Pop',
    'B01001_026E': 'Female_Pop',
    'B02001_003E': 'Black_Pop',
    'B03002_012E': 'Hispanic_Pop',
    'B19013_001E': 'Med_HH_Income',
    'B19301_001E': 'Per_Cap_Income',
    'B17001_001E': 'Poverty_Status',
    'B25034_001E': 'Year_Built',
    'B25024_001E': 'Units_Struct',
    'B18101_001E': 'Tot_Civil_Noninst_Pop',
    'B18101_002E': 'Disability_Pop',
    'B25044_001E': 'Vehicles_Avail',
    'B28002_001E': 'Total_HH',
    'B28002_004E': 'Broadband_HH'
})

# Specify the output directory and shapefile path
output_directory = 'Study_Area_ACS'
shapefile_path = os.path.join(output_directory, 'Nebraska_ACS_data.shp')

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save to shapefile
combined_df.to_file(shapefile_path, driver='ESRI Shapefile', mode='w')

print(f'Shapefile saved to {shapefile_path}')

