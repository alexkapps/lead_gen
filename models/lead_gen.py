#PURPOSE: generates dataset (shapefile) of leads for ag sales team
#NAME: Alex Kappel
#CONTACT: apkappel@gmail.com
#----------------------------------------

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#sets inputs
silo_path = '../inputs/silos_ilmenard/silos_ilmenard.shp'
parcel_path = '../inputs/parcels_ilmenard/parcels_ilmenard.shp'
farmer_id_str = 'OWNER'
farm_acres_id_str = 'FARM_ACRES'
size_id_str = 'DIAMETER'
cluster_size = 3
cluster_distance_meters = 20
granger_silo_lookup_path = '../inputs/wide_corrugation_bin_data.csv'
shp_output_path = '../outputs/lead_gen.shp'
csv_output_path = '../outputs/lead_gen.csv'

#finds farm parcels, returns farm polygons and summed farm acreage grouped by owner - also provides owner contact info
def return_farmer_gdf(parcel_shp_path, farmer_id_str, farm_acres_id_str):
	#load and clean parcel data and fields of interest
	parcel_df = gpd.read_file(parcel_shp_path)
	parcel_df[farmer_id_str] = parcel_df[farmer_id_str].str.strip()
	#assume NaN farm acres data is '0'
	parcel_df[farm_acres_id_str] = parcel_df[farm_acres_id_str].fillna(value=0)
	#subset parcel df for records with farm acres >0
	farm_parcels_df = parcel_df[parcel_df[farm_acres_id_str]>0]
	#subset df to fields of interest
	farm_parcels_by_owner_df = farm_parcels_df[[farmer_id_str,farm_acres_id_str,'geometry']]
	#create df for contact info
	farm_parcels_contact_df = farm_parcels_df[[farmer_id_str,'MAILTO_ADD','MAILTO_CSZ']]
	#set index to farmer ID for easy joining later
	farm_parcels_contact_df = farm_parcels_contact_df.set_index(farmer_id_str)
	#remove duplicates in farmer contact info
	farm_parcels_contact_df = farm_parcels_contact_df[~farm_parcels_contact_df.index.duplicated()]
	#group farm parcels by owner name, sum farm acres, dissolve geoms into single poly
	owner_farm_acres_and_geom = farm_parcels_by_owner_df.dissolve(by=farmer_id_str, aggfunc='sum')
	owner_farm_acres_and_geom.sort_values(by=[farm_acres_id_str], inplace=True, ascending=False)
	#join contact info back to farmer name
	owner_farm_acres_and_geom = owner_farm_acres_and_geom.join(farm_parcels_contact_df)
	#return geodataframe of farmers (owners), their parcel footprint, summed farm acres, and contact info
	return owner_farm_acres_and_geom

#filters silo shapes for those with the correct diameter (4-20), and within a group of at least 3 (within 40 meters of each other) 
#uses spatial reference epsg=32616 for distance calcs (http://spatialreference.org/ref/epsg/wgs-84-utm-zone-16n/)
def return_silo_gdf(silo_shp_path, size_id_str, cluster_size, cluster_distance_meters):
	silo_df = gpd.read_file(silo_shp_path)
	silo_df['unique_id'] = silo_df.index
	#4 to 20 meters in diameter, 3 or more
	silo_size_subset_df = silo_df[(silo_df[size_id_str]>=4) & (silo_df[size_id_str]<=20)]
	#create 'reasonable distance' buffer around each potential silo - will use to determine proximity
	silo_subset_proj = silo_size_subset_df.copy()
	#remember to project data to meters!
	silo_subset_proj['geometry'] = silo_subset_proj['geometry'].to_crs(epsg=32616)
	silo_subset_proj['geometry'] = silo_subset_proj.geometry.buffer(cluster_distance_meters)
	#spatial match, exclude matches to self, group by, those with >2 matches are candidates
	silo_silo_spatial_matches = gpd.sjoin(silo_subset_proj, silo_subset_proj)
	silo_silo_spatial_matches = silo_silo_spatial_matches[['unique_id_left', 'unique_id_right']]
	count_per_silo = silo_silo_spatial_matches.groupby(['unique_id_left']).size()
	#filters for silos in clusters of at least 'cluster size argument'
	silo_id_series = count_per_silo[count_per_silo>=cluster_size]
	silo_id_list = silo_id_series.index.tolist()
	#filters original shapefile by ID list of detected silos
	screened_silo_df = silo_df[silo_df['unique_id'].isin(silo_id_list)]
	#return geodataframe of screened silos
	return screened_silo_df

#returns dict of slopes and intercepts for silo diameter (x) to bushel (y) relationship for each Granger tier
#https://www.brockmfg.com/uploads/pdf/BR_2286_201702_Brock_Non_Stiffened_Storage_Capacities_Fact_Sheet_EM.pdf
#https://stackoverflow.com/questions/39624462/how-to-do-power-curve-fitting-in-python
def diameter_bushel_curve_by_granger_tier(granger_silo_lookup_path):
	#loads granger silo dataset (see link above)
	granger_lookup_df = pd.read_csv(granger_silo_lookup_path)
	#isolates fields of interest
	granger_lookup_df = granger_lookup_df[['diameter_m', 'tiers', 'bushels']]
	#casts tiers to string
	granger_lookup_df['tiers'] = granger_lookup_df['tiers'].astype('str')
	#crosstab to produce bushels by diameter and tier
	bushels_by_tier_diameter = pd.crosstab(granger_lookup_df.diameter_m, granger_lookup_df.tiers, values=granger_lookup_df.bushels, aggfunc=sum)
	#creates field from crosstab output index of diameter_m	
	bushels_by_tier_diameter['diameter_m'] = bushels_by_tier_diameter.index
	#apply natural log to all values
	bushels_by_tier_diameter_log = bushels_by_tier_diameter.apply(np.log)
	#create empty dictionary to store curve coefficient and intercept, for each tier
	power_curve_defs_dict = dict()
	#for each tier, produce a power curve - and add to dictionary
	for tier in granger_lookup_df.tiers.unique():
		#subset dataframe to include only current tier data
		bushels_by_tier_diameter_log_subset = bushels_by_tier_diameter_log[[tier, 'diameter_m']]
		#remove null values, as linear regression will not allow them
		bushels_by_tier_diameter_log_subset = bushels_by_tier_diameter_log_subset[bushels_by_tier_diameter_log_subset[tier].notnull()]
		#create linear regression model
		reg = linear_model.LinearRegression()
		#fit model to tier's data points
		reg.fit(bushels_by_tier_diameter_log_subset[['diameter_m']], bushels_by_tier_diameter_log_subset[tier])
		#store model components (slope and intercept) in dictionary
		power_curve_defs_dict[tier] = [reg.coef_[0], np.exp(reg.intercept_)]
	#return dictionary of curve definitions for each tier
	return power_curve_defs_dict

#creates lookup dict of min/max granger tiers available for each diameter
def min_max_granger_tier_by_diameter(granger_silo_lookup_path):
	#loads granger silo dataset
	granger_lookup_df = pd.read_csv(granger_silo_lookup_path)
	#isolates fields of interest
	granger_lookup_df = granger_lookup_df[['diameter_m', 'tiers', 'bushels']]
	#creates empty dictionary to store min and max granger tiers available for each granger diameter
	min_max_tier_dict = dict()
	#for each diameter in the granger data, find the minimum and the maximum tier available
	for diameter in granger_lookup_df.diameter_m.unique():
		#find the min tier for all diameters
		granger_diam_min_tier = granger_lookup_df.groupby('diameter_m').min()
		#isolate the specific min value for current iterate diameter
		min_tier = granger_diam_min_tier.at[diameter, 'tiers']
		#find the max tier or all diameters
		granger_diam_max_tier = granger_lookup_df.groupby('diameter_m').max()
		#isolate the specific max value for current iterate diameter
		max_tier = granger_diam_max_tier.at[diameter, 'tiers']
		#store min and max tier per granger diameter in dictionary
		min_max_tier_dict[diameter] = [min_tier, max_tier]
	#return dictionary with diameter(key) and [min_tier,max_tier] (value)
	return min_max_tier_dict

#applies curve (from min and max tier levels) to apply a low and high bushel estimate to each silo
def silo_min_max_bushel_est(screened_silo_df, power_curve_defs_dict, min_max_tier_dict):
	#create copy of silo df to host estimates output
	screened_silo_df_est = screened_silo_df.copy()
	#iterate over all of the silos, pull all the correct parts for the curve calculation,
	#and log the minimum and maximum estimates
	for index, row in screened_silo_df_est.iterrows():
		#find the Granger diameter (key in the dict) that is nearest to the actual silo diameter
		#assumes this will be the most relevant granger based model to use
		nearest_key = min(min_max_tier_dict, key=lambda x:abs(x-row.DIAMETER))
		#pull the min and max from the dict using the nearest granger diameter key
		nearest_key_min_max = min_max_tier_dict[nearest_key]
		#set the nearest granger diam to the silo df
		screened_silo_df_est.at[index, 'nearest_granger_diam'] = nearest_key
		#set the min granger tier relevant to the nearest granger diam
		screened_silo_df_est.at[index, 'min_tier'] = str(nearest_key_min_max[0])
		#set the max granger tier relevant to the nearest granger diam
		screened_silo_df_est.at[index, 'max_tier'] = str(nearest_key_min_max[1])

		#pull the silo diameter value
		diameter = screened_silo_df_est.at[index, 'DIAMETER']
		#pull the min tier curve's slope
		min_slope = power_curve_defs_dict[str(nearest_key_min_max[0])][0]
		#pull the min tier curve's intercept
		min_intercept = power_curve_defs_dict[str(nearest_key_min_max[0])][1]
		#apply the min curve to the diameter to estimate the min # of bushels
		screened_silo_df_est.at[index, 'min_bushels_est'] = min_intercept*(diameter**min_slope)

		#pull the max tier curve's slope
		max_slope = power_curve_defs_dict[str(nearest_key_min_max[1])][0]
		#pull the max tier curve's intercept
		max_intercept = power_curve_defs_dict[str(nearest_key_min_max[1])][1]
		#apply the max curve to the diameter to estimate the max # of bushels
		screened_silo_df_est.at[index, 'max_bushels_est'] = max_intercept*(diameter**max_slope)

	#return silo df that now features min and max bushel estimates
	return screened_silo_df_est

#perform a spatial match to match silos (now with estimates) to each farmers polygon area
def match_silo_estimates_to_farmers(screened_silo_df_est, owner_farm_acres_and_geom_df):
	#spatial join between farmers geom and silo geom - this matches all silos to a source farmer
	farm_to_silo_match = gpd.sjoin(owner_farm_acres_and_geom_df, screened_silo_df_est, how='left')
	#pulls the min and max bushels estimates for each farmer (the index)
	farm_to_silo_match = farm_to_silo_match[['min_bushels_est', 'max_bushels_est']]
	#groups by the farmer (index) and sums the silo estimates for all that farmer's silos
	silo_bushel_est_by_farmer = farm_to_silo_match.groupby(farm_to_silo_match.index).sum()
	#join these owner-level min and max bushel sum estimates back to the original owner level database we produced
	bushel_estimates_by_farmer_geom = owner_farm_acres_and_geom_df.join(silo_bushel_est_by_farmer)
	#create owner field from index
	bushel_estimates_by_farmer_geom['owner'] = bushel_estimates_by_farmer_geom.index
	#save final dataframe to shapefile format - this will get used in the Tableau dashboard
	bushel_estimates_by_farmer_geom.to_file(shp_output_path)
	#drops geometry for impending csv export
	bushel_estimates_by_farmer_geom_csv = bushel_estimates_by_farmer_geom.drop(['geometry'], axis=1)
	#exports as csv, drops index, rearranges columns
	bushel_estimates_by_farmer_geom_csv.to_csv(csv_output_path, index=False, columns=['owner','MAILTO_ADD','MAILTO_CSZ','FARM_ACRES','min_bushels_est','max_bushels_est'])
	#prints 'finished' message
	print("completed leads report - find at 'outputs/lead_gen.shp' and 'outputs/lead_gen.csv'")
	#returns output dataframe just in case :)
	return bushel_estimates_by_farmer_geom


#run all functions
owner_farm_acres_and_geom_df = return_farmer_gdf(parcel_path, farmer_id_str, farm_acres_id_str)
screened_silo_df = return_silo_gdf(silo_path, size_id_str, cluster_size, cluster_distance_meters)
power_curve_defs_dict = diameter_bushel_curve_by_granger_tier(granger_silo_lookup_path)
min_max_tier_dict = min_max_granger_tier_by_diameter(granger_silo_lookup_path)
screened_silo_df_est = silo_min_max_bushel_est(screened_silo_df, power_curve_defs_dict, min_max_tier_dict)
bushel_estimates_by_farmer_geom = match_silo_estimates_to_farmers(screened_silo_df_est, owner_farm_acres_and_geom_df)





