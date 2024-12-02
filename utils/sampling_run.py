# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:30:26 2023

@author: benjaminstoeckigt
"""

import os
import glob
import geopandas as gpd
import random
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import tempfile
import subprocess
from rasterio.transform import from_origin

#shapely.speedups.disable()
#os.chdir(r"E:\++++Promotion\Verwaltung\Publikation_1\Workflow_Scripts\final")



def generate_points(tolerance, raster, aoi_shp, total_points, value_ranges, distance, existing_points_gdf=None):
    # Open the raster file
    with rasterio.open(raster) as src:
        raster_crs = src.crs  # Get the coordinate reference system of the raster

        # Load the Area Of Interest (AOI) shapefile
        aoi_gdf = gpd.read_file(aoi_shp)
        # Reproject the AOI shapefile to match the CRS of the raster, if necessary
        if aoi_gdf.crs != raster_crs:
            aoi_gdf = aoi_gdf.to_crs(raster_crs)

        # Extract the geometry of AOI as shapes
        shapes = [geometry for geometry in aoi_gdf.geometry]
        # Mask the raster using AOI and get the corresponding array and transformation
        raster_image, raster_transform = rasterio.mask.mask(src, shapes, crop=True, filled=True)
        raster_image = raster_image[0]  # Get the first band
        nodata_value = src.nodata  # Get the nodata value for the raster

        skip_factor = distance // 10

        if skip_factor > 1:
            #raster_image[::skip_factor, :] = nodata_value
            #raster_image[:, ::skip_factor] = nodata_value

            # Create an empty mask of the same shape as raster_image, initially setting everything to True (mark as NoData)
            mask = np.ones_like(raster_image, dtype=bool)
            # Generate arrays of row indices and column indices using np.meshgrid
            rows, cols = np.meshgrid(np.arange(raster_image.shape[0]), np.arange(raster_image.shape[1]), indexing='ij')
            # Update the mask to False only where both row and column indices are multiples of skip_factor
            mask[(rows % skip_factor == 0) & (cols % skip_factor == 0)] = False
            # Apply the mask to the raster_image by setting True positions in the mask to nodata_value
            raster_image[mask] = nodata_value

        #if sentinel20m == True:
         #   # Set every second row and column to nodata
          #  raster_image[::2, :] = nodata_value
           # raster_image[:, ::2] = nodata_value

        points = []  # List to store generated points
        ranges = []  # List to store the value ranges corresponding to the generated points
        existing_coords = set()  # Set to store coordinates of points to ensure uniqueness

        # If there are existing points, store their coordinates in existing_coords set
        if existing_points_gdf is not None:
            for point in existing_points_gdf.geometry:
                existing_coords.add((point.x, point.y))

        # Loop through each specified value range
        for value_range in value_ranges:
            min_val, max_val, proportion = value_range  # Unpack the value range
            # Find the indices where raster values are within the current range and not equal to nodata
            indices = np.where((raster_image >= min_val) & (raster_image < max_val) & (raster_image != nodata_value))

            # Calculate the number of points to generate for the current value range
            num_points_range = int(total_points * proportion)
            # Oversample by 50% to create a buffer for uniqueness checks
            extra_points = int(num_points_range * 1.5)
            # Randomly sample indices, ensuring we don't exceed the number of available indices
            sample_indices = random.sample(range(len(indices[0])), min(extra_points, len(indices[0])))

            added_points = 0  # Counter for the number of points successfully added for this value range
            # Loop through the sampled indices to generate points
            for idx in sample_indices:
                # Stop if we have already added the desired number of points for this value range
                if added_points >= num_points_range:
                    break

                # Get the row and column from indices
                row, col = indices[0][idx], indices[1][idx]
                # Transform row and column to x and y coordinates
                x_coord, y_coord = rasterio.transform.xy(raster_transform, row, col, offset='center')
                point = Point(x_coord, y_coord)  # Create a Shapely Point object

                # Check for uniqueness of the point
                if (x_coord, y_coord) not in existing_coords:
                    # If there are existing points, check if the point is not too close to them
                    if existing_points_gdf is not None:
                        buffered_point = point.buffer(tolerance)
                        if existing_points_gdf.sindex.query(buffered_point).size == 0:
                            # If point is unique and not too close to existing ones, add it to the list
                            points.append(point)
                            ranges.append("tcd " + str(value_range))
                            existing_coords.add((x_coord, y_coord))
                            added_points += 1  # Increment the counter
                    else:
                        # If there are no existing points to check against, simply add the point to the list
                        points.append(point)
                        ranges.append("vegh " + str(value_range))
                        existing_coords.add((x_coord, y_coord))
                        added_points += 1  # Increment the counter

    return points, ranges  # Return the list of generated points and corresponding value ranges


def generate_points_based_on_distance(tolerance, aoi_shp, total_points, distance):
    # Load the Area Of Interest (AOI) shapefile
    aoi_gdf = gpd.read_file(aoi_shp)

    # Get the extent of the AOI geometry
    aoi_geometry = aoi_gdf.unary_union
    minx, miny, maxx, maxy = aoi_geometry.bounds

    width = int((maxx - minx) / distance)
    height = int((maxy - miny) / distance)

    transform = from_origin(minx, maxy, distance, distance)

    # Create a blank raster array
    raster_image = np.ones((height, width))

    points = []

    # Generate total_points by randomly selecting raster cells within AOI geometry
    for _ in range(total_points):
        row, col = random.randint(0, height - 1), random.randint(0, width - 1)
        x_coord, y_coord = rasterio.transform.xy(transform, row, col, offset='center')
        point = Point(x_coord, y_coord)

        while raster_image[row, col] == 0 or not aoi_geometry.contains(point):
            row, col = random.randint(0, height - 1), random.randint(0, width - 1)
            x_coord, y_coord = rasterio.transform.xy(transform, row, col, offset='center')
            point = Point(x_coord, y_coord)

        points.append(point)

    return points  # Return the list of generated points

def sampling(project_name,process_folder,aoi_files,output_n,output_n_m,percent,distance,
    value_ranges_raster1,value_ranges_raster2,raster_files1,raster_files2,**kwargs):

    tolerance = 2
    output_folder = f"{process_folder}/results/_SamplingPoints/{project_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #print(vegh_files)
    #print(tcd_files)
    #print(aoi_files)

    if raster_files1 is None:
        raster_files1 = [None] * len(aoi_files)
    if raster_files2 is None:
        raster_files2 = [None] * len(aoi_files)

    for vegh_file, tcd_file, aoi_file, city, year in zip(raster_files1, raster_files2, aoi_files, output_n, output_n_m):
        print(f"started calculating points for {aoi_file}")

        if vegh_file == None and tcd_file == None:
            print(f"sampling without grids for stratification")
            aoi_gdf = gpd.read_file(aoi_file)
            # Check if 'area' column exists, if not calculate area in m²
            if 'area' not in aoi_gdf.columns:
                print("area column not found in shapefile ... calculating")
                # Assuming the CRS of the GeoDataFrame is in meters.
                # If not, you'll need to project it to an appropriate CRS first.
                aoi_gdf['area'] = aoi_gdf.geometry.area

            # ###START ADJUSTED
            # all_points = []  # List to collect all generated points
            # all_ranges = []  # List to collect value ranges if applicable
            #
            # for index, row in aoi_gdf.iterrows():
            #     polygon_area = row['area']
            #     total_points = round(
            #         polygon_area * percent * 0.01 * (0.25 if sentinel20m else 1))  # Adjusting for Sentinel pixel size
            #
            #     print("Sampling with grids for stratification")
            #     points, ranges = generate_points(sentinel20m, tolerance, vegh_file if vegh_file else tcd_file,
            #                                      row['geometry'], total_points,
            #                                      value_ranges_vegh if vegh_file else value_ranges_tcd)
            #     all_ranges.extend(ranges)
            #     all_points.extend(points)
            # # Creating a GeoDataFrame from all points
            # gdf = gpd.GeoDataFrame(geometry=all_points)
            # if all_ranges:
            #     gdf['val_range'] = all_ranges
            # if vegh_file:
            #     gdf.crs = rasterio.open(vegh_file).crs
            # else:
            #     gdf.crs = aoi_gdf.crs
            # # Create output filename based on city and year
            # output_filename = os.path.join(output_folder, f"{city}_{year}_points.shp")
            # # Save GeoDataFrame as a new Shapefile
            # gdf.to_file(output_filename)
            # print(f"Finished processing for {city} {year}")


            area_sum = aoi_gdf['area'].sum()
            total_points = round(area_sum * percent * 0.01) #area_sum(m²) * ProzentAnteil * 1/100, weil Sentinel Pixel 100m² * 1/4, weil Sentinel 20 Meter Pixel

            all_points = generate_points_based_on_distance(tolerance, aoi_file, total_points, distance)
            gdf = gpd.GeoDataFrame(geometry=all_points)
            gdf.crs = aoi_gdf.crs
            # Create output filename based on city and year
            output_filename = os.path.join(output_folder, f"{city}_{year}_points.shp")

            # Save GeoDataFrame as a new Shapefile
            gdf.to_file(output_filename)

            print(f"Finished processing for {city} {year}")

        else:
            print(f"sampling with grids for stratification")
            aoi_gdf = gpd.read_file(aoi_file)

            if 'area' not in aoi_gdf.columns:
                print("area column not found in shapefile ... calculating")
                # Assuming the CRS of the GeoDataFrame is in meters.
                # If not, you'll need to project it to an appropriate CRS first.
                aoi_gdf['area'] = aoi_gdf.geometry.area

            area_sum = aoi_gdf['area'].sum()

            total_points = round(area_sum * percent * 0.01 * (100 / (distance * distance))) #area_sum(m²) * ProzentAnteil * 1/100, weil Sentinel Pixel 100m² * 1/4, weil Sentinel 20 Meter Pixel
            #if sentinel20m == True:
             #   total_points = round(area_sum * percent * 0.01 * 0.25) #area_sum(m²) * ProzentAnteil * 1/100, weil Sentinel Pixel 100m² * 1/4, weil Sentinel 20 Meter Pixel
            #else:
             #   total_points = round(area_sum * percent * 0.01) #area_sum(m²) * ProzentAnteil * 1/100, weil Sentinel Pixel 100m²

            if tcd_file == None:
                total_points_vegh = int(total_points)
            else:
                total_points_vegh = int(total_points * 0.5)
                total_points_tcd = int(total_points * 0.5)

            points_vegh, ranges_vegh = generate_points(tolerance, vegh_file, aoi_file, total_points_vegh, value_ranges_raster1, distance)

            if tcd_file == None:
                all_points = points_vegh
                all_ranges = ranges_vegh

            else:
                points_vegh_gdf = gpd.GeoDataFrame(geometry=points_vegh)
                points_vegh_gdf.sindex
                points_tcd, ranges_tcd = generate_points(tolerance, tcd_file, aoi_file, total_points_tcd, value_ranges_raster2, distance, existing_points_gdf=points_vegh_gdf)
                all_points = points_vegh + points_tcd
                all_ranges = ranges_vegh + ranges_tcd

            gdf = gpd.GeoDataFrame(geometry=all_points)
            gdf['val_range'] = all_ranges
            gdf.crs = rasterio.open(vegh_file).crs
            # Create output filename based on city and year
            output_filename = os.path.join(output_folder, f"{city}_{year}_points.shp")

            # Save GeoDataFrame as a new Shapefile
            gdf.to_file(output_filename)

            print(f"Finished processing for {city} {year}")
            # print stratifications:
            analyze_shapefiles(f'{process_folder}/results/_SamplingPoints/{project_name}')


def modify_and_run_script(script_path, shp_value, tif_value, output_value):
    with open(script_path, 'r') as original_script:
        content = original_script.read()

        # Replace the relevant lines
        content = content.replace('shp = r"s"', f'shp = r"{shp_value}"')
        content = content.replace('tif = r"t"', f'tif = r"{tif_value}"')
        content = content.replace('output = r"o"', f'output = r"{output_value}"')

        # Write the modified content to a temporary script
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        with open(temp_file.name, 'w') as modified_script:
            modified_script.write(content)
        temp_file.close()
    # Execute the temporary script
    subprocess.run(['python', temp_file.name])
    # time.sleep(1)
    # Clean up
    os.remove(temp_file.name)


def extract_ref(project_name,process_folder,raster_path,column_name,**kwargs):
    scripts_skel = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/force/skel"

    script_path = f"{scripts_skel}/zonal_rasterstats_mp.py"
    shapefile_path = sorted(glob.glob(f"{process_folder}/results/_SamplingPoints/{project_name}/*shp"))
    o_folder = f"{process_folder}/results/_SamplingPoints/{project_name}"
    for shape, raster in zip(shapefile_path, raster_path):
        print(f"extracting for {shape}")

        shape_name = os.path.splitext(os.path.basename(shape))[0]
        if not os.path.exists(o_folder):
            print(f"output folder doesnt exist ... creating {o_folder}")
            os.makedirs(o_folder)
        # Call the function

        # Check if the CRS matches EPSG:3035
        gdf = gpd.read_file(shape)
        if gdf.crs != "EPSG:3035":
            print(f"detected crs {gdf.crs} reprojecting to EPSG:3035")
            gdf = gdf.to_crs("EPSG:3035")
            gdf.to_file(shape.replace(".shp","_3035.shp"))
            shape = shape.replace(".shp","_3035.shp")

        modify_and_run_script(script_path, shape, raster, f"{o_folder}/{shape_name}.shp")

        # Read the shapefile with geopandas
        gdf = gpd.read_file(f"{o_folder}/{shape_name}.shp")

        # Extract raster values using point_query
        # raster_values = point_query(gdf.geometry, raster)

        # Add the raster values to the GeoDataFrame
        gdf[column_name] = gdf["value"]
        gdf = gdf[["geometry", column_name]]

        # Create new columns for X and Y coordinates
        gdf['X'] = gdf.geometry.x
        gdf['Y'] = gdf.geometry.y

        # Reorder the columns
        gdf = gdf[['X', 'Y', column_name, 'geometry']]

        # Count rows before dropping
        initial_row_count = len(gdf)
        # Drop rows where column_name is empty
        gdf = gdf.dropna(subset=[column_name])
        # Count rows after dropping
        final_row_count = len(gdf)
        # Calculate number of dropped rows
        dropped_rows = initial_row_count - final_row_count
        # Print the number of dropped rows
        print(f"{dropped_rows} rows dropped because of missing values!")

        # Drop the CID column
        gdf_csv = gdf.drop('geometry', axis=1)
        # Export the file as CSV without index and header
        if not os.path.exists(o_folder):
            print(f"output folder doesnt exist ... creating {o_folder}")
            os.makedirs(o_folder)

        gdf.to_file(f"{o_folder}/{shape_name}_extract.shp", index=False, header=False, sep=' ')
        gdf_csv.to_csv(f"{o_folder}/{shape_name}_extract.csv", index=False, header=False, sep=' ')
        #os.remove(f"{o_folder}/{shape_name}.shp")



def analyze_shapefiles(directory, column_name='val_range', prefixes=None):
    if prefixes is None:
        prefixes = ['vegh', 'tcd']  # Default prefixes to look for

    # Store results in a dictionary
    results = {}

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".shp"):
            filepath = os.path.join(directory, filename)
            try:
                # Load the shapefile
                gdf = gpd.read_file(filepath)

                # Initialize a dictionary to hold counts for this file
                file_results = {'filename': filename, 'data': {}}

                # Process each prefix
                for prefix in prefixes:
                    # Filter data for current prefix
                    prefix_data = gdf[gdf[column_name].str.startswith(prefix)]

                    # Count the points for each unique value
                    prefix_counts = prefix_data[column_name].value_counts()

                    # Store results
                    file_results['data'][prefix] = prefix_counts.to_dict()

                # Append the file results to the main results dictionary
                results[filename] = file_results

            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

    # Print results
    for file, data in results.items():
        print(f"File: {file}")
        for prefix, counts in data['data'].items():
            print(f"Counts for '{prefix}...' values:")
            for value, count in counts.items():
                print(f"{value}: {count}")
            print()


