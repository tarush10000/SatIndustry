from datetime import datetime, date
import os
import random
import re
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import requests
import logging
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def get_weather_data(request):
    """
    Fetches current weather data from OpenWeatherMap API.
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return JsonResponse({'error': 'Latitude and longitude are required parameters.'}, status=400)

    api_key = settings.OPENWEATHERAPI_KEY # Securely access API key from settings
    if not api_key:
        logger.error("OpenWeatherAPI_KEY is not set in Django settings.")
        return JsonResponse({'error': 'API key not configured on server.'}, status=500)


    current_weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'

    try:
        response = requests.get(current_weather_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        weather_data = response.json()
        return JsonResponse(weather_data)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching weather data: {http_err}")
        return JsonResponse({'error': 'Failed to fetch weather data from OpenWeatherMap'}, status=response.status_code)
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception fetching weather data: {req_err}")
        return JsonResponse({'error': 'Error communicating with OpenWeatherMap service'}, status=503)
    except Exception as e:
        logger.exception("Unexpected error fetching weather data:") # Log full exception
        return JsonResponse({'error': 'Internal server error occurred'}, status=500)


def get_air_pollution_data(request):
    """
    Fetches air pollution data from OpenWeatherMap API.
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return JsonResponse({'error': 'Latitude and longitude are required parameters.'}, status=400)

    api_key = settings.OPENWEATHERAPI_KEY # Securely access API key
    if not api_key:
        logger.error("OPENWEATHERAPI_KEY is not set in Django settings.")
        return JsonResponse({'error': 'API key not configured on server.'}, status=500)

    air_pollution_url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}'

    try:
        response = requests.get(air_pollution_url)
        response.raise_for_status() # Raise HTTPError for bad responses
        air_pollution_data = response.json()
        return JsonResponse(air_pollution_data)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching air pollution data: {http_err}")
        return JsonResponse({'error': 'Failed to fetch air pollution data from OpenWeatherMap'}, status=response.status_code)
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception fetching air pollution data: {req_err}")
        return JsonResponse({'error': 'Error communicating with OpenWeatherMap service'}, status=503)
    except Exception as e:
        logger.exception("Unexpected error fetching air pollution data:") # Log full exception
        return JsonResponse({'error': 'Internal server error occurred'}, status=500)

def get_historical_air_quality_data(request):
    """
    Fetches historical air quality data from OpenWeatherMap API for the past 5 years.
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return JsonResponse({'error': 'Latitude and longitude are required parameters.'}, status=400)

    api_key = settings.OPENWEATHERAPI_KEY
    if not api_key:
        logger.error("OPENWEATHERAPI_KEY is not set in Django settings.")
        return JsonResponse({'error': 'API key not configured on server.'}, status=500)

    air_quality_data = []

    def fetch_batch_data(start_date, end_date): # NOTE: Removed 'async' keyword here - it's now a regular function
        url = f'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_date}&end={end_date}&appid={api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get('list', [])
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching historical AQI batch: {http_err}")
            return []
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request exception historical AQI batch: {req_err}")
            return []
        except Exception as e:
            logger.exception("Unexpected error fetching historical AQI batch:")
            return []


    for year_offset in range(5):
        year = date.today().year - year_offset
        start_date = int(datetime(year, 1, 1).timestamp())
        end_date = int(datetime(year, 12, 31).timestamp())

        batch_data = fetch_batch_data(start_date, end_date) # NOTE: Direct function call, no 'await'

        if batch_data:
            yearly_aqi_values = [entry.get('main', {}).get('aqi') for entry in batch_data] # More robustly handle missing 'main' or 'aqi'
            yearly_aqi_values = [aqi for aqi in yearly_aqi_values if aqi is not None] # Filter out None values

            if yearly_aqi_values:
                average_aqi = sum(yearly_aqi_values) / len(yearly_aqi_values)
                air_quality_data.append({
                    'year': year,
                    'averageAQI': f"{average_aqi:.2f}"
                })

    return JsonResponse({'airQualityData': air_quality_data})

def get_cloud_layer_url(request):
    """
    Returns the URL for the OpenWeatherMap cloud tile layer, with API key from backend.
    """
    api_key = settings.OPENWEATHERAPI_KEY
    if not api_key:
        logger.error("OPENWEATHERAPI_KEY is not set in Django settings.")
        return JsonResponse({'error': 'API key not configured on server.'}, status=500)

    tile_url = f'https://tile.openweathermap.org/map/clouds/{{z}}/{{x}}/{{y}}.png?appid={api_key}' # Using f-string for URL
    return JsonResponse({'tileUrl': tile_url})

def get_wind_layer_url(request):
    """
    Returns the URL for the OpenWeatherMap wind tile layer, with API key from backend.
    """
    api_key = settings.OPENWEATHERAPI_KEY
    if not api_key:
        logger.error("OPENWEATHERAPI_KEY is not set in Django settings.")
        return JsonResponse({'error': 'API key not configured on server.'}, status=500)

    tile_url = f'https://tile.openweathermap.org/map/wind/{{z}}/{{x}}/{{y}}.png?appid={api_key}' # Using f-string for URL
    return JsonResponse({'tileUrl': tile_url})

def get_planet_imagery_analysis(request):
    """
    Performs Planet imagery analysis (NDVI, change detection) for a given location using direct API requests.
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return JsonResponse({'error': 'Latitude and longitude are required parameters.'}, status=400)

    planet_api_key = settings.PLANET_API_KEY
    if not planet_api_key:
        logger.error("PLANET_API_KEY is not set in Django settings.")
        return JsonResponse({'error': 'Planet API key not configured on server.'}, status=500)

    try:
        # --- Functions from your script adapted for Django view ---
        def authenticate():
            return {"Authorization": f"api-key {planet_api_key}"}

        def fetch_scene_data(point_geom, start_date, end_date, sun_elevation_range):
            url = "https://api.planet.com/data/v1/quick-search"
            headers = authenticate()
            payload = {
                "item_types": ["PSScene"],
                "filter": {
                    "type": "AndFilter",
                    "config": [
                        {
                            "type": "DateRangeFilter",
                            "field_name": "acquired",
                            "config": {
                                "gte": start_date,
                                "lte": end_date
                            }
                        },
                        {
                            "type": "RangeFilter",
                            "field_name": "cloud_cover",
                            "config": {
                                "lte": 0.05  # Less than or equal to 5% cloud cover - Adjusted to 5%
                            }
                        },
                        {
                            "type": "RangeFilter",
                            "field_name": "sun_elevation",
                            "config": {
                                "gte": sun_elevation_range[0],
                                "lte": sun_elevation_range[1]
                            }
                        },
                        {
                            "type": "GeometryFilter",
                            "field_name": "geometry",
                            "config": point_geom # Use point_geom directly
                        },
                        {
                            "type": "RangeFilter",
                            "field_name": "gsd", # Ground Sample Distance
                            "config": {
                                "lte": 5   # Resolution less than 5 meters/pixel
                            }
                        }
                    ]
                },
            }
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                search_results = response.json()
                # Extract relevant info and return as list of dictionaries
                features_data = []
                if search_results and search_results.get("features"):
                    for feature in search_results["features"]:
                        features_data.append({
                            "item_id": feature["id"],
                            "acquired_date": feature.get("properties", {}).get("acquired", "Unknown Date"),
                            "asset_url": f"https://api.planet.com/data/v1/item-types/PSScene/items/{feature['id']}/assets/" # Construct asset URL directly
                        })
                return features_data # Return list of dictionaries
            else:
                logger.error(f"Error fetching scene data: {response.status_code}")
                logger.error(f"Error details: {response.text}")
                return None

        def activate_asset(item_id, asset_type):
            url = f"https://api.planet.com/data/v1/item-types/PSScene/items/{item_id}/assets/"
            headers = authenticate()
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                logger.error(f"Error fetching assets: {response.status_code}")
                logger.error(f"Details: {response.text}")
                return None

            assets = response.json()
            if asset_type not in assets:
                logger.error(f"Asset type {asset_type} not found.")
                return None

            # Activate the asset
            activation_url = assets[asset_type]["_links"]["activate"]
            requests.post(activation_url, headers=headers)

            # Wait for activation to complete (with logging)
            while True:
                response = requests.get(url, headers=headers)
                assets = response.json()
                if assets[asset_type]["status"] == "active":
                    return assets[asset_type].get("location")
                elif assets[asset_type]["status"] == "inactive":
                    logger.error("Activation failed.")
                    return None
                else:
                    r = random.randint(1, 10000)
                    if r > 9900:
                        logger.info("Waiting for activation...") # Using logger for info

        def download_image(asset_url, destination_folder="planet_images"): # Destination folder added
            print(f"Downloading image from {asset_url}...") # Using print for debugging
            os.makedirs(destination_folder, exist_ok=True) # Ensure folder exists
            match = re.search(r'item_id=([^&]+)', asset_url)
            if match:
                filename = f"{match.group(1)}_basic_analytic_8b.tif"
            else:
                filename = f"planet_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif" # More robust filename
            file_path = os.path.join(destination_folder, filename)
            if not os.path.exists(file_path):
                logger.info(f"Downloading {filename}...") # Using logger for info
                # Use streaming to handle large files
                response = requests.get(asset_url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded {filename}") # Using logger for info
                else:
                    logger.error(f"Failed to download. Status code: {response.status_code}") # Using logger for error
                    return None
            return file_path

        def calculate_ndvi(file_path):
            try:
                with rasterio.open(file_path) as src:
                    if src.count >= 4:
                        red = src.read(3).astype(np.float32)
                        nir = src.read(4).astype(np.float32)

                        ndvi = (nir - red) / (nir + red + 1e-8) # Added small epsilon to prevent divide by zero
                        ndvi_normalized = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-8) # Normalize NDVI
                        return ndvi_normalized
                    else:
                        logger.warning("Insufficient bands for NDVI calculation in {}".format(file_path)) # Using logger for warning
                        return None
            except rasterio.RasterioIOError as e: # Catch rasterio specific errors
                logger.error(f"Rasterio error while processing {file_path}: {e}")
                return None

        def convert_tif_to_jpg(file_path, output_path):
            with rasterio.open(file_path) as src:
                if src.count >= 3:
                    red = src.read(3)
                    green = src.read(2)
                    blue = src.read(1)
                    
                    red_normalized = (red - red.min()) / (red.max() - red.min())
                    green_normalized = (green - green.min()) / (green.max() - green.min())
                    blue_normalized = (blue - blue.min()) / (blue.max() - blue.min())
                    
                    rgb_image = np.dstack((red_normalized, green_normalized, blue_normalized))
                    
                    plt.imsave(output_path, rgb_image)
                    print(f"Converted {file_path} to {output_path}")
                    return output_path # Return the output path
                else:
                    print(f"Insufficient bands for true color image in {file_path}")
                    return None

        # 1. Define Point Geometry for search
        point_geom = {
            "type": "Point",
            "coordinates": [float(lon), float(lat)]
        }

        # 2. Define Date Ranges and Sun Elevation
        sun_elevation_range = [30, 50] # Example sun elevation range
        oldest_start_date = "2018-01-01T00:00:00Z" # Adjusted start date
        pre_2023_end_date = "2022-12-31T23:59:59Z"
        post_2023_start_date = "2024-01-01T00:00:00Z" # Adjusted start date
        most_recent_end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ') # Current time as end date


        # 3. Fetch Scene Data for two time periods
        data_before_2023 = fetch_scene_data(point_geom, oldest_start_date, pre_2023_end_date, sun_elevation_range)
        data_after_2023 = fetch_scene_data(point_geom, post_2023_start_date, most_recent_end_date, sun_elevation_range)


        downloaded_files = []
        rgb_image_paths = {} # Dictionary to store RGB image paths

        # 4. Process Images Before 2023
        if data_before_2023:
            for feature in data_before_2023["features"]:
                item_id = feature["id"]
                acquired_date = feature.get("properties", {}).get("acquired", "Unknown Date")

                logger.info(f"Processing item before 2023: {item_id}, Acquired: {acquired_date}") # Using logger

                asset_url = activate_asset(item_id, "basic_analytic_4b") # Using basic_analytic_4b as in script
                if asset_url:
                    logger.info(f"Asset URL (before 2023): {asset_url}") # Using logger
                    downloaded_file = download_image(asset_url)
                    if downloaded_file:
                        downloaded_files.append(downloaded_file)

                        # Convert to RGB and store path
                        oldest_jpg_path = downloaded_file.replace(".tif", "_rgb.jpg")
                        rgb_image_paths['oldest'] = convert_tif_to_jpg(downloaded_file, oldest_jpg_path) # Store path


        # 5. Process Images After 2023
        if data_after_2023 and data_after_2023.get("features"):
            for feature in data_after_2023["features"]:
                item_id = feature["id"]
                acquired_date = feature.get("properties", {}).get("acquired", "Unknown Date")

                logger.info(f"Processing item after 2023: {item_id}, Acquired: {acquired_date}") # Using logger

                asset_url = activate_asset(item_id, "basic_analytic_4b") # Using basic_analytic_4b as in script
                if asset_url:
                    logger.info(f"Asset URL (after 2023): {asset_url}") # Using logger
                    downloaded_file = download_image(asset_url)
                    if downloaded_file:
                        downloaded_files.append(downloaded_file)
                        # Convert to RGB and store path
                        most_recent_jpg_path = downloaded_file.replace(".tif", "_rgb.jpg")
                        rgb_image_paths['most_recent'] = convert_tif_to_jpg(downloaded_file, most_recent_jpg_path) # Store path


        # 6. NDVI Calculation and Change Detection if we have two downloaded files
        ndvi_change_summary = {} # Initialize even if no NDVI is calculated

        if len(downloaded_files) == 2:
            try:
                oldest_file_path = downloaded_files[0]
                most_recent_file_path = downloaded_files[1]

                oldest_ndvi = calculate_ndvi(oldest_file_path)
                most_recent_ndvi = calculate_ndvi(most_recent_file_path)

                if oldest_ndvi is not None and most_recent_ndvi is not None: # Proceed only if both NDVI are calculated
                    ndvi_change = most_recent_ndvi - oldest_ndvi

                    ndvi_change_summary = {
                        'min_change': float(np.min(ndvi_change)),
                        'max_change': float(np.max(ndvi_change)),
                        'average_change': float(np.mean(ndvi_change)),
                    }
                else:
                    return JsonResponse({'error': 'Could not calculate NDVI for both images.'}, status=500) # More specific error

            except Exception as ndvi_e: # Catch NDVI calculation errors
                logger.error(f"Error during NDVI calculation: {ndvi_e}")
                return JsonResponse({'error': 'Error during NDVI calculation.'}, status=500)


        elif downloaded_files: # If only one or zero files are downloaded, report accordingly
            return JsonResponse({'error': 'Insufficient PlanetScope imagery found for change detection (Need 2 images).'}, status=404)
        else:
            return JsonResponse({'error': 'No PlanetScope imagery found for this location.'}, status=404)


        # 7. Prepare JSON Response - Include RGB image paths
        response_data = {
            'ndvi_change_summary': ndvi_change_summary,
            'status': 'Analysis successful',
            'oldest_rgb_image_url': rgb_image_paths.get('oldest'), # Include RGB image paths in response
            'most_recent_rgb_image_url': rgb_image_paths.get('most_recent'),
        }
        return JsonResponse(response_data)


    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception during Planet API interaction: {req_err}")
        return JsonResponse({'error': 'Error communicating with Planet API'}, status=503)
    except Exception as e:
        logger.exception("Unexpected error during Planet imagery analysis:")
        return JsonResponse({'error': 'Internal server error during Planet analysis'}, status=500)


def home(request): # Make sure your home view is still here
    return render(request, 'mainapp/home.html')

def about(request):
    return render(request, 'mainapp/about.html')

def report_form(request):
    if request.method == 'POST':
        # Handle form submission
        pass
    return render(request, 'mainapp/report.html')

def data_search(request):
    return render(request, 'mainapp/data.html')