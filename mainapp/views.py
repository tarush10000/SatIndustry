from datetime import datetime, date, time
import os
import random
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import requests
import logging
import requests
import numpy as np

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
        print(response)
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

# Pollution Industry Data
import json
import random
from django.http import JsonResponse
from django.urls import reverse

def get_polluting_industries(request):
    industry_type = request.GET.get('industry', None)
    if industry_type:
        industries_data = {
            "Cement": [
                {"name": "Lafarge Holcim", "location": "Zurich, Switzerland"},
                {"name": "Anhui Conch Cement", "location": "Wuhu, China"},
                {"name": "China National Building Material", "location": "Beijing, China"},
                {"name": "HeidelbergCement", "location": "Heidelberg, Germany"},
                {"name": "Cemex", "location": "Monterrey, Mexico"},
                {"name": "Shree Cement", "location": "Beawar, India"},
                {"name": "UltraTech Cement", "location": "Mumbai, India"},
            ],
            "Power plant": [
                {"name": "Taichung Power Plant", "location": "Taichung, Taiwan"},
                {"name": "Drax Power Station", "location": "Selby, UK"},
                {"name": "Belchatow Power Station", "location": "Bełchatów, Poland"},
                {"name": "Ratcliffe-on-Soar Power Station", "location": "Nottinghamshire, UK"},
                {"name": "Kozienice Power Station", "location": "Kozienice, Poland"},
                {"name": "Rihand Super Thermal Power Station", "location": "Sonbhadra, India"},
                {"name": "Mundra Thermal Power Station", "location": "Mundra, India"},
            ],
            "Tannery": [
                {"name": "Hazaribagh Tanneries (formerly)", "location": "Dhaka, Bangladesh"},
                {"name": "Kanpur Leather Cluster", "location": "Kanpur, India"},
                {"name": "Fez Leather Souk", "location": "Fez, Morocco"},
                {"name": "Sava Leather Tannery Zone", "location": "Addis Ababa, Ethiopia"},
                {"name": "Rinos Leather", "location": "Leon, Mexico"},
                {"name": "Basukinath Tanners", "location": "Basukinath, India"},
                {"name": "Ambur Leather Cluster", "location": "Ambur, India"},
            ],
            "Steel": [
                {"name": "ArcelorMittal", "location": "Luxembourg City, Luxembourg"},
                {"name": "China Baowu Steel Group", "location": "Shanghai, China"},
                {"name": "Nippon Steel Corporation", "location": "Tokyo, Japan"},
                {"name": "POSCO", "location": "Pohang, South Korea"},
                {"name": "JSW Steel", "location": "Mumbai, India"},
                {"name": "Tata Steel", "location": "Mumbai, India"},
                {"name": "Steel Authority of India Limited (SAIL)", "location": "New Delhi, India"},
            ],
        }

        if industry_type in industries_data:
            industries = industries_data[industry_type]
            if len(industries) > 4:
                sampled_industries = random.sample(industries, 4)
            else:
                sampled_industries = industries

            results = []
            for industry in sampled_industries:
                data_page_url = reverse('data_search') + f'?location={industry["location"]}'
                results.append({"name": industry["name"], "location": industry["location"], "url": data_page_url})

            return JsonResponse({"industries": results})
        else:
            return JsonResponse({"error": "Invalid industry type"}, status=400)
    else:
        return JsonResponse({"error": "Industry type not provided"}, status=400)
    
import os
import requests
from django.http import JsonResponse
from django.conf import settings
from requests.structures import CaseInsensitiveDict

# Models Code Ishnanvi
import joblib
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configuration for the model
SEQ_LENGTH = 72
N_FEATURES = 11
N_TARGETS = 5
FEATURES = ["temperature", "humidity", "wind_speed", "pressure", "co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]
TARGETS = ["so2", "no2", "co", "pm2_5", "pm10"]

# Function to load the appropriate model based on industry
def load_industry_model(industry):
    if 'cement' in industry.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_cement_model.pkl')
    elif 'power plant' in industry.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_powerplant_model.pkl')
    elif 'tannery' in industry.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_tannery_model.pkl')
    elif 'steel' in industry.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_steel_model.pkl')
    else:
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_cement_model.pkl')
    try:
        print("Loading model from:", model_path)
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading model for {industry}: {e}")
        return None

def fetch_historical_weather(lat, lon, days=10, api_key=None):
    API_KEY = settings.OPENWEATHERAPI_KEY
    yesterday = datetime.now().date() - timedelta(days=1)
    end_date = datetime.combine(yesterday, datetime.min.time())
    start_date = end_date - timedelta(days=days - 1)
    all_data = []
    for i in range(days):
        current_date = end_date - timedelta(days=i)
        timestamp = int(current_date.timestamp())
        response = requests.get(
            "https://history.openweathermap.org/data/2.5/history/city",
            params={
                "lat": lat,
                "lon": lon,
                "type": "hour",
                "start": timestamp,
                "cnt": 24,
                "appid": API_KEY,
                "units": "metric"
            }
        )
        if response.status_code == 200:
            data = response.json()
            for record in data.get("list", []):
                all_data.append({
                    "timestamp": pd.to_datetime(record["dt"], unit='s', utc=True),
                    "temperature": record["main"]["temp"],
                    "humidity": record["main"]["humidity"],
                    "wind_speed": record["wind"]["speed"],
                    "pressure": record["main"]["pressure"]
                })
        else:
            print(f"Error fetching weather data: {response.text}")
        # time.sleep(1) # Consider removing or adjusting sleep for production
    return pd.DataFrame(all_data)

# Function to fetch historical air pollution data (modified to be reusable)
def fetch_historical_air_pollution(lat, lon, days=10, api_key=None):
    API_KEY = settings.OPENWEATHERAPI_KEY
    yesterday = datetime.now().date() - timedelta(days=1)
    end_date = datetime.combine(yesterday, datetime.min.time())
    start_date = end_date - timedelta(days=days - 1)
    all_data = []
    for i in range(days):
        current_date = end_date - timedelta(days=i)
        timestamp = int(current_date.timestamp())
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/air_pollution/history",
            params={
                "lat": lat,
                "lon": lon,
                "start": timestamp,
                "end": timestamp + 86400,
                "appid": API_KEY
            }
        )
        if response.status_code == 200:
            data = response.json()
            for record in data.get("list", []):
                all_data.append({
                    "timestamp": datetime.utcfromtimestamp(record["dt"]),
                    "co": record["components"]["co"],
                    "no2": record["components"]["no2"],
                    "so2": record["components"]["so2"],
                    "o3": record["components"]["o3"],
                    "pm2_5": record["components"]["pm2_5"],
                    "pm10": record["components"]["pm10"],
                    "nh3": record["components"]["nh3"]
                })
        else:
            print(f"Error fetching air pollution data: {response.text}")
        # time.sleep(1)
    return pd.DataFrame(all_data)

def preprocess_data(weather_df, air_pollution_df):
    final_df = pd.merge(weather_df, air_pollution_df, on="timestamp", how="inner")
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    final_df = final_df.set_index('timestamp').sort_index()
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    final_df = final_df.resample('H').first()
    for col in final_df.columns:
        final_df[col] = final_df[col].fillna(method='ffill', limit=2)
        final_df[col] = final_df[col].interpolate(method='time', limit_direction='both')
    def clip_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return series.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    pollution_cols = ['co', 'no2', 'so2', 'o3', 'pm2_5', 'pm10', 'nh3']
    final_df[pollution_cols] = final_df[pollution_cols].apply(clip_outliers)
    final_df['humidity'] = final_df['humidity'].clip(0, 100)
    final_df['pressure'] = final_df['pressure'].clip(800, 1200)
    final_df['wind_speed'] = final_df['wind_speed'].clip(0, 100)

    scaler = MinMaxScaler()
    final_df[FEATURES] = scaler.fit_transform(final_df[FEATURES])
    scaler_no2 = StandardScaler()
    final_df["no2"] = scaler_no2.fit_transform(final_df[["no2"]])
    return final_df

def create_sequences(data: pd.DataFrame, seq_length: int):
    if len(data) < seq_length:
        print(f"Error: Data length ({len(data)}) is not greater than sequence length ({seq_length}).")
        return np.array([])
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length]
        sequences.append(seq.values)
    return np.array(sequences)

def get_coordinates_here(request):
    location_name = request.GET.get('q')
    industry = request.GET.get('industry')
    print(f"Searching for location: {location_name}, Industry: {industry}")
    if location_name and industry:
        api_key = settings.GEOAPIFY_API_KEY
        base_url = 'https://api.geoapify.com/v1/geocode/search'
        params = {
            'text': industry + ',' + location_name,
            'apiKey': api_key
        }
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"

        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get('features'):
                feature = data['features'][0]
                longitude = feature['geometry']['coordinates'][0]
                latitude = feature['geometry']['coordinates'][1]
                display_name = feature['properties']['formatted']

                # Load the appropriate model
                model = load_industry_model(industry)
                print(model)
                print(f"Model loaded for industry: {industry}")
                if model:
                    # Fetch historical data for the location
                    openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
                    weather_df = fetch_historical_weather(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(weather_df)
                    air_pollution_df = fetch_historical_air_pollution(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(air_pollution_df)
                    if not weather_df.empty and not air_pollution_df.empty:
                        print("Data fetched successfully.")
                        final_df = preprocess_data(weather_df, air_pollution_df)
                        X_new = create_sequences(final_df[FEATURES], SEQ_LENGTH)
                        print(f"Data shape: {X_new.shape}")
                        if X_new.size > 0:
                            predictions = model.predict(X_new)
                            # Assuming you want to return the last prediction for the next hour
                            last_prediction = predictions[-1].tolist() if predictions.size > 0 else []
                            print(f"Predictions: {last_prediction}")
                            return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'predictions': last_prediction, 'targets': TARGETS})
                        else:
                            return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': 'Not enough data to make a prediction.'})
                    else:
                        return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': 'Could not fetch historical data.'})

                else:
                    return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': f'Model not found for industry: {industry}'})

            else:
                return JsonResponse({'error': f'Location "{location_name}" not found'}, status=404)
        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': f'Error fetching data from Geoapify API: {e}'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Error processing request: {e}'}, status=500)
    else:
        return JsonResponse({'error': 'Missing location or industry query parameter'}, status=400)