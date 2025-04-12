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
from django.views.decorators.http import require_GET


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

    api_key = settings.OPENWEATHERAPI_KEY
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

import random
from django.http import JsonResponse
from django.urls import reverse

def get_polluting_industries(request):
    industry_type = request.GET.get('industry', None)
    if industry_type:
        industries_data = {
            "Cement": [
                {"name": "UltraTech Cement", "location": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
                {"name": "Shree Cement", "location": "Beawar, India", "latitude": 26.1000, "longitude": 74.3200},
                {"name": "Ambuja Cements", "location": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
                {"name": "ACC Limited", "location": "Mumbai, India", "latitude": 19.20398870503228, "longitude": 72.96402733045294},
                {"name": "Dalmia Bharat Limited", "location": "New Delhi, India", "latitude": 28.7041, "longitude": 77.1025},
                {"name": "JK Cement", "location": "Kanpur, India", "latitude": 26.4499, "longitude": 80.3319},
                {"name": "Birla Corporation", "location": "Kolkata, India", "latitude": 22.5726, "longitude": 88.3639},
            ],
            "Power plant": [
                {"name": "Singrauli Super Thermal Power Station", "location": "Singrauli, India", "latitude": 24.2000, "longitude": 82.6800},
                {"name": "Vindhyachal Thermal Power Station", "location": "Singrauli, India", "latitude": 24.1300, "longitude": 82.6700},
                {"name": "Mundra Thermal Power Station", "location": "Mundra, India", "latitude": 22.8400, "longitude": 69.7300},
                {"name": "Sasan Ultra Mega Power Project", "location": "Sasan, India", "latitude": 24.0500, "longitude": 82.7900},
                {"name": "Korba Super Thermal Power Station", "location": "Korba, India", "latitude": 22.3500, "longitude": 82.6800},
                {"name": "Rihand Super Thermal Power Station", "location": "Sonbhadra, India", "latitude": 24.0300, "longitude": 83.0200},
                {"name": "Farakka Super Thermal Power Station", "location": "Farakka, India", "latitude": 24.8300, "longitude": 87.9300},
            ],
            "Tannery": [
                {"name": "Kanpur Leather Cluster", "location": "Kanpur, India", "latitude": 26.4499, "longitude": 80.3319},
                {"name": "Ambur Leather Cluster", "location": "Ambur, India", "latitude": 12.7900, "longitude": 78.7100},
                {"name": "Ranipet Leather Industrial Estate", "location": "Ranipet, India", "latitude": 12.9700, "longitude": 79.3300},
                {"name": "Dindigul Leather Cluster", "location": "Dindigul, India", "latitude": 10.3800, "longitude": 77.9900},
                {"name": "Vaniyambadi Leather Cluster", "location": "Vaniyambadi, India", "latitude": 12.6700, "longitude": 78.6200},
                {"name": "Basukinath Tanners", "location": "Basukinath, India", "latitude": 24.1600, "longitude": 87.2300},
                {"name": "Kolkata Leather Complex", "location": "Kolkata, India", "latitude": 22.5726, "longitude": 88.3639},
            ],
            "Steel": [
                {"name": "JSW Steel", "location": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
                {"name": "Tata Steel", "location": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
                {"name": "Steel Authority of India Limited (SAIL)", "location": "New Delhi, India", "latitude": 28.7041, "longitude": 77.1025},
                {"name": "ArcelorMittal Nippon Steel India", "location": "Mumbai, India", "latitude": 19.0760, "longitude": 72.8777},
                {"name": "Jindal Steel and Power", "location": "New Delhi, India", "latitude": 28.7041, "longitude": 77.1025},
                {"name": "Rashtriya Ispat Nigam Limited (RINL)", "location": "Visakhapatnam, India", "latitude": 17.6868, "longitude": 83.2185},
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
                results.append({
                    "name": industry["name"],
                    "location": industry["location"],
                    "latitude": industry["latitude"],
                    "longitude": industry["longitude"],
                    "url": data_page_url,
                })

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
import google.generativeai as genai

# Configuration for the model
SEQ_LENGTH = 72
N_FEATURES = 11
N_TARGETS = 5
FEATURES = ["temperature", "humidity", "wind_speed", "pressure", "co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]
TARGETS = ["so2", "no2", "co", "pm2_5", "pm10"]

# Function to load the appropriate model based on industry
def load_industry_model(industry, location):
    api = settings.GEMINI_API_KEY
    genai.configure(api_key=api)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
            f"For the given industry: {industry} situated at {location} "
            f"Return in 2 words whether the industry is a Cement Industry, Power Plant, Tannery, Steel Plant or something else",
        )
    print(response.text)
    response = response.text
    if 'cement' in response.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_cement_model.pkl')
    elif 'power' in response.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_powerplant_model.pkl')
    elif 'tannery' in response.lower():
        model_path = os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_tannery_model.pkl')
    elif 'steel' in response.lower():
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
    print("Preprocessing started...")
    print("Weather DataFrame:")
    print(weather_df.head())
    print("Air Pollution DataFrame:")
    print(air_pollution_df.head())
    print("Weather DataFrame Timestamp dtype:", weather_df['timestamp'].dtype)
    print("Air Pollution DataFrame Timestamp dtype:", air_pollution_df['timestamp'].dtype)
    print("Length of weather_df:", len(weather_df))
    print("Length of air_pollution_df:", len(air_pollution_df))

    # Make the 'timestamp' column in air_pollution_df timezone-aware in UTC
    air_pollution_df['timestamp'] = pd.to_datetime(air_pollution_df['timestamp'], utc=True)
    print("Air Pollution DataFrame Timestamp dtype after conversion:", air_pollution_df['timestamp'].dtype)

    final_df = pd.merge(weather_df, air_pollution_df, on="timestamp", how="inner")
    print("5678")
    print(final_df.head())
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

def mitigation_stratergies(industry, location, predicted_air_quality, current_air_quality, current_date):
    api = settings.GEMINI_API_KEY
    genai.configure(api_key=api)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"For the given industry: {industry} at {location}. " \
             f"The current air quality is: {current_air_quality}. " \
             f"The predicted air quality for the next hour is: {predicted_air_quality} on {current_date}. " \
             f"Generate a concise mitigation strategy (max 3 sentences) to reduce the predicted pollution levels in the area."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating mitigation strategies: {e}")
        return "Failed to generate mitigation strategies."

def get_coordinates_here(request):
    location_name = request.GET.get('q')
    industry = request.GET.get('industry')
    print(f"Searching for location: {location_name}, Industry: {industry}")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"For the given industry: {industry} at {location_name} "
        f"Return in a comma separated format: Name of Place, City, State, Country. NO OTHER TEXT OR INFORMATION",
    ).text
    print(response)
    name, city, state, country = response.split(",")
    location_name = f"{name}, {city}, {state}, {country}"
    if location_name and industry:
        api_key = settings.GEOAPIFY_API_KEY
        base_url = 'https://api.geoapify.com/v1/geocode/search'
        params = {
            'apiKey': api_key,
            'name': name,
            'city': city,
            'country': country
        }
        params2 = {
            'apiKey': api_key,
            'city' : city,
            'country': country
        }
        headers = CaseInsensitiveDict()
        headers["Accept"] = "application/json"
        try:
            response = requests.get(base_url, params=params, headers=headers)
            print(response)
            if response.status_code != 200:
                response2 = requests.get(base_url, params=params2, headers=headers)
                print(response2)
                response.raise_for_status()
            else:
                response.raise_for_status()

            data = response.json()
            if data.get('features'):
                feature = data['features'][0]
                longitude = feature['geometry']['coordinates'][0]
                latitude = feature['geometry']['coordinates'][1]
                display_name = feature['properties']['formatted']

                print(f"Coordinates: {latitude}, {longitude}")
                print(f"Display Name: {display_name}")

                # Load the appropriate model
                model = load_industry_model(industry, location_name)
                print(f"Model loaded for industry: {industry}")
                print(model)
                if model:
                    # Fetch historical data for the location
                    openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
                    weather_df = fetch_historical_weather(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(weather_df)
                    air_pollution_df = fetch_historical_air_pollution(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(air_pollution_df)
                    if not weather_df.empty and not air_pollution_df.empty:
                        print("Data fetched successfully.")
                        print(f"Weather Data: {weather_df.head()}")
                        print(f"Air Pollution Data: {air_pollution_df.head()}")
                        final_df = preprocess_data(weather_df, air_pollution_df)
                        print(f"Preprocessed Data: {final_df.head()}")
                        print(f"Final Data Shape: {final_df.shape}")
                        print("Data preprocessed successfully.")
                        X_new = create_sequences(final_df[FEATURES], SEQ_LENGTH)
                        print(f"Sequences created: {X_new.shape}")
                        print(f"Data shape: {X_new.shape}")
                        if X_new.size > 0:
                            predictions = model.predict(X_new)
                            # Assuming you want to return the last prediction for the next hour
                            last_prediction = predictions[-1].tolist() if predictions.size > 0 else []
                            print(f"Predictions: {last_prediction}")

                            # Fetch current air pollution data for mitigation strategies
                            current_air_pollution_response = requests.get(
                                f'http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={settings.OPENWEATHERAPI_KEY}'
                            )
                            current_air_pollution_data = current_air_pollution_response.json().get('list', [])[0].get('components', {}) if current_air_pollution_response.status_code == 200 and current_air_pollution_response.json().get('list') else {}

                            current_air_quality_str = ", ".join([f"{k}: {v}" for k, v in current_air_pollution_data.items()])
                            predicted_air_quality_str = ", ".join([f"{TARGETS[i]}: {last_prediction[i]:.2f}" for i in range(len(TARGETS))])
                            current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            mitigation_strategies = mitigation_stratergies(industry, location_name, predicted_air_quality_str, current_air_quality_str, current_date_str)

                            return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'predictions': last_prediction, 'targets': TARGETS, 'mitigationStrategies': mitigation_strategies})
                        else:
                            return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': 'Not enough data to make a prediction.'})
                    else:
                        return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': 'Could not fetch historical data.'})
                else:
                    return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': display_name, 'error': f'Model not loading'})
            else:
                return JsonResponse({'error': f'Location "{location_name}" not found'}, status=404)
        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': f'Error fetching data from Geoapify API: {e}'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Error processing request: {e}'}, status=500)
    else:
        return JsonResponse({'error': 'Missing location or industry query parameter'}, status=400)
    
    
@require_GET
def get_predictions_and_mitigation(request):
    industry = request.GET.get('industry')
    latitude_str = request.GET.get('latitude')
    longitude_str = request.GET.get('longitude')

    if not industry or not latitude_str or not longitude_str:
        return JsonResponse({'error': 'Missing industry, latitude, or longitude'}, status=400)

    try:
        latitude = float(latitude_str)
        longitude = float(longitude_str)
    except ValueError:
        return JsonResponse({'error': 'Invalid latitude or longitude'}, status=400)

    location_name = industry # Use the fetched display name for model loading

    # Load the appropriate model
    model = load_industry_model(industry, location_name)
    print(f"Model loaded for industry: {industry}")
    print(model)
    if model:
        # Fetch historical data for the location
        openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        weather_df = fetch_historical_weather(latitude, longitude, days=10, api_key=openweather_api_key)
        print(weather_df)
        air_pollution_df = fetch_historical_air_pollution(latitude, longitude, days=10, api_key=openweather_api_key)
        print(air_pollution_df)
        if not weather_df.empty and not air_pollution_df.empty:
            print("Data fetched successfully.")
            print(f"Weather Data: {weather_df.head()}")
            print(f"Air Pollution Data: {air_pollution_df.head()}")
            final_df = preprocess_data(weather_df, air_pollution_df)
            print(f"Preprocessed Data: {final_df.head()}")
            print(f"Final Data Shape: {final_df.shape}")
            print("Data preprocessed successfully.")
            X_new = create_sequences(final_df[FEATURES], SEQ_LENGTH)
            print(f"Sequences created: {X_new.shape}")
            print(f"Data shape: {X_new.shape}")
            if X_new.size > 0:
                predictions = model.predict(X_new)
                # Assuming you want to return the last prediction for the next hour
                last_prediction = predictions[-1].tolist() if predictions.size > 0 else []
                print(f"Predictions: {last_prediction}")

                # Fetch current air pollution data for mitigation strategies
                current_air_pollution_response = requests.get(
                    f'http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={settings.OPENWEATHERAPI_KEY}'
                )
                current_air_pollution_data = current_air_pollution_response.json().get('list', [])[0].get('components', {}) if current_air_pollution_response.status_code == 200 and current_air_pollution_response.json().get('list') else {}

                current_air_quality_str = ", ".join([f"{k}: {v}" for k, v in current_air_pollution_data.items()])
                predicted_air_quality_str = ", ".join([f"{TARGETS[i]}: {last_prediction[i]:.2f}" for i in range(len(TARGETS))])
                current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                mitigation_strategies = mitigation_stratergies(industry, location_name, predicted_air_quality_str, current_air_quality_str, current_date_str)

                return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': industry, 'predictions': last_prediction, 'targets': TARGETS, 'mitigationStrategies': mitigation_strategies})
            else:
                return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': industry, 'error': 'Not enough data to make a prediction.'})
        else:
            return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': industry, 'error': 'Could not fetch historical data.'})
    else:
        return JsonResponse({'latitude': latitude, 'longitude': longitude, 'displayName': industry, 'error': f'Model not loading'})