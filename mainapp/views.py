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
from sklearn.metrics import classification_report


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
                {"name": "UltraTech Cement", "location": "Mumbai, India", "latitude": 19.250548044827784, "longitude": 72.843470708654},
                {"name": "Shree Cement", "location": "Beawar, India", "latitude": 26.08134560261072, "longitude": 74.38362414837279},
                {"name": "Ambuja Cements", "location": "Mumbai, India", "latitude": 19.002223730181832, "longitude": 73.03229075158546},
                {"name": "ACC Limited", "location": "Mumbai, India", "latitude": 19.20398870503228, "longitude": 72.96402733045294},
                {"name": "Dalmia Bharat Limited", "location": "New Delhi, India", "latitude": 28.631640375927617, "longitude": 77.22651347711746},
                {"name": "Birla Corporation", "location": "Kolkata, India", "latitude": 22.54410699789887, "longitude": 88.36505952212816},
                {"name": "Ramco Cements", "location": "Chennai, India", "latitude": 13.052844109796666, "longitude": 80.26891335323748},
                {"name": "India Cements", "location": "Chennai, India", "latitude": 13.027089962137735, "longitude": 80.26822670771199},
                {"name": "Orient Cement", "location": "Hyderabad, India", "latitude": 17.543981808592296, "longitude": 78.50716275672416},
            ],
            "Power plant": [
                {"name": "Singrauli Super Thermal Power Station", "location": "Singrauli, India", "latitude": 24.120970609267747, "longitude": 82.693090450962},
                {"name": "Vindhyachal Thermal Power Station", "location": "Singrauli, India", "latitude": 24.099208199421874, "longitude": 82.67562493562268},
                {"name": "Adani Power Limited Thermal Power Station", "location": "Mundra, India", "latitude": 22.823099198964474, "longitude": 69.55286753561526},
                {"name": "Reliance Sasan Ultra Mega Power Project", "location": "Sasan, India", "latitude": 23.97961345204601, "longitude": 82.62691587324335},
                {"name": "NTPC Korba Super Thermal Power Station", "location": "Korba, India", "latitude": 22.386074765444544, "longitude": 82.68343993560642},
                {"name": "NTPC Rihand Super Thermal Power Station", "location": "Sonbhadra, India", "latitude": 24.03475499179369, "longitude": 82.81009557609129},
                {"name": "NTPC Farakka Super Thermal Power Station", "location": "Farakka, India", "latitude": 24.773510320104375, "longitude": 87.89702967793565},
                {"name": "NTPC Kahalgaon Super Thermal Power Station", "location": "Kahalgaon, India", "latitude": 25.237547780776634, "longitude": 87.26742238164822},
                {"name": "Mejia Thermal Power Station", "location": "Mejia, India", "latitude": 23.463950366922987, "longitude": 87.13179192027268},
                {"name": "Bokaro Thermal Power Station(BTPS)", "location": "Bokaro, India", "latitude": 23.78491635197923, "longitude": 85.87825394912915},
            ],
            "Tannery": [
                {"name": "Kanpur Leather Cluster", "location": "Kanpur, India", "latitude": 26.36976374936636, "longitude": 80.26920083376281},
                {"name": "Ambur Leather Cluster", "location": "Ambur, India", "latitude": 12.78555512112967, "longitude": 78.72022853513525},
                {"name": "Ranipet Leather Industrial Estate", "location": "Ranipet, India", "latitude": 12.97106389983351, "longitude":  79.29919181579598},
                {"name": "Khaja Moideen Leather Tannery", "location": "Pallapatti, India", "latitude": 10.356997769578435, "longitude": 77.94141596531541},
                {"name": "Vaniyambadi Leather Cluster", "location": "Vaniyambadi, India", "latitude": 12.687264434487899, "longitude": 78.62721565806284},
                {"name": "Kolkata Leather Complex", "location": "Kolkata, India", "latitude": 22.497891155851956, "longitude": 88.50977035342015},
                {"name": "Sathik Tanning Company", "location": "Chennai, India", "latitude": 13.08812194991736, "longitude": 80.26703312413918},
                {"name": "Tarun Tanning Industries", "location": "Agra, India", "latitude": 27.199730382769566, "longitude": 78.00412893972815},
                {"name": "Jalandhar Leather Cluster", "location": "Jalandhar, India", "latitude": 31.33931478701642, "longitude": 75.53465738274737},
            ],
            "Steel": [
                {"name": "JSW Steel (Vijayanagar)", "location": "Toranagallu, Karnataka", "latitude": 15.183917077333625, "longitude": 76.65964272213336},
                {"name": "Tata Steel (Jamshedpur)", "location": "Jamshedpur, Jharkhand", "latitude": 22.791260006474772, "longitude": 86.19700630146932},
                {"name": "Steel Authority of India Limited (Bhilai Steel Plant)", "location": "Bhilai, Chhattisgarh", "latitude": 21.195491173487618, "longitude": 81.38497629329093},
                {"name": "ArcelorMittal Nippon Steel India (Hazira)", "location": "Hazira, Gujarat", "latitude": 21.12898485336033, "longitude": 72.65563334310765},
                {"name": "Jindal Steel and Power (Raigarh)", "location": "Raigarh, Chhattisgarh", "latitude": 21.92472666919913, "longitude": 83.34714539825828},
                {"name": "Rashtriya Ispat Nigam Limited (Visakhapatnam Steel Plant)", "location": "Visakhapatnam, Andhra Pradesh", "latitude": 17.608092080562546, "longitude":  83.20527959331055},
                {"name": "JSW Steel (Dolvi)", "location": "Dolvi, Maharashtra", "latitude": 18.697496965527414, "longitude": 73.03436209513367},
                {"name": "Tata Steel (Kalinganagar)", "location": "Kalinganagar, Odisha", "latitude": 20.97631553868611, "longitude": 86.00537702027131},
                {"name": "Steel Authority of India Limited (Rourkela Steel Plant)", "location": "Rourkela, Odisha", "latitude": 22.220419641479797, "longitude": 84.8602984933045},
                {"name": "ArcelorMittal Nippon Steel India (Parabur)", "location": "Parabur, Odisha", "latitude": 20.47683633448304, "longitude": 86.61441202993318},
                {"name": "Jindal Steel and Power (Angul)", "location": "Angul, Odisha", "latitude": 20.90531336448541, "longitude": 85.01361316259757},
                {"name": "Vedanta Limited (Electrosteel Steels Limited)", "location": "Bokaro Steel City, Jharkhand", "latitude": 23.724801520873896, "longitude": 86.29281301094312},
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import google.generativeai as genai

# Configuration for the model
SEQ_LENGTH = 72
N_FEATURES = 11
N_TARGETS = 5
FEATURES = ["temperature", "humidity", "wind_speed", "pressure", "co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]
TARGETS = ["so2", "no2", "co", "pm2_5", "pm10"]

def fetch_historical_weather(lat, lon, days=20, api_key=None):
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
def fetch_historical_air_pollution(lat, lon, days=20, api_key=None):
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
            if response.status_code != 200:
                response2 = requests.get(base_url, params=params2, headers=headers)
                response.raise_for_status()
            else:
                response.raise_for_status()

            data = response.json()
            if data.get('features'):
                feature = data['features'][0]
                longitude = feature['geometry']['coordinates'][0]
                latitude = feature['geometry']['coordinates'][1]
                display_name = feature['properties']['formatted']

                # Load the appropriate model
                model = load_clustering_model(industry, location_name)
                if model:
                    # Fetch historical data for the location
                    openweather_api_key = os.getenv('OPENWEATHERAPI_KEY')
                    weather_df = fetch_historical_weather(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(weather_df)
                    air_pollution_df = fetch_historical_air_pollution(latitude, longitude, days=10, api_key=openweather_api_key)
                    print(air_pollution_df)
                    if not weather_df.empty and not air_pollution_df.empty:
                        final_df = preprocess_data(weather_df, air_pollution_df)
                        X_new = create_sequences(final_df[FEATURES], SEQ_LENGTH)
                        if X_new.size > 0:
                            predictions = model.predict(X_new)
                            # Assuming you want to return the last prediction for the next hour
                            last_prediction = predictions[-1].tolist() if predictions.size > 0 else []
                        
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

def identify_industry(industry, location):
    api = settings.GEMINI_API_KEY
    genai.configure(api_key=api)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
            f"For the given industry: {industry} situated at {location} "
            f"Return in 2 words whether the industry is a Cement Industry, Power Plant, Tannery, Steel Plant or something else",
        )
    print(response.text)
    response = response.text
    return response

def fetch_data_for_3_months(LAT, LON, API_KEY):
    print("prcoesseing")
    import requests
    import pandas as pd
    import time
    from datetime import datetime, timedelta

    WEATHER_API = "https://history.openweathermap.org/data/2.5/history/city"
    AIR_POLLUTION_API = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    # Function to fetch historical weather data
    def fetch_historical_weather(lat, lon, days=182):
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        all_data = []

        for i in range(days):
            timestamp = int((end_date - timedelta(days=i)).timestamp())
            print(i,"temp")
            response = requests.get(
                WEATHER_API,
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
                        "timestamp": datetime.utcfromtimestamp(record["dt"]),
                        "temperature": record["main"]["temp"],
                        "humidity": record["main"]["humidity"],
                        "wind_speed": record["wind"]["speed"],
                        "pressure": record["main"]["pressure"]
                    })
            else:
                print(f"Error fetching weather data: {response.text}")
            
             # Avoid rate limit issues

        return pd.DataFrame(all_data)

    # Function to fetch historical air pollution data
    def fetch_historical_air_pollution(lat, lon, days=182):
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        all_data = []

        for i in range(days):
            timestamp = int((end_date - timedelta(days=i)).timestamp())
            print(i,"pollutatnat")
            response = requests.get(
                AIR_POLLUTION_API,
                params={
                    "lat": lat,
                    "lon": lon,
                    "start": timestamp,
                    "end": timestamp + 86400,  # 1-day range
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


        return pd.DataFrame(all_data)

    # Fetch weather and air pollution data
    weather_df = fetch_historical_weather(LAT, LON, days=20)
    air_pollution_df = fetch_historical_air_pollution(LAT, LON, days=20)

    # Merge datasets on timestamp
    final_df = pd.merge(weather_df, air_pollution_df, on="timestamp", how="inner")

    # Save as CSV
    # final_df.to_csv("steel_plant2.csv", index=False)
    
    return final_df

@require_GET
def get_predictions_and_mitigation(request):
    industry = request.GET.get('industry')
    latitude_str = request.GET.get('latitude')
    longitude_str = request.GET.get('longitude')
    location = request.GET.get('location')

    if not industry or not latitude_str or not longitude_str:
        return JsonResponse({'error': 'Missing industry, latitude, or longitude'}, status=400)

    try:
        latitude = float(latitude_str)
        longitude = float(longitude_str)
    except ValueError:
        return JsonResponse({'error': 'Invalid latitude or longitude'}, status=400)


    industry_type = identify_industry(industry, location)


    openweather_api_key = settings.OPENWEATHERAPI_KEY

    final_df1 = fetch_data_for_3_months(latitude, longitude, openweather_api_key)
    # Anomaly Detection
    anomaly_df = perform_anomaly_detection(final_df1.copy()) # Pass a copy to avoid modifying original df

    # Clustering
    kmeans, scaler, pca, industry_summary = load_clustering_model(industry_type)
    clustering_results = perform_clustering(kmeans, scaler, pca, industry_summary, final_df1.copy())
    pca_coords, cluster_stats, industry_stats, industry_daily_mean, industry_24hr_mean, exceeding, avg_cluster_pollution, new_avg = clustering_results

    # LSTM
    model = load_lstm_model(industry_type)
    lstm_results = perform_lstm(model, final_df1.copy())
    future_preds, future_timestamps, y_test, y_pred, test_timestamps, targets_lstm = lstm_results

    print(f"Model loaded for industry: {industry}")
    # Fetch current air pollution data for mitigation strategies
    current_air_pollution_response = requests.get(
        f'http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={settings.OPENWEATHERAPI_KEY}'
    )
    current_air_pollution_data = current_air_pollution_response.json().get('list', [])[0].get('components', {}) if current_air_pollution_response.status_code == 200 and current_air_pollution_response.json().get('list') else {}
    print(current_air_pollution_data)
    
    current_air_quality_str = ", ".join([f"{k}: {v}" for k, v in current_air_pollution_data.items()])
    predicted_air_quality_str = ""
    if future_preds is not None and len(future_preds) > 0:
        first_prediction = future_preds[0]
        predicted_air_quality_str = ", ".join([f"{targets_lstm[i]}: {first_prediction[i]:.2f}" for i in range(len(targets_lstm))])
    else:
        predicted_air_quality_str = "No future predictions available."
        
    print(predicted_air_quality_str)
    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mitigation_strategies = mitigation_stratergies(industry, location, str(predicted_air_quality_str), str(current_air_quality_str), current_date_str)
    print(mitigation_strategies)

    response_data = {
        'latitude': latitude,
        'longitude': longitude,
        'displayName': industry,
        'targets': TARGETS,
        'mitigationStrategies': mitigation_strategies,
        'pca_coords': pca_coords.tolist(),
        'cluster_stats': cluster_stats.to_dict(),
        'industry_stats': industry_stats.to_dict(),
        'industry_daily_mean': industry_daily_mean.to_list(), # Assuming this is a Series
        'industry_24hr_mean': industry_24hr_mean.to_dict(),
        'exceeding_limits': exceeding,
        'average_cluster_pollution': avg_cluster_pollution.to_dict(),
        'average_industry_pollution': new_avg,
        'anomaly_data': anomaly_df[['timestamp', 'co', 'no2', 'so2', 'o3', 'pm2_5', 'pm10', 'nh3', 'anomaly']].to_dict(orient='records'),
        'lstm_predictions': future_preds.tolist(),
        'lstm_timestamps': [pd.to_datetime(ts).isoformat() for ts in future_timestamps],
        'lstm_actual': y_test.tolist(),
        'lstm_predicted': y_pred.tolist(),
        'lstm_test_timestamps': [pd.to_datetime(ts).isoformat() for ts in test_timestamps],
        'lstm_targets': targets_lstm
    }
    return JsonResponse(response_data)

# Function to load the appropriate model based on industry
def load_clustering_model(response):
    print("Loading clustering model") # Added print statement for clarity
    try:
        if 'cement' in response.lower():
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'kmeans_cement_model.pkl'))
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'scaler_cement.pkl'))
            pca = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'pca_cement.pkl'))
            industry_summary = pd.read_csv(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'industry_clusters_cement.csv'))
        elif 'power' in response.lower():
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'kmeans_power_model.pkl'))
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'scaler_power.pkl'))
            pca = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'pca_power.pkl'))
            industry_summary = pd.read_csv(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'industry_clusters_power.csv'))
        elif 'tannery' in response.lower():
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'kmeans_tannery_model.pkl'))
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'scaler_tannery.pkl'))
            pca = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'pca_tannery.pkl'))
            industry_summary = pd.read_csv(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'industry_clusters_tannery.csv'))
        elif 'steel' in response.lower():
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'kmeans_steel_model.pkl'))
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'scaler_steel.pkl'))
            pca = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'pca_steel.pkl'))
            industry_summary = pd.read_csv(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'industry_clusters_steel.csv'))
        else:
            kmeans = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'kmeans_cement_model.pkl'))
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'scaler_cement.pkl'))
            pca = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'pca_cement.pkl'))
            industry_summary = pd.read_csv(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'industry_clusters_cement.csv'))
        return kmeans, scaler, pca, industry_summary
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading clustering model: {e}") # More specific error message
        return None

def load_lstm_model(response):
    print("Loading LSTM model") # Added print statement for clarity
    try:
        if 'cement' in response.lower():
            model = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_cement_model_new.pkl'))
        elif 'power' in response.lower():
            model = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_power_model_new.pkl'))
        elif 'tannery' in response.lower():
            model = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_tannery_model_new.pkl'))
        elif 'steel' in response.lower():
            model = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_steel_model_new.pkl'))
        else:
            model = joblib.load(os.path.join(settings.BASE_DIR, 'SatIndustry', 'mainapp', 'models', 'trained_cement_model_new.pkl'))
        print("returning LSTM model")
        return model
    except FileNotFoundError:
        print(" notLoading LSTM model")
        return None
    except Exception as e:
        print("not  Loading LSTM model")
        print(f"Error loading LSTM model: {e}") # More specific error message
        return None

def perform_clustering(kmeans, scaler, pca, industry_summary, new_data):
    # Features to use
    features = ["temperature", "humidity", "wind_speed", "pressure",
                "co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]
    pollution_features = ["co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]

    # Scale and transform new data
    scaled_new = scaler.transform(new_data[features])
    predicted_cluster = kmeans.predict(scaled_new)[0]
    pca_coords = pca.transform(scaled_new)[0]
    # --- Pollution Stats ---
    print(f"\n🔷 New Industry is classified into Cluster {predicted_cluster}")

    # Cluster Stats (3-months)
    cluster_data = industry_summary[industry_summary["cluster"] == predicted_cluster]
    cluster_stats = cluster_data[pollution_features].agg(["mean", "min", "max"])

    # New Industry Stats (3-months)
    industry_stats = new_data[pollution_features].agg(["mean", "min", "max"])

    print("\n📊 Cluster Pollution Statistics (3-month aggregate):")
    print(cluster_stats.T.round(2))

    print("\n🏭 New Industry Pollution Statistics (3-month data):")
    print(industry_stats.T.round(2))

    # --- CPCB Regulatory Standards (24-hour average in µg/m³) ---
    cpcb_limits = {
        "co": 2000,     # converted mg/m³ → µg/m³
        "no2": 80,
        "so2": 80,
        "o3": 100,
        "pm2_5": 60,
        "pm10": 100,
        "nh3": 400
    }

    print("\n📏 CPCB Permissible Limits (24-hr average):")
    for pol, limit in cpcb_limits.items():
        print(f"{pol.upper():<6}: {limit} µg/m³")

    # --- Check for Pollutants Exceeding Limits ---
    industry_daily_mean = new_data.groupby(new_data['timestamp'].dt.date)[pollution_features].mean().mean() # Calculate daily mean and then overall mean
    # --- Take only last 24 hours of data ---
    last_24hr_data = new_data.tail(24)
    industry_24hr_mean = last_24hr_data[pollution_features].mean()

    # --- Check for Pollutants Exceeding Limits ---
    print("\n📏 CPCB Permissible Limits (24-hr average):")
    for pol, limit in cpcb_limits.items():
        print(f"{pol.upper():<6}: {limit} µg/m³")

    exceeding = []
    for pol, limit in cpcb_limits.items():
        val = industry_24hr_mean[pol]
        print(pol,val)
        if val > limit:
            exceeding.append({'pollutant': pol.upper(), 'value': round(val, 2), 'limit': limit}) # Store as dictionary

    if exceeding:
        print("\n⚠️ Pollutants exceeding CPCB limits (based on last 24 hours):")
        for item in exceeding:
            print(f"• {item['pollutant']}: {item['value']} µg/m³ > {item['limit']} µg/m³")
    else:
        print("\n✅ All pollutants are within CPCB permissible limits (last 24 hours).")

    # --- Optional: Overall Pollution Risk ---
    avg_cluster_pollution = industry_summary.groupby("cluster")[pollution_features].mean().mean(axis=1)
    new_avg = industry_daily_mean.mean()
    print(f"\n📌 Average Pollution of New Industry: {new_avg:.2f}")

    if new_avg > avg_cluster_pollution.max():
        print("⚠️ This industry is more harmful than any existing cluster.")
    elif new_avg > avg_cluster_pollution.mean():
        print("⚠️ This industry is relatively harmful.")
    else:
        print("✅ This industry is within normal pollution levels.")
    return pca_coords, cluster_stats, industry_stats, industry_daily_mean, industry_24hr_mean, exceeding, avg_cluster_pollution, new_avg

def perform_anomaly_detection(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    features = df.drop(columns=["timestamp"])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(features_scaled)
    df["anomaly"] = df["anomaly"].map({1: 1, -1: 0}) # Changed mapping to 1 for anomaly, 0 for normal

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, df["anomaly"], test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("Anomaly Detection Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Anomaly detection results calculated.")
    return df

def perform_lstm(model, df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    import matplotlib.pyplot as plt
    from datetime import timedelta
    from sklearn.metrics import classification_report
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Define features and targets
    features = ["temperature", "humidity", "wind_speed", "pressure", "co", "no2", "so2", "o3", "pm2_5", "pm10", "nh3"]
    targets = ["so2", "no2", "co", "pm2_5", "pm10"]

    no2_series = df["no2"].copy()

    # Detect NO2 anomalies using z-score method
    def detect_anomalies_zscore(series, threshold=3):
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std
        return np.where(np.abs(z_scores) > threshold)[0]

    # Detect
    no2_anomalies = detect_anomalies_zscore(no2_series, threshold=3)
    print(f"NO2 anomalies detected: {len(no2_anomalies)} at indices {no2_anomalies[:10]}")

    # Create a new column for corrected values (initialize with original values)
    df["no2_corrected"] = df["no2"]

    # Replace anomaly values with NaN for interpolation
    df.loc[no2_anomalies, "no2_corrected"] = np.nan

    # Interpolate using time-based or index-based method (choose one)
    df["no2_corrected"] = df["no2_corrected"].interpolate(method="linear", limit_direction="both")
    df["no2"]=df["no2_corrected"]


    so2_anomalies = detect_anomalies_zscore(df["so2"], threshold=3)
    print(f"SO₂ anomalies detected: {len(so2_anomalies)} at indices {so2_anomalies[:10]}")

    # Replace with interpolated values
    df["so2_corrected"] = df["so2"]
    df.loc[so2_anomalies, "so2_corrected"] = np.nan
    df["so2_corrected"] = df["so2_corrected"].interpolate(method="linear", limit_direction="both")
    df["so2"] = df["so2_corrected"]

    # Normalize features
    print("is this this")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    scaled_df["timestamp"] = df["timestamp"].values
    scaler_no2 = StandardScaler() # Separate scaler for NO2
    df["no2"] = scaler_no2.fit_transform(df[["no2"]])
    # Split into training and testing based on timestamp
    latest_time = scaled_df["timestamp"].max()
    cutoff_time = latest_time - timedelta(days=18)
    train_df = scaled_df[scaled_df["timestamp"] < cutoff_time].drop(columns=["timestamp"])
    test_df = scaled_df[scaled_df["timestamp"] >= cutoff_time].drop(columns=["timestamp"])
    n_input = 24

    # Prepare data for training
    print("is it timesieries")

    train_generator = TimeseriesGenerator(train_df.values, train_df[targets].values, length=n_input, batch_size=1)

    print("is it epoch")
    model.fit(train_generator, epochs=4, verbose=0)
    print("done")# Reduced verbosity
    # Prepare test data (manual windowing)
    X_test, y_test = [], []
    test_values = test_df.values
    for i in range(len(test_values) - n_input):
        X_test.append(test_values[i:i+n_input])
        y_test.append(test_values[i+n_input][[features.index(t) for t in targets]])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Predict
    y_pred = model.predict(X_test)
    test_timestamps = scaled_df[scaled_df["timestamp"] >= cutoff_time]["timestamp"].values[n_input:]
    # Predict next 3 days (72 hours ahead)
    future_steps = 72
    last_input_seq = test_df.values[-n_input:].copy()  # Last known window

    future_preds = []

    for _ in range(future_steps):
        pred = model.predict(last_input_seq.reshape(1, n_input, -1), verbose=0)[0]
        future_preds.append(pred)

        # Create the next input by appending prediction and removing oldest
        next_input = np.zeros_like(last_input_seq[0])
        for idx, target in enumerate(targets):
            target_index = features.index(target)
            next_input[target_index] = pred[idx]

        # For non-target features, keep previous last values
        for idx in range(len(features)):
            if features[idx] not in targets:
                next_input[idx] = last_input_seq[-1][idx]

        last_input_seq = np.vstack([last_input_seq[1:], next_input])

    future_preds = np.array(future_preds)

    # Convert last_timestamp to pandas Timestamp for compatibility with timedelta
    if len(test_timestamps) > 0:
        last_timestamp = pd.Timestamp(test_timestamps[-1])

        # Generate future timestamps (assuming hourly predictions)
        future_timestamps = [last_timestamp + timedelta(hours=i + 1) for i in range(future_steps)]
    else:
        future_timestamps = []
        print("Warning: No test timestamps available for generating future timestamps.")


    # Classification Report (Quantile binning: 3 classes - Low, Medium, High)
    print("\n=== LSTM Classification Reports (Quantile Binning) ===\n")
    for i, target in enumerate(targets):
        actual_bins = pd.qcut(y_test[:, i], q=3, labels=["Low", "Medium", "High"], duplicates='drop')
        pred_bins = pd.qcut(y_pred[:, i], q=3, labels=["Low", "Medium", "High"], duplicates='drop')

        # Ensure both series have the same length after quantile binning
        min_len = min(len(actual_bins), len(pred_bins))
        actual_bins = actual_bins[:min_len]
        pred_bins = pred_bins[:min_len]

        if len(actual_bins) > 0 and len(pred_bins) > 0 and len(np.unique(actual_bins)) > 1 and len(np.unique(pred_bins)) > 1:
            print(f"\nTarget: {target}")
            print(classification_report(actual_bins, pred_bins, digits=3, zero_division=0))
        else:
            print(f"\nTarget: {target} - Insufficient unique values for classification report.")

    print("LSTM predictions performed.")
    return future_preds, future_timestamps, y_test, y_pred, test_timestamps, targets