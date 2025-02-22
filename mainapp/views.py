from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'mainapp/index.html')

from django.http import JsonResponse
from django.conf import settings
import requests
import logging

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

def home(request): # Make sure your home view is still here
    return render(request, 'mainapp/index.html')