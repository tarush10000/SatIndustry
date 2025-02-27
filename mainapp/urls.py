from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('report/', views.report_form, name='report_form'),
    path('data/', views.data_search, name='data_search'),
    path('weather-data/', views.get_weather_data, name='weather_data'),
    path('air-pollution-data/', views.get_air_pollution_data, name='air_pollution_data'),
    path('historical-air-quality-data/', views.get_historical_air_quality_data, name='historical_air_quality_data'),
    path('cloud-layer-url/', views.get_cloud_layer_url, name='cloud_layer_url'),
    path('wind-layer-url/', views.get_wind_layer_url, name='wind_layer_url'),
    path('planet-analysis/', views.get_planet_imagery_analysis, name='planet_analysis'),
]