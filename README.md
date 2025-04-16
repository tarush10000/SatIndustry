# SatIndustry

## Overview
![image](https://github.com/user-attachments/assets/ba43917c-8b1f-4310-9074-ba563dab68c8)

**SatIndustry** is a web application developed as part of the Technical Answers to Real-World Problems (TARP) subject at VIT Vellore. It utilizes satellite data and machine learning to monitor, predict, and analyze environmental pollution — especially industrial air pollution — in real-time. The platform is designed to provide transparency, promote sustainable practices, and enable data-driven policy interventions.

## Project Goals

The primary goals of SatIndustry are:

- **Industrial Pollution Monitoring:** Enable scalable, real-time tracking of industrial emissions across India using remote sensing and ML.
- **Predictive Analytics:** Use time-series forecasting models like LSTM to predict pollution levels, aiding early interventions.
- **Actionable Insights for Stakeholders:** Offer interpretable dashboards for governments, industries, and the public to track pollution patterns and risk zones.
- **Environmental Impact Assessment:** Use NDVI (Normalized Difference Vegetation Index) to assess pollution's effect on surrounding ecosystems.

## Key Features
![image](https://github.com/user-attachments/assets/10617e09-1fb4-4f28-b92a-2da25759b568)

![image](https://github.com/user-attachments/assets/478cc190-ac41-49bd-b7fb-e7dce5baa0e1)

![image](https://github.com/user-attachments/assets/479c6a8a-5aa4-4e07-a4e7-d930931e6b4a)

SatIndustry incorporates the following core features:

- **Multi-Pollutant Monitoring:** Tracks 8 critical pollutants — CO, NO, NO₂, SO₂, O₃, PM2.5, PM10, and NH₃ — using satellite and sensor fusion.
- **Time-Series Forecasting:** LSTM models predict pollutant levels up to 4 days in advance with an 85% accuracy rate.
- **Anomaly Detection:** Random Forest and Isolation Forest models flag unusual pollution spikes.
- **Hotspot Clustering:** K-Means identifies high-risk pollution zones; clustering insights guide mitigation efforts.
- **Real-Time Dashboard:** Interactive heatmaps, trend graphs, and AQI tables updated every minute.
- **User-Centric Mapping:** Leaflet-based maps enable location-aware queries and overlay toggles for AQI, wind, temperature, and vegetation layers.

## System Architecture

The platform is built on a layered architecture:

1. **Data Acquisition Layer:** Satellite imagery and APIs for weather and pollution data (NO₂, PM2.5, SO₂, O₃, etc.).
2. **Processing & Analytics Layer:**
   - **LSTM** for pollutant forecasting
   - **Random Forest** for anomaly detection
   - **K-Means Clustering** for identifying pollution clusters
   - **NDVI** for environmental degradation analysis
3. **Visualization Layer:** Real-time dashboards with interactive maps and visual summaries for stakeholder decisions.

## Machine Learning Models

- **LSTM**: Achieved 87.7% classification accuracy across multiple pollutants (PM10, NO₂, SO₂, etc.)
- **Random Forest & Isolation Forest**: Used for anomaly detection across pollutants
- **K-Means**: Clustered 60 industries of 4 types (Cement, Tannery, Steel and Power Plant) into distinct zones based on pollution severity

## Technology Stack

- **Frontend:** HTML, CSS, JavaScript (Leaflet.js, Chart.js, ApexCharts)
- **Backend:** Python (Django)
- **ML/AI:** TensorFlow/Keras for LSTM, Scikit-learn for clustering and classification
- **Visualization:** Real-time dashboards with geospatial overlays

## Open Source APIs Used

This project makes extensive use of the following open-source APIs and datasets:

- **OpenWeatherMap API** – Real-time weather and temperature data.
- **WAQI (World Air Quality Index) API** – Real-time AQI and pollutant concentration data.
- **Sentinel (ESA Copernicus)** – Satellite imagery for pollutant and NO₂ tracking.
- **Geoapify API** – Geocoding, map tiles, and place search functionality.
- **Google Gemini API** – AI-based response generation for summarizing and interpreting results.

## Comparative Advantage

- **Broader Pollutant Range:** Covers 8+ pollutants and vegetation metrics
- **Real-Time + Historical Insights:** Merges current data with 20-year trends
- **Low Latency:** < 1 minute for most updates
- **Public Transparency:** Data accessible via a user-friendly, responsive interface
- **Scalability:** Adaptable for national-scale deployments and various industries

## Team Members

*   **Tarush Agarwal:** Full Stack Developer and AI Engineer - Architect of machine learning pipelines and satellite data analysis systems.
*   **Ishanvi Mishra:** AI & ML Engineer - Specialist in emission pattern recognition and regulatory compliance analysis.
*   **Ansh Chaturvedi:** Environmental Data Scientist - Developer of the real-time monitoring platform and data visualization tools.

## Project Status

**Actively in development** — with successful pilot implementations in multiple industrial zones.

## Future Enhancements

Potential future developments for SatIndustry include:

*   **Advanced Analytics:** Implementing more sophisticated AI/ML models for predictive environmental analysis and anomaly detection.
*   **Expanded Data Layers:** Integrating additional environmental datasets, such as water quality indices, deforestation rates, and biodiversity metrics.
*   **User Accounts and Personalization:**  Adding user accounts to enable personalized dashboards, saved locations, and custom data alerts.
*   **Algorithm Optimization** to reduce computational costs  

## Acknowledgements
Special thanks to **VIT Vellore** and the **TARP initiative** for supporting this endeavor. We also appreciate the contributions from open-source communities and APIs like Planet Labs, OpenWeatherMap, WAQI, Sentinel, Geoapify, and Gemini that empower real-world impact.
