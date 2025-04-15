document.addEventListener('DOMContentLoaded', function () {
    const map = L.map('map').setView([20.5937, 78.9629], 5); // Default center: India

    let currentMarker;

    const infoPanel = document.getElementById('info-panel');
    const currentTemp = document.getElementById('current-temp');
    const aqiTable = document.getElementById('aqi-table');
    const coordinatesDiv = document.getElementById('coordinates');

    // Left Prediction Panel Elements
    const predictionPanel = document.getElementById('prediction-panel');
    const closePredictionBtn = document.getElementById('closePredictionBtn');
    const predictionDataContainer = document.getElementById('prediction-data');
    const mitigationStrategiesContainer = document.getElementById('mitigation-strategies');

    // Add map layers, tile layers, etc. (Keep these as they are)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

    let cloudsLayer;
    let windLayer;

    // Fetch tile layer URLs from Django backend and initialize layers
    async function initializeTileLayers() {
        try {
            // Fetch cloud layer URL
            const cloudsUrlResponse = await fetch('/cloud-layer-url/');
            if (!cloudsUrlResponse.ok) {
                throw new Error(`Failed to fetch cloud layer URL: ${cloudsUrlResponse.status} ${cloudsUrlResponse.statusText}`);
            }
            const cloudsUrlData = await cloudsUrlResponse.json();
            const cloudsTileUrl = cloudsUrlData.tileUrl;
            cloudsLayer = L.tileLayer(cloudsTileUrl); // Initialize cloudsLayer with URL from backend


            // Fetch wind layer URL
            const windUrlResponse = await fetch('/wind-layer-url/');
            if (!windUrlResponse.ok) {
                throw new Error(`Failed to fetch wind layer URL: ${windUrlResponse.status} ${windUrlResponse.statusText}`);
            }
            const windUrlData = await windUrlResponse.json();
            const windTileUrl = windUrlData.tileUrl;
            windLayer = L.tileLayer(windTileUrl);          // Initialize windLayer with URL from backend


        } catch (error) {
            console.error('Error initializing tile layers:', error);
            alert('Failed to initialize weather tile layers.'); // Or handle error more gracefully
        }
    }

    initializeTileLayers(); // Call the initialization function


    // Toggle Layers (Keep the toggle logic, but now using the initialized layers)
    document.getElementById('toggle-clouds').addEventListener('click', () => {
        console.log("Toggle Clouds button clicked!");
        if (map.hasLayer(cloudsLayer)) {
            map.removeLayer(cloudsLayer);
        } else {
            map.addLayer(cloudsLayer);
        }
    });

    document.getElementById('toggle-wind').addEventListener('click', () => {
        console.log("Toggle Wind button clicked!");
        if (map.hasLayer(windLayer)) {
            map.removeLayer(windLayer);
        } else {
            map.addLayer(windLayer);
        }
    });

    document.getElementById('toggle-aqi').addEventListener('click', () => {
        console.log("Toggle AQI button clicked!");
        if (map.hasLayer(waqiLayer)) {
            map.removeLayer(waqiLayer);
        } else {
            map.addLayer(waqiLayer);
        }
    });

    const cloudsButton = document.getElementById('toggle-clouds');
    const windButton = document.getElementById('toggle-wind');
    const aqiButton = document.getElementById('toggle-aqi');

    function toggleButton(button) {
        button.classList.toggle('toggled');
    }

    cloudsButton.addEventListener('click', () => toggleButton(cloudsButton));
    windButton.addEventListener('click', () => toggleButton(windButton));
    aqiButton.addEventListener('click', () => toggleButton(aqiButton));


    const WAQI_URL = 'https://tiles.waqi.info/tiles/usepa-aqi/{z}/{x}/{y}.png?token=_TOKEN_ID_';
    const WAQI_ATTR = 'Air Quality Tiles &copy; <a href="http://waqi.info">waqi.info</a>';
    const waqiLayer = L.tileLayer(WAQI_URL, { attribution: WAQI_ATTR });


    function addMarker(lat, lon, title) {
        if (currentMarker) map.removeLayer(currentMarker);
        currentMarker = L.marker([lat, lon]).addTo(map).bindPopup(title).openPopup();
        map.setView([lat, lon], 13);
    }

    function toggleSpinner(show) {
        document.getElementById('spinner').style.display = show ? 'flex' : 'none';
    }


    async function fetchHistoricalData(lat, lon) {
        const historicalTemps = [];
        const baseUrl = 'https://archive-api.open-meteo.com/v1/archive';

        for (let yearOffset = 1; yearOffset <= 20; yearOffset++) {
            const year = new Date().getFullYear() - yearOffset;
            const startDate = `${year}-01-01`; // Start of the year (YYYY-MM-DD format)
            const endDate = `${year}-01-31`;   // End of the year (YYYY-MM-DD format)

            const lat2 = Math.round(lat * 100) / 100; // Round latitude to 2 decimal places
            const lon2 = Math.round(lon * 100) / 100; // Round longitude to 2 decimal places

            const historicalUrl = `${baseUrl}?latitude=${lat2}&longitude=${lon2}&start_date=${startDate}&end_date=${endDate}&hourly=temperature_2m`;

            try {
                const response = await fetch(historicalUrl);
                const data = await response.json();

                if (data.hourly && data.hourly.temperature_2m) {
                    const temps = data.hourly.temperature_2m;
                    const yearlyAvg = temps.reduce((sum, temp) => sum + temp, 0) / temps.length;
                    historicalTemps.push({ year, temp: yearlyAvg.toFixed(2) });
                } else {
                    console.warn(`No temperature data found for ${year}.`);
                }
            } catch (error) {
                console.error('Error fetching historical data for year ' + year, error);
            }
        }
        return historicalTemps;
    }

    let currentChart = null;
    let currentChart2 = null;

    async function createAirQualityChart(lat, lon) {
        if (currentChart2) {
            currentChart2.destroy();
            currentChart2 = null;
        }

        const historicalAQIDataUrl = `/historical-air-quality-data/?lat=${lat}&lon=${lon}`; // URL to Django historical AQI view

        try {
            const response = await fetch(historicalAQIDataUrl);
            if (!response.ok) {
                const message = `Error fetching historical air quality data: ${response.status} ${response.statusText}`;
                throw new Error(message);
            }
            const responseData = await response.json();
            const airQualityData = responseData.airQualityData; // Access airQualityData from response JSON

            const ctx = document.getElementById('chart2').getContext('2d');

            currentChart2 = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: airQualityData.map(entry => entry.year),
                    datasets: [{
                        label: 'Average Air Quality Index (AQI)',
                        data: airQualityData.map(entry => entry.averageAQI),
                        borderColor: '#ff5722',
                        backgroundColor: 'rgba(255, 87, 34, 0.2)',
                        fill: true,
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Year',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'AQI',
                            },
                            ticks: {
                                beginAtZero: true,
                            }
                        }
                    }
                }
            });


        } catch (error) {
            console.error('Error fetching historical air quality data:', error);
            alert('Failed to fetch historical air quality data.');
        }
    }

    function createChart(data) {
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
        const ctx = document.getElementById('chart').getContext('2d');
        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(entry => entry.year),
                datasets: [{
                    label: 'Average Temperature (°C)',
                    data: data.map(entry => entry.temp),
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ddd',
                        },
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Year',
                            color: '#ddd',
                        },
                        ticks: {
                            color: '#ddd',
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Temperature (°C)',
                            color: '#ddd',
                        },
                        ticks: {
                            color: '#ddd',
                        },
                    },
                },
            },
        });
    }

    function openPredictionPanel() {
        predictionPanel.classList.add('open');
    }

    function closePredictionPanel() {
        predictionPanel.classList.remove('open');
    }

    if (closePredictionBtn) {
        closePredictionBtn.addEventListener('click', closePredictionPanel);
    }

    const urlParams = new URLSearchParams(window.location.search);
    const latitudeFromURL = urlParams.get('latitude');
    const longitudeFromURL = urlParams.get('longitude');
    const industryNameFromURL = urlParams.get('industry');
    const locationNameFromURL = urlParams.get('location');

    if (latitudeFromURL && longitudeFromURL && industryNameFromURL) {
        const lat = parseFloat(latitudeFromURL);
        const lon = parseFloat(longitudeFromURL);
        const displayName = industryNameFromURL;
        const locationName = locationNameFromURL || industryNameFromURL;

        if (!isNaN(lat) && !isNaN(lon)) {
            console.log("Latitude from URL:", lat);
            console.log("Longitude from URL:", lon);
            console.log("Industry from URL:", displayName);

            if (typeof addMarker === 'function') {
                addMarker(lat, lon, displayName);
            }
            if (typeof fetchData === 'function') {
                fetchData(lat, lon);
            }
            document.getElementById('search-input').value = displayName;
            fetchPredictionsAndMitigation(displayName, lat, lon, locationName);
        } else {
            console.error("Invalid latitude or longitude in the URL.");
            alert("Invalid coordinates provided in the URL.");
        }
    } else {
        console.log("Latitude, longitude, or industry not found in the URL.");
    }

    document.getElementById('search-button').addEventListener('click', () => {
        const location = document.getElementById('search-input').value;
        if (location) {
            fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${location}`)
                .then(response => response.json())
                .then(data => {
                    if (data && data.length > 0) {
                        const { lat, lon, display_name } = data[0];
                        addMarker(lat, lon, display_name);
                        fetchData(lat, lon);
                    } else {
                        alert('Location not found');
                    }
                })
                .catch(error => {
                    console.error('Error fetching location:', error);
                    alert('Failed to fetch location.');
                });
        } else {
            alert('Please enter a location.');
        }
    });

    const searchLocation = (locationName, industry, successCallback, failureCallback) => {
        let url = `/get_coordinates_here/?q=${encodeURIComponent(locationName)}`;
        console.log("searchLocation URL:", url);
        if (industry) {
            url += `&industry=${encodeURIComponent(industry)}`;
        }
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || 'Geocoding API error');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.latitude && data.longitude) {
                    successCallback(data.latitude, data.longitude, data.displayName);
                    fetchPredictionsAndMitigation(data.displayName, data.latitude, data.longitude);
                } else {
                    failureCallback(data.error || `Location "${locationName}" not found.`);
                }
            })
            .catch(error => {
                console.error('Error fetching location:', error);
                failureCallback(`Failed to fetch location for "${locationName}".`);
            });
    };

    function createDataTable(title, data, description = "") {
        let html = `<div class="info-header">${title}</div>`;
        if (description) {
            html += `<p class="info-description">${description}</p>`;
        }
        html += '<table class="data-table">';
        if (Object.keys(data).length > 0) {
            const keys = Object.keys(data);
            const firstItem = data[keys[0]];
    
            // Create table header
            html += '<thead><tr>';
            if (typeof firstItem === 'object' && firstItem !== null) {
                html += '<th>Pollutant</th>';
                for (const key in firstItem) {
                    if (firstItem.hasOwnProperty(key)) {
                        html += `<th>${key.charAt(0).toUpperCase() + key.slice(1)}</th>`; // Capitalize keys
                    }
                }
            } else {
                html += '<th>Cluster</th><th>Average Pollution</th>';
            }
            html += '</tr></thead><tbody>';
    
            // Create table rows
            for (const key in data) {
                if (data.hasOwnProperty(key)) {
                    html += '<tr>';
                    if (typeof firstItem === 'object' && firstItem !== null) {
                        html += `<td>${key.toUpperCase()}</td>`; // Pollutant name
                        for (const stat in firstItem) {
                            if (firstItem.hasOwnProperty(stat)) {
                                html += `<td>${data[key][stat].toFixed(2)}</td>`; // Display with 2 decimal places
                            }
                        }
                    } else {
                        html += `<td>${key}</td><td>${data[key].toFixed(2)}</td>`; // Cluster ID and average
                    }
                    html += '</tr>';
                }
            }
            html += '</tbody></table>';
        } else {
            html += '<p>No data available.</p>';
        }
        return html;
    }
    
    function createComparisonTable(industryStats, clusterStats) {
        let html = '<div class="info-header">Industry vs. Cluster (Mean Values)</div>';
        html += '<table class="data-table">';
        html += '<thead><tr><th>Pollutant</th><th>Industry Mean</th><th>Cluster Mean</th></tr></thead><tbody>';
    
        for (const pollutant in industryStats) {
            if (industryStats.hasOwnProperty(pollutant) && clusterStats.hasOwnProperty(pollutant)) {
                html += '<tr>';
                html += `<td>${pollutant.toUpperCase()}</td>`;
                html += `<td>${industryStats[pollutant].mean.toFixed(2)}</td>`;
                html += `<td>${clusterStats[pollutant].mean.toFixed(2)}</td>`;
                html += '</tr>';
            }
        }
    
        html += '</tbody></table>';
        return html;
    }

    function fetchPredictionsAndMitigation(industryName, latitude, longitude, location) {
        const url = `/get_predictions_and_mitigation/?industry=${encodeURIComponent(industryName)}&latitude=${latitude}&longitude=${longitude}&location=${encodeURIComponent(location)}`;
        console.log("Fetching predictions and mitigation from:", url);
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || 'Failed to fetch predictions and mitigation');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log("Full Prediction Data:", data); // Log the entire data object
                const predictionDataContainer = document.getElementById('prediction-data');
                const mitigationStrategiesContainer = document.getElementById('mitigation-strategies');
                // console.log("Mitigation Strategies:", data.mitigationStrategies);
    
                // Display mitigation strategies
                if (data.mitigationStrategies) {
                    mitigationStrategiesContainer.innerHTML = `<p>${data.mitigationStrategies}</p>`;
                } else {
                    mitigationStrategiesContainer.innerHTML = '<p>No mitigation strategies available.</p>';
                }
    
                // --- Log the new data received ---
                // console.log("PCA Coordinates:", data.pca_coords);
                // console.log("Cluster Statistics:", data.cluster_stats);
                // console.log("Industry Statistics:", data.industry_stats);
                // console.log("Industry Daily Mean Pollution:", data.industry_daily_mean);
                // console.log("Industry 24-Hour Mean Pollution:", data.industry_24hr_mean);
                // console.log("Exceeding CPCB Limits:", data.exceeding_limits);
                // console.log("Average Cluster Pollution:", data.average_cluster_pollution);
                // console.log("Average Industry Pollution:", data.average_industry_pollution);
                // console.log("Anomaly Data:", data.anomaly_data);
                // console.log("LSTM Predictions:", data.lstm_predictions);
                // console.log("LSTM Timestamps:", data.lstm_timestamps);
                // console.log("LSTM Actual Values:", data.lstm_actual);
                // console.log("LSTM Predicted Values:", data.lstm_predicted);
                // console.log("LSTM Test Timestamps:", data.lstm_test_timestamps);
                // console.log("LSTM Targets:", data.lstm_targets);
    
                let tableHTML = '';
                tableHTML += createDataTable('Cluster Statistics', data.cluster_stats, `Average pollution statistics for the cluster that ${industryName} belongs to.`);
                tableHTML += createDataTable('Industry Statistics', data.industry_stats, `Pollution statistics for the selected industry: ${industryName}.`);
                tableHTML += createDataTable('Average Cluster Pollution', data.average_cluster_pollution);

                // Create the comparison table
                tableHTML += createComparisonTable(data.industry_stats, data.cluster_stats);

                let otherInfoHTML = '';
                otherInfoHTML += `<div class="info-header">Average Industry Pollution</div><p>${data.average_industry_pollution}</p>`;
                // otherInfoHTML += `<div class="info-header">Exceeding CPCB Limits</div><p>${data.exceeding_limits}</p>`;
    
                predictionDataContainer.innerHTML = tableHTML + otherInfoHTML;
    
                // --- Prepare data for charts ---
                const anomalyData = data.anomaly_data;
                const lstmTimestamps = data.lstm_timestamps;
                const lstmPredictions = data.lstm_predictions;
                const lstmActual = data.lstm_actual;
                const lstmPredicted = data.lstm_predicted;
                const lstmTestTimestamps = data.lstm_test_timestamps;
                const lstmTargets = data.lstm_targets;
    
                const createApexPollutantChart = (target, anomalyData, lstmTimestamps, lstmPredictions, lstmActual, lstmPredicted, lstmTestTimestamps, lstmTargets) => {
                    const chartId = `chart-${target.replace(/[^a-zA-Z0-9]/g, '')}`;
                    let chartContainer = document.getElementById(chartId);
    
                    if (!chartContainer) {
                        chartContainer = document.createElement('div');
                        chartContainer.id = chartId;
                        chartContainer.classList.add('chart-container'); // Add a class for styling if needed
                        predictionDataContainer.appendChild(chartContainer);
                    }
    
                    const targetIndex = lstmTargets.indexOf(target);
                    const actualData = lstmActual.map(item => item[targetIndex]);
                    const predictedData = lstmPredicted.map(item => item[targetIndex]);
                    const forecastData = lstmPredictions.map(item => item[targetIndex]);
    
                    const anomalyPoints = anomalyData
                        .filter(item => item.anomaly && item[target.toLowerCase().replace('.', '_')])
                        .map(item => ({
                            x: new Date(item.timestamp).getTime(), // ApexCharts uses milliseconds for dates
                            y: item[target.toLowerCase().replace('.', '_')]
                        }));
    
                    const options = {
                        series: [
                            {
                                name: `Actual ${target}`,
                                data: lstmTestTimestamps.map((ts, index) => [new Date(ts).getTime(), actualData[index]])
                            },
                            {
                                name: `Predicted ${target}`,
                                data: lstmTestTimestamps.map((ts, index) => [new Date(ts).getTime(), predictedData[index]])
                            },
                            {
                                name: `Forecasted ${target}`,
                                data: lstmTimestamps.map((ts, index) => [new Date(ts).getTime(), forecastData[index]])
                            },
                            {
                                name: 'Anomalies',
                                type: 'scatter',
                                data: anomalyPoints,
                                show: false
                            }
                        ],
                        chart: {
                            id: chartId,
                            type: 'line',
                            height: 350, // Adjust as needed
                            toolbar: {
                                show: true
                            },
                            foreColor: '#fff', // Set default text color to white
                            background: '#111' // Optional: Set a dark background for better contrast
                        },
                        xaxis: {
                            type: 'datetime',
                            labels: {
                                style: {
                                    colors: '#fff' // White color for x-axis labels
                                },
                                format: 'dd MMM HH:mm' // Customize date format
                            },
                            max: new Date().getTime() + (4 * 24 * 60 * 60 * 1000) // Set max date to 4 days after present
                        },
                        yaxis: {
                            title: {
                                text: target,
                                style: {
                                    color: '#fff' // White color for y-axis title
                                }
                            },
                            labels: {
                                style: {
                                    colors: '#fff' // White color for y-axis labels
                                }
                            },
                            decimalsInFloat: 2
                        },
                        markers: {
                            size: [4, 4, 4, 8], // Adjust marker sizes for each series
                            colors: ['blue', 'orange', 'green', 'red']
                        },
                        stroke: {
                            curve: 'smooth'
                        },
                        tooltip: {
                            style: {
                                background: 'rgba(50, 50, 50)', // Grey background with some transparency
                                color: '#000', // White text color for tooltip
                                borderColor: '#777'
                            }
                        },
                        legend: {
                            labels: {
                                colors: '#fff' // White color for legend text
                            }
                        },
                        grid: {
                            borderColor: '#444' // Optional: Adjust grid line color for better visibility on dark background
                        }
                    };
    
                    const chart = new ApexCharts(chartContainer, options);
                    chart.render();
                };
    
                lstmTargets.forEach(target => {
                    createApexPollutantChart(target, anomalyData, lstmTimestamps, lstmPredictions, lstmActual, lstmPredicted, lstmTestTimestamps, lstmTargets);
                });
    
    
                if (typeof openPredictionPanel === 'function') {
                    openPredictionPanel();
                }
    
                // // You'll now likely want to use the LSTM predictions for your primary prediction display
                // let predictionHTML = '<ul>';
                // if (data.lstm_predictions && data.lstm_targets && data.lstm_predictions.length > 0) {
                //     // Assuming you want to display the first prediction in the LSTM output as the "current" prediction
                //     // You might need to adjust this based on how you want to represent the LSTM predictions
                //     const firstPrediction = data.lstm_predictions[0];
                //     for (let i = 0; i < data.lstm_targets.length; i++) {
                //         predictionHTML += `<li>${data.lstm_targets[i]}: ${firstPrediction[i].toFixed(2)}</li>`;
                //     }
                // } else {
                //     predictionHTML += '<li>No LSTM prediction data available.</li>';
                // }
                // predictionDataContainer.innerHTML = predictionHTML;
    
    
                // if (typeof openPredictionPanel === 'function') {
                //     openPredictionPanel();
                // }
            })
            .catch(error => {
                console.error("Error fetching predictions and mitigation:", error);
                alert("Failed to load predictions and mitigation data.");
            });
    }

    
// Handle Location Button
document.getElementById('location-button').addEventListener('click', () => {
        console.log("Location button clicked!"); // ADDED console.log
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(position => {
                const { latitude, longitude } = position.coords;
                addMarker(latitude, longitude, 'Your Location');
                fetchData(latitude, longitude);
            }, error => {
                console.error('Error fetching geolocation:', error);
                alert('Failed to get your location.');
            });
        } else {
            alert('Geolocation is not supported by your browser.');
        }
    });
    
    
    // CLOSE BUTTON
    document.getElementById('closeBtn').addEventListener('click', () => {
        console.log("Close button clicked!"); // ADDED console.log
        panel.style.display = 'none';
    });

    async function fetchData(lat, lon) {
        console.log("fetchData function called with lat:", lat, "lon:", lon);
        toggleSpinner(true);
        lat = parseFloat(lat.toFixed(2));
        lon = parseFloat(lon.toFixed(2));
        const weatherDataUrl = `/weather-data/?lat=${lat}&lon=${lon}`;
        const airPollutionDataUrl = `/air-pollution-data/?lat=${lat}&lon=${lon}`;

        try {
            if (isNaN(lat) || isNaN(lon)) {
                throw new Error('Invalid latitude or longitude');
            }

            // Fetch current weather data
            const weatherResponse = await fetch(weatherDataUrl);
            if (!weatherResponse.ok) {
                const errorDetail = await weatherResponse.json();
                throw new Error(`Failed to fetch weather data: ${weatherResponse.status} - ${errorDetail.error || weatherResponse.statusText}`);
            }
            const weatherData = await weatherResponse.json();
            currentTemp.innerText = `${weatherData.main.temp.toFixed(1)}°C`;

            // Fetch air pollution data
            const airPollutionResponse = await fetch(airPollutionDataUrl);
            if (!airPollutionResponse.ok) {
                const errorDetail = await airPollutionResponse.json();
                throw new Error(`Failed to fetch air pollution data: ${airPollutionResponse.status} - ${errorDetail.error || airPollutionResponse.statusText}`);
            }
            const airPollutionData = await airPollutionResponse.json();
            const components = airPollutionData.list[0].components;

            aqiTable.innerHTML = `
                <tr>
                    <th>Pollutant</th>
                    <th>Level (µg/m³)</th>
                </tr>
                ${Object.entries(components).map(([pollutant, level]) => {
                const classification = level > 150 ? 'danger' : level > 50 ? 'warning' : 'safe';
                return `<tr><td>${pollutant.toUpperCase()}</td><td class="${classification}">${level}</td></tr>`;
            }).join('')}
            `;

            coordinatesDiv.innerHTML = `Latitude: ${lat.toFixed(2)}, Longitude: ${lon.toFixed(2)}`;

            // Create historical temperature chart
            const historicalData = await fetchHistoricalData(lat, lon);
            createChart(historicalData);

            // Create air quality chart
            createAirQualityChart(lat, lon);
            console.log("Air Quality Chart created!");
            // Ensure the info panel is visible
            infoPanel.style.display = 'flex';

        } catch (error) {
            console.error('Error fetching data:', error);
            alert('Failed to fetch data.');
        } finally {
            toggleSpinner(false);
        }
    }
});