const map = L.map('map').setView([20.5937, 78.9629], 5); // Default center: India

let currentMarker;

const infoPanel = document.getElementById('info-panel');
const currentTemp = document.getElementById('current-temp');
const aqiTable = document.getElementById('aqi-table');
const coordinatesDiv = document.getElementById('coordinates');
const planetPanel = document.getElementById('planet-panel');
const planetResultsDiv = document.getElementById('planet-results');

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
        windLayer = L.tileLayer(windTileUrl);     // Initialize windLayer with URL from backend


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



// Handle Search Button
document.getElementById('search-button').addEventListener('click', () => {
    console.log("Search button clicked!"); // ADDED console.log
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

    const weatherDataUrl = `/weather-data/?lat=${lat}&lon=${lon}`;
    const airPollutionDataUrl = `/air-pollution-data/?lat=${lat}&lon=${lon}`;

    try {
        lat = parseFloat(lat);
        lon = parseFloat(lon);

        if (isNaN(lat) || isNaN(lon)) {
            throw new Error('Invalid latitude or longitude');
        }

        // Fetch current weather data
        const weatherResponse = await fetch(weatherDataUrl);
        if (!weatherResponse.ok) { /* ... error handling ... */ }
        const weatherData = await weatherResponse.json();
        currentTemp.innerText = `${weatherData.main.temp.toFixed(1)}°C`;

        // Fetch air pollution data
        const airPollutionResponse = await fetch(airPollutionDataUrl);
        if (!airPollutionResponse.ok) { /* ... error handling ... */ }
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

        // **Trigger Planet API analysis AFTER successful AQI fetch**
        console.log("Triggering Planet API analysis...");
        fetchPlanetData(lat, lon); // Call fetchPlanetData here

    } catch (error) {
        console.error('Error fetching data:', error);
        alert('Failed to fetch data.');
    } finally {
        toggleSpinner(false);
    }
}

async function fetchPlanetData(lat, lon) {
    toggleSpinner(true);
    console.log("fetchPlanetData function called with lat:", lat, "lon:", lon);
    planetPanel.style.display = 'flex'; // Show Planet Panel
    planetResultsDiv.innerHTML = "<p>Analyzing Planet Imagery...</p>"; // Initial message

    const planetAnalysisUrl = `/planet-analysis/?lat=${lat}&lon=${lon}`;

    try {
        const response = await fetch(planetAnalysisUrl);
        if (!response.ok) {
            const message = `Error fetching Planet analysis: ${response.status} ${response.statusText}`;
            throw new Error(message);
        }
        const planetData = await response.json();
        console.log("Planet Analysis Data:", planetData);

        console.log("Planet Analysis Data:", planetData);
        // Display Planet Analysis Results in the left panel
        let planetResultsHTML = `<h3>Vegetation Change Summary</h3>`;
        if (planetData.ndvi_change_summary) {
            planetResultsHTML += `
                <p>Min NDVI Change: ${planetData.ndvi_change_summary.min_change.toFixed(3)}</p>
                <p>Max NDVI Change: ${planetData.ndvi_change_summary.max_change.toFixed(3)}</p>
                <p>Average NDVI Change: ${planetData.ndvi_change_summary.average_change.toFixed(3)}</p>
            `;
        } else if (planetData.error) {
            planetResultsHTML = `<p>Error during Planet analysis: ${planetData.error}</p>`;
        } else {
            planetResultsHTML = "<p>No Planet analysis summary available.</p>"; // Fallback message
        }
        planetResultsDiv.innerHTML = planetResultsHTML; // Update planet-results div

        // Display RGB Images and Metadata
        const oldestRgbImage = document.getElementById('oldest-rgb-image');
        const recentRgbImage = document.getElementById('recent-rgb-image');
        const oldestImageDate = document.getElementById('oldest-image-date');
        const recentImageDate = document.getElementById('recent-image-date');
        const oldestImageId = document.getElementById('oldest-image-id');
        const recentImageId = document.getElementById('recent-image-id');


        if (planetData.oldest_rgb_image_url) {
            oldestRgbImage.src = planetData.oldest_rgb_image_url;
            oldestRgbImage.style.display = 'block';

            // Format and display date and ID
            oldestImageDate.textContent = `Acquired: ${new Date(planetData.oldest_image_date).toLocaleDateString()}`; // Format date nicely
            oldestImageId.textContent = `Item ID: ${planetData.oldest_image_id}`;

            oldestImageDate.style.display = 'block'; // Show date
            oldestImageId.style.display = 'block';   // Show ID

        } else {
            oldestRgbImage.style.display = 'none';
            oldestImageDate.style.display = 'none'; // Hide date if no image
            oldestImageId.style.display = 'none';   // Hide ID if no image
            console.warn("Oldest RGB Image URL not found in response.");
        }

        if (planetData.most_recent_rgb_image_url) {
            recentRgbImage.src = planetData.most_recent_rgb_image_url;
            recentRgbImage.style.display = 'block';

            // Format and display date and ID
            recentImageDate.textContent = `Acquired: ${new Date(planetData.most_recent_image_date).toLocaleDateString()}`; // Format date nicely
            recentImageId.textContent = `Item ID: ${planetData.most_recent_image_id}`;

            recentImageDate.style.display = 'block'; // Show date
            recentImageId.style.display = 'block';   // Show ID
        } else {
            recentRgbImage.style.display = 'none';
            recentImageDate.style.display = 'none'; // Hide date if no image
            recentImageId.style.display = 'none';   // Hide ID if no image
            console.warn("Most Recent RGB Image URL not found in response.");
        }


        planetResultsDiv.innerHTML = ''; // Clear loading message after successful load
        planetResultsDiv.appendChild(document.querySelector('.rgb-images')); // Re-append rgb-images div
        planetResultsDiv.appendChild(ndviSummaryDiv); // Re-append ndviSummaryDiv


    } catch (error) {
        console.error('Error fetching Planet data:', error);
        planetResultsDiv.innerHTML = `<p>Failed to fetch Planet analysis data: ${error.message}</p>`; // Display error in panel
        alert('Failed to fetch Planet analysis data.');
    } finally {
        toggleSpinner(false);
    }
}

// CLOSE PLANET PANEL BUTTON
document.getElementById('closePlanetBtn').addEventListener('click', () => {
    console.log("Close Planet Panel button clicked!");
    planetPanel.style.display = 'none';
});