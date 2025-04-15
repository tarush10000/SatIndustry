const backToTopButton = document.getElementById('backToTop');

window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
        backToTopButton.classList.add('visible');
    } else {
        backToTopButton.classList.remove('visible');
    }
});

backToTopButton.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const industryButtons = document.querySelectorAll('.industry-button');
    const industryResultsDiv = document.getElementById('industry-results');

    industryButtons.forEach(button => {
        button.addEventListener('click', function() {
            console.log('Button clicked:', this.getAttribute('data-industry'));
            const industryType = this.getAttribute('data-industry');
            fetch(`/get_polluting_industries/?industry=${industryType}`)
                .then(response => response.json())
                .then(data => {
                    if (data.industries) {
                        // Create the table
                        const table = document.createElement('table');
                        table.id = 'industry-results-table';
                        const thead = document.createElement('thead');
                        const headerRow = document.createElement('tr');
                        const nameHeader = document.createElement('th');
                        nameHeader.textContent = 'Industry Name';
                        const locationHeader = document.createElement('th');
                        locationHeader.textContent = 'Location';
                        headerRow.appendChild(nameHeader);
                        headerRow.appendChild(locationHeader);
                        thead.appendChild(headerRow);
                        table.appendChild(thead);

                        const tbody = document.createElement('tbody');
                        data.industries.forEach(industry => {
                            const row = document.createElement('tr');
                            row.dataset.url = industry.url; // Store the URL in a data attribute
                            row.dataset.lat = industry.latitude; // Store latitude in row.dataset.lat
                            row.dataset.lon = industry.longitude; // Store longitude in row.dataset.lon
                            row.dataset.industry = industry.name; // Store industry name in row.dataset.industry

                            const nameCell = document.createElement('td');
                            nameCell.textContent = industry.name;
                            row.appendChild(nameCell);

                            const locationCell = document.createElement('td');
                            locationCell.textContent = industry.location;
                            row.appendChild(locationCell);

                            tbody.appendChild(row);
                        });
                        table.appendChild(tbody);

                        // Add the table to the results div
                        industryResultsDiv.innerHTML = ''; // Clear previous results
                        industryResultsDiv.appendChild(table);

                        // Add event listener for row clicks to redirect to data page
                        tbody.addEventListener('click', function(event) {
                            const clickedRow = event.target.closest('tr');
                            if (clickedRow && clickedRow.dataset.url && clickedRow.dataset.lat && clickedRow.dataset.lon) {
                                const industryName = clickedRow.querySelector('td:first-child').textContent;
                                const latitude = clickedRow.dataset.lat;
                                const longitude = clickedRow.dataset.lon;
                                const location = clickedRow.querySelector('td:nth-child(2)').textContent;
                                console.log('Row clicked:', industryName, latitude, longitude, location);
                                window.location.href = `${clickedRow.dataset.url}&industry=${encodeURIComponent(industryName)}&latitude=${latitude}&longitude=${longitude}&location=${encodeURIComponent(location)}`;
                            }
                        });

                    } else if (data.error) {
                        industryResultsDiv.innerHTML = `<p class="error-message">${data.error}</p>`;
                    }
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    industryResultsDiv.innerHTML = '<p class="error-message">Failed to fetch industry data.</p>';
                });
        });
    });

    const manualSearchButton = document.getElementById('manual-search-button');
    const manualSearchInput = document.getElementById('manual-search-input');

    if (manualSearchButton) {
        manualSearchButton.addEventListener('click', function() {
            const location = manualSearchInput.value;
            if (location) {
                const dataSearchUrl = "{% url 'data_search' %}";
                window.location.href = `/data?location=${encodeURIComponent(location)}`;
                console.log('Searching for:', location);
                console.log('Redirecting to:', `${dataSearchUrl}?location=${encodeURIComponent(location)}`);
            } else {
                alert('Please enter a location to search.');
            }
        });
    }
});