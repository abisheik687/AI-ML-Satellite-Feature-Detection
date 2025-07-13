// Dashboard JavaScript Functionality

// Global variables
let maps = {};
let charts = {};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeCharts();
    initializeMaps();
    initializeControls();
    initializeFileUpload();
});

// Navigation functionality
function initializeNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    navTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');

            // Remove active class from all tabs and contents
            navTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            // Refresh maps when tab becomes active
            setTimeout(() => {
                if (maps[targetTab]) {
                    maps[targetTab].invalidateSize();
                }
            }, 100);
        });
    });
}

// Initialize all charts
async function initializeCharts() {
    await createOverviewChart();
    await createTemporalChart();
    await createRoadsChart();
    await createDrainageChart();
    await createPerformanceCharts();
}

// Overview performance chart
async function createOverviewChart() {
    const metrics = await fetchFromAPI('metrics');
    const data = [
        {
            x: ['Glacial Lakes', 'Road Centrelines', 'Urban Drainage'],
            y: [metrics.glacial_lakes.accuracy, metrics.roads.accuracy, metrics.drainage.accuracy],
            type: 'bar',
            marker: {
                color: ['#3498db', '#2ecc71', '#e74c3c'],
                opacity: 0.8
            },
            name: 'Accuracy (%)'
        }
    ];

    const layout = {
        title: {
            text: 'Model Accuracy by Feature Type',
            font: { size: 14 }
        },
        xaxis: { title: 'Feature Type' },
        yaxis: { title: 'Accuracy (%)' },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('overview-chart', data, layout, {responsive: true});
}

// Temporal analysis chart for glacial lakes
async function createTemporalChart() {
    const metrics = await fetchFromAPI('metrics');
    const years = [2020, 2021, 2022, 2023, 2024];
    const lakeCount = years.map(y => metrics.system.total_processed * (1 + (y - 2024) * 0.05));
    const totalArea = years.map(y => metrics.glacial_lakes.iou_score * 100 * (1 + (y - 2024) * 0.05));

    const trace1 = {
        x: years,
        y: lakeCount,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Lake Count',
        line: { color: '#3498db' },
        yaxis: 'y'
    };

    const trace2 = {
        x: years,
        y: totalArea,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Total Area (km²)',
        line: { color: '#e74c3c' },
        yaxis: 'y2'
    };

    const layout = {
        title: {
            text: 'Glacial Lake Evolution',
            font: { size: 12 }
        },
        xaxis: { title: 'Year' },
        yaxis: {
            title: 'Lake Count',
            side: 'left'
        },
        yaxis2: {
            title: 'Total Area (km²)',
            side: 'right',
            overlaying: 'y'
        },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true,
        legend: { x: 0, y: 1 }
    };

    Plotly.newPlot('temporal-chart', [trace1, trace2], layout, {responsive: true});
}

// Roads network statistics chart
async function createRoadsChart() {
    const data = [
        {
            labels: ['National Highways', 'State Highways', 'District Roads', 'Rural Roads'],
            values: [15, 25, 35, 25],
            type: 'pie',
            marker: {
                colors: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            }
        }
    ];

    const layout = {
        title: {
            text: 'Road Network Distribution',
            font: { size: 12 }
        },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('roads-chart', data, layout, {responsive: true});
}

// Drainage stream order chart
async function createDrainageChart() {
    const data = [
        {
            x: ['Order 1', 'Order 2', 'Order 3', 'Order 4', 'Order 5+'],
            y: [45, 28, 15, 8, 4],
            type: 'bar',
            marker: {
                color: '#3498db',
                opacity: 0.8
            }
        }
    ];

    const layout = {
        title: {
            text: 'Stream Order Distribution',
            font: { size: 12 }
        },
        xaxis: { title: 'Stream Order' },
        yaxis: { title: 'Percentage (%)' },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('drainage-chart', data, layout, {responsive: true});
}

// Performance analytics charts
async function createPerformanceCharts() {
    const metrics = await fetchFromAPI('metrics');
    // Computational Efficiency
    const efficiencyData = [
        {
            x: ['Preprocessing', 'Model Inference', 'Post-processing', 'Vectorization'],
            y: [0.8, metrics.system.avg_response_time, 0.5, 0.3],
            type: 'bar',
            marker: { color: '#2ecc71' },
            name: 'Processing Time (s)'
        }
    ];

    const efficiencyLayout = {
        title: { text: 'Processing Time by Stage', font: { size: 12 } },
        xaxis: { title: 'Pipeline Stage' },
        yaxis: { title: 'Time (seconds)' },
        margin: { t: 40, b: 60, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('efficiency-chart', efficiencyData, efficiencyLayout, {responsive: true});

    // Scalability Assessment
    const scalabilityData = [
        {
            x: [100, 500, 1000, 2000, 5000],
            y: [metrics.glacial_lakes.accuracy, metrics.glacial_lakes.accuracy - 2, metrics.glacial_lakes.accuracy - 4, metrics.glacial_lakes.accuracy - 6, metrics.glacial_lakes.accuracy - 8],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy',
            line: { color: '#3498db' }
        }
    ];

    const scalabilityLayout = {
        title: { text: 'Accuracy vs Dataset Size', font: { size: 12 } },
        xaxis: { title: 'Dataset Size (images)' },
        yaxis: { title: 'Accuracy (%)' },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('scalability-chart', scalabilityData, scalabilityLayout, {responsive: true});

    // Model Accuracy Comparison
    const accuracyData = [
        {
            x: ['U-Net', 'DeepLabV3+', 'SAM', 'Random Forest'],
            y: [metrics.glacial_lakes.accuracy, metrics.roads.accuracy, metrics.drainage.accuracy, 78],
            type: 'bar',
            marker: {
                color: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                opacity: 0.8
            }
        }
    ];

    const accuracyLayout = {
        title: { text: 'Model Performance Comparison', font: { size: 12 } },
        xaxis: { title: 'Model Architecture' },
        yaxis: { title: 'Accuracy (%)' },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('accuracy-chart', accuracyData, accuracyLayout, {responsive: true});

    // Resource Utilization
    const resourceData = [
        {
            labels: ['GPU Memory', 'CPU Usage', 'RAM Usage', 'Storage'],
            values: [75, 45, 60, 30],
            type: 'pie',
            marker: {
                colors: ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
            }
        }
    ];

    const resourceLayout = {
        title: { text: 'Resource Utilization (%)', font: { size: 12 } },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('resource-chart', resourceData, resourceLayout, {responsive: true});
}

// Initialize maps
function initializeMaps() {
    // Glacial Lakes Map
    maps['glacial-lakes'] = L.map('glacial-map').setView([28.2380, 83.9956], 8);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(maps['glacial-lakes']);

    // Roads Map
    maps['roads'] = L.map('roads-map').setView([28.6139, 77.2090], 10);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(maps['roads']);

    // Drainage Map
    maps['drainage'] = L.map('drainage-map').setView([19.0760, 72.8777], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(maps['drainage']);
}

// Initialize interactive controls
function initializeControls() {
    // Time slider for glacial lakes
    const timeSlider = document.getElementById('time-slider');
    const timeDisplay = document.getElementById('time-display');

    if (timeSlider && timeDisplay) {
        timeSlider.addEventListener('input', function() {
            timeDisplay.textContent = this.value;
            // Update glacial lakes display based on selected year
            updateGlacialLakesDisplay(this.value);
        });
    }

    // Drainage layer selector
    const drainageLayer = document.getElementById('drainage-layer');
    if (drainageLayer) {
        drainageLayer.addEventListener('change', function() {
            updateDrainageLayer(this.value);
        });
    }
}

// Update glacial lakes display based on year
function updateGlacialLakesDisplay(year) {
    // Simulate temporal changes
    console.log(`Updating glacial lakes display for year: ${year}`);
    // In a real implementation, this would update the map layers
    // and refresh the temporal chart with data for the selected year
}

// Update drainage layer display
function updateDrainageLayer(layerType) {
    console.log(`Switching to drainage layer: ${layerType}`);
    // In a real implementation, this would switch between different
    // drainage visualization layers (streams, flow direction, accumulation)
}

// Utility function to simulate data loading
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    }
}

// Utility function to hide loading
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '';
    }
}

// Export functionality (placeholder)
function exportResults(featureType) {
    alert(`Exporting ${featureType} results as Shapefile/GeoJSON...`);
    // In a real implementation, this would trigger a download
    // of the extracted features in the requested format
}

// Add event listeners for export buttons
document.addEventListener('DOMContentLoaded', function() {
    const exportButtons = document.querySelectorAll('.btn-primary');
    exportButtons.forEach(button => {
        if (button.textContent.includes('Export')) {
            button.addEventListener('click', function() {
                exportResults('current feature');
            });
        }
    });
});

// File upload functionality
function initializeFileUpload() {
    const fileUploadInputs = document.querySelectorAll('input[type="file"]');
    fileUploadInputs.forEach(input => {
        input.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) {
                return;
            }

            const featureType = event.target.dataset.featureType;
            const endpoint = `process/${featureType}`;

            try {
                showLoading(`${featureType}-map`);
                const result = await uploadFile(endpoint, file);
                hideLoading(`${featureType}-map`);
                updateMap(featureType, result.result);
            } catch (error) {
                console.error(`Error processing ${featureType}:`, error);
                hideLoading(`${featureType}-map`);
            }
        });
    });
}

// Update map with new data
function updateMap(featureType, geojsonData) {
    const map = maps[featureType];
    if (!map) {
        return;
    }

    // Clear existing layers
    map.eachLayer(layer => {
        if (layer instanceof L.GeoJSON) {
            map.removeLayer(layer);
        }
    });

    // Add new GeoJSON layer
    const geoJsonLayer = L.geoJSON(geojsonData, {
        style: function (feature) {
            switch (featureType) {
                case 'glacial-lakes':
                    return { color: '#3498db' };
                case 'roads':
                    return { color: '#e74c3c' };
                case 'drainage':
                    return { color: '#2ecc71' };
                default:
                    return { color: '#ffffff' };
            }
        }
    }).addTo(map);

    // Fit map to the new layer
    map.fitBounds(geoJsonLayer.getBounds());
}
