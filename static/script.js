/* Main Application JavaScript */

const API_URL = 'http://localhost:5000';
let selectedFile = null;
let currentChart = null;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const classifyBtn = document.getElementById('classifyBtn');
const btnText = document.getElementById('btnText');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const predictionsList = document.getElementById('predictionsList');
const topClass = document.getElementById('topClass');
const topConfidence = document.getElementById('topConfidence');
const errorMessage = document.getElementById('errorMessage');
const chartSection = document.getElementById('chartSection');

// Event Listeners
uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);

imageInput.addEventListener('change', handleFileSelect);
changeImageBtn.addEventListener('click', resetImage);
classifyBtn.addEventListener('click', classifyImage);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.backgroundColor = '#f1f5ff';
    uploadBox.style.borderColor = '#8b5cf6';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.backgroundColor = '';
    uploadBox.style.borderColor = '';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.backgroundColor = '';
    uploadBox.style.borderColor = '';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
}

// File Selection Handler
function handleFileSelect() {
    const files = imageInput.files;
    if (files.length === 0) return;

    selectedFile = files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewBox.style.display = 'block';
        classifyBtn.disabled = false;
        btnText.textContent = 'Classify Image';
        resultsSection.style.display = 'none';
        chartSection.style.display = 'none';
        errorMessage.style.display = 'none';
    };

    reader.readAsDataURL(selectedFile);
}

// Reset Image
function resetImage() {
    selectedFile = null;
    imageInput.value = '';
    previewBox.style.display = 'none';
    uploadBox.style.display = 'block';
    resultsSection.style.display = 'none';
    chartSection.style.display = 'none';
    errorMessage.style.display = 'none';
    classifyBtn.disabled = true;
    btnText.textContent = 'Upload an image to classify';
}

// Classify Image
async function classifyImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    // Show loading state
    classifyBtn.disabled = true;
    loader.style.display = 'inline-block';
    btnText.textContent = 'Classifying...';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Classification failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`Connection error: ${error.message}`);
    } finally {
        // Reset button state
        classifyBtn.disabled = false;
        loader.style.display = 'none';
        btnText.textContent = 'Classify Image';
    }
}

// Display Results
function displayResults(data) {
    if (!data.predictions || data.predictions.length === 0) {
        showError('No predictions received');
        return;
    }

    // Show results section
    resultsSection.style.display = 'block';

    // Display top prediction
    const topPred = data.predictions[0];
    topClass.textContent = topPred.class_name;
    topConfidence.textContent = topPred.confidence;

    // Display all predictions
    predictionsList.innerHTML = '';
    data.predictions.forEach((pred, index) => {
        const predItem = createPredictionItem(pred, index + 1);
        predictionsList.appendChild(predItem);
    });

    // Display chart
    displayConfidenceChart(data.predictions);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create Prediction Item
function createPredictionItem(prediction, rank) {
    const item = document.createElement('div');
    item.className = 'prediction-item';

    const percentage = (parseFloat(prediction.confidence) * 100).toFixed(1);

    item.innerHTML = `
        <div class="prediction-rank">${rank}</div>
        <div class="prediction-info">
            <div class="prediction-name">${prediction.class_name}</div>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="prediction-percentage">${prediction.confidence}</div>
        </div>
    `;

    return item;
}

// Display Confidence Chart
function displayConfidenceChart(predictions) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');

    // Destroy previous chart if exists
    if (currentChart) {
        currentChart.destroy();
    }

    const labels = predictions.map(p => p.class_name);
    const data = predictions.map(p => parseFloat(p.confidence) * 100);

    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: [
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(168, 85, 247, 0.8)',
                    'rgba(217, 70, 239, 0.8)',
                    'rgba(236, 72, 153, 0.8)',
                ],
                borderColor: [
                    'rgb(99, 102, 241)',
                    'rgb(139, 92, 246)',
                    'rgb(168, 85, 247)',
                    'rgb(217, 70, 239)',
                    'rgb(236, 72, 153)',
                ],
                borderWidth: 2,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        suffix: '%'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    }
                },
                y: {
                    grid: {
                        display: false,
                    }
                }
            }
        }
    });

    chartSection.style.display = 'block';
}

// Show Error
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    resultsSection.style.display = 'block';
}

// Initialize App
window.addEventListener('load', async () => {
    // Check if API is running
    try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('API connection failed:', error);
        showError('Could not connect to API. Make sure the server is running.');
    }
});

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());
