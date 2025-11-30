// Bitcoin Price Predictor 2025 JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log("Bitcoin Predictor 2025 Initialized");
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Fetch current Bitcoin price
    fetchCurrentBitcoinPrice();
    
    // Refresh price every 30 seconds
    setInterval(fetchCurrentBitcoinPrice, 30000);

    // Form submission handler
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            generatePrediction();
        });
    }

    // Add real-time date to generation time
    updateGenerationTime();
});

// Global variables
let currentBitcoinPrice = 45280.50;
let predictionChart = null;

function updateGenerationTime() {
    const now = new Date();
    const options = { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    const generationTimeElement = document.getElementById('generationTime');
    if (generationTimeElement) {
        generationTimeElement.textContent = now.toLocaleString('en-US', options);
    }
}

async function fetchCurrentBitcoinPrice() {
    try {
        console.log("Fetching current Bitcoin price...");
        const response = await fetch('/api/current_price');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();

        if (data.success) {
            currentBitcoinPrice = data.price;
            const currentPriceElement = document.getElementById('currentPrice');
            const marketCapElement = document.getElementById('marketCap');
            const aiStatusElement = document.getElementById('aiStatus');
            
            if (currentPriceElement) {
                currentPriceElement.textContent = `$${data.price.toLocaleString()}`;
            }
            
            if (marketCapElement) {
                marketCapElement.textContent = data.market_cap;
            }
            
            // Format and color the price change
            const changeElement = document.getElementById('priceChange');
            if (changeElement) {
                const change = data.change_24h;
                const changeText = change >= 0 ? `+${change}%` : `${change}%`;
                
                changeElement.textContent = changeText;
                changeElement.className = change >= 0 ? 'fw-bold fs-5 price-up' : 'fw-bold fs-5 price-down';
            }
            
            // Update AI status
            if (aiStatusElement) {
                aiStatusElement.textContent = 'Ready';
                aiStatusElement.className = 'fw-bold fs-5 text-success';
            }
            
            console.log("Current price updated successfully:", data.price);
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Error fetching current price:', error);
        // Fallback to mock data
        currentBitcoinPrice = 45280.50;
        const currentPriceElement = document.getElementById('currentPrice');
        const changeElement = document.getElementById('priceChange');
        const marketCapElement = document.getElementById('marketCap');
        const aiStatusElement = document.getElementById('aiStatus');
        
        if (currentPriceElement) currentPriceElement.textContent = '$45,280.50';
        if (changeElement) {
            changeElement.textContent = '+2.5%';
            changeElement.className = 'fw-bold fs-5 price-up';
        }
        if (marketCapElement) marketCapElement.textContent = '$885B';
        if (aiStatusElement) {
            aiStatusElement.textContent = 'Ready (Demo)';
            aiStatusElement.className = 'fw-bold fs-5 text-warning';
        }
    }
}

async function generatePrediction() {
    const form = document.getElementById('predictionForm');
    if (!form) {
        console.error('Prediction form not found');
        return;
    }
    
    const formData = new FormData(form);
    
    const predictionData = {
        model_type: formData.get('model_type'),
        prediction_days: formData.get('prediction_days')
    };

    console.log("Generating prediction with data:", predictionData);

    // Show loading spinner and hide results
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const predictButton = document.getElementById('predictButton');
    
    if (loadingSpinner) loadingSpinner.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
    if (predictButton) {
        predictButton.disabled = true;
        predictButton.innerHTML = '<i class="fas fa-cog fa-spin me-2"></i>Processing...';
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Prediction response:", result);

        if (result.success) {
            displayResults(result);
        } else {
            showError('Prediction failed: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Network error: ' + error.message);
    } finally {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        if (predictButton) {
            predictButton.disabled = false;
            predictButton.innerHTML = '<i class="fas fa-rocket me-2"></i>Generate AI Prediction';
        }
    }
}

function displayResults(result) {
    console.log("Displaying results:", result);
    
    // Update generation time
    updateGenerationTime();
    
    // Update model and period info
    const modelUsedElement = document.getElementById('modelUsed');
    const periodUsedElement = document.getElementById('periodUsed');
    
    if (modelUsedElement) modelUsedElement.textContent = result.model_used + ' Neural Network';
    if (periodUsedElement) periodUsedElement.textContent = result.prediction_days;

    // Update summary statistics
    const avgPriceElement = document.getElementById('avgPrice');
    const minPriceElement = document.getElementById('minPrice');
    const maxPriceElement = document.getElementById('maxPrice');
    const trendIndicatorElement = document.getElementById('trendIndicator');
    
    if (avgPriceElement) avgPriceElement.textContent = `$${result.summary.avg_price.toLocaleString()}`;
    if (minPriceElement) minPriceElement.textContent = `$${result.summary.min_price.toLocaleString()}`;
    if (maxPriceElement) maxPriceElement.textContent = `$${result.summary.max_price.toLocaleString()}`;

    // Calculate and display trend
    if (trendIndicatorElement) {
        const priceChangePercent = ((result.summary.last_price - currentBitcoinPrice) / currentBitcoinPrice * 100).toFixed(2);
        const trend = result.summary.last_price > currentBitcoinPrice ? '↗️' : 
                     result.summary.last_price < currentBitcoinPrice ? '↘️' : '→';
        trendIndicatorElement.textContent = `${trend} ${Math.abs(priceChangePercent)}%`;
    }

    // Update AI insights
    updateAIInsights(result);

    // Create chart
    createPredictionChart(result);

    // Populate predictions table
    populatePredictionsTable(result);

    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.style.opacity = '0';
        
        // Animate fade in
        setTimeout(() => {
            resultsSection.style.opacity = '1';
            resultsSection.style.transition = 'opacity 0.5s ease-in';
        }, 100);

        // Scroll to results
        resultsSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
    
    console.log("Results displayed successfully");
}

function updateAIInsights(result) {
    const insightsElement = document.getElementById('aiInsights');
    if (!insightsElement) return;
    
    const trend = result.summary.last_price > currentBitcoinPrice ? 'bullish' : 
                 result.summary.last_price < currentBitcoinPrice ? 'bearish' : 'neutral';
    
    const priceChangePercent = ((result.summary.last_price - currentBitcoinPrice) / currentBitcoinPrice * 100).toFixed(2);
    const volatility = ((result.summary.max_price - result.summary.min_price) / result.summary.avg_price * 100).toFixed(1);
    
    let insightText = `The ${result.model_used} neural network predicts a ${trend} trend `;
    insightText += `(${priceChangePercent}% change) from current price over the next ${result.prediction_days} days. `;
    insightText += `Expected price range: $${result.summary.min_price.toLocaleString()} - $${result.summary.max_price.toLocaleString()}. `;
    insightText += `Market volatility is projected at ${volatility}%. `;
    
    if (trend === 'bullish') {
        insightText += `The AI detects positive momentum patterns similar to historical bull markets.`;
    } else if (trend === 'bearish') {
        insightText += `The model identifies resistance levels that may limit upward movement.`;
    } else {
        insightText += `The market shows consolidation patterns with balanced buying and selling pressure.`;
    }
    
    insightsElement.textContent = insightText;
}

function createPredictionChart(result) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (predictionChart) {
        predictionChart.destroy();
    }

    // Add current price as the first data point
    const allDates = ['Now'].concat(result.predictions.map(p => {
        const date = new Date(p.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }));
    
    const allPrices = [currentBitcoinPrice].concat(result.predictions.map(p => p.price));

    // Calculate trend for point colors
    const pointBackgroundColors = allPrices.map((price, index) => {
        if (index === 0) return '#007bff'; // Blue for current price
        const prevPrice = allPrices[index - 1];
        return price > prevPrice ? '#28a745' : price < prevPrice ? '#dc3545' : '#6c757d';
    });

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: [{
                label: 'Bitcoin Price (USD)',
                data: allPrices,
                borderColor: '#f7931a',
                backgroundColor: 'rgba(247, 147, 26, 0.1)',
                borderWidth: 4,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: pointBackgroundColors,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Bitcoin Price Forecast - Current vs Prediction',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    padding: 20
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            if (context.dataIndex === 0) {
                                return `Current: $${context.parsed.y.toLocaleString()}`;
                            } else {
                                return `Predicted: $${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Timeline',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price (USD)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index',
            }
        }
    });
}

function populatePredictionsTable(result) {
    const tbody = document.getElementById('predictionsBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';

    // Add current price as first row
    const currentRow = document.createElement('tr');
    currentRow.className = 'table-primary';
    currentRow.innerHTML = `
        <td class="fw-bold">Current</td>
        <td class="fw-bold fs-6">$${currentBitcoinPrice.toLocaleString()}</td>
        <td class="price-neutral fw-bold">0.00%</td>
        <td class="price-neutral fw-bold">⚡</td>
    `;
    tbody.appendChild(currentRow);

    result.predictions.forEach((prediction, index) => {
        const row = document.createElement('tr');
        
        // Calculate trend
        const priceChangeFromCurrent = ((prediction.price - currentBitcoinPrice) / currentBitcoinPrice * 100);
        let trendIcon = '';
        let trendClass = '';
        
        if (priceChangeFromCurrent > 5) {
            trendIcon = '🚀 📈';
            trendClass = 'price-up';
        } else if (priceChangeFromCurrent > 2) {
            trendIcon = '📈 ↗️';
            trendClass = 'price-up';
        } else if (priceChangeFromCurrent < -5) {
            trendIcon = '⚠️ 📉';
            trendClass = 'price-down';
        } else if (priceChangeFromCurrent < -2) {
            trendIcon = '📉 ↘️';
            trendClass = 'price-down';
        } else {
            trendIcon = '➡️';
            trendClass = 'price-neutral';
        }

        row.innerHTML = `
            <td class="fw-bold">${prediction.date}</td>
            <td class="fw-bold fs-6">$${prediction.price.toLocaleString()}</td>
            <td class="${trendClass} fw-bold">
                ${priceChangeFromCurrent >= 0 ? '+' : ''}${priceChangeFromCurrent.toFixed(2)}%
            </td>
            <td class="${trendClass} fw-bold">${trendIcon}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function showError(message) {
    // Remove any existing alerts
    const existingAlerts = document.querySelectorAll('.alert.alert-danger');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
    alertDiv.innerHTML = `
        <strong><i class="fas fa-exclamation-triangle me-2"></i>Error!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    const resultsSection = document.getElementById('resultsSection');
    if (container) {
        if (resultsSection && resultsSection.style.display !== 'none') {
            container.insertBefore(alertDiv, resultsSection);
        } else {
            container.appendChild(alertDiv);
        }
    }
    
    // Auto remove after 8 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 8000);
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter or Cmd+Enter to generate prediction
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generatePrediction();
    }
});