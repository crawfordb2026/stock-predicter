document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predictBtn');
    const stockSymbol = document.getElementById('stockSymbol');
    const period = document.getElementById('period');
    
    // Figure out the right API URL based on where we're running
    const API_CONFIG = {
        // Use whatever URL we're currently on
        baseUrl: `${window.location.protocol}//${window.location.host}`,
        // Backup plan for local development
        fallbackUrl: 'http://127.0.0.1:8080'
    };
    
    let priceChart = null;

    predictBtn.addEventListener('click', async function() {
        const symbol = stockSymbol.value;
        const selectedPeriod = period.value;
        
        // Show that we're working on it
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';
        
        try {
            console.log('Sending prediction request...');
            
            // Try our main URL first, then fallback for local dev
            let apiUrl = API_CONFIG.baseUrl;
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                apiUrl = API_CONFIG.fallbackUrl;
            }
            
            console.log(`Making request to: ${apiUrl}/predict`);
            
            const response = await fetch(`${apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    period: selectedPeriod
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Network response was not ok');
            }
            
            const data = await response.json();
            console.log('Received prediction data:', data);
            updateUI(data);
        } catch (error) {
            console.error('Error details:', error);
            alert(`Error: ${error.message}`);
        } finally {
            // Reset the button no matter what happened
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict';
        }
    });

    function updateUI(data) {
        // Fill in the current price
        document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
        
        // Show our prediction
        document.getElementById('predictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
        
        // Show the expected change with proper color coding
        const changeElement = document.getElementById('expectedChange');
        const change = data.expected_change;
        changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
        changeElement.className = `value ${change >= 0 ? 'positive' : 'negative'}`;
        
        // Show how confident we are
        const confidence = data.prediction_confidence || 0;
        document.getElementById('predictionConfidence').textContent = `${confidence.toFixed(1)}%`;
        
        // Fill in all the technical analysis stuff
        const technicalAnalysis = data.technical_analysis || {};
        document.getElementById('rsi').textContent = `${(technicalAnalysis.rsi || 0).toFixed(2)}`;
        document.getElementById('macd').textContent = `${(technicalAnalysis.macd || 0).toFixed(2)}`;
        document.getElementById('supportLevel').textContent = `$${(technicalAnalysis.support_level || 0).toFixed(2)}`;
        document.getElementById('resistanceLevel').textContent = `$${(technicalAnalysis.resistance_level || 0).toFixed(2)}`;
        
        // Fill in the risk metrics
        const riskMetrics = data.risk_metrics || {};
        console.log('Risk Metrics received:', riskMetrics);  // For debugging
        
        document.getElementById('volatility').textContent = `${(riskMetrics.volatility || 0).toFixed(2)}%`;
        document.getElementById('sharpeRatio').textContent = (riskMetrics.sharpe_ratio || 0).toFixed(2);
        document.getElementById('maxDrawdown').textContent = `${(riskMetrics.max_drawdown || 0).toFixed(2)}%`;
        
        const priceMomentum = riskMetrics.price_momentum || 0;
        const volumeTrend = riskMetrics.volume_trend || 0;
        console.log('Price Momentum:', priceMomentum);  // Debug info
        console.log('Volume Trend:', volumeTrend);  // Debug info
        
        document.getElementById('priceMomentum').textContent = `${priceMomentum > 0 ? '+' : ''}${priceMomentum.toFixed(2)}%`;
        document.getElementById('volumeTrend').textContent = `${volumeTrend > 0 ? '+' : ''}${volumeTrend.toFixed(2)}%`;
        
        // Update the price chart
        updateChart(data.chart_data);
    }

    function updateChart(chartData) {
        // Simple fallback chart using HTML/CSS instead of Chart.js
        const chartContainer = document.getElementById('priceChart');
        
        // Clear any existing chart
        chartContainer.innerHTML = '';
        
        // Create a simple line chart using divs and CSS
        const actualPrices = chartData.actual.filter(price => price !== null);
        const dates = chartData.dates.slice(0, actualPrices.length);
        
        if (actualPrices.length === 0) {
            chartContainer.innerHTML = '<p>No chart data available</p>';
            return;
        }
        
        // Find min and max for scaling
        const minPrice = Math.min(...actualPrices);
        const maxPrice = Math.max(...actualPrices);
        const priceRange = maxPrice - minPrice;
        
        // Add predicted price if available
        const predictedPrice = chartData.predicted[chartData.predicted.length - 1];
        const allPrices = [...actualPrices];
        if (predictedPrice !== null) {
            allPrices.push(predictedPrice);
        }
        
        // Create chart HTML
        let chartHTML = `
            <div class="simple-chart">
                <div class="chart-title">Stock Price Chart (Recent Trading Days)</div>
                <div class="chart-container">
                    <div class="price-line">
        `;
        
        // Add price points
        actualPrices.forEach((price, index) => {
            const x = (index / (actualPrices.length - 1)) * 100;
            const y = ((price - minPrice) / priceRange) * 70; // 70% of height for data, inverted
            chartHTML += `<div class="price-point actual" style="left: ${x}%; bottom: ${y + 10}%;" title="$${price.toFixed(2)}"></div>`;
        });
        
        // Add predicted price point if available
        if (predictedPrice !== null) {
            const y = ((predictedPrice - minPrice) / priceRange) * 70; // Inverted positioning
            chartHTML += `<div class="price-point predicted" style="left: 100%; bottom: ${y + 10}%;" title="Predicted: $${predictedPrice.toFixed(2)}"></div>`;
        }
        
        chartHTML += `
                    </div>
                    <div class="chart-labels">
                        <div class="price-labels">
                            <div class="price-label">$${maxPrice.toFixed(2)}</div>
                            <div class="price-label">$${minPrice.toFixed(2)}</div>
                        </div>
                        <div class="date-labels">
                            <span>Start</span>
                            <span>Latest</span>
                            ${predictedPrice !== null ? '<span class="predicted-label">Prediction</span>' : ''}
                        </div>
                    </div>
                </div>
                <div class="chart-legend">
                    <span class="legend-item"><span class="legend-color actual"></span> Historical Prices</span>
                    ${predictedPrice !== null ? '<span class="legend-item"><span class="legend-color predicted"></span> Prediction</span>' : ''}
                </div>
            </div>
        `;
        
        chartContainer.innerHTML = chartHTML;
    }

    // Handle the model info tabs switching
    const modelTabs = document.querySelectorAll('.model-tab');
    const modelInfos = document.querySelectorAll('.model-info');

    modelTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Clear all active states
            modelTabs.forEach(t => t.classList.remove('active'));
            modelInfos.forEach(info => info.classList.remove('active'));

            // Set the clicked tab as active
            tab.classList.add('active');

            // Show the corresponding model info
            const modelId = tab.getAttribute('data-model');
            document.getElementById(`${modelId}-info`).classList.add('active');
        });
    });
}); 