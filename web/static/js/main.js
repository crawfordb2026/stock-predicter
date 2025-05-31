document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predictBtn');
    const stockSymbol = document.getElementById('stockSymbol');
    const period = document.getElementById('period');
    
    // Dynamic API URL configuration
    const API_CONFIG = {
        // Try to use the same host as the current page, but with port 5000
        baseUrl: `${window.location.protocol}//${window.location.hostname}:5000`,
        // Fallback to localhost if needed
        fallbackUrl: 'http://127.0.0.1:5000'
    };
    
    let priceChart = null;

    predictBtn.addEventListener('click', async function() {
        const symbol = stockSymbol.value;
        const selectedPeriod = period.value;
        
        // Show loading state
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';
        
        try {
            console.log('Sending prediction request...');
            
            // Try the dynamic API URL first, fallback to localhost
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
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict';
        }
    });

    function updateUI(data) {
        // Update current price
        document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
        
        // Update predicted price
        document.getElementById('predictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
        
        // Update expected change
        const changeElement = document.getElementById('expectedChange');
        const change = data.expected_change;
        changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
        changeElement.className = `value ${change >= 0 ? 'positive' : 'negative'}`;
        
        // Update prediction confidence
        document.getElementById('predictionConfidence').textContent = `${(data.prediction_confidence || 0).toFixed(1)}%`;
        
        // Update technical analysis
        const technicalAnalysis = data.technical_analysis || {};
        document.getElementById('rsi').textContent = `${(technicalAnalysis.rsi || 0).toFixed(2)}`;
        document.getElementById('macd').textContent = `${(technicalAnalysis.macd || 0).toFixed(2)}`;
        document.getElementById('supportLevel').textContent = `$${(technicalAnalysis.support_level || 0).toFixed(2)}`;
        document.getElementById('resistanceLevel').textContent = `$${(technicalAnalysis.resistance_level || 0).toFixed(2)}`;
        
        // Update risk metrics
        const riskMetrics = data.risk_metrics || {};
        console.log('Risk Metrics received:', riskMetrics);  // Debug log
        
        document.getElementById('volatility').textContent = `${(riskMetrics.volatility || 0).toFixed(2)}%`;
        document.getElementById('sharpeRatio').textContent = (riskMetrics.sharpe_ratio || 0).toFixed(2);
        document.getElementById('maxDrawdown').textContent = `${(riskMetrics.max_drawdown || 0).toFixed(2)}%`;
        
        const priceMomentum = riskMetrics.price_momentum || 0;
        const volumeTrend = riskMetrics.volume_trend || 0;
        console.log('Price Momentum:', priceMomentum);  // Debug log
        console.log('Volume Trend:', volumeTrend);  // Debug log
        
        document.getElementById('priceMomentum').textContent = `${priceMomentum > 0 ? '+' : ''}${priceMomentum.toFixed(2)}%`;
        document.getElementById('volumeTrend').textContent = `${volumeTrend > 0 ? '+' : ''}${volumeTrend.toFixed(2)}%`;
        
        // Update chart
        updateChart(data.chart_data);
    }

    function updateChart(chartData) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        if (priceChart) {
            priceChart.destroy();
        }
        
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.dates,
                datasets: [
                    {
                        label: 'Actual Price',
                        data: chartData.actual,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Predicted Price',
                        data: chartData.predicted,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Price Prediction'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }

    // Add model tab switching functionality
    const modelTabs = document.querySelectorAll('.model-tab');
    const modelInfos = document.querySelectorAll('.model-info');

    modelTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and infos
            modelTabs.forEach(t => t.classList.remove('active'));
            modelInfos.forEach(info => info.classList.remove('active'));

            // Add active class to clicked tab
            tab.classList.add('active');

            // Show corresponding model info
            const modelId = tab.getAttribute('data-model');
            document.getElementById(`${modelId}-info`).classList.add('active');
        });
    });
}); 