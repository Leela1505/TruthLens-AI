/* TruthLens AI - Dashboard & Admin Charts Handler */

document.addEventListener('DOMContentLoaded', function() {
    const chartCanvas = document.getElementById('predictionChart');
    if (!chartCanvas) return;

    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const ctx = chartCanvas.getContext('2d');
            
            const isUserChart = chartCanvas.getAttribute('data-type') === 'user';
            const stats = isUserChart ? data.user : data.global;
            
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Real News', 'Fake News'],
                    datasets: [{
                        data: [stats.real, stats.fake],
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.85)',
                            'rgba(239, 68, 68, 0.85)'
                        ],
                        borderColor: [
                            '#10b981',
                            '#ef4444'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#94a3b8',
                                font: {
                                    size: 13,
                                    family: 'Inter'
                                }
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1e293b',
                            titleColor: '#f8fafc',
                            bodyColor: '#cbd5e1',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1
                        }
                    },
                    cutout: '70%'
                }
            });
        })
        .catch(err => console.error('Failed to load chart stats:', err));
});
