// Heatmap visualization using Plotly.js

const HeatmapViz = {
    chart: null,
    data: null,
    
    initialize(containerId, data) {
        console.log('Initializing heatmap with:', data);
        this.data = data;
        
        // Create all visualizations with error handling
        try {
            // Calendar heatmap
            const heatmapContainer = document.getElementById(containerId);
            if (heatmapContainer) {
                this._createCalendarHeatmap(containerId);
                console.log('✅ Calendar heatmap initialized');
            } else {
                console.warn(`⚠️ Container ${containerId} not found for calendar heatmap`);
            }
            
            // Hourly distribution
            const hourlyContainer = document.getElementById(`${containerId}-hourly`);
            if (hourlyContainer) {
                this._createHourlyDistribution(`${containerId}-hourly`);
                console.log('✅ Hourly distribution initialized');
            } else {
                console.warn(`⚠️ Container ${containerId}-hourly not found for hourly distribution`);
            }
            
            // Activity trends
            const trendsContainer = document.getElementById('activity-trends-viz');
            if (trendsContainer) {
                this._createActivityTrends('activity-trends-viz');
                console.log('✅ Activity trends initialized');
            } else {
                console.warn('⚠️ Container activity-trends-viz not found for activity trends');
            }
            
            // Handle window resize for all charts
            window.addEventListener('resize', () => {
                if (heatmapContainer) {
                    Plotly.relayout(containerId, {
                        width: heatmapContainer.clientWidth
                    });
                }
                if (hourlyContainer) {
                    Plotly.relayout(`${containerId}-hourly`, {
                        width: hourlyContainer.clientWidth
                    });
                }
                if (trendsContainer) {
                    Plotly.relayout('activity-trends-viz', {
                        width: trendsContainer.clientWidth
                    });
                }
            });
        } catch (error) {
            console.error('❌ Error initializing heatmap visualizations:', error);
            throw error;
        }
    },
    
    _createCalendarHeatmap(containerId) {
        // Process daily activity data
        const dates = this.data.daily_activity.map(d => d.date);
        const counts = this.data.daily_activity.map(d => d.count);
        
        // Calculate color scale range
        const maxCount = Math.max(...counts);
        const colorScale = [
            [0, '#1a237e'],      // Dark blue for no activity
            [0.3, '#303f9f'],    // Medium blue for low activity
            [0.7, '#3f51b5'],    // Light blue for medium activity
            [1, '#5c6bc0']       // Very light blue for high activity
        ];
        
        const layout = {
            title: {
                text: 'Daily Activity Calendar',
                font: {
                    size: 14,
                    color: '#ffffff'
                }
            },
            paper_bgcolor: '#111827',  // Matches bg-gray-900
            plot_bgcolor: '#111827',
            font: {
                color: '#ffffff',
                size: 11
            },
            height: 180,
            width: document.getElementById(containerId).clientWidth,
            margin: {
                l: 40,
                r: 40,
                t: 30,
                b: 20
            },
            xaxis: {
                showgrid: false,
                tickangle: 0,
                tickfont: {
                    size: 10
                },
                tickformat: '%b %Y',
                nticks: 6,
                tickcolor: '#666'
            },
            yaxis: {
                showgrid: false,
                tickfont: {
                    size: 10
                },
                tickcolor: '#666'
            },
            coloraxis: {
                colorbar: {
                    title: {
                        text: 'Bookmarks',
                        font: {
                            size: 10,
                            color: '#ffffff'
                        }
                    },
                    tickfont: {
                        size: 9,
                        color: '#ffffff'
                    },
                    len: 0.5
                }
            }
        };

        const trace = {
            type: 'heatmap',
            x: dates,
            y: ['Activity'],
            z: [counts],
            colorscale: colorScale,
            showscale: true,
            hoverongaps: false,
            hovertemplate: 'Date: %{x}<br>Bookmarks: %{z}<extra></extra>'
        };
        
        Plotly.newPlot(containerId, [trace], layout, {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false
        });
    },
    
    _createHourlyDistribution(containerId) {
        // Process hourly distribution data
        const hours = this.data.hourly_distribution.map(d => d.hour);
        const hourCounts = this.data.hourly_distribution.map(d => d.count);
        
        const layout = {
            title: {
                text: 'Hourly Distribution',
                font: {
                    size: 14,
                    color: '#ffffff'
                }
            },
            paper_bgcolor: '#111827',
            plot_bgcolor: '#111827',
            font: {
                color: '#ffffff',
                size: 11
            },
            height: 180,
            width: document.getElementById(containerId).clientWidth,
            margin: {
                l: 40,
                r: 40,
                t: 30,
                b: 40
            },
            xaxis: {
                title: {
                    text: 'Hour of Day',
                    font: {
                        size: 10
                    },
                    standoff: 10
                },
                showgrid: true,
                gridcolor: '#374151',
                tickcolor: '#666',
                range: [-0.5, 23.5],
                tickfont: {
                    size: 10
                },
                dtick: 3,
                tickformat: '%H:00'
            },
            yaxis: {
                title: {
                    text: 'Bookmark Count',
                    font: {
                        size: 10
                    },
                    standoff: 10
                },
                showgrid: true,
                gridcolor: '#374151',
                tickcolor: '#666',
                tickfont: {
                    size: 10
                }
            },
            bargap: 0.1
        };
        
        const trace = {
            type: 'bar',
            x: hours,
            y: hourCounts,
            marker: {
                color: '#3f51b5',
                opacity: 0.8
            },
            hovertemplate: 'Hour: %{x}:00<br>Bookmarks: %{y}<extra></extra>'
        };
        
        Plotly.newPlot(containerId, [trace], layout, {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false
        });
    },

    _createActivityTrends(containerId) {
        // Calculate 7-day moving average
        const dates = this.data.daily_activity.map(d => d.date);
        const counts = this.data.daily_activity.map(d => d.count);
        const movingAvg = this._calculateMovingAverage(counts, 7);
        
        const layout = {
            title: {
                text: 'Activity Trends (7-day average)',
                font: {
                    size: 14,
                    color: '#ffffff'
                }
            },
            paper_bgcolor: '#111827',
            plot_bgcolor: '#111827',
            font: {
                color: '#ffffff',
                size: 11
            },
            height: 180,
            width: document.getElementById(containerId).clientWidth,
            margin: {
                l: 40,
                r: 40,
                t: 30,
                b: 20
            },
            xaxis: {
                showgrid: true,
                gridcolor: '#374151',
                tickcolor: '#666',
                tickfont: {
                    size: 10
                },
                tickformat: '%b %Y',
                nticks: 6
            },
            yaxis: {
                title: {
                    text: 'Avg. Bookmarks',
                    font: {
                        size: 10
                    },
                    standoff: 10
                },
                showgrid: true,
                gridcolor: '#374151',
                tickcolor: '#666',
                tickfont: {
                    size: 10
                }
            },
            showlegend: false
        };
        
        const trace = {
            type: 'scatter',
            x: dates,
            y: movingAvg,
            mode: 'lines',
            line: {
                color: '#5c6bc0',
                width: 2
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(92, 107, 192, 0.2)',
            hovertemplate: 'Date: %{x}<br>Avg. Bookmarks: %{y:.1f}<extra></extra>'
        };
        
        Plotly.newPlot(containerId, [trace], layout, {
            responsive: true,
            displayModeBar: false,
            scrollZoom: false
        });
    },

    _calculateMovingAverage(data, window) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - window + 1);
            const end = i + 1;
            const slice = data.slice(start, end);
            const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
            result.push(avg);
        }
        return result;
    },
    
    update(data) {
        console.log('Updating activity visualizations with:', data);
        this.data = data;
        
        // Update all visualizations
        this._createCalendarHeatmap('heatmap-viz');
        this._createHourlyDistribution('heatmap-viz-hourly');
        this._createActivityTrends('activity-trends-viz');
    },
    
    destroy() {
        // Clean up Plotly elements
        if (this.chart) {
            Plotly.purge('heatmap-viz');
            Plotly.purge('heatmap-viz-hourly');
            Plotly.purge('activity-trends-viz');
            this.chart = null;
        }
    }
}; 