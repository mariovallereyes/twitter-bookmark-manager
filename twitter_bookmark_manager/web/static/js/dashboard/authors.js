// Author visualizations using ECharts

const AuthorViz = {
    chart: null,
    data: null,
    
    initialize(containerId, data) {
        console.log('Initializing author visualization with:', data);
        this.data = data;
        
        try {
            // Check if container exists
            const container = document.getElementById(containerId);
            if (!container) {
                console.warn(`⚠️ Container ${containerId} not found for author visualization`);
                return;
            }
            
            // Initialize ECharts instance
            this.chart = echarts.init(container);
            
            // Process data for visualization
            const authors = this.data.top_authors.map(author => ({
                name: author.display_name || author.username,
                username: author.username,
                value: author.bookmark_count,
                engagement: author.engagement_metrics
            }));
            
            // Configure the chart
            const option = {
                backgroundColor: '#1f2937', // Matches bg-gray-800
                title: {
                    text: 'Top Authors by Bookmarks',
                    subtext: `Total Authors: ${this.data.total_authors}`,
                    left: 'center',
                    top: 20,
                    textStyle: {
                        color: '#fff'
                    },
                    subtextStyle: {
                        color: '#aaa'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'shadow'
                    },
                    formatter: (params) => {
                        const author = authors.find(a => a.name === params[0].name);
                        return `
                            <strong>${author.name}</strong><br/>
                            @${author.username}<br/>
                            Bookmarks: ${author.value}<br/>
                            Bookmarks/Month: ${author.engagement.bookmarks_per_month.toFixed(2)}<br/>
                            First Bookmark: ${author.engagement.first_bookmark}<br/>
                            Last Bookmark: ${author.engagement.last_bookmark}
                        `;
                    }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    axisLabel: {
                        color: '#fff'
                    },
                    splitLine: {
                        lineStyle: {
                            color: '#374151' // Matches border-gray-700
                        }
                    }
                },
                yAxis: {
                    type: 'category',
                    data: authors.map(a => a.name),
                    axisLabel: {
                        color: '#fff',
                        formatter: value => value
                    },
                    splitLine: {
                        show: false
                    }
                },
                series: [
                    {
                        name: 'Bookmarks',
                        type: 'bar',
                        data: authors.map(a => ({
                            value: a.value,
                            itemStyle: {
                                color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                    { offset: 0, color: '#3182ce' },
                                    { offset: 1, color: '#63b3ed' }
                                ])
                            }
                        })),
                        label: {
                            show: true,
                            position: 'right',
                            color: '#fff'
                        }
                    }
                ]
            };
            
            // Set the configuration and render
            this.chart.setOption(option);
            console.log('✅ Author visualization initialized');
            
            // Handle window resize
            window.addEventListener('resize', () => {
                if (this.chart) {
                    this.chart.resize();
                }
            });
        } catch (error) {
            console.error('❌ Error initializing author visualization:', error);
            throw error;
        }
    },
    
    update(data) {
        console.log('Updating author visualization with:', data);
        this.data = data;
        this.initialize('author-viz', data);
    },
    
    destroy() {
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }
}; 