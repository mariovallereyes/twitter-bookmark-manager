// Category visualizations using ECharts

const CategoryViz = {
    pieChart: null,
    wordCloud: null,
    data: null,
    
    initializePieChart(containerId, data) {
        console.log('Initializing category pie chart with:', data);
        this.data = data;
        
        try {
            // Check if container exists
            const container = document.getElementById(containerId);
            if (!container) {
                console.warn(`⚠️ Container ${containerId} not found for pie chart`);
                return;
            }
            
            // Initialize ECharts instance
            this.pieChart = echarts.init(container);
            
            // Process data for visualization
            const categories = this.data.distribution.map(cat => ({
                name: cat.name,
                value: cat.count,
                percentage: cat.percentage
            }));
            
            // Configure the chart
            const option = {
                backgroundColor: '#1f2937', // Matches bg-gray-800
                title: {
                    text: `Total Categories: ${this.data.total_categories}`,
                    subtext: `Total Bookmarks: ${this.data.total_bookmarks}`,
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
                    trigger: 'item',
                    formatter: (params) => {
                        const category = categories.find(c => c.name === params.name);
                        return `
                            <strong>${params.name}</strong><br/>
                            Bookmarks: ${params.value}<br/>
                            Percentage: ${category.percentage}%
                        `;
                    }
                },
                legend: {
                    type: 'scroll',
                    orient: 'vertical',
                    right: 10,
                    top: 20,
                    bottom: 20,
                    textStyle: {
                        color: '#fff'
                    }
                },
                series: [
                    {
                        name: 'Categories',
                        type: 'pie',
                        radius: ['40%', '70%'],
                        center: ['40%', '50%'],
                        avoidLabelOverlap: true,
                        itemStyle: {
                            borderRadius: 10,
                            borderColor: '#1f2937',
                            borderWidth: 2
                        },
                        label: {
                            show: false,
                            position: 'center'
                        },
                        emphasis: {
                            label: {
                                show: true,
                                fontSize: 20,
                                fontWeight: 'bold',
                                formatter: (params) => {
                                    const category = categories.find(c => c.name === params.name);
                                    return `${params.name}\n${category.percentage}%`;
                                }
                            }
                        },
                        data: categories
                    }
                ]
            };
            
            // Set the configuration and render
            this.pieChart.setOption(option);
            console.log('✅ Pie chart initialized');
            
            // Handle window resize
            window.addEventListener('resize', () => {
                if (this.pieChart) {
                    this.pieChart.resize();
                }
            });
            
            // Add click handler for category selection
            this.pieChart.on('click', (params) => {
                if (params.seriesType === 'pie') {
                    // Update word cloud for selected category
                    this.updateWordCloud(params.name);
                }
            });
        } catch (error) {
            console.error('❌ Error initializing pie chart:', error);
            throw error;
        }
    },
    
    initializeWordCloud(containerId, data) {
        console.log('Initializing word cloud with:', data);
        
        try {
            // Check if container exists
            const container = document.getElementById(containerId);
            if (!container) {
                console.warn(`⚠️ Container ${containerId} not found for word cloud`);
                return;
            }
            
            this.wordCloud = echarts.init(container);
            
            // Initialize with empty state - will be updated on category selection
            const option = {
                backgroundColor: '#1f2937', // Matches bg-gray-800
                title: {
                    text: 'Select a category to view word cloud',
                    left: 'center',
                    top: 20,
                    textStyle: {
                        color: '#fff'
                    }
                }
            };
            
            this.wordCloud.setOption(option);
            console.log('✅ Word cloud initialized');
            
            // Handle window resize
            window.addEventListener('resize', () => {
                if (this.wordCloud) {
                    this.wordCloud.resize();
                }
            });
        } catch (error) {
            console.error('❌ Error initializing word cloud:', error);
            throw error;
        }
    },
    
    async updateWordCloud(category) {
        try {
            // Fetch word cloud data for selected category
            const response = await fetch(`/api/dashboard/categories/${category}/wordcloud`);
            const data = await response.json();
            
            if (data.status === 'success') {
                const option = {
                    backgroundColor: '#1a1a1a',
                    title: {
                        text: `Word Cloud: ${category}`,
                        subtext: `${data.data.metadata.total_bookmarks} bookmarks, ${data.data.metadata.unique_words} unique words`,
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
                        show: true,
                        formatter: (params) => `${params.name}: ${params.value} occurrences`
                    },
                    series: [{
                        type: 'wordCloud',
                        shape: 'circle',
                        left: 'center',
                        top: 'center',
                        width: '90%',
                        height: '90%',
                        right: null,
                        bottom: null,
                        sizeRange: [12, 60],
                        rotationRange: [-45, 45],
                        rotationStep: 15,
                        gridSize: 8,
                        drawOutOfBound: false,
                        textStyle: {
                            fontFamily: 'Inter',
                            fontWeight: 'normal',
                            color: function () {
                                return 'rgb(' + [
                                    Math.round(Math.random() * 100 + 155),
                                    Math.round(Math.random() * 100 + 155),
                                    Math.round(Math.random() * 100 + 155)
                                ].join(',') + ')';
                            }
                        },
                        emphasis: {
                            textStyle: {
                                fontWeight: 'bold',
                                shadowBlur: 10,
                                shadowColor: '#333'
                            }
                        },
                        data: data.data.word_frequencies
                    }]
                };
                
                this.wordCloud.setOption(option);
            }
        } catch (error) {
            console.error('Error updating word cloud:', error);
        }
    },
    
    updatePieChart(data) {
        console.log('Updating pie chart with:', data);
        this.data = data;
        this.initializePieChart('category-viz', data);
    },
    
    destroy() {
        if (this.pieChart) {
            this.pieChart.dispose();
            this.pieChart = null;
        }
        if (this.wordCloud) {
            this.wordCloud.dispose();
            this.wordCloud = null;
        }
    }
}; 