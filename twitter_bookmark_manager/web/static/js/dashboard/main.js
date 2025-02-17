// Main dashboard initialization and state management

document.addEventListener('alpine:init', () => {
    Alpine.data('dashboardApp', () => ({
        // Initialize all required properties with default values
        filters: {
            dateRange: 'all',
            category: 'all'
        },
        stats: {
            total_bookmarks: 0,
            total_categories: 0,
            unique_authors: 0,
            monthly_activity: 0,
            last_updated: null
        },
        categories: [],
        isLoading: true,
        error: null,
        loadingStates: {
            heatmap: true,
            categories: true,
            authors: true,
            topics: true
        },
        errors: {},

        // Initialize component and load data immediately
        init() {
            console.log('🚀 Initializing dashboard...');
            this.loadData();
        },

        async loadData() {
            console.log('📊 Starting to load dashboard data...');
            this.isLoading = true;
            this.error = null;
            this._resetErrors();
            
            try {
                const params = new URLSearchParams({
                    dateRange: this.filters.dateRange,
                    category: this.filters.category
                });
                
                console.log('🔄 Fetching data from /dashboard/api/data');
                const response = await fetch(`/dashboard/api/data?${params}`);
                console.log('📥 Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                console.log('📦 Received data:', result);
                
                if (result.status === 'success' && result.data) {
                    console.log('✅ Data fetch successful, updating state...');
                    // Update stats
                    this.stats = {
                        ...this.stats,
                        ...result.data.statistics
                    };
                    
                    // Log the data we're working with
                    console.log('📊 Stats updated:', this.stats);
                    console.log('🎨 Visualizations data:', result.data.visualizations);
                    
                    // Update filter options before initializing visualizations
                    if (result.data.visualizations?.categories?.distribution) {
                        console.log('🔄 Updating filter options with categories:', 
                            result.data.visualizations.categories.distribution);
                        this.updateFilterOptions(result.data);
                    } else {
                        console.warn('⚠️ No category distribution data found');
                    }
                    
                    // Initialize visualizations with the data
                    console.log('🎨 Initializing visualizations...');
                    await this.initializeVisualizations(result.data);
                } else {
                    console.error('❌ Data fetch failed:', result.message);
                    this.error = result.message || 'Failed to load dashboard data';
                }
            } catch (error) {
                console.error('❌ Error loading dashboard data:', error);
                this.error = 'Error loading dashboard data. Please try again later.';
            } finally {
                console.log('⏱️ Finished loading data. Loading state:', this.isLoading);
                this.isLoading = false;
            }
        },

        async initializeVisualizations(data) {
            console.log('🎨 Starting visualization initialization...', data);
            try {
                const visualizations = data.visualizations;
                
                // Initialize heatmap visualization
                if (visualizations?.heatmap) {
                    console.log('📊 Initializing heatmap with:', visualizations.heatmap);
                    this.loadingStates.heatmap = true;
                    await this._initializeComponent(() => 
                        HeatmapViz.initialize('heatmap-viz', visualizations.heatmap),
                        'heatmap'
                    );
                }
                
                // Initialize category visualization
                if (visualizations?.categories) {
                    console.log('🔄 Initializing categories with:', visualizations.categories);
                    this.loadingStates.categories = true;
                    await this._initializeComponent(() => {
                        CategoryViz.initializePieChart('category-viz', visualizations.categories);
                        CategoryViz.initializeWordCloud('wordcloud-viz', visualizations.categories);
                    }, 'categories');
                }
                
                // Initialize author visualization
                if (visualizations?.authors) {
                    console.log('👥 Initializing authors with:', visualizations.authors);
                    this.loadingStates.authors = true;
                    await this._initializeComponent(() => 
                        AuthorViz.initialize('author-viz', visualizations.authors),
                        'authors'
                    );
                }
                
                // Initialize topic visualization
                if (visualizations?.topics) {
                    console.log('📝 Initializing topics with:', visualizations.topics);
                    this.loadingStates.topics = true;
                    await this._initializeComponent(() => 
                        TopicViz.initialize('topic-viz', visualizations.topics),
                        'topics'
                    );
                }
                
                console.log('✅ All visualizations initialized successfully');
            } catch (error) {
                console.error('❌ Error initializing visualizations:', error);
                this.error = 'Error initializing dashboard visualizations';
            }
        },

        async _initializeComponent(initFn, component) {
            console.log(`🔄 Initializing ${component} component...`);
            try {
                await initFn();
                this.loadingStates[component] = false;
                this.errors[component] = null;
                console.log(`✅ ${component} initialized successfully`);
            } catch (error) {
                console.error(`❌ Error initializing ${component}:`, error);
                this.errors[component] = `Failed to load ${component} visualization`;
                this.loadingStates[component] = false;
            }
        },

        _resetErrors() {
            console.log('🧹 Resetting all errors...');
            this.error = null;
            Object.keys(this.errors).forEach(key => {
                this.errors[key] = null;
            });
        },

        updateFilterOptions(data) {
            console.log('🔄 Updating filter options with data:', data);
            // Update category filter options
            const categorySelect = document.querySelector('#category-filter');
            if (categorySelect && data.visualizations?.categories?.distribution) {
                const categories = data.visualizations.categories.distribution;
                console.log('📋 Categories found:', categories);
                
                categorySelect.innerHTML = '<option value="all">All Categories</option>';
                
                if (Array.isArray(categories)) {
                    categories.forEach(category => {
                        if (category && category.name) {
                            const option = document.createElement('option');
                            option.value = category.name;
                            option.textContent = category.name;
                            categorySelect.appendChild(option);
                        }
                    });
                    console.log('✅ Category filter options updated with:', categories.map(c => c.name));
                } else {
                    console.warn('⚠️ Categories data is not an array:', categories);
                }
            } else {
                console.warn('⚠️ Could not update category filter options:', {
                    hasSelect: !!categorySelect,
                    hasCategories: !!data.visualizations?.categories?.distribution
                });
            }
        },

        async applyFilters() {
            console.log('🔄 Applying filters:', this.filters);
            const params = new URLSearchParams({
                dateRange: this.filters.dateRange,
                category: this.filters.category
            });
            
            // Set loading states for all visualizations
            Object.keys(this.loadingStates).forEach(key => {
                this.loadingStates[key] = true;
            });
            this._resetErrors();
            
            try {
                console.log('📤 Sending filter request...');
                const response = await fetch(`/dashboard/api/data?${params}`);
                console.log('📥 Filter response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('📦 Received filtered data:', data);
                
                if (data.status === 'success') {
                    console.log('✅ Applying filtered data to visualizations...');
                    // Update each visualization with filtered data
                    await Promise.all([
                        this._updateComponent(() => 
                            HeatmapViz.update(data.data.visualizations.heatmap),
                            'heatmap'
                        ),
                        this._updateComponent(() => 
                            CategoryViz.updatePieChart(data.data.visualizations.categories),
                            'categories'
                        ),
                        this._updateComponent(() => 
                            AuthorViz.update(data.data.visualizations.authors),
                            'authors'
                        ),
                        this._updateComponent(() => 
                            TopicViz.update(data.data.visualizations.topics),
                            'topics'
                        )
                    ]);
                    console.log('✅ All visualizations updated with filtered data');
                } else {
                    console.error('❌ Filter application failed:', data.message);
                    this.error = data.message || 'Failed to apply filters';
                }
            } catch (error) {
                console.error('❌ Error applying filters:', error);
                this.error = 'Error updating visualizations with filters';
            }
        },

        async _updateComponent(updateFn, component) {
            console.log(`🔄 Updating ${component} component...`);
            try {
                await updateFn();
                this.loadingStates[component] = false;
                this.errors[component] = null;
                console.log(`✅ ${component} updated successfully`);
            } catch (error) {
                console.error(`❌ Error updating ${component}:`, error);
                this.errors[component] = `Failed to update ${component} visualization`;
                this.loadingStates[component] = false;
            }
        }
    }));
}); 