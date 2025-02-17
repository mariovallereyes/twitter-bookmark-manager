// Topic visualizations using D3.js

const TopicViz = {
    svg: null,
    simulation: null,
    data: null,
    width: 800,
    height: 600,
    
    initialize(containerId, data) {
        console.log('Initializing topic visualization with:', data);
        this.data = data;
        
        try {
            // Check if container exists
            const container = document.getElementById(containerId);
            if (!container) {
                console.warn(`⚠️ Container ${containerId} not found for topic visualization`);
                return;
            }
            
            // Check if we have topic data
            if (!this.data.topics || this.data.topics.length === 0) {
                console.log('No topic data available, showing placeholder');
                this._showPlaceholder(container);
                return;
            }
            
            // Clear previous visualization
            container.innerHTML = '';
            
            // Set dimensions
            this.width = container.clientWidth;
            this.height = container.clientHeight;
            
            // Create SVG
            this.svg = d3.select(`#${containerId}`)
                .append('svg')
                .attr('width', this.width)
                .attr('height', this.height)
                .attr('class', 'bg-gray-800'); // Match container background
                
            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.5, 3])
                .on('zoom', (event) => {
                    this.svg.select('g').attr('transform', event.transform);
                });
                
            this.svg.call(zoom);
            
            // Create the visualization
            this._createForceGraph();
            console.log('✅ Topic visualization initialized');
            
            // Handle window resize
            window.addEventListener('resize', () => {
                this.width = container.clientWidth;
                this.height = container.clientHeight;
                this.svg
                    .attr('width', this.width)
                    .attr('height', this.height);
                if (this.simulation) {
                    this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
                    this.simulation.alpha(0.3).restart();
                }
            });
        } catch (error) {
            console.error('❌ Error initializing topic visualization:', error);
            throw error;
        }
    },
    
    _showPlaceholder(container) {
        // Clear previous content
        container.innerHTML = '';
        
        // Create placeholder message
        const placeholder = document.createElement('div');
        placeholder.className = 'flex flex-col items-center justify-center h-full text-gray-400';
        placeholder.innerHTML = `
            <svg class="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p class="text-lg font-medium">No Topics Available</p>
            <p class="text-sm mt-2">Topics will appear here once enough data is collected</p>
            <p class="text-xs mt-4">Last analyzed: ${this.data.metadata?.last_updated || 'Never'}</p>
        `;
        
        container.appendChild(placeholder);
    },
    
    _createForceGraph() {
        try {
            // Process data
            const nodes = this.data.topics.map(topic => ({
                id: topic.name,
                value: topic.value,
                sentiment: topic.sentiment,
                categories: topic.categories
            }));
            
            const links = this.data.relationships.map(rel => ({
                source: rel.source,
                target: rel.target,
                value: rel.strength
            }));
            
            // Create force simulation
            this.simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id))
                .force('charge', d3.forceManyBody().strength(-100))
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.value) * 2));
                
            // Create container group
            const g = this.svg.append('g');
            
            // Create links
            const link = g.append('g')
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('stroke', '#4b5563') // text-gray-600
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.sqrt(d.value));
                
            // Create nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(nodes)
                .join('circle')
                .attr('r', d => Math.sqrt(d.value) * 2)
                .attr('fill', d => {
                    // Color based on sentiment
                    const sentiment = d.sentiment;
                    if (sentiment > 0.3) return '#10b981'; // text-emerald-500
                    if (sentiment < -0.3) return '#ef4444'; // text-red-500
                    return '#3b82f6'; // text-blue-500
                })
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5)
                .call(this._drag(this.simulation));
                
            // Add labels
            const label = g.append('g')
                .selectAll('text')
                .data(nodes)
                .join('text')
                .text(d => d.id)
                .attr('font-size', '10px')
                .attr('fill', '#fff')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em');
                
            // Add tooltips
            node.append('title')
                .text(d => `${d.id}\nValue: ${d.value}\nSentiment: ${d.sentiment.toFixed(2)}\nCategories: ${d.categories.join(', ')}`);
                
            // Update positions on tick
            this.simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                    
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                    
                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
        } catch (error) {
            console.error('❌ Error creating force graph:', error);
            throw error;
        }
    },
    
    _drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    },
    
    update(data) {
        console.log('Updating topic visualization with:', data);
        this.data = data;
        this.initialize('topic-viz', data);
    },
    
    destroy() {
        if (this.simulation) {
            this.simulation.stop();
            this.simulation = null;
        }
        if (this.svg) {
            this.svg.remove();
            this.svg = null;
        }
    }
}; 