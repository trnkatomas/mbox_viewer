
        // --- D3 CHARTING LOGIC ---

        /**
         * Renders the D3.js charts in the statistics dashboard.
         */
        async function loadStats() {
            // Dummy Data

            const volumeData = await d3.json("/api/stats/data/dates_size");
            //console log(volumeData_new);

            /*const volumeData = [
                { date: new Date('2025-10-24'), count: 55 },
                { date: new Date('2025-10-25'), count: 68 },
                { date: new Date('2025-10-26'), count: 80 },
                { date: new Date('2025-10-27'), count: 72 },
                { date: new Date('2025-10-28'), count: 95 },
                { date: new Date('2025-10-29'), count: 110 },
                { date: new Date('2025-10-30'), count: 105 }
            ];*/

            const senderData = [
                { domain: "corp.com", count: 350 },
                { domain: "client.io", count: 120 },
                { domain: "spam.net", count: 50 },
                { domain: "hr.com", count: 80 },
                { domain: "other", count: 400 },
            ];



            // --- 1. Daily Email Volume (Line Chart) ---
            const drawLineChart = (data, selector) => {
                const container = document.querySelector(selector);
                if (!container) return;

                // Clear previous tooltips
                d3.selectAll(".tooltip").remove();

                const margin = { top: 10, right: 30, bottom: 40, left: 60 };
                const width = container.clientWidth - margin.left - margin.right;
                const height = Math.max(300, 400 - margin.top - margin.bottom);

                d3.select(selector).select("svg").remove();

                const svg = d3.select(selector)
                    .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                        .attr("transform", `translate(${margin.left},${margin.top})`);

                const x = d3.scaleTime()
                    .domain(d3.extent(data, d => d.date))
                    .range([0, width]);
                svg.append("g")
                    .attr("transform", `translate(0,${height})`)
                    .call(d3.axisBottom(x).ticks(d3.timeDay.every(1)).tickFormat(d3.timeFormat("%b %d")))
                    .selectAll("text")
                    .style("fill", "#6b7280"); // Darker axis text

                const y = d3.scaleLinear()
                    .domain([0, d3.max(data, d => d.count) * 1.1])
                    .range([height, 0]);
                svg.append("g")
                    .call(d3.axisLeft(y))
                    .selectAll("text")
                    .style("fill", "#6b7280"); // Darker axis text

                svg.append("path")
                    .datum(data)
                    .attr("fill", "none")
                    .attr("stroke", "#1e40af") // Dark blue line
                    .attr("stroke-width", 3)
                    .attr("d", d3.line()
                        .x(d => x(d.date))
                        .y(d => y(d.count))
                    );

                // Tooltip setup (updated for light theme)
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip fixed p-2 bg-white border border-gray-300 rounded-lg text-xs text-gray-900 shadow-xl pointer-events-none opacity-0 transition-opacity duration-200");

                svg.selectAll("dot")
                    .data(data)
                    .enter().append("circle")
                    .attr("r", 5)
                    .attr("cx", d => x(d.date))
                    .attr("cy", d => y(d.count))
                    .attr("fill", "#059669") // Dark green dots
                    .on("mouseover", function(event, d) {
                        d3.select(this).attr("r", 8).attr("fill", "#f59e0b"); // Amber hover

                        tooltip.style("opacity", 1)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 15) + "px")
                            .html(`Date: ${d3.timeFormat("%b %d")(d.date)}<br>Count: ${d.count}`);

                        d3.select(this.parentNode).append("line")
                            .attr("class", "hover-line")
                            .attr("x1", x(d.date))
                            .attr("x2", x(d.date))
                            .attr("y1", height)
                            .attr("y2", y(d.count))
                            .attr("stroke", "#f59e0b")
                            .attr("stroke-width", 1)
                            .attr("stroke-dasharray", "4,4");
                    })
                    .on("mouseout", function() {
                        d3.select(this).attr("r", 5).attr("fill", "#059669");
                        tooltip.style("opacity", 0);
                        d3.select(".hover-line").remove();
                    });
            };

            // --- 2. Emails by Sender Domain (Pie Chart) ---
            const drawPieChart = (data, selector) => {
                const size = 300;
                const outerRadius = size / 2;
                const innerRadius = outerRadius * 0.6;

                // Clear previous tooltips
                d3.selectAll(".tooltip").remove();

                d3.select(selector).select("svg").remove();
                d3.select(selector).select(".legend-container").remove();

                const svg = d3.select(selector)
                    .append("svg")
                        .attr("width", size)
                        .attr("height", size)
                    .append("g")
                        .attr("transform", `translate(${size / 2}, ${size / 2})`);

                // Using a color scheme that looks good on light background
                const color = d3.scaleOrdinal()
                    .domain(data.map(d => d.domain))
                    .range(d3.schemeSpectral[data.length]); // A diverse scheme

                const pie = d3.pie()
                    .value(d => d.count);

                const arc = d3.arc()
                    .innerRadius(innerRadius)
                    .outerRadius(outerRadius);

                const g = svg.selectAll(".arc")
                    .data(pie(data))
                    .enter().append("g")
                    .attr("class", "arc");

                g.append("path")
                    .attr("d", arc)
                    .attr("fill", d => color(d.data.domain))
                    .attr("stroke", "#ffffff") // White border between slices
                    .style("stroke-width", "2px")
                    .on("mouseover", function(event, d) {
                        d3.select(this).transition()
                            .duration(100)
                            .attr("d", d3.arc().innerRadius(innerRadius).outerRadius(outerRadius * 1.05));

                        const tooltip = d3.select("body").append("div")
                            .attr("class", "tooltip fixed p-2 bg-gray-900 border border-gray-700 rounded-lg text-xs text-white shadow-xl pointer-events-none opacity-0 transition-opacity duration-200")
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 15) + "px")
                            .style("opacity", 1)
                            .html(`Domain: <strong>${d.data.domain}</strong><br>Emails: ${d.data.count}`);
                    })
                    .on("mouseout", function() {
                        d3.select(this).transition()
                            .duration(100)
                            .attr("d", arc);
                        d3.select(".tooltip").remove();
                    });

                // Add Legend
                const legendContainer = document.querySelector(selector);
                const legend = d3.select(legendContainer)
                    .append("div")
                    .attr("class", "legend-container mt-4 flex flex-wrap justify-center");

                data.forEach((d) => {
                    legend.append("div")
                        .attr("class", "flex items-center mr-4 mb-2")
                        .html(`
                            <span class="w-3 h-3 rounded-full mr-2" style="background-color: ${color(d.domain)};"></span>
                            <span class="text-xs text-gray-700">${d.domain} (${d.count})</span>
                        `);
                });
            };

            drawLineChart(volumeData, '#volume-chart');
            drawPieChart(senderData, '#sender-chart');

            // Re-draw charts on resize to maintain responsiveness
            window.addEventListener('resize', () => {
                drawLineChart(volumeData, '#volume-chart');
            });
        }



