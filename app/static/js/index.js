
const tasks = document.querySelectorAll('.task');
tasks.forEach(task =>{
  task.addEventListener('click', () => {
    tasks.forEach(i => i.classList.remove('active'));
    task.classList.add('active')
  });
});

document.getElementById('split-btn').addEventListener('click', function () {
    const input = document.getElementById('text-input').value;
    const active_btn = document.querySelector('.task.active');
    const process_type = active_btn.textContent

    if (process_type === 'Tokenize'){
      get_tokens(input)
    } 
    else if (process_type === 'Stemming'){
      get_stems(input)
    }
    else if (process_type === 'POS'){
      get_pos(input)
    }
    else if (process_type === 'NER'){
      get_ner(input)
    }
    else if(process_type == 'Embedding'){
      get_embed(input)
    }
  });

async function get_tokens(user_input) {
  try{
    const response = await fetch(
      '/tokenize',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: user_input}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const output = document.getElementById('output-section');
    output.innerHTML = '';
    const responseData = await response.json();
    const tokens = responseData['tokens']

    tokens.forEach(token => {
      const div = document.createElement('div');
      div.className = 'token';
      div.textContent = token;
      output.appendChild(div);
    });

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}

async function get_stems(user_input) {
  try{
    const response = await fetch(
      '/stemmize',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: user_input}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const output = document.getElementById('output-section');
    output.innerHTML = '';
    const responseData = await response.json();
    const tokens = responseData['stems']

    tokens.forEach(token => {
      const div = document.createElement('div');
      div.className = 'token';
      div.textContent = token;
      output.appendChild(div);
    });

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}

async function get_pos(user_input) {
  try{
    const response = await fetch(
      '/pos',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: user_input}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const output = document.getElementById('output-section');
    output.innerHTML = '';
    const responseData = await response.json();
    const tokens = responseData['pos']

    tokens.forEach(token => {
      const div = document.createElement('div');
      div.className = 'token';
    
      const span1 = document.createElement('span');
      span1.className = 'entity';
      span1.textContent = token[0];
      
      const span2 = document.createElement('span');
      span2.className = 'recognition';
      span2.textContent = token[1];
    
      div.textContent = '';
      div.appendChild(span1);
      div.appendChild(span2);
    
      output.appendChild(div);
    });

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}

async function get_ner(user_input) {
  try{
    const response = await fetch(
      '/ner',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: user_input}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const output = document.getElementById('output-section');
    output.innerHTML = '';
    const responseData = await response.json();
    const tokens = responseData['ner']

    tokens.forEach(token => {
      const div = document.createElement('div');
      div.className = 'token';
    
      // Create the first new element
      const span1 = document.createElement('span');
      span1.className = 'entity';
      span1.textContent = token[0];
      
      // Create the second new element
      const span2 = document.createElement('span');
      span2.className = 'recognition';
      span2.textContent = token[1];
    
      div.textContent = '';
      div.appendChild(span1);
      div.appendChild(span2);
    
      output.appendChild(div);
    });

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}

async function get_embed(user_input) {
  try{
    const response = await fetch(
      '/tokenize',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: user_input}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const output = document.getElementById('output-section');
    output.innerHTML = '';
    const responseData = await response.json();
    const tokens = responseData['tokens']

    get_embeddings(tokens)

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}

async function get_embeddings(token_list) {
  try{
    const response = await fetch(
      '/embed',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({text: token_list}),
      });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const responseData = await response.json();
    const embeddingsData = responseData['reduced']

    console.log(embeddingsData) 
    
    // ------------ Drawing Graph in the web page ------------ //

    const container = d3.select("#output-section");

    // Create a tooltip div (outside the SVG)
    const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("opacity", 0);

    // Function to draw the visualization
    function drawVisualization() {
      container.select("svg").remove();

      const containerWidth = container.node().clientWidth;
      const containerHeight = container.node().clientHeight;

      const margin = { top: 20, right: 30, bottom: 40, left: 50 };
      const width = containerWidth - margin.left - margin.right;
      const height = containerHeight - margin.top - margin.bottom;

      const svg = container
          .append("svg")
          .attr("width", containerWidth)
          .attr("height", containerHeight)
          .append("g")
          .attr("transform", `translate(${margin.left},${margin.top})`);

      const xValues = embeddingsData.map(d => d.embedding[0]);
      const yValues = embeddingsData.map(d => d.embedding[1]);

      const xExtent = d3.extent(xValues);
      const yExtent = d3.extent(yValues);

      const xScale = d3.scaleLinear()
          .domain([xExtent[0] - 1, xExtent[1] + 1])
          .range([0, width]);

      const yScale = d3.scaleLinear()
          .domain([yExtent[0] - 1, yExtent[1] + 1])
          .range([height, 0]);

      const xAxis = d3.axisBottom(xScale);
      const yAxis = d3.axisLeft(yScale);

      svg.append("g")
          .attr("class", "axis x-axis")
          .attr("transform", `translate(0,${height})`)
          .call(xAxis);

      svg.append("g")
          .attr("class", "axis y-axis")
          .call(yAxis);

      svg.selectAll(".origin-line")
          .data(embeddingsData)
          .enter()
          .append("line")
          .attr("class", "origin-line")
          .attr("x1", xScale(0))
          .attr("y1", yScale(0))
          .attr("x2", d => xScale(d.embedding[0]))
          .attr("y2", d => yScale(d.embedding[1]));
      // ---------------------------------------------

      svg.append("line")
          .attr("class", "origin-axis")
          .attr("x1", 0)
          .attr("y1", yScale(0))
          .attr("x2", width)
          .attr("y2", yScale(0));

      svg.append("line")
          .attr("class", "origin-axis")
          .attr("x1", xScale(0))
          .attr("y1", 0)
          .attr("x2", xScale(0))
          .attr("y2", height);
      // ----------------------------------------------------

      svg.selectAll(".data-point")
          .data(embeddingsData)
          .enter()
          .append("circle")
          .attr("class", "data-point")
          .attr("cx", d => xScale(d.embedding[0]))
          .attr("cy", d => yScale(d.embedding[1]))
          .attr("r", 5)
          // Add mouseover event for tooltip
          .on("mouseover", function(event, d) {
              tooltip.transition()
                  .duration(200)
                  .style("opacity", .9);
              tooltip.html(d.word)
                  .style("left", (event.pageX) + "px")
                  .style("top", (event.pageY - 28) + "px");
          })
          // Add mouseout event to hide tooltip
          .on("mouseout", function(d) {
              tooltip.transition()
                  .duration(500)
                  .style("opacity", 0); // Hide tooltip
          });

      svg.selectAll(".word-label")
          .data(embeddingsData)
          .enter()
          .append("text")
          .attr("class", "word-label")
          .attr("x", d => xScale(d.embedding[0]) + 10)
          .attr("y", d => yScale(d.embedding[1]) + 3)
          .text(d => d.word);

      // Add axis labels (optional)
      svg.append("text")
          .attr("transform", `translate(${width/2},${height + margin.bottom - 5})`)
          .style("text-anchor", "middle")
          .text("Embedding Dimension 1");

      svg.append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 0 - margin.left)
          .attr("x", 0 - (height / 2))
          .attr("dy", "1em")
          .style("text-anchor", "middle")
          .text("Embedding Dimension 2");
  }

  drawVisualization();
  window.addEventListener('resize', drawVisualization);

  // ------------ End Drawing Graph in the web page ------------ //

  } catch (error){
    console.error('Error posting data to server: ', error);
    throw error;
  } 
}