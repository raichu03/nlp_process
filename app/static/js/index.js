
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