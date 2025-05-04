
document.getElementById('split-btn').addEventListener('click', function () {
    const input = document.getElementById('text-input').value;
    const output = document.getElementById('output-section');
    output.innerHTML = ''; // Clear previous results

    // Split by spaces or commas
    const tokens = input.split(/[\s,]+/).filter(Boolean);
    console.log(output)
    tokens.forEach(token => {
      const div = document.createElement('div');
      div.className = 'token';
      div.textContent = token;
      output.appendChild(div);
    });
  });