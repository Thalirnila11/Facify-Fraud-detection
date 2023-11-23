document.addEventListener('DOMContentLoaded', function() {
    const button = document.getElementById('recognizeButton');
    button.addEventListener('click', function() {
        console.log('Clicked');
        const username = document.getElementById('name').value;
        const email = document.getElementById('email').value;

        fetch('http://localhost:3000/recognize-face', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username: username, email: email }) // Pass the email as JSON data
        })
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error('Error: ' + response.status);
            }
        })
        .then(data => {
            console.log(data);
            alert('Face recognition result: ' + data);
        })
        .catch(error => {
            console.error(error);
        });
    });
});

    // Get form values
    var formData = {
      
      phone: document.getElementById('phone').value,
      address: document.getElementById('address').value,
      city: document.getElementById('city').value,
      state: document.getElementById('state').value,
      zip: document.getElementById('zip').value,
      cardholder: document.getElementById('cardholder').value,
      cardnumber: document.getElementById('cardnumber').value,
      expmonth: document.getElementById('expmonth').value,
      expyear: document.getElementById('expyear').value,
      cvv: document.getElementById('cvv').value
    };
    
    