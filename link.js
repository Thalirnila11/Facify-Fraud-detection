document.addEventListener('DOMContentLoaded', function() {
    const button = document.getElementById('recognizeButton');
    button.addEventListener('click', function() {
        console.log('Clicked');
        const username = document.getElementById('usernameInput').value;
        const email = document.getElementById('emailInput').value;

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
            // Check if the result is successful and load the HTML page
            if (data === 'success') {
                window.location.href = 'http://localhost:3000/verification_successful'; // Replace with your success page URL
            }
        })
        .catch(error => {
            console.error(error);
        });
    });
});
