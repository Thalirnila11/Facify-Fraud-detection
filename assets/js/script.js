// Handle form submission
document.getElementById('checkout-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission
    
    // Get form values
    var formData = {
      
      cardname: document.getElementById('card holder name').value,
      cardnumber: document.getElementById('cardnumber').value,
      expmonth: document.getElementById('expmonth').value,
       expyear: document.getElementById('expyear').value,
      cvv: document.getElementById('cvv').value
    };
    
    // Perform further actions with the form data
    console.log(formData);
  });