document.addEventListener('DOMContentLoaded', function () {
  async function sendEmail(email) {
    try {
      const response = await fetch('http://localhost:3000/send-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ recipient: email, subject: 'OTP Verification' })
      });

      if (response.ok) {
        console.log('Email sent successfully');


      } else {
        console.error('Error sending email');
      }
    } catch (error) {
      console.error('Error sending email:', error);
    }
  }

  const sendemail = document.getElementById("sendmail");
  sendemail.addEventListener('click', function () {
    const emailInput = document.getElementById('emailInput');
    const email = emailInput.value;


    sendEmail(email);
    var otpContainer = document.getElementById("otpContainer");
    otpContainer.classList.remove("hidden");
  })

  const verifyotp = document.getElementById("verify");
  verifyotp.addEventListener('click', function () {
    console.log("verify clicked")
    const enteredOTP = document.getElementById('otpInput').value;
    if (enteredOTP !== "") {
      const emailInput = document.getElementById('emailInput').value;
      const data = {
        email: emailInput,
        otp: enteredOTP.toString()
      };

      try {
        const response = fetch('http://localhost:3000/verify-otp', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(data)
        }).then(response => {
          if (response.ok) {
            return response.text();
          } else {
            throw new Error('Error: ' + response.status);
          }
        }).then(data => {
          if (data === "Otp verified") {
            window.location.href= "success.html"
          } else {
            alert("otp not verified")
          }
        })


      } catch (error) {
        console.error('Error verifying OTP:', error);
        alert('Error verifying OTP');
      }
    } else {
      // No OTP entered
      alert('Please enter the OTP');
    }


  })


})



