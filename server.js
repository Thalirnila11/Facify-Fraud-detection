const express = require('express');
const { spawn } = require('child_process');
const app = express();
const cors = require('cors');
const port = 3000;

app.use(cors());
app.use(express.json());

// Import the nodemailer module
const nodemailer = require('nodemailer');
const transporter = nodemailer.createTransport({
    // Configure your email service provider here
    service: 'gmail',
    auth: {
        user: 'preethaabu04@gmail.com',
        pass: 'udjcimxmecrjlnnk'
    }
});

// Generate the email content and send it to the provided email address
function sendEmail(username, email) {
    const link = `http://localhost:${port}/scan-face?name=${username}`;
    const mailOptions = {
        from: 'preethaabu04@gmail.com',
        to: email,
        subject: 'Face Recognition Link',
        html: `<p>Click the following link to perform face recognition:</p><a href="${link}">${link}</a>`
    };

    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.error('Error sending email:', error);
        } else {
            console.log('Email sent:', info.response);
        }
    });
}

app.post('/recognize-face', (req, res) => {
    const name = req.body.username;
    const email = req.body.email;

    console.log('Button clicked. Sending email with the link...');
    console.log('Name:', name);

    sendEmail(name, email); // Send the email with the link

    res.send('Email sent. Please check your inbox to proceed with face recognition.');
});

app.get('/scan-face', (req, res) => {
    const name = req.query.name;
    let responseData = ''; // Declare responseData variable

    // Execute the Python script as a child process
    const pythonScript = spawn('python', ['./face_recognition.py', name]);

    pythonScript.stdout.on('data', (data) => {
        responseData += data.toString(); // Accumulate the data
    });

    pythonScript.on('error', (error) => {
        console.error(`Error executing Python script: ${error.message}`);
        res.status(500).send('An error occurred during script execution');
    });

    pythonScript.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
        if (code === 0) {
            res.send('success'); // Send 'success' if the face recognition was successful
          } else {
            res.send('failure'); // Send 'failure' if the face recognition failed
          }
    });
});

app.get('/verification_successful', (req, res) => {
    res.sendFile(__dirname + '/verification_success.html'); // Replace with the path to your verification success HTML file
});


app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
})














