const express = require('express');
const nodemailer = require('nodemailer');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const generateOTP= require('./otpGenerator')


const app = express();
const uri = 'mongodb+srv://preethaabu04:12345@cluster0.dagks1r.mongodb.net/otp?retryWrites=true&w=majority';
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors());



app.post('/send-email', async (req, res) => {
  console.log("req received")
  try {
    const { recipient, subject } = req.body;
    const otp = generateOTP();

    // Store the OTP in MongoDB
    await client.connect();
    const db = client.db('otp');
    const otpCollection = db.collection('otps');
    await otpCollection.insertOne({ email: recipient, otp });

    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: 'preethaabu04@gmail.com',
        pass: 'udjcimxmecrjlnnk'
      }
    });

    const mailOptions = {
      from: 'preethaabu04@gmail.com',
      to: recipient,
      subject: subject,
      html: `Your OTP: ${otp}`
    };

    // Send the email
    const info = await transporter.sendMail(mailOptions);
    console.log('Email sent:', info.response);

    res.status(200).json({ success: true, message: 'Email sent successfully', otp });
  } catch (error) {
    console.error('Error sending email:', error);
    res.status(500).json({ success: false, message: 'Error sending email' });
  } finally {
    await client.close();
  }
});

// Verify OTP
app.post('/verify-otp', async (req, res) => {
  try {
    const { email, otp } = req.body;
    await client.connect();
    const db = client.db('otp');
    const otpCollection = db.collection('otps');

    // Find the stored OTP for the email
    const storedOTP = await otpCollection.findOne({ email },{ sort: { _id: -1 }});
    console.log(storedOTP.otp)
    console.log(otp)
    console.log(typeof storedOTP.otp)
    console.log(typeof otp)

    if (storedOTP.otp === otp) {
      // OTP verification successful
      console.log("verified")
      res.send("Otp verified");
    } else {
      // Invalid OTP
      res.send('Invalid OTP');
    }
  }  finally {
    await client.close();
  }
});

// Start the Express server
app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
