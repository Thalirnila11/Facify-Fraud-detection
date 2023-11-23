import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Function to capture frames from the webcam
def capture_frames():
    camera = cv2.VideoCapture(0)  # Open the webcam (change the index if you have multiple cameras)

    while True:
        success, frame = camera.read()  # Read a frame from the webcam

        if not success:
            break

        # Process the frame or perform any desired operations

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the JPEG-encoded frame as a byte array
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.errorhandler(500)
def handle_internal_server_error(e):
    return "Internal Server Error: {}".format(e), 500

if __name__ == '__main__':
    app.run(debug=True)
