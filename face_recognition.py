from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
import sys
import dlib
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import keyboard
import requests
import webbrowser

sys.path.append('./FaceSpoof')
name=sys.argv[1]

print("Name:", name)
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from sklearn.model_selection import ParameterGrid
modeldir = './model/20180402-114759.pb'
pruned_modeldir = './model/pruned_model.pb'
# modeldir =r'E:\Projects\augmentation\head-position_estimation\model\20180402-114759.pb'
# pruned_modeldir = './model/pruned_model.pb'
threshold = 0.01

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        facenet.load_model(modeldir)
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        weights_dict = {}
        for op in graph.get_operations():
            for tensor in op.values():
                if isinstance(tensor, tf.Variable):
                    weight_name = tensor.name.replace("/read", "")
                    weights_dict[weight_name] = tensor

        for weight_name, weight_tensor in weights_dict.items():
            weight_values = sess.run(weight_tensor)
            mask = np.abs(weight_values) > threshold
            pruned_weights = weight_values * mask
            sess.run(tf.assign(weight_tensor, pruned_weights))

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ["embeddings"])
        with tf.gfile.GFile(pruned_modeldir, "wb") as f:
            f.write(output_graph_def.SerializeToString())

print("Model pruned and saved to:", pruned_modeldir)

video = 0
#modeldir = r'E:\Projects\augmentation\head-position_estimation\model\20180402-114759.pb'
# classifier_filename = './class/classifier.pkl'
# npy = r'E:\Projects\augmentation\head-position_estimation\npy'
# train_img = r'E:\Projects\augmentation\head-position_estimation\train_img'

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = './train_img'
verified_face_path = None

#anti_spoof_model_dir = r"E:\python\facenet_live\liveness_integration\FaceSpoof\resources\anti_spoof_models"
anti_spoof_model_dir = './FaceSpoof/resources/anti_spoof_models'

# Initialize the anti-spoof model
anti_spoof_model = AntiSpoofPredict(0)  # Set the device ID according to your systemqqq

# Initialize the image cropper
image_cropper = CropImage()

root = tk.Tk()
root.withdraw()

# Define the hyperparameters and their possible values
hyperparameters = {
    'face_detection_threshold': [0.6, 0.7, 0.8],
    'liveness_threshold': [0.9, 0.95, 0.99]
}

# Generate all combinations of hyperparameters
param_grid = ParameterGrid(hyperparameters)

# Initialize variables to store the best hyperparameters and their performance
best_hyperparameters = None
best_accuracy = 0.0

#predictor_path = r'E:\Projects\augmentation\head-position_estimation\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'
predictor_path = './shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        minsize = 30  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps' threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size = 1000  # 1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        count = 0
        color = ""
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        threshold = [0.9, 0.95, 0.99]
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        while True:
            ret, frame = video_capture.read()
            display = frame


            # Perform histogram equalization on gray frame (hidden processing)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

            timer = time.time()
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(det[i][0])
                    ymin = int(det[i][1])
                    xmax = int(det[i][2])
                    ymax = int(det[i][3])

            # Detect frontal faces using dlib
            frontal_faces = face_detector(frame)

            for face in frontal_faces:
                xmin = face.left()
                ymin = face.top()
                xmax = face.right()
                ymax = face.bottom()

                try:
                    # inner exception
                    if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                        print('Face is very close!')
                        continue

                    face_image = display[ymin:ymax, xmin:xmax, :]
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    landmarks = predictor(face_gray, face)
                    landmarks_points = []
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        landmarks_points.append((x, y))
                    for point in landmarks_points:
                        cv2.circle(display, point, 2, (0, 0,255), -1)

                    # Perform liveness detection on the recognized face
                    img = face_rgb
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

                    # Perform liveness detection using the anti-spoof model
                    image_bbox = anti_spoof_model.get_bbox(img)
                    prediction = np.zeros((1, 3))

                    for model_name in os.listdir(anti_spoof_model_dir):
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": img,
                            "bbox": image_bbox,
                            "scale": scale,
                            "out_w": w_input,
                            "out_h": h_input,
                            "crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img_cropped = image_cropper.crop(**param)
                        prediction += anti_spoof_model.predict(img_cropped,
                                                               os.path.join(anti_spoof_model_dir, model_name))

                    # Draw the result of liveness detection
                    label = np.argmax(prediction)
                    value = prediction[0][label] / 2
                    if label == 1:
                        liveness_label = "real"
                        color = (0, 255, 0)  # Green color for real face
                    else:
                        liveness_label = "fake"
                        color = (0, 0, 255)
                    cropped.append(frame[ymin:ymax, xmin:xmax, :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    # cv2.putText(display, liveness_label, (xmin, ymin - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, thickness=1, lineType=1)
                    cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # boxing face
                    for H_i in HumanNames:
                        if HumanNames[best_class_indices[0]] == H_i:
                            result_names = HumanNames[best_class_indices[0]]
                            print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(
                                HumanNames[best_class_indices[0]], best_class_probabilities[0]))
                            if label == 1:

                                if result_names == name:
                                    count = count + 1
                                    if count >= 2:
                                        print("Verification Successful")
                                        color = (0, 255, 0)
                                        name = str(HumanNames[best_class_indices[0]])
                                        verified_face_path = os.path.join(train_img, result_names + ".jpg")

                                         # Send a request to the Flask web server
                                        response = requests.get('http://localhost:3000/verification_successful')
                                        if response.status_code == 200:
                                            print("HTML file loaded successfully")
                                        else:
                                            print("Failed to load HTML file")
                                        # Path to the HTML file you want to open
                                        html_file_path = "./verification_success.html"

                                        # Open the HTML file in a new browser tab
                                        webbrowser.open("file://" + html_file_path, new=2)

                                else:
                                    if count >= 0:
                                        count = count - 1
                                    print("Person not verified")
                                    color = (0, 0, 255)
                                    verified_face_path = None
                                    messagebox.showinfo("Verification", "Face is not verified!\nTry again")

                                cv2.rectangle(display, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                                cv2.putText(display, result_names, (xmin, ymin - 5),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 0), thickness=1, lineType=1)

                    endtimer = time.time()
                    fps = 1 / (endtimer - timer)
                    cv2.rectangle(display, (15, 30), (135, 60), (0, 255, 255), -1)
                    cv2.putText(display, "fps: {:.2f}".format(fps), (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    cv2.imshow("Face Recognition", display)

                except:
                    print("error")


            if keyboard.is_pressed('q'):
                break
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

        video_capture.release()
        cv2.destroyAllWindows()