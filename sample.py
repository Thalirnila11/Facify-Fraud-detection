from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training
import os
from head_position import face_scan
from preprocess import preprocesses
from data_augmentation import augmentation
import cv2

# Enter the user
name = "Thalir"
# Path for saving the scanned image
real_folder_path = f'./train_img/real{name}'
spoof_folder_path = f'./train_img/spoof{name}'

# Paths for Data preprocessing
real_input_datadir = './train_img/real'
real_output_datadir = './aligned_img/real'

# Paths for training
real_datadir = './aligned_img'
modeldir = './model/20180402-114759.pb'
real_classifier_filename = './class/classifier.pkl'

frame_count = 25

def scan_and_save_real_images(real_folder_path):
    # Create the output directory if it doesn't exist
    os.makedirs(real_folder_path, exist_ok=True)
    print("Folder created successfully!")
    face_scan(name)
    print("Face scanned successfully")
    # augmentation(name)
    print("Face Augmentation completed")

    obj = preprocesses(real_input_datadir, real_output_datadir)
    nrof_images_total, nrof_successfully_aligned = obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    augmentation(name)

    print("Training Start")
    obj = training(real_datadir, modeldir, real_classifier_filename)
    get_file = obj.main_train()
    print('Saved classifier model to file "%s"' % get_file)

def extract_and_save_spoof_frames(spoof_folder_path):
    # Create the output directory if it doesn't exist
    os.makedirs(spoof_folder_path, exist_ok=True)

    video = cv2.VideoCapture(0)  # Use index 0 for the default webcam

    count = 1

    while count <= frame_count:
        success, frame = video.read()
        name = os.path.join(spoof_folder_path, f"frame_{count}.jpg")

        if success:
            cv2.imwrite(name, frame)
            print(f"Frame {count} Extracted and Saved Successfully")
            count += 1
        else:
            break

    video.release()

def main():
    if not os.path.exists(real_folder_path):
        scan_and_save_real_images(real_folder_path)
    else:
        print("Folder already exists, cannot scan!")
        obj = preprocesses(real_input_datadir, real_output_datadir)
        nrof_images_total, nrof_successfully_aligned = obj.collect_data()

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

        augmentation(name)

        print("Training Start")
        obj = training(real_datadir, modeldir, real_classifier_filename)
        get_file = obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)

    extract_and_save_spoof_frames(spoof_folder_path)

    sys.exit("All Done")

main()
