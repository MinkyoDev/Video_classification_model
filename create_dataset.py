import os
import shutil
import cv2
import numpy as np
import random
from ultralytics import YOLO

class CreateDataset():
    def __init__(self):
        # Initialize the YOLO object detection model
        self.model = YOLO("yolov8s.pt")
    
    def dir_create_and_seperate(self, label_list, src_folder, dic_folder):
        for label in label_list:
            # Specify the source folder for the label
            src_folder_label = f"{src_folder}/{label}"

            # Specify the target folders (train, validation, test) for the label
            train_folder = f"{dic_folder}/train/{label}"
            val_folder = f"{dic_folder}/val/{label}"
            test_folder = f"{dic_folder}/test/{label}"

            # Create the target folders if they don't exist
            for folder in [train_folder, val_folder, test_folder]:
                if not os.path.exists(folder):
                    os.makedirs(folder)

            # Get a list of files and shuffle them randomly
            files = [f for f in os.listdir(src_folder_label) if os.path.isfile(os.path.join(src_folder_label, f))]
            random.shuffle(files)

            # Set the split ratios
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15

            # Calculate the number of files for each set
            num_files = len(files)
            train_count = int(num_files * train_ratio)
            val_count = int(num_files * val_ratio)
            test_count = num_files - train_count - val_count

            # Copy files to their respective folders
            for i, file in enumerate(files):
                src = os.path.join(src_folder_label, file)
                if i < train_count:
                    dst = os.path.join(train_folder, file)
                elif i < train_count + val_count:
                    dst = os.path.join(val_folder, file)
                else:
                    dst = os.path.join(test_folder, file)
                
                shutil.copy(src, dst)
            print(f'Seperate done ({label})  -  train_count : {train_count}, val_count : {val_count}, test_count : {test_count}')


    def get_new_file_name(self, label, dst, file_name):
        index = 0
        base_name, ext = os.path.splitext(file_name)
        new_file_name = file_name

        while os.path.exists(os.path.join(dst, new_file_name)):
            # new_file_name = f"{base_name.split('_')[0]}_aug_{index}{ext}"
            new_file_name = f'{label}_aug_{index}.mp4'
            index += 1

        return new_file_name


    def augment_video(self, label_list, dic_folder, aug_size):
        for label in label_list:
            print(f'augment start ({label})')

            # Specify the source folder for the current label
            path = f'{dic_folder}/train/{label}'
            file_list = os.listdir(path)

            # Specify the output directory for augmented videos of the current label
            output_dir = f'{dic_folder}/train/{label}'

            # Perform augmentation for a specified number of times (aug_size)
            for i in range(1, aug_size):
                for file_name in file_list:
                    input_video = os.path.join(path, file_name)
                    
                    # Generate a unique filename for the augmented video
                    new_file_name = self.get_new_file_name(label, output_dir, file_name)
                    output_video = os.path.join(output_dir, new_file_name)

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir) 
                    
                    # Generate random augmentation parameters
                    angle = round(random.uniform(-10, 10), 2)
                    brightness_factor = round(random.uniform(-30, 30), 2)
                    red = round(random.uniform(-30, 30), 2)
                    blue = round(random.uniform(-30, 30), 2)
                    green = round(random.uniform(-30, 30), 2)

                    # Create a video capture object
                    cap = cv2.VideoCapture(input_video)

                    # Prepare a video writer object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))

                    while True:
                        ret, frame = cap.read()
                        
                        if not ret:
                            break

                        # Adjust brightness
                        if brightness_factor >= 0:
                            frame = cv2.add(frame, np.ones(frame.shape, dtype='uint8') * np.uint8(brightness_factor))
                        else:
                            frame = cv2.subtract(frame, np.ones(frame.shape, dtype='uint8') * np.uint8(abs(brightness_factor)))

                        # Adjust colors
                        blue_, green_, red_ = cv2.split(frame)
                        red_ = cv2.add(red_, red)
                        blue_ = cv2.subtract(blue_, blue)
                        green_ = cv2.add(green_, green)  
                        frame = cv2.merge((blue_, green_, red_))
                        
                        # Get the frame size
                        (h, w) = frame.shape[:2]

                        # Set the rotation center to the frame's center
                        center = (w / 2, h / 2)

                        # Get the rotation matrix
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)

                        # Perform rotation
                        rotated = cv2.warpAffine(frame, M, (w, h))

                        # Write the rotated frame to the output video
                        out.write(rotated)

                    cap.release()
                    out.release()
                print(f'Augment done  -  {label} / {i}')


    def video_crop(self, input_dataset, output_dataset):
        # Get a list of subdirectories in the input dataset directory (e.g., different types)
        type_list = os.listdir(input_dataset)
        for type in type_list:
            input_dataset_type = os.path.join(input_dataset, type)
            output_dataset_type = os.path.join(output_dataset, type)

            # Get a list of labels (e.g., different categories) within the current type
            label_list = os.listdir(input_dataset_type)
            for label in label_list:
                print(f"Croping...   {type}-{label}")

                # Set input and output directories for the current label
                input_dataset_type_label = os.path.join(input_dataset_type, label)
                output_dataset_type_label = os.path.join(output_dataset_type, label)
                
                if not os.path.exists(output_dataset_type_label):
                    os.makedirs(output_dataset_type_label)

                # Get a list of video files for the current label
                file_names = os.listdir(input_dataset_type_label)
                for file_name in file_names:
                    input_file = os.path.join(input_dataset_type_label, file_name)
                    output_file = os.path.join(output_dataset_type_label, file_name)

                    # Create a video capture object
                    cap = cv2.VideoCapture(input_file)

                    # Get the frames per second (fps) of the input video
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    # Prepare a video writer object for the output video
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_file, fourcc, fps, (224, 224))

                    last_cropped_frame = None

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Predict objects in the current frame using the YOLO model
                        results = self.model.predict(frame, conf=0.5, classes=0, verbose=False)
                        try:
                            cords = results[0].boxes[0].xyxy.tolist()
                            if len(results) > 0:
                                # Calculate the center coordinates of the detected object
                                center_x = (cords[0][0] + cords[0][2]) / 2
                                center_y = (cords[0][1] + cords[0][3]) / 2
                                # Crop a region around the detected object and write it to the output video
                                cropped_frame = frame[int(center_y) - 112:int(center_y) + 112, int(center_x) - 112:int(center_x) + 112]
                                last_cropped_frame = cropped_frame
                                out.write(cropped_frame)
                            elif last_cropped_frame is not None:
                                # Use the last cropped frame if object detection fails
                                out.write(last_cropped_frame)
                        except:
                            # Use the last cropped frame if an exception occurs
                            out.write(last_cropped_frame)

                    cap.release()
                    out.release()
            print(f"Crop done") 


    def delete_short_videos(self, input_dataset, min_duration=2.0):
        print(input_dataset)
        # Get a list of subdirectories in the input dataset directory (e.g., different types)
        type_list = os.listdir(input_dataset)
        for type in type_list:
            input_dataset_type = os.path.join(input_dataset, type)

            # Get a list of labels (e.g., different categories) within the current type
            label_list = os.listdir(input_dataset_type)
            for label in label_list:
                print(f"file checking ...   {type}-{label}")

                # Set the input directory for the current label
                input_dataset_type_label = os.path.join(input_dataset_type, label)

                # Get a list of video files for the current label
                file_names = os.listdir(input_dataset_type_label)
                for file_name in file_names:
                    file_ = os.path.join(input_dataset_type_label, file_name)
                    
                    # Check if the file is a video file
                    if os.path.isfile(file_) and file_name.endswith('.mp4'):
                        # Create a video capture object
                        cap = cv2.VideoCapture(file_)
                        
                        # Get the frames per second (fps) and frame count of the video
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        cap.release()
                        
                        # Delete the video file if its FPS is 0 or its duration is less than the minimum duration
                        if fps == 0 or (frame_count / max(fps, 1)) < min_duration:
                            os.remove(file_)
                            print(f"Deleted: {file_name} (Duration: {(frame_count / max(fps, 1)):.2f} seconds)")
                    


if __name__ == '__main__':
    label_list = ['catch', 'put', 'normal', 'insert']

    src_dir = f"data/origin_data/640x480_fps30"
    aug_size = 6
    dic_dir = f"data/data_set/data_set_aug{aug_size}"


    create_dataset = CreateDataset()

    create_dataset.dir_create_and_seperate(label_list, src_dir, dic_dir)

    create_dataset.augment_video(label_list, dic_dir, aug_size)

    create_dataset.video_crop(dic_dir, f"{dic_dir}_crop")

    create_dataset.delete_short_videos(f"{dic_dir}_crop", min_duration=2.0)