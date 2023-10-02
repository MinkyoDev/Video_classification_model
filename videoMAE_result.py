from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToPILImage, ToTensor
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import datetime, timedelta
import torch
import os


class OneVideoClassification():
    def __init__(self, model_ckpt):
        # Initialize the image processor and model using the provided checkpoint.
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, ignore_mismatched_sizes=True)

        # Extract mean, standard deviation, and image size information for preprocessing.
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)
        self.num_frames_to_sample = self.model.config.num_frames

       
    def load_video(self, video_file):
        # Load a video file and prepare it for classification.
        self.video_file = video_file
        self.video_file_path = '/'.join(video_file.split('/')[:-1])
        self.video_file_name = video_file.split('/')[-1].split('.')[0]
    
        video, _, _ = read_video(video_file, pts_unit="sec")
        self.video = video.permute(3, 0, 1, 2)

        return self.video


    def transform_video(self, video):
        # Transform the video frames for preprocessing.
        video = video.permute(1, 0, 2, 3)   # Convert from (C, T, H, W) to (T, C, H, W)
        transformed_video = []
        for frame in video:
            frame = frame / 255.0   # Normalize pixel values to the [0, 1] range
            frame = ToPILImage()(frame)   # Convert the frame to a PIL image
            frame = Resize(self.resize_to, antialias=True)(frame)    # Resize the frame
            frame = ToTensor()(frame)   # Convert the frame back to a tensor
            frame = Normalize(self.mean, self.std)(frame)   # Apply mean and standard deviation normalization
            transformed_video.append(frame)
        transformed_video = torch.stack(transformed_video)

        return transformed_video.permute(1, 0, 2, 3)   # Convert back to (C, T, H, W) format


    def preprocessing(self):
        # Apply transformations to the video frames for preprocessing.
        val_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(self.transform_video),
                        ]
                    ),
                ),
            ])
        self.video_tensor = val_transform({"video": self.video})["video"]

        return self.video_tensor


    def run_inference(self, model, video):
        # Run inference on the preprocessed video using the provided model.
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": perumuted_sample_test_video.unsqueeze(0)}   # Add a batch dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}   # Move inputs to the appropriate device
        model = model.to(device)   # Move the model to the appropriate device

        with torch.no_grad():
            outputs = model(**inputs)   # Perform inference
            logits = outputs.logits

        return logits

    def predict(self):
        # Preprocess the video and make a prediction.
        self.preprocessing()
        logits = self.run_inference(self.model, self.video_tensor)
        predicted_class_idx = logits.argmax(-1).item()   # Get the index of the predicted class

        return self.model.config.id2label[predicted_class_idx]   # Retrieve the class label for the prediction


class MultiVideoClassification():
    def __init__(self, model_ckpt):
        # Initialize the image processor and model using the provided checkpoint.
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, ignore_mismatched_sizes=True)

        # Extract mean, standard deviation, and image size information for preprocessing.
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)
        self.num_frames_to_sample = self.model.config.num_frames

       
    def load_videos(self, video_dir):
        # Load a directory of video files for classification.
        self.video_dir = video_dir
        self.video_list = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]


    def transform_video(self, video):
        # Transform the video frames for preprocessing.
        video = video.permute(1, 0, 2, 3)   # Convert from (C, T, H, W) to (T, C, H, W)
        transformed_video = []
        for frame in video:
            frame = frame / 255.0   # Normalize pixel values to the [0, 1] range
            frame = ToPILImage()(frame)   # Convert the frame to a PIL image
            frame = Resize(self.resize_to, antialias=True)(frame)   # Resize the frame
            frame = ToTensor()(frame)   # Convert the frame back to a tensor
            frame = Normalize(self.mean, self.std)(frame)   # Apply mean and standard deviation normalization
            transformed_video.append(frame)
        transformed_video = torch.stack(transformed_video)

        return transformed_video.permute(1, 0, 2, 3)   # Convert back to (C, T, H, W) format


    def preprocessing(self):
        # Apply transformations to the video frames for preprocessing.
        val_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(self.transform_video),
                        ]
                    ),
                ),
            ])
        self.video_tensor = val_transform({"video": self.video})["video"]

        return self.video_tensor


    def run_inference(self, model, video):
        # Run inference on the preprocessed video using the provided model.
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": perumuted_sample_test_video.unsqueeze(0)}   # Add a batch dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}   # Move inputs to the appropriate device
        model = model.to(device)   # Move the model to the appropriate device

        with torch.no_grad():
            outputs = model(**inputs)   # Perform inference
            logits = outputs.logits

        return logits

    def predict(self, date, duration, format_str):
         # Make predictions for each video in the directory.
        result_dic = {}
        for video_name in self.video_list:
            video_file = os.path.join(self.video_dir, video_name)
            video, _, _ = read_video(video_file, pts_unit="sec")
            self.video = video.permute(3, 0, 1, 2)
    
            self.preprocessing()
            logits = self.run_inference(self.model, self.video_tensor)
            predicted_class_idx = logits.argmax(-1).item()
            result = self.model.config.id2label[predicted_class_idx]

            video_index = int(video_name.split('.')[0].split('_')[1])
            seg_sec = video_index * duration
            new_date = date + timedelta(seconds=seg_sec)
            key_time = new_date.strftime(format_str)

            result_dic[key_time] = [video_index, result]

        return result_dic


if __name__ == '__main__':
    model_ckpt = "./models/20230914/checkpoint-426"   # best

    format_str = "%Y-%m-%d-%H-%M-%S"
    start_time="2023-09-12-14-44-14"
    duration = 4

    date=datetime.strptime(start_time, format_str)


    # OneVideoClassification
    input_file = 'data/test_data/multi_cam/insert_1_crop/crop_002.mp4'

    video_classification = OneVideoClassification(model_ckpt)
    video_classification.load_video(input_file)
    result = video_classification.predict()
    print(result)


    # MultiVideoClassification
    input_folder = 'data/test_data/multi_cam/insert_1_crop'

    video_classification = MultiVideoClassification(model_ckpt)
    video_classification.load_videos(input_folder)
    person_data = video_classification.predict(date, duration, format_str)
    print(person_data)