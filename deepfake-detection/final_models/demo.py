import numpy as np
import torch
import cv2
import os
import argparse
import shutil
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

class SimpleCNN(nn.Module):
    """CNN Classifier: Image → Class Logits (Real/Fake)"""
    def __init__(self, input_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def extract_frames_as_tensor(path):
        
        if os.path.exists("temp/1"):
            shutil.rmtree("temp/1")
        os.makedirs("temp/1")

        # Open the video file
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {path}")
            return

        print("Extracting Frames...")
        frame_count = 0
        while True:
            # Read a frame
            ret, frame = cap.read()

            # If frame is not read successfully, break the loop
            if not ret:
                break

            # Save the frame
            frame_filename = os.path.join("temp/1", f"frame_{frame_count:05d}.jpg")

            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Done...Extracted {frame_count} frames")


        print("Converting to tensors...")
        #each image will be 64x64
        image_size = 64

        demo_image_dataset = ImageFolder(root="temp",
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        features_extract = []
        for i in tqdm(range(len(demo_image_dataset))):
            t = demo_image_dataset[i]
            features_extract.append(t[0])

        X = torch.stack(features_extract, dim = 0)

        print("...Done")

        return X

def image_to_tensor(path):
    image_size = 64
    pil_image = Image.open(path)
    transform=transforms.Compose([transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
    image_tensor = transform(pil_image)
    return image_tensor.unsqueeze(0)

def classify_video(frames, model):

    model.eval()

    with torch.no_grad():

        logits = model(frames)

    probs = torch.softmax(logits, dim=1)

    preds = torch.argmax(probs, dim=1).numpy()

    fake_video_prob = np.mean(preds)

    # cumulative_probs = torch.sum(probs, dim=0).numpy()

    # fake_video_prob = cumulative_probs[1] / np.sum(cumulative_probs)

    return fake_video_prob

def classify_image(image, model):

    model.eval()

    with torch.no_grad():

        logits = model(image)

    probs = torch.softmax(logits, dim=1).numpy()

    return probs[0][1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--image', type=str)

    args = parser.parse_args()

    video_filepath = args.video

    image_filepath = args.image



    model = SimpleCNN()
    state_dict = torch.load("final_cnn_state.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["cnn_state_dict"])

    if video_filepath is not None:
        if not video_filepath.endswith(".mp4"):
            print("Invalid file type (.mp4 only)")
            exit()
        X = extract_frames_as_tensor(video_filepath)

        
        confidence = classify_video(X, model)

        print("Deep Fake Probability = {}%".format(round(confidence * 100, 2)))

    if image_filepath is not None:
        if not (image_filepath.endswith(".jpg") or image_filepath.endswith(".png")):
            print("Invalid file type (jpg and png only)")
            exit()
        X = image_to_tensor(image_filepath)

        print("Loaded image as Tensor...")
        print("Evaluating with Soft Margin CNN Classifer...")
        confidence = classify_image(X, model)
        print("Deep Fake Probability = {}%".format(round(confidence * 100, 2)))




