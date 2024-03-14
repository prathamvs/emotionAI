import cv2
import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm.notebook import tqdm

from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.editor import VideoFileClip

import torch
from facenet_pytorch import (MTCNN)

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

from PIL import Image, ImageDraw

# Set cache directories for XDG and Hugging Face Hub
os.environ['XDG_CACHE_HOME'] = '/home/msds2023/jlegara/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/msds2023/jlegara/.cache'

scene = 'videos/output1.avi'

def video_capture():
    
    # Open the video file
    video_capture = cv2.VideoCapture(scene)

    # Get the frames per second of the video
    vid_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Initialize an empty list to store frames
    video_data = []

    # Read frames from the video
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If the frame was not successfully read, break the loop
        if not ret:
            break

        # Convert the frame from BGR to RGB (OpenCV uses BGR, but MoviePy uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the list
        video_data.append(frame_rgb)

    # Release the video capture object
    video_capture.release()

    # Convert the list of frames to a numpy array
    video_data = np.array(video_data)
    # print(video_data.shape)
    
    return video_data

def cnn():
    # Initialize MTCNN model for single face cropping
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=200,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False
    )

    # Load the pre-trained model and feature extractor
    extractor = AutoFeatureExtractor.from_pretrained(
        "trpakov/vit-face-expression"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "trpakov/vit-face-expression"
    )
    
    # print(mtcnn , extractor , model)
    return mtcnn , extractor , model


## Detect Emotions
def detect_emotions(image):
    """
    Detect emotions from a given image.
    Returns a tuple of the cropped face image and a
    dictionary of class probabilities.
    """
    mtcnn , extractor, model = cnn()
    temporary = image.copy()

    # Detect faces in the image using the MTCNN group model
    sample = mtcnn.detect(temporary)
    if sample[0] is not None:
        box = sample[0][0]

        # Crop the face
        face = temporary.crop(box)

        # Pre-process the face
        inputs = extractor(images=face, return_tensors="pt")

        # Run the image through the model
        outputs = model(**inputs)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits,
                                                    dim=-1)

        # Retrieve the id2label attribute from the configuration
        config = AutoConfig.from_pretrained(
            "trpakov/vit-face-expression"
        )
        id2label = config.id2label

        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().numpy().tolist()[0]

        # Map class labels to their probabilities
        class_probabilities = {
            id2label[i]: prob for i, prob in enumerate(probabilities)
        }
        
        print(face,class_probabilities)
        return face, class_probabilities
    return None, None

## Combined images
def create_combined_image(face, class_probabilities):
    """
    Create an image combining the detected face and a barplot
    of the emotion probabilities.

    Parameters:
    face (PIL.Image): The detected face.
    class_probabilities (dict): The probabilities of each
        emotion class.

    Returns:
    np.array: The combined image as a numpy array.
    """
    # Define colors for each emotion
    colors = {
        "angry": "red",
        "disgust": "green",
        "fear": "gray",
        "happy": "yellow",
        "neutral": "purple",
        "sad": "blue",
        "surprise": "orange"
    }
    palette = [colors[label] for label in class_probabilities.keys()]

    # Create a figure with 2 subplots: one for the
    # face image, one for the barplot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Display face on the left subplot
    axs[0].imshow(np.array(face))
    axs[0].axis('off')

    # Create a barplot of the emotion probabilities
    # on the right subplot
    sns.barplot(ax=axs[1],
                y=list(class_probabilities.keys()),
                x=[prob * 100 for prob in class_probabilities.values()],
                palette=palette,
                orient='h')
    axs[1].set_xlabel('Probability (%)')
    axs[1].set_title('Emotion Probabilities')
    axs[1].set_xlim([0, 100])  # Set x-axis limits

    # Convert the figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img  = img.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    print(img)
    return img


if __name__== '__main__':
    video_data = video_capture()
    frame=video_data[70]
    print(frame)
    image = Image.fromarray(frame)
    detect_emotions(image)