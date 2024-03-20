import streamlit as st
import matplotlib.pyplot as plt

import cv2
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm.notebook import tqdm

import torch
from facenet_pytorch import (MTCNN)

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)

from PIL import Image
from moviepy.editor import VideoFileClip, ImageSequenceClip

## Import modules
import av_recorder
from av_seprator import seperator
from audio_app import inference
import video_app as vp


os.environ['XDG_CACHE_HOME'] = '/home/msds2023/jlegara/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/msds2023/jlegara/.cache'


## Input file with audio
input_file = 'videos/output_with_audio.mp4'



scene = 'videos/output.avi'
def video_app():
    ## Input load video
    
    combined_images = vp.proba(scene)
    _,vid_fps = vp.video_capture(scene)
    
    return combined_images,vid_fps

def main():
    st.title("Emotion Detection App")

    # Button to start camera and record
    if st.button("Start Camera"):
        av_recorder.start_rec()
        
        ## Seperate audio and video
        seperator(input_file)
        
        # st.write("Done Recording")
        combined_images,vid_fps = video_app()
        skips=2
        
        clip_with_plot = ImageSequenceClip(combined_images,
                                   fps=vid_fps/skips)
        
        # video_file = "temp_video.mp4"
        # clip_with_plot.write_videofile(scene,codec='libx264')
        clip_with_plot.ipython_display(width=700,maxduration=120)
        
        graph = '__temp__.mp4'
        # Display the video in Streamlit
        st.video(graph)
        


if __name__ == "__main__":
    main()

   