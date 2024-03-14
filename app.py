import streamlit as st
# import numpy as np
import av_recorder
from av_seprator import seperator
from audio_app import inference
import matplotlib.pyplot as plt

## Input file
input_file = 'videos/output_with_audio.mp4'

def func_call():
    ## Get details related to audio
    _,text,sent,perc = inference(input_file,'Sentiment Only')
    
    return sent,perc

# Function to simulate emotion detection
# def detect_emotions():
#     # emotions = ['happy', 'sad', 'neutral', 'anger', 'surprise', 'fear', 'disgust']
#     emotions,val = func_call()
#     percentages = val*100
    
#     return emotions,percentages
    # return dict(zip(emotions, percentages))
    

def plot_custom_graph(x_values, y_values):
    # Plot the data using Matplotlib
    plt.bar(x_values, y_values)
    plt.xlabel('Emotions')
    plt.ylabel('Percentage')
    plt.title('Custom Graph')

    # Display the plot using Streamlit
    st.pyplot()

# UI elements
def main():
    st.title("Emotion Detection App")

    # Button to start camera and record
    if st.button("Start Camera"):
        av_recorder.start_rec()

        ## Seperate audio and video file
        seperator(input_file)
        
        
        emotions,val = func_call()
        st.text(f"{emotions}:{val*100}")
        x_values = [emotions]
        y_values = [val]
        
        plot_custom_graph(x_values,y_values)
        # Display emotions graph
        # st.subheader("Emotions Detected")
        # emotions_data = detect_emotions()
        # st.bar_chart(emotions_data)
    
if __name__ == "__main__":
    main()