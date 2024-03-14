from moviepy.editor import VideoFileClip

def seperator(input_file):

    # Path to the input video file
    # input_video_path = 'videos/output_with_audio.mp4'

    # Path to save the extracted audio file
    output_audio_path = 'audios/output_audio.wav'

    # Load the video clip
    video_clip = VideoFileClip(input_file)

    # Extract audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio clip to a WAV file
    audio_clip.write_audiofile(output_audio_path)

    # Close the video clip
    video_clip.close()

# print("Audio extracted and saved to:", output_audio_path)