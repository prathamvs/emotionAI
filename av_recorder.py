import cv2
import numpy as np
import pyaudio
import wave
import threading
import subprocess
import os
import time

# Video settings
output_file = 'videos/output.avi'
fps = 30
width = 640
height = 480

# Audio settings
audio_filename = 'audios/temp_audio.wav'
duration = 20  # seconds
sample_rate = 44100
channels = 2  # stereo
chunk = 1024

class AudioRecorder(threading.Thread):
    def __init__(self, filename):
        threading.Thread.__init__(self)
        self.filename = filename
        self.frames = []

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk)

        print("Recording audio...")
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            self.frames.append(data)

        print("Finished recording audio.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

class VideoRecorder:
    def __init__(self, filename, fps, width, height):
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()

def start_rec():
    # Create video recorder
    video_recorder = VideoRecorder(output_file, fps, width, height)

    # Start audio recorder
    audio_recorder = AudioRecorder(audio_filename)
    audio_recorder.start()

    # Start video capture

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or time.time() - start_time >= duration:
            break

        # Write video frame
        video_recorder.write_frame(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video recorder
    video_recorder.release()

    # Wait for audio recorder to finish
    audio_recorder.join()

    # Merge audio and video using ffmpeg
    output_filename = 'videos/output_with_audio.mp4'
    subprocess.call(['ffmpeg', '-y', '-i', output_file, '-i', audio_filename, '-c:v', 'copy', '-c:a', 'aac', output_filename])

    # Cleanup temporary files
    os.remove(output_file)
    os.remove(audio_filename)

    # Release OpenCV resources
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    start_rec()