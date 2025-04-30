from moviepy import VideoFileClip

clip = VideoFileClip("video.webm")
clip.write_videofile("video.mp4", codec="libx264", audio_codec="aac")
