import glob
import os

RESOLUTION = {
    "l": "480p15",
    "m": "720p30",
    "h": "1080p60",
    "p": "1440p60",
    "k": "2160p60",
}
QUALITY = "l"

parts = [
    "section0",
    "section1",
    "section2",
    "section3",
    "section4",
    "section6",
    "section7",
]

# for part in parts:
#     os.system(f"manim -q{QUALITY} {part}.py")

with open("video_files.txt", "w") as file:
    for part in parts:
        pattern = f"media/videos/{part}/{RESOLUTION[QUALITY]}/*.mp4"
        for path in sorted(glob.glob(pattern)):
            file.write(f"file '{path}'\n")

os.system(f"ffmpeg -y -f concat -i video_files.txt output_{RESOLUTION[QUALITY]}.mp4")
