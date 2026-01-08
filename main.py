import os
import subprocess

VIDEO_FILE = "video_files.txt"
RESOLUTION = {
    "l": "480p30",
    "m": "720p30",
    "h": "1080p60",
    "p": "1440p60",
    "k": "2160p60",
}
QUALITY = "h"

parts = [
    "section0",
    "section1",
    "section2",
    "section3",
    "section4",
    "section6",
    "section7",
]

for part in parts:
    os.system(f"manim -q{QUALITY} {part}.py")

with open(VIDEO_FILE, "w") as file:
    file.write(
        "\n".join(
            f"file '{subprocess.check_output(
        f"ls media/videos/{part}/{RESOLUTION[QUALITY]}/*.mp4", shell=True
    ).decode().strip()}'"
            for part in parts
        )
    )

os.system(f"ffmpeg -y -f concat -i video_files.txt output_{RESOLUTION[QUALITY]}.mp4")
