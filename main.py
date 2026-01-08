import os
import subprocess

VIDEO_FILE = "video_files.txt"

parts = ["section0", "section1", "section2", "section3", "section4", "section5"]

for part in parts:
    os.system(f"manim -qh {part}.py")

with open(VIDEO_FILE, "w") as file:
    file.write(
        "\n".join(
            f"file '{subprocess.check_output(
        f"ls media/videos/{part}/1080p60/*.mp4", shell=True
    ).decode().strip()}'"
            for part in parts
        )
    )

os.system("ffmpeg -f concat -i video_files.txt output.mp4")
