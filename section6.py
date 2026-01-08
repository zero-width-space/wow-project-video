from utils import BaseSection, VideoMobject
from manim import *


class Demonstration(BaseSection):
    def construct(self):
        self.show_section_title("Demonstration", "")

        top_text = Paragraph(
            "Here is a demonstration of our model against a human player",
            "(sped up for brevity, but run in real time)",
            font_size=30,
        ).to_edge(UP)
        self.play(Write(top_text))

        video = (
            VideoMobject("demonstration.mp4", speed=3)
            .scale_to_fit_height(5.5)
            .to_edge(DOWN)
        )
        self.add(video)
        self.wait_until(lambda: video.finished)
        self.play(FadeOut(video))

        new_text = Paragraph(
            "As you can see, even though it lost in the end, it still",
            "can close to a draw",
            font_size=30,
        ).to_edge(UP)
        self.play(Transform(top_text, new_text))
        self.wait(2)
        self.fade_out()
