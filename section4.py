from utils import BaseSection
from manim import *


class Demonstration(BaseSection):
    def construct(self):
        self.show_section_title("Demonstration", "")

        top_text = Paragraph(
            "Here is a demonstration of our model against a human player",
            font_size=30,
        ).to_edge(UP)
        self.play(Write(top_text))

        body_text = Paragraph(
            "Insert video here",
            "Insert video here",
            "Insert video here",
            "Insert video here",
            "Insert video here",
            font="monospace",
            font_size=32,
        ).scale(0.5)
        self.play(Write(body_text))
        self.wait()
        self.fade_out()
