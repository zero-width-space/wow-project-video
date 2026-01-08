from utils import BaseSection
from manim import *


class Opening(BaseSection):
    def construct(self) -> None:
        self.show_section_title("Building a chess engine", "Video made with Manim")
        body_text = Paragraph(
            "Made by",
            "Li Dianheng",
            "Tan Le Qian",
            "Miyazaki Keishi",
            font_size=48,
            alignment="center",
        )
        self.play(Write(body_text))
        self.fade_out()
