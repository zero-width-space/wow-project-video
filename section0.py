from utils import BaseSection
from manim import *


class Opening(BaseSection):
    def construct(self) -> None:
        self.show_section_title(
            "Making an AI chess engine",
            "Li Dianheng, Miyazaki Keishi, Tan Le Qian",
            "Made with Manim",
            time=1
        )
