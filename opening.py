from utils import BaseSection
from manim import *


class Opening(BaseSection):
    def construct(self) -> None:
        self.show_section_title("Building a chess engine", "Video made with Manim")
