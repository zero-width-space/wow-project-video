from utils import BaseSection
from manim import *


class Ending(BaseSection):
    def construct(self):
        self.show_section_title("Ending", "Thanks and acknowledgements")

        top_text = Paragraph(
            "Thank you for watching this video!",
            font_size=30,
        ).to_edge(UP)
        self.play(Write(top_text))

        body_text = Paragraph(
            "Insert credits here",
            "Insert credits here",
            "Insert credits here",
            "Insert credits here",
            "Insert credits here",
            font="monospace",
            font_size=32,
        ).scale(0.5)
        self.play(Write(body_text))
        self.wait()
        self.fade_out()
