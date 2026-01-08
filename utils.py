from manim import *
from video import VideoMobject


class BaseSection(Scene):
    def show_section_title(self, text, subtext=""):
        title = Text(text, font_size=48).move_to(UP * 0.5)
        subtitle = Text(subtext, font_size=20).next_to(title, DOWN)
        underline = Underline(title)
        self.play(Write(title), Create(underline))
        self.play(Write(subtitle))
        self.wait(0.5)
        self.fade_out()

    def fade_out(self):
        self.play(*[FadeOut(obj) for obj in self.mobjects])
