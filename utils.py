from manim import *
from video import VideoMobject


class BaseSection(Scene):
    def show_section_title(self, text, subtext="", subsubtext=""):
        title = Text(text, font_size=48).move_to(UP * 0.5)
        underline = Underline(title)
        self.play(Write(title), Create(underline))
        if subtext:
            subtitle = Text(subtext, font_size=40).scale(0.5).next_to(title, DOWN)
            self.play(Write(subtitle))
        if subsubtext:
            subsubtitle = (
                Text(subsubtext, color=GRAY, font_size=32)
                .scale(0.5)
                .next_to(subtitle, DOWN)
            )
            self.play(Write(subsubtitle))
        self.wait(0.5)
        self.fade_out()

    def fade_out(self):
        self.play(*[FadeOut(obj) for obj in self.mobjects])
