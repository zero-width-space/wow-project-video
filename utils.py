from manim import *
from video import VideoMobject


class BaseSection(Scene):
    def show_section_title(self, text, subtext="", subsubtext="", time=1.0):
        title = Text(text, font_size=48).move_to(UP * 0.5)
        underline = Underline(title)

        self.play(
            Write(title),
            Create(underline),
            run_time=1.0 / time,
        )

        if subtext:
            subtitle = Text(subtext, font_size=40).scale(0.5).next_to(title, DOWN)
            self.play(Write(subtitle), run_time=1.0 / time)

        if subsubtext:
            subsubtitle = (
                Text(subsubtext, color=GRAY, font_size=32)
                .scale(0.5)
                .next_to(subtitle, DOWN)
            )
            self.play(Write(subsubtitle), run_time=1.0 / time)

        self.wait(0.5 / time)
        self.fade_out(time=time)

    def fade_out(self, time=1.0):
        self.play(
            *[FadeOut(obj) for obj in self.mobjects],
            run_time=1.0 / time,
        )
