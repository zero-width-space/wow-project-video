from utils import BaseSection, VideoMobject
from manim import *


class ChessIntro(BaseSection):
    def construct(self):
        self.show_section_title("Introduction to chess", "A brief overview")
        text = Paragraph(
            "Chess is a 2 player game where the goal is to capture,",
            'or "checkmate", the other player\'s king',
            font_size=30,
        ).to_edge(UP)
        video = VideoMobject("checkmate.mp4").scale_to_fit_height(5).to_edge(DOWN)
        self.add(video)
        self.play(Write(text))
        self.wait(5)
        text = Text("Checkmate!", font_size=16).next_to(video, UP)
        self.play(Write(text))
        self.wait()
        self.fade_out()

        text = Paragraph(
            "Different pieces can move in different ways",
            font_size=30,
        ).to_edge(UP)
        video = VideoMobject("pieces.mp4").scale_to_fit_height(5).to_edge(DOWN)
        self.add(video)
        self.play(Write(text))
        self.wait(5)
        self.fade_out()

        text = Paragraph(
            "A chess engine is a chess playing program that tries to",
            "play the best possible move in response to the user to",
            "maximise its chances of winning",
            font_size=30,
        ).to_edge(UP)
        video = VideoMobject("stockfish.mp4").scale_to_fit_height(5).to_edge(DOWN)
        self.add(video)
        self.play(Write(text))
        self.wait(1)
        text1 = Paragraph(
            "Here is an example using an engine known as Stockfish", font_size=30
        ).to_edge(UP)
        self.play(Transform(text, text1))
        self.wait(6)
        self.fade_out()
