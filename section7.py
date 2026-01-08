from utils import BaseSection, VideoMobject
from manim import *


class Ending(BaseSection):
    def construct(self):
        self.show_section_title("The End", "Thanks and acknowledgements")

        body_text = Paragraph(
            "Our model is now up on Lichess",
            "(https://lichess.org/@/ChessEngine_AI)",
            font_size=30,
        ).to_edge(UP)
        self.play(Write(body_text))
        self.wait(1.2)
        video = (
            VideoMobject("fools_mate_engine.mp4").scale_to_fit_height(5.5).to_edge(DOWN)
        )
        self.add(video)
        self.wait_until(lambda: video.finished)
        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not body_text])

        new_text = Paragraph(
            "This project would not have been possible without",
            "the searchless_chess project",
            "(https://github.com/google-deepmind/searchless_chess)",
            "and their paper, dataset and GitHub repository",
            font_size=32,
        ).to_edge(UP)
        self.play(Transform(body_text, new_text))
        self.wait(1.2)

        image = ImageMobject("pdf.png").scale_to_fit_height(5).to_edge(DOWN)
        self.play(FadeIn(image))

        self.wait(2)
        self.play(FadeOut(image))

        new_text = Paragraph(
            "Christopher and Hugh Bishop's book on deep learning also",
            "served as a good primer on neural networks and deep learning",
            font_size=32,
        ).to_edge(UP)
        self.play(Transform(body_text, new_text))
        self.wait(1.2)

        image = ImageMobject("bishop_book.jpg").scale_to_fit_height(5).to_edge(DOWN)
        self.play(FadeIn(image))

        self.wait(2)
        self.play(FadeOut(image))

        new_text = Paragraph(
            "Last but not least, 3Blue1Brown's multiple YouTube series on AI",
            "and other related fields served as good intuition for our project",
            font_size=32,
        ).to_edge(UP)
        self.play(Transform(body_text, new_text))
        self.wait(1.2)

        image = ImageMobject("3b1b.png").scale_to_fit_height(5).to_edge(DOWN)
        self.play(FadeIn(image))

        self.wait(2)
        self.play(FadeOut(image))

        new_text = Paragraph("Thank you once again for watching!", font_size=48)
        underline = Underline(new_text)
        self.play(Transform(body_text, new_text), Create(underline))
        self.wait()
        self.fade_out()
