from utils import BaseSection
from parse_loss import get_data
from manim import *


class OurModel(BaseSection):
    def construct(self):
        self.show_section_title("Our model", "Training process and results")

        top_text = Paragraph(
            "Due to time constraints, we cannot create a model large enough",
            "for self-play to be effective",
            font_size=30,
        ).to_edge(UP)

        self.play(Write(top_text))
        self.wait()

        new_text = Paragraph(
            "Therefore, we decided to use distillation, where we train a",
            "weaker model using data from a stronger model",
            font_size=30,
        ).to_edge(UP)
        self.play(Transform(top_text, new_text))

        left_box = (
            RoundedRectangle(width=4, height=4, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .shift(4 * LEFT + DOWN)
        )
        logo = ImageMobject("stockfish.png").scale_to_fit_height(4).move_to(left_box)
        logo_text = Text("Stockfish", font_size=24).next_to(logo, UP)
        right_box = (
            RoundedRectangle(width=4, height=4, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .shift(4 * RIGHT + DOWN)
        )
        text = Text("Our model", font_size=24).move_to(right_box)
        arrow = Arrow(left_box.get_edge_center(RIGHT), right_box.get_edge_center(LEFT))
        arrow_text = (
            Text("Generate training data", font_size=32).scale(0.5).next_to(arrow, UP)
        )
        self.play(
            Create(left_box),
            FadeIn(logo),
            Write(logo_text),
            Create(right_box),
            Write(text),
        )
        self.wait()
        self.play(Create(arrow), Write(arrow_text))

        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        new_text = Paragraph(
            "Here is a snippet of some training",
            font_size=30,
        ).to_edge(UP)
        self.play(Transform(top_text, new_text))

        body_text = Paragraph(
            "Insert something here",
            "Insert something here",
            "Insert something here",
            "Insert something here",
            "Insert something here",
            font="monospace",
            font_size=32,
        ).scale(0.5)
        self.play(Write(body_text))
        self.wait()

        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        new_text = Paragraph(
            "Here is a graph showing the loss of our model,",
            "which measures how much our model fails",
            font_size=30,
        ).to_edge(UP)
        self.play(Transform(top_text, new_text))

        axes = Axes(
            x_range=(0, 1200000, 50000),
            y_range=(1, 5, 0.2),
            x_length=10,
            y_length=5,
            x_axis_config={
                "numbers_to_include": np.arange(0, 1200001, 200000),
                "numbers_with_elongated_ticks": np.arange(0, 1200001, 100000),
                "label_direction": DOWN,
            },
            y_axis_config={"numbers_to_include": np.arange(1, 5.01, 0.5)},
        ).next_to(top_text, DOWN, buff=0.5)

        _, smooth_data = get_data()

        smooth_graph = axes.plot_line_graph(
            x_values=[s for s, _ in smooth_data],
            y_values=[l for _, l in smooth_data],
            add_vertex_dots=True,
            line_color=GRAY,
            vertex_dot_radius=0.04,
        )

        labels = axes.get_axis_labels(
            Text("Training steps", font_size=32).scale(0.5),
            Text("Loss (lower is better)", font_size=32).scale(0.5),
        )
        self.play(Create(axes), Create(labels))
        self.wait()
        self.play(Create(smooth_graph))
        self.wait(2)

        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        new_text = Paragraph(
            "Here is a graph showing the accuracy of our model in",
            "Lichess puzzles, seperated by puzzle difficulty rating",
            font_size=30,
        ).to_edge(UP)
        self.play(Transform(top_text, new_text))

        bar_chart = BarChart(
            values=[1, 2, 3, 4, 5],
            bar_names=["0-1", "1-2", "2-3", "3-4"],
            x_length=10,
            y_length=5,
        ).next_to(top_text, DOWN, buff=0.5)
        self.play(Create(bar_chart))
        self.wait(2)

        self.fade_out()
