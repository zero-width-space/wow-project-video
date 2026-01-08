from manim import *


class TransformerIllustration(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"

        # -----------------------------
        # Input tokens
        # -----------------------------
        token_texts = ["The", "model", "learns", "relationships"]
        tokens = VGroup()

        for word in token_texts:
            box = RoundedRectangle(
                width=2.2,
                height=0.9,
                corner_radius=0.15,
                stroke_color=BLUE,
                fill_color=BLUE_E,
                fill_opacity=0.8,
            )
            text = Text(word, font_size=28)
            text.move_to(box.get_center())
            tokens.add(VGroup(box, text))

        tokens.arrange(RIGHT, buff=0.4)
        tokens.to_edge(UP)

        input_label = Text("Input tokens", font_size=30).next_to(tokens, UP)

        self.play(
            FadeIn(input_label),
            LaggedStart(*[FadeIn(t) for t in tokens], lag_ratio=0.2),
        )
        self.wait(0.5)

        # -----------------------------
        # Attention visualization
        # -----------------------------
        attention_lines = VGroup()
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j:
                    line = Line(
                        tokens[i].get_bottom(),
                        tokens[j].get_bottom(),
                        stroke_opacity=0.3,
                        stroke_width=2,
                        color=YELLOW,
                    )
                    attention_lines.add(line)

        attention_label = Text("Self-Attention", font_size=30)
        attention_label.next_to(attention_lines, DOWN)

        self.play(Create(attention_lines), FadeIn(attention_label))
        self.wait(1)
        self.play(FadeOut(attention_lines), FadeOut(attention_label))

        # -----------------------------
        # Transformer block
        # -----------------------------
        block = RoundedRectangle(
            width=9,
            height=2.5,
            corner_radius=0.3,
            stroke_color=GREEN,
            fill_color=GREEN_E,
            fill_opacity=0.85,
        )

        block_text = Text(
            "Transformer Block\n(Self-Attention + Feedforward)",
            font_size=32,
            line_spacing=0.9,
        ).move_to(block.get_center())

        transformer = VGroup(block, block_text)
        transformer.move_to(ORIGIN)

        arrows_down = VGroup()
        for token in tokens:
            arrow = Arrow(
                token.get_bottom(),
                block.get_top(),
                buff=0.1,
                stroke_width=3,
                color=WHITE,
            )
            arrows_down.add(arrow)

        self.play(LaggedStart(*[Create(a) for a in arrows_down], lag_ratio=0.15))
        self.play(FadeIn(transformer))
        self.wait(1)

        # -----------------------------
        # Output tokens
        # -----------------------------
        output_tokens = tokens.copy()
        output_tokens.to_edge(DOWN)

        output_label = Text("Output representations", font_size=30).next_to(
            output_tokens, DOWN
        )

        arrows_up = VGroup()
        for token in output_tokens:
            arrow = Arrow(
                block.get_bottom(),
                token.get_top(),
                buff=0.1,
                stroke_width=3,
                color=WHITE,
            )
            arrows_up.add(arrow)

        self.play(LaggedStart(*[Create(a) for a in arrows_up], lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(t) for t in output_tokens], lag_ratio=0.2))
        self.play(FadeIn(output_label))

        self.wait(2)
