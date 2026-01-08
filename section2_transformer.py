"""
Manim scene: "TransformerChessScene"

What this does:
- Visualizes a chessboard (8x8) with piece tokens (simplified as colored circles).
- Shows an example transformer encoder applying self-attention between tokens.
- Animates attention weights as animated curved arrows whose thickness and opacity reflect attention strength.
- Shows a policy head: after attention, generates a probability distribution over candidate moves and overlays arrows on the board sized by probability.
- Shows a value head: a numeric gauge indicating the network's value estimate for the position.

Usage (Manim Community edition, e.g. v0.16+ or v0.17):
    manim -pql transformer_chess_viz.py TransformerChessScene

Notes / customization:
- All model weights here are synthetic / randomly generated for demonstration.
- You can plug in real attention matrices and policy/value outputs by replacing the random generators.
- The code focuses on clarity and visual storytelling rather than raw performance.

"""

from manim import *
import numpy as np

BOARD_SIZE = 6  # use 6x6 for readability in animation; change to 8 for full board
SQUARE_SIZE = 0.55


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TransformerChessScene(Scene):
    def construct(self):
        # Title
        title = Text(
            "Transformer in a Chess Engine: Attention, Policy & Value"
        ).to_edge(UP)
        self.play(FadeIn(title))

        # Create visual board and tokens
        board_group, square_centers = self.create_board(BOARD_SIZE)
        self.play(Create(board_group), run_time=1.5)

        tokens = self.create_tokens(square_centers)
        self.add(tokens)

        # Show embeddings (as small labels) for a few tokens
        embedding_labels = self.show_embeddings(tokens, sample_indices=[0, 5, 10])
        self.wait(0.6)

        # Show a multi-head attention block animation
        self.show_multi_head_attention(tokens, square_centers)

        # Show policy head producing move probabilities
        self.show_policy_overlay(square_centers)

        # Show value head (gauge)
        self.show_value_gauge()

        self.wait(2)

    # ---------------- board and tokens ----------------
    def create_board(self, n):
        squares = VGroup()
        centers = []
        for i in range(n):
            for j in range(n):
                sq = Square(side_length=SQUARE_SIZE)
                sq.move_to(
                    (
                        (j - (n - 1) / 2) * SQUARE_SIZE,
                        ((n - 1) / 2 - i) * SQUARE_SIZE,
                        0,
                    )
                )
                color = LIGHT_BROWN if (i + j) % 2 == 0 else DARK_BROWN
                sq.set_fill(color, opacity=1)
                sq.set_stroke(BLACK, 0.5)
                squares.add(sq)
                centers.append(sq.get_center())
        board = VGroup(squares).arrange_in_grid(
            rows=n, cols=n
        )  # used only for grouping
        # override positions because arrange_in_grid repositions things; we already placed
        board = squares
        # small coordinate labels
        files = VGroup()
        for idx, c in enumerate(centers):
            # optional: show coordinates
            pass
        return board, centers

    def create_tokens(self, centers):
        # We'll create a few token circles to represent pieces/embeddings
        tokens = VGroup()
        n = len(centers)
        rng = np.random.RandomState(1)
        # choose a handful of occupied squares
        occupied = rng.choice(range(n), size=min(12, n // 2), replace=False)
        for i in range(n):
            if i in occupied:
                circ = Circle(radius=SQUARE_SIZE * 0.34)
                circ.move_to(centers[i])
                # color to indicate side: white-ish or black-ish
                circ.set_fill(WHITE if (i % 2 == 0) else GREY, opacity=1)
                circ.set_stroke(BLACK, 1)
                # label with a token id
                lbl = Text(f"t{i}", font_size=18)
                lbl.move_to(centers[i] + UP * 0.02)
                tokens.add(VGroup(circ, lbl))
        return tokens

    def show_embeddings(self, tokens, sample_indices=[0, 1, 2]):
        labels = VGroup()
        for idx in sample_indices:
            if idx < len(tokens):
                lbl = MathTex(f"e_{{{idx}}}")
                lbl.scale(0.45)
                lbl.next_to(tokens[idx][0], RIGHT, buff=0.08)
                labels.add(lbl)
        self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.2))
        return labels

    # ---------------- attention visuals ----------------
    def show_multi_head_attention(self, tokens, centers):
        # Simulate a single encoder layer with multiple heads and animate one head
        heads = 4
        n_tokens = len(tokens)
        # pick one "query" token to visualize from
        query_idx = 2 if n_tokens > 2 else 0

        header = (
            Text("Self-Attention (one layer, multiple heads)", font_size=24)
            .to_edge(UP)
            .shift(DOWN * 0.8)
        )
        self.play(Write(header))

        # Create a panel that shows head selection and a matrix view
        panel = RoundedRectangle(corner_radius=0.2, height=2.4, width=4.8)
        panel.to_edge(RIGHT).shift(DOWN * 0.4)
        self.play(Create(panel))

        head_labels = VGroup(*[Button(text=f"Head {i}") for i in range(heads)])
        # arrange them vertically inside panel
        for i, hl in enumerate(head_labels):
            hl.move_to(panel.get_center() + LEFT * 0.8 + UP * (0.6 - i * 0.7))
        self.play(LaggedStartMap(FadeIn, head_labels, lag_ratio=0.12))

        # show initial attention matrix for the first head
        weights = self.synthetic_attention(n_tokens, seed=10)
        matrix_mob = self.matrix_mobject(
            weights, title=f"Head 0 attention (query t{query_idx})"
        )
        matrix_mob.next_to(panel, LEFT, buff=0.3)
        self.play(FadeIn(matrix_mob))

        # Animate attention lines on the board for the selected query token
        att_lines = self.attention_lines_from_query(
            centers, query_idx, weights[query_idx]
        )
        # self.play(LaggedStartMap(GrowArrow, att_lines, lag_ratio=0.08))
        # highlight query token
        self.play(tokens[query_idx][0].animate.scale(1.08), run_time=0.4)

        # cycle through heads with updated weights
        for h in range(1, heads):
            new_w = self.synthetic_attention(n_tokens, seed=10 + h)
            # update matrix and attention on board
            new_matrix = self.matrix_mobject(
                new_w, title=f"Head {h} attention (query t{query_idx})"
            )
            new_matrix.next_to(panel, LEFT, buff=0.3)

            # create new att lines
            new_att_lines = self.attention_lines_from_query(
                centers, query_idx, new_w[query_idx]
            )
            # crossfade matrix and replace arrows
            self.play(
                Transform(matrix_mob, new_matrix),
                *[ReplacementTransform(a, b) for a, b in zip(att_lines, new_att_lines)],
                run_time=1.2,
            )
            att_lines = new_att_lines
            self.wait(0.6)

        # show multi-head aggregation: average attention
        avg_w = np.mean(
            [self.synthetic_attention(n_tokens, seed=10 + h) for h in range(heads)],
            axis=0,
        )
        avg_matrix = self.matrix_mobject(avg_w, title="Aggregated attention")
        avg_matrix.next_to(panel, LEFT, buff=0.3)
        agg_lines = self.attention_lines_from_query(
            centers, query_idx, avg_w[query_idx]
        )
        self.play(
            Transform(matrix_mob, avg_matrix),
            *[ReplacementTransform(a, b) for a, b in zip(att_lines, agg_lines)],
            run_time=1.0,
        )
        self.wait(0.6)

        # Clean up header + panel for next stage
        self.play(
            FadeOut(header), FadeOut(panel), FadeOut(head_labels), FadeOut(matrix_mob)
        )
        for a in agg_lines:
            self.play(FadeOut(a))

    def synthetic_attention(self, n, seed=0):
        rng = np.random.RandomState(seed)
        # produce a full attention matrix (n x n) where rows sum to 1
        raw = rng.rand(n, n) * (np.abs(np.linspace(-1, 1, n)) + 0.2)
        # emphasise local context: add Gaussian around diagonal
        for i in range(n):
            raw[i] += np.exp(-((np.arange(n) - i) ** 2) / (2 * (n / 6) ** 2)) * 2.0
        # softmax row-wise
        mat = np.array([softmax(raw[i]) for i in range(n)])
        return mat

    def matrix_mobject(self, mat, title="attention"):
        # small visual matrix using rectangles whose fill intensity corresponds to values
        n = mat.shape[0]
        cell_size = 0.25
        cells = VGroup()
        for i in range(n):
            row = VGroup()
            for j in range(n):
                rect = Square(side_length=cell_size)
                rect.set_fill(BLUE, opacity=float(mat[i, j]) * 0.9)
                rect.set_stroke(GREY_E, 0.2)
                rect.move_to(
                    (j - (n - 1) / 2) * cell_size * RIGHT
                    + ((n - 1) / 2 - i) * cell_size * UP
                )
                row.add(rect)
            cells.add(row)
        matrix = VGroup(*cells).arrange(DOWN, buff=0)
        matrix.scale(1)
        title_m = Text(title, font_size=18)
        box = VGroup(matrix, title_m).arrange(DOWN, buff=0.12)
        return box

    def attention_lines_from_query(self, centers, qidx, weights_row):
        # produce arrows from query token to every other token with thickness/opac reflecting weight
        arrows = VGroup()
        qpos = centers[qidx]
        max_width = 6.0
        for j, w in enumerate(weights_row):
            if j == qidx:
                # self-attention can be shown as a circle
                circ = Circle(radius=0.12)
                circ.move_to(qpos + RIGHT * SQUARE_SIZE * 0.02)
                circ.set_stroke(RED, width=2 * float(w * 2))
                circ.set_fill(RED, opacity=0.1)
                arrows.add(circ)
                continue
            target = centers[j]
            vec = target - qpos
            # an arrow that curves slightly depending on distance
            mid = (
                (qpos + target) / 2 + OUT * 0.0 + np.array([0, 0.05 * (j % 3 - 1), 0.0])
            )
            path = CubicBezier(qpos, qpos + UP * 0.08, target + UP * 0.08, target)
            arr = CurvedArrow(start_point=qpos, end_point=target, angle=0.5)
            # instead of built-in arrow, make a simple arc using Line then add arrow tip
            line = Line(qpos, target)
            # visual thickness scaled
            stroke_w = float(1.0 + (w * max_width))
            line.set_stroke(RED, width=stroke_w, opacity=min(1.0, float(w * 3)))
            arrows.add(line)
        return arrows

    # ---------------- policy visualization ----------------
    def show_policy_overlay(self, centers):
        title = (
            Text("Policy Head → Candidate Moves & Probabilities", font_size=22)
            .to_edge(UP)
            .shift(DOWN * 0.8)
        )
        self.play(Write(title))

        # pick a source square and propose a few target squares
        n = len(centers)
        rng = np.random.RandomState(42)
        source = rng.randint(0, n)
        # propose 6 candidate moves
        candidates = rng.choice(
            [i for i in range(n) if i != source], size=6, replace=False
        )
        logits = rng.randn(len(candidates)) * 1.4 + np.linspace(-1, 1, len(candidates))
        probs = softmax(logits)

        move_arrows = VGroup()
        for i, tgt in enumerate(candidates):
            arr = Arrow(start=centers[source], end=centers[tgt], buff=0.12)
            thickness = 2 + 6 * probs[i]
            arr.set_stroke(BLUE, width=thickness, opacity=0.9)
            label = MathTex(f"{probs[i]:.2f}")
            label.scale(0.4)
            label.next_to(arr.get_center(), RIGHT, buff=0.06)
            move_arrows.add(VGroup(arr, label))

        # animate arrows appearing from smallest prob to largest
        order = np.argsort(probs)
        for idx in order:
            self.play(
                GrowArrow(move_arrows[idx][0]),
                FadeIn(move_arrows[idx][1]),
                run_time=0.45,
            )
        self.wait(0.8)

        # highlight top-k
        topk = np.argsort(probs)[-2:][::-1]
        for idx in topk:
            self.play(move_arrows[idx][0].animate.scale(1.05), run_time=0.25)

        # leave them on screen a bit and fade title
        self.play(FadeOut(title))

    # ---------------- value visualization ----------------
    def show_value_gauge(self):
        # small meter showing scalar value estimate in [-1, 1]
        title = (
            Text("Value Head → Position Evaluation", font_size=22)
            .to_edge(UP)
            .shift(DOWN * 0.8)
        )
        self.play(Write(title))

        gauge = VMobject()
        # draw semicircle gauge
        arc = Arc(radius=1.1, start_angle=PI, angle=PI)
        ticks = VGroup()
        for t in np.linspace(-1, 1, 9):
            a = np.interp(t, [-1, 1], [PI, 0])
            inner = np.array([np.cos(a) * 0.95, np.sin(a) * 0.95, 0])
            outer = np.array([np.cos(a) * 1.15, np.sin(a) * 1.15, 0])
            ticks.add(Line(inner, outer))
        gauge = VGroup(arc, ticks)
        gauge.to_edge(DOWN).shift(UP * 0.5)
        label = Text("Value", font_size=16)
        label.next_to(gauge, DOWN)
        self.play(Create(gauge), FadeIn(label))

        # fake predicted value from network
        predicted_value = 0.42
        # animate needle
        start_angle = PI  # -1 corresponds to leftmost
        end_angle = np.interp(predicted_value, [-1, 1], [PI, 0])
        needle = Line(
            ORIGIN,
            np.array([np.cos(start_angle) * 0.9, np.sin(start_angle) * 0.9, 0]),
            stroke_width=6,
            color=YELLOW,
        )
        needle.move_to(gauge.get_center())
        needle.add_tip = True
        self.play(GrowFromCenter(needle))
        self.play(
            Rotate(
                needle, angle=end_angle - start_angle, about_point=gauge.get_center()
            ),
            run_time=1.6,
        )

        # numeric display
        num = DecimalNumber(predicted_value, num_decimal_places=2)
        num.next_to(label, RIGHT, buff=0.6)
        self.play(Write(num))

        self.wait(0.6)
        self.play(FadeOut(title))


# Supporting UI Button (simple)
class Button(VMobject):
    def __init__(self, text="Button", width=1.2, height=0.4, **kwargs):
        super().__init__(**kwargs)
        rect = RoundedRectangle(corner_radius=0.08, height=height, width=width)
        lbl = Text(text, font_size=16)
        self.add(rect, lbl)
        lbl.move_to(rect.get_center())


# Colors
LIGHT_BROWN = "#f0d9b5"
DARK_BROWN = "#b58863"
