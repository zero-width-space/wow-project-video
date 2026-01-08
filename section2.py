from utils import BaseSection
from manim import *
import chess
import chess.svg
from pathlib import Path
import tempfile
import uuid


class EngineIntro(BaseSection):
    def show_stockfish_tree(self):
        # =============================
        # GLOBAL LOOK + SAFE LAYOUT
        # =============================

        # Move to corner and become small

        # =============================
        # 2) BOARD: REAL POSITION + "INGESTION" EDGE SWEEP
        # =============================
        fen = "r2q1rk1/pp2bppp/2n1pn2/2bp4/2B5/2N1PN2/PPQ2PPP/R1B2RK1 w - - 0 9"
        board = chess.Board(fen)

        svg_text = chess.svg.board(board=board, size=720, coordinates=False)

        tmp_dir = Path(tempfile.gettempdir())
        svg_path = tmp_dir / f"manim_chess_board_{uuid.uuid4().hex}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        board_svg = SVGMobject(str(svg_path))
        board_svg.set_z_index(5)
        board_svg.scale_to_fit_height(3)
        board_svg.to_edge(LEFT, buff=0.55).shift(0.5 * DOWN)

        self.play(FadeIn(board_svg, shift=RIGHT * 0.15), run_time=0.55)

        # --- Build edge overlay aligned to board bounding box
        overlay = VGroup().set_z_index(20)  # above board pieces
        left, right, top, bottom = (
            board_svg.get_left(),
            board_svg.get_right(),
            board_svg.get_top(),
            board_svg.get_bottom(),
        )
        xmin, xmax = left[0], right[0]
        ymin, ymax = bottom[1], top[1]
        w, h = (xmax - xmin), (ymax - ymin)
        sq_w, sq_h = w / 8.0, h / 8.0

        def square_corners(file_idx, rank_idx_top0):
            # rank_idx_top0: 0..7 (0 is top, 7 is bottom)
            x0 = xmin + file_idx * sq_w
            x1 = x0 + sq_w
            y1 = ymax - rank_idx_top0 * sq_h
            y0 = y1 - sq_h
            return (x0, y0), (x1, y0), (x1, y1), (x0, y1)

        edges = []
        ordering = []

        # order by top-left -> bottom-right (f + r), with micro-stagger by edge
        edge_order = {"top": 0, "right": 1, "bottom": 2, "left": 3}

        for r in range(8):  # 0 top .. 7 bottom
            for f in range(8):  # 0 left .. 7 right
                (x0, y0), (x1, y0b), (x1b, y1), (x0b, y1b) = square_corners(f, r)
                bl = np.array([x0, y0, 0])
                br = np.array([x1, y0, 0])
                tr = np.array([x1, y1, 0])
                tl = np.array([x0, y1, 0])

                e_top = Line(tr, tl)
                e_right = Line(br, tr)
                e_bottom = Line(bl, br)
                e_left = Line(tl, bl)

                per_sq = [
                    ("top", e_top),
                    ("right", e_right),
                    ("bottom", e_bottom),
                    ("left", e_left),
                ]
                for name, e in per_sq:
                    e.set_stroke(color=WHITE, width=2.0, opacity=0.0)
                    edges.append(e)
                    ordering.append((f + r, edge_order[name]))

        # Sort edges to sweep diagonally
        edges_sorted = [e for _, e in sorted(zip(ordering, edges), key=lambda x: x[0])]
        overlay.add(*edges_sorted)
        self.add(overlay)

        # Smooth sweep: fade in with lag, then fade out with lag
        sweep_in = LaggedStart(
            *[e.animate.set_stroke(opacity=0.85) for e in edges_sorted],
            lag_ratio=0.006,
            run_time=0.9,
        )
        sweep_out = LaggedStart(
            *[e.animate.set_stroke(opacity=0.0) for e in edges_sorted],
            lag_ratio=0.006,
            run_time=0.85,
        )
        self.play(sweep_in)
        self.play(sweep_out)

        # Make room: dim board and remove overlay
        self.play(
            board_svg.animate.set_opacity(0.25),
            FadeOut(overlay),
            run_time=0.5,
        )

        # =============================
        # 3) SEARCH TREE: DFS + CLEAR ALPHA-BETA PRUNING
        # Depth = 3 plies (root MAX -> MIN -> MAX -> leaf evals)
        # We'll show a *strong* prune: left subtree sets alpha high, right subtree quickly sets beta low.
        # =============================

        # ---- Layering
        Z_EDGE = 2
        Z_NODE = 6
        Z_TEXT = 25
        Z_TAG = 40

        def node(radius=0.23):
            c = Circle(radius=radius)
            c.set_stroke(WHITE, 2)
            c.set_fill("#1b2230", opacity=1.0)
            c.set_z_index(Z_NODE)
            return c

        def connect(a, b):
            ln = Line(a.get_center(), b.get_center())
            ln.set_stroke(WHITE, 2, opacity=0.55)
            ln.set_z_index(Z_EDGE)
            return ln

        def val_above(n, dy=0.32):
            return n.get_center() + UP * dy

        def score_text(s, size=22):
            # Keep colors subtle but readable (engine-ish)
            if s.startswith("+"):
                col = GREEN_B
            elif s.startswith("-"):
                col = RED_B
            else:
                col = WHITE
            t = Text(s, font_size=size, color=col)
            t.set_z_index(Z_TEXT)
            return t

        def tag_text(s, size=18, color=YELLOW):
            t = Text(s, font_size=size, color=color)
            t.set_z_index(Z_TAG)
            return t

        def flash_path(objs, color=YELLOW, width=4, rt=0.22):
            # objs: list of Lines
            return AnimationGroup(
                *[
                    o.animate.set_stroke(color=color, width=width, opacity=0.95)
                    for o in objs
                ],
                run_time=rt,
                lag_ratio=0.0,
            )

        # ---- Layout: place tree center-right, below logo safe zone
        tree = VGroup().set_z_index(3)
        origin = RIGHT * 2.5 + DOWN * 1.5

        # Vertical levels
        Y0 = origin[1] + 2.55  # root
        Y1 = origin[1] + 1.25  # MIN
        Y2 = origin[1] + -0.1  # MAX
        Y3 = origin[1] + -1.8  # leaves

        # Horizontal positions (spaced to avoid overlaps)
        x_root = origin[0] + 0.0
        x_l1 = origin[0] + [-1.9, 1.9]  # A (left), B (right)
        x_l2 = origin[0] + [-2.8, -1.0, 1.0, 2.8]  # A1, A2, B1, B2
        x_l3 = origin[0] + [
            -3.35,
            -2.25,
            -1.55,
            -0.45,
            0.45,
            1.55,
            2.25,
            3.35,
        ]  # 8 leaves

        # Create nodes
        root = node(0.26).move_to([x_root, Y0, 0])
        l1 = [node().move_to([x, Y1, 0]) for x in x_l1]  # MIN nodes
        l2 = [node().move_to([x, Y2, 0]) for x in x_l2]  # MAX nodes
        l3 = [node(0.20).move_to([x, Y3, 0]) for x in x_l3]  # leaf eval nodes

        # Add into group
        tree.add(root, *l1, *l2, *l3)

        # Create edges
        e_root_A = connect(root, l1[0])
        e_root_B = connect(root, l1[1])

        e_A_A1 = connect(l1[0], l2[0])
        e_A_A2 = connect(l1[0], l2[1])
        e_B_B1 = connect(l1[1], l2[2])
        e_B_B2 = connect(l1[1], l2[3])

        # Leaves under each MAX (2 each)
        e_A1_L1 = connect(l2[0], l3[0])
        e_A1_L2 = connect(l2[0], l3[1])
        e_A2_L1 = connect(l2[1], l3[2])
        e_A2_L2 = connect(l2[1], l3[3])

        e_B1_L1 = connect(l2[2], l3[4])
        e_B1_L2 = connect(l2[2], l3[5])
        e_B2_L1 = connect(l2[3], l3[6])
        e_B2_L2 = connect(l2[3], l3[7])

        edges_all = [
            e_root_A,
            e_root_B,
            e_A_A1,
            e_A_A2,
            e_B_B1,
            e_B_B2,
            e_A1_L1,
            e_A1_L2,
            e_A2_L1,
            e_A2_L2,
            e_B1_L1,
            e_B1_L2,
            e_B2_L1,
            e_B2_L2,
        ]
        tree.add(*edges_all)

        # Show tree
        self.play(FadeIn(tree, shift=UP * 0.15), run_time=0.65)

        # ---- Alpha/Beta small HUD near root + current MIN node (kept minimal)
        alpha_hud = (
            tag_text("α = -∞", size=18, color=YELLOW)
            .next_to(root, LEFT, buff=0.55)
            .shift(UP * 0.15)
        )
        beta_hud = (
            tag_text("β = +∞", size=18, color=BLUE_B)
            .next_to(root, RIGHT, buff=0.55)
            .shift(UP * 0.15)
        )
        alpha_hud.set_z_index(Z_TAG)
        beta_hud.set_z_index(Z_TAG)
        self.play(FadeIn(alpha_hud), FadeIn(beta_hud), run_time=0.25)

        # ---- Helper: “visit” highlight node + edge briefly
        def visit(node_obj, edge_obj=None, rt=0.22):
            anims = []
            anims.append(node_obj.animate.set_fill("#2a3347", opacity=1.0))
            if edge_obj is not None:
                anims.append(
                    edge_obj.animate.set_stroke(color=YELLOW, width=4, opacity=0.95)
                )
            return AnimationGroup(*anims, run_time=rt, lag_ratio=0.0)

        def unvisit(node_obj, edge_obj=None, rt=0.18):
            anims = []
            anims.append(node_obj.animate.set_fill("#1b2230", opacity=1.0))
            if edge_obj is not None:
                anims.append(
                    edge_obj.animate.set_stroke(color=WHITE, width=2, opacity=0.55)
                )
            return AnimationGroup(*anims, run_time=rt, lag_ratio=0.0)

        # ---- Leaf eval values chosen to force a strong prune
        # Left subtree yields high value => alpha becomes +0.65
        # Right subtree first explored child yields beta = +0.40 => prune sibling branch
        leaf_vals = [
            "+0.75",
            "+0.60",  # A1 leaves => A1 (MAX) = +0.75
            "+0.65",
            "+0.55",  # A2 leaves => A2 (MAX) = +0.65 => A (MIN) = min(0.75,0.65)=0.65 => alpha=0.65
            "+0.40",
            "+0.30",  # B1 leaves => B1 (MAX) = +0.40 => at B (MIN): beta=0.40 <= alpha => prune B2
            "+0.10",
            "-0.20",  # B2 leaves (never visited; pruned)
        ]

        # Store displayed texts for potential fade/prune
        leaf_texts = [None] * 8
        l2_texts = [None] * 4
        l1_texts = [None] * 2
        root_text = None

        # ---- DFS WALK (explicit)
        # ROOT -> A (MIN)
        self.play(visit(root), run_time=0.15)
        self.play(visit(l1[0], e_root_A), run_time=0.25)

        # A -> A1 (MAX) -> leaves 0,1
        self.play(visit(l2[0], e_A_A1), run_time=0.25)

        # Leaf 0
        self.play(visit(l3[0], e_A1_L1), run_time=0.22)
        t0 = score_text(leaf_vals[0], size=20).move_to(val_above(l3[0], 0.26))
        leaf_texts[0] = t0
        self.play(FadeIn(t0, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[0], e_A1_L1), run_time=0.12)

        # Leaf 1
        self.play(visit(l3[1], e_A1_L2), run_time=0.22)
        t1 = score_text(leaf_vals[1], size=20).move_to(val_above(l3[1], 0.26))
        leaf_texts[1] = t1
        self.play(FadeIn(t1, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[1], e_A1_L2), run_time=0.12)

        # Backup to A1 (MAX): +0.75
        a1_val = score_text("+0.75", size=22).move_to(val_above(l2[0], 0.32))
        l2_texts[0] = a1_val
        self.play(TransformFromCopy(t0, a1_val), run_time=0.25)
        self.play(unvisit(l2[0], e_A_A1), run_time=0.14)

        # A -> A2 (MAX) -> leaves 2,3
        self.play(visit(l2[1], e_A_A2), run_time=0.25)

        # Leaf 2
        self.play(visit(l3[2], e_A2_L1), run_time=0.22)
        t2 = score_text(leaf_vals[2], size=20).move_to(val_above(l3[2], 0.26))
        leaf_texts[2] = t2
        self.play(FadeIn(t2, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[2], e_A2_L1), run_time=0.12)

        # Leaf 3
        self.play(visit(l3[3], e_A2_L2), run_time=0.22)
        t3 = score_text(leaf_vals[3], size=20).move_to(val_above(l3[3], 0.26))
        leaf_texts[3] = t3
        self.play(FadeIn(t3, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[3], e_A2_L2), run_time=0.12)

        # Backup to A2 (MAX): +0.65
        a2_val = score_text("+0.65", size=22).move_to(val_above(l2[1], 0.32))
        l2_texts[1] = a2_val
        self.play(TransformFromCopy(t2, a2_val), run_time=0.25)
        self.play(unvisit(l2[1], e_A_A2), run_time=0.14)

        # Backup to A (MIN): min(+0.75, +0.65) = +0.65
        A_val = score_text("+0.65", size=24).move_to(val_above(l1[0], 0.34))
        l1_texts[0] = A_val
        self.play(TransformFromCopy(a2_val, A_val), run_time=0.25)

        # Update alpha at root: α = +0.65
        new_alpha = tag_text("α = +0.65", size=18, color=YELLOW).move_to(alpha_hud)
        self.play(Transform(alpha_hud, new_alpha), run_time=0.22)

        self.play(unvisit(l1[0], e_root_A), run_time=0.14)

        # ROOT -> B (MIN)
        self.play(visit(l1[1], e_root_B), run_time=0.25)

        # B -> B1 (MAX) -> leaves 4,5
        self.play(visit(l2[2], e_B_B1), run_time=0.25)

        # Leaf 4
        self.play(visit(l3[4], e_B1_L1), run_time=0.22)
        t4 = score_text(leaf_vals[4], size=20).move_to(val_above(l3[4], 0.26))
        leaf_texts[4] = t4
        self.play(FadeIn(t4, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[4], e_B1_L1), run_time=0.12)

        # Leaf 5
        self.play(visit(l3[5], e_B1_L2), run_time=0.22)
        t5 = score_text(leaf_vals[5], size=20).move_to(val_above(l3[5], 0.26))
        leaf_texts[5] = t5
        self.play(FadeIn(t5, shift=UP * 0.08), run_time=0.18)
        self.play(unvisit(l3[5], e_B1_L2), run_time=0.12)

        # Backup to B1 (MAX): +0.40
        b1_val = score_text("+0.40", size=22).move_to(val_above(l2[2], 0.32))
        l2_texts[2] = b1_val
        self.play(TransformFromCopy(t4, b1_val), run_time=0.25)
        self.play(unvisit(l2[2], e_B_B1), run_time=0.14)

        # Now at B (MIN), beta becomes +0.40. Since beta <= alpha (+0.65), prune B2.
        beta_at_B = (
            tag_text("β = +0.40", size=18, color=BLUE_B)
            .next_to(l1[1], RIGHT, buff=0.35)
            .shift(UP * 0.1)
        )
        beta_at_B.set_z_index(Z_TAG)
        self.play(FadeIn(beta_at_B, shift=UP * 0.08), run_time=0.18)

        prune_tag = tag_text("β ≤ α  →  prune", size=18, color=RED_B).next_to(
            l2[3], UP, buff=0.25
        )
        prune_tag.set_z_index(Z_TAG)

        # Build a big red X over B2 (and fade its whole subtree)
        def red_x_over(mobj, scale=0.35):
            c = mobj.get_center()
            x1 = Line(c + LEFT * 0.35 + UP * 0.35, c + RIGHT * 0.35 + DOWN * 0.35)
            x2 = Line(c + LEFT * 0.35 + DOWN * 0.35, c + RIGHT * 0.35 + UP * 0.35)
            X = VGroup(x1, x2).set_stroke(RED_B, 5).scale(scale / 0.35)
            X.set_z_index(Z_TAG)
            return X

        pruned_nodes = VGroup(l2[3], l3[6], l3[7])
        pruned_edges = VGroup(e_B_B2, e_B2_L1, e_B2_L2)

        Xb2 = red_x_over(l2[3], scale=0.55)

        # Prune animation: show reason tag, then fade subtree + drop X
        self.play(FadeIn(prune_tag, shift=UP * 0.08), run_time=0.18)
        self.play(
            FadeOut(pruned_edges, shift=DOWN * 0.05),
            pruned_nodes.animate.set_opacity(0.18),
            FadeIn(Xb2),
            run_time=0.45,
        )

        # Backup to B (MIN): with only one explored child, it is +0.40
        B_val = score_text("+0.40", size=24).move_to(val_above(l1[1], 0.34))
        l1_texts[1] = B_val
        self.play(TransformFromCopy(b1_val, B_val), run_time=0.25)

        self.play(unvisit(l1[1], e_root_B), run_time=0.14)

        # ROOT backup (MAX): max(A=+0.65, B=+0.40) => +0.65
        root_val = score_text("+0.65", size=28)
        root_val.set_color(YELLOW)  # highlight best move result at root
        root_val.move_to(val_above(root, 0.38))
        root_text = root_val
        self.play(FadeIn(root_val, scale=1.1), run_time=0.25)

        # Highlight winning line: root -> A -> A2 (since A is MIN, it would choose +0.65 via A2)
        # This is just a visual: the "principal variation" style highlight.
        pv_edges = [e_root_A, e_A_A2]
        self.play(flash_path(pv_edges, color=YELLOW, width=5, rt=0.30))
        self.play(
            *[
                e.animate.set_stroke(color=YELLOW, width=4, opacity=0.95)
                for e in pv_edges
            ],
            run_time=0.25,
        )

        # Keep the end frame briefly
        self.wait(0.8)

    def show_alphazero_flowchart(self):
        center = 2 * LEFT + DOWN
        boxes = [
            RoundedRectangle(width=3, height=1, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 2 * UP),
            RoundedRectangle(width=3, height=1, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 3 * RIGHT),
            RoundedRectangle(width=3, height=1, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 2 * DOWN),
            RoundedRectangle(width=3, height=1, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 3 * LEFT),
        ]

        texts = [
            Text("Initialise neural network", font_size=32).scale(0.5),
            Text("Self-play", font_size=32).scale(0.5),
            Text("Generate training data", font_size=32).scale(0.5),
            Text("Train neural network", font_size=32).scale(0.5),
        ]

        arrows = [
            CurvedArrow(
                boxes[0].get_edge_center(RIGHT),
                boxes[1].get_edge_center(UP),
                angle=-PI / 2,
            ),
            CurvedArrow(
                boxes[1].get_edge_center(DOWN),
                boxes[2].get_edge_center(RIGHT),
                angle=-PI / 2,
            ),
            CurvedArrow(
                boxes[2].get_edge_center(LEFT),
                boxes[3].get_edge_center(DOWN),
                angle=-PI / 2,
            ),
            CurvedArrow(
                boxes[3].get_edge_center(UP),
                boxes[0].get_edge_center(LEFT),
                angle=-PI / 2,
            ),
        ]

        for box, text, arrow in zip(boxes, texts, arrows):
            text.move_to(box)
            self.play(Create(box), Write(text))
            self.play(Create(arrow))

    def show_transformer(self):
        tokens = VGroup()

        for _ in range(6):
            box = RoundedRectangle(
                width=1.5,
                height=0.5,
                corner_radius=0.15,
                stroke_color=BLUE,
                fill_color=BLUE_E,
                fill_opacity=0.8,
            )
            tokens.add(box)

        tokens.arrange(RIGHT, buff=0.4)
        tokens.to_edge(UP).shift(2 * DOWN)

        input_label = Text("Input tokens", font_size=32).scale(0.5).next_to(tokens, UP)

        self.play(
            Write(input_label),
            LaggedStart(*[Create(t) for t in tokens], lag_ratio=0.2),
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

        attention_label = Text("Self-Attention", font_size=32).scale(0.5)
        attention_label.next_to(attention_lines, DOWN)

        self.play(Create(attention_lines), Write(attention_label))
        self.wait(1)
        self.play(FadeOut(attention_lines), FadeOut(attention_label))

        # -----------------------------
        # Transformer block
        # -----------------------------
        block = RoundedRectangle(
            width=9,
            height=1,
            corner_radius=0.3,
            stroke_color=GREEN,
            fill_color=GREEN_E,
            fill_opacity=0.85,
        )

        block_text = Paragraph(
            "Transformer Block",
            "(Self-Attention + Feedforward)",
            alignment="center",
            font_size=32,
        ).move_to(block.get_center())

        transformer = VGroup(block, block_text)
        transformer.next_to(tokens, DOWN).shift(1 * DOWN)

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
        self.play(Create(transformer))
        self.wait(1)

        # -----------------------------
        # Output tokens
        # -----------------------------
        output_tokens = tokens.copy()
        output_tokens.next_to(transformer, DOWN).shift(1 * DOWN)

        output_label = (
            Text("Output representations", font_size=32)
            .scale(0.5)
            .next_to(output_tokens, DOWN)
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
        self.play(LaggedStart(*[Create(t) for t in output_tokens], lag_ratio=0.2))
        self.play(Write(output_label))

        self.wait(2)

    def construct(self):
        self.show_section_title(
            "Introduction to chess engines", "A more detailed overview"
        )

        top_text = Paragraph(
            "There are 2 main types of chess engines", font_size=30
        ).to_edge(UP)

        self.play(Write(top_text))
        self.wait()

        new_text = Paragraph("1. Traditional engines", font_size=30).to_edge(UP)
        self.play(Transform(top_text, new_text))

        body_text = Paragraph(
            "Traditional engines such as Stockfish use a tree-search",
            "algorithm known as alpha-beta pruning to find the best",
            "possible move, evaluating many chess positions",
            font_size=30,
        ).next_to(top_text, DOWN)
        logo = (
            ImageMobject("stockfish.png")
            .scale_to_fit_height(3)
            .to_edge(RIGHT)
            .shift(DOWN * 1.75)
        )
        self.play(Write(body_text), FadeIn(logo))

        self.play(FadeOut(logo))
        self.show_stockfish_tree()

        new_text = Paragraph(
            "However, this is slow due to the exponentially increasing",
            "number of positions that have to be searched",
            font_size=30,
        ).next_to(top_text, DOWN)
        self.play(Transform(body_text, new_text))

        self.wait()
        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        new_text = Paragraph("2. Neural networks", font_size=30).to_edge(UP)
        self.play(Transform(top_text, new_text))
        self.wait()

        body_text = Paragraph(
            "In recent years, neural networks have been applied to chess",
            "engines like AlphaZero to reduce the total number of positions",
            "to search",
            font_size=30,
        ).next_to(top_text, DOWN)
        logo = (
            ImageMobject("alphazero.png")
            .scale_to_fit_height(3)
            .to_edge(RIGHT)
            .shift(DOWN * 1.75)
        )
        self.play(Write(body_text), FadeIn(logo))
        self.wait()

        new_text = Paragraph(
            "The neural network needs to be trained in order to play well,",
            "which uses a lot of compute power",
            font_size=30,
        ).next_to(top_text, DOWN)
        self.play(Transform(body_text, new_text))

        self.show_alphazero_flowchart()

        new_text = Paragraph(
            "For example, AlphaZero uses self-play, which requires a large",
            "neural network and needs powerful hardware to train",
            font_size=30,
        ).next_to(top_text, DOWN)
        self.play(Transform(body_text, new_text))

        self.wait()

        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        body_text = Paragraph(
            "Neural networks use a transformer, which is a neural network",
            "architecture that is also used in large language models",
            font_size=30,
        ).next_to(top_text, DOWN)
        self.play(Write(body_text))

        self.show_transformer()

        self.play(
            *[FadeOut(obj) for obj in self.mobjects if obj not in (top_text, body_text)]
        )

        new_text = Paragraph(
            "For more information, check out the YouTube series on deep learning",
            "by 3Blue1Brown",
            font_size=30,
        ).next_to(top_text, DOWN)
        self.play(Transform(body_text, new_text))

        thumbnail = ImageMobject("transformer.jpg").scale_to_fit_height(5).to_edge(DOWN)
        self.play(FadeIn(thumbnail))
        self.wait()

        self.fade_out()
