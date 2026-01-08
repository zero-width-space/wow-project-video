from utils import BaseSection
from manim import *


# ============================================================
# Replacement: drop-in scene that runs your provided code
# (kept the animation content the same; only wrapped in a method)
# ============================================================
class StockfishExplainer(Scene):
    def construct(self):
        # -----------------------------
        # Z-LAYERS
        # -----------------------------
        Z_BOARD = 5
        Z_OVERLAY = 25
        Z_UI = 40
        Z_LOGO = 100
        Z_TREE_TEXT = 80

        # =============================
        # 1) LOGO
        # =============================
        logo = ImageMobject("./stockfish_logo.png")
        logo.set_z_index(Z_LOGO)
        logo.scale(0.001).move_to(ORIGIN)

        self.play(FadeIn(logo, scale=1.05), run_time=0.15)
        self.play(
            logo.animate.scale(900),
            rate_func=rate_functions.ease_out_back,
            run_time=0.9,
        )

        logo_target = logo.copy()
        logo_target.scale(0.18).to_corner(UR, buff=0.25).shift(DOWN * 0.05).shift(
            LEFT * 0.05
        )

        # Faster move-to-corner transform (grow is fine; move is faster)
        self.play(
            Transform(logo, logo_target),
            run_time=0.6,
            rate_func=rate_functions.ease_in_out_cubic,
        )

        # =============================
        # 2) CHESS BOARD (SVG) + FEATURE SWEEP + EVAL BAR
        # =============================
        import chess
        import chess.svg
        from pathlib import Path
        import tempfile
        import uuid
        import numpy as np

        fen = "rnbq1rk1/pp2bppp/4pn2/2p5/2BPP3/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 8"
        board = chess.Board(fen)

        svg_text = chess.svg.board(board=board, size=740, coordinates=False)

        tmp_dir = Path(tempfile.gettempdir())
        svg_path = tmp_dir / f"manim_chess_board_{uuid.uuid4().hex}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        board_svg = SVGMobject(str(svg_path))
        board_svg.set_z_index(Z_BOARD)
        board_svg.scale_to_fit_height(4.95)

        # Place on the left, but vertically CENTER the board in the frame
        board_svg.to_edge(LEFT, buff=0.55)
        board_svg.move_to([board_svg.get_center()[0], 0.0, 0.0])  # center Y = 0

        self.play(FadeIn(board_svg, shift=RIGHT * 0.15), run_time=0.65)

        # Pause to look at the board before the "taking in" grid sweep
        self.wait(1.1)

        # ---- Overlay grid edges ABOVE board
        overlay = VGroup().set_z_index(Z_OVERLAY)

        left, right = board_svg.get_left(), board_svg.get_right()
        top, bottom = board_svg.get_top(), board_svg.get_bottom()

        xmin, xmax = left[0], right[0]
        ymin, ymax = bottom[1], top[1]
        w, h = (xmax - xmin), (ymax - ymin)
        sq_w, sq_h = w / 8.0, h / 8.0

        def square_corners(file_idx, rank_idx_top0):
            x0 = xmin + file_idx * sq_w
            x1 = x0 + sq_w
            y1 = ymax - rank_idx_top0 * sq_h
            y0 = y1 - sq_h
            bl = np.array([x0, y0, 0])
            br = np.array([x1, y0, 0])
            tr = np.array([x1, y1, 0])
            tl = np.array([x0, y1, 0])
            return bl, br, tr, tl

        edges = []
        delays = []
        for r in range(8):
            for f in range(8):
                bl, br, tr, tl = square_corners(f, r)
                e_top = Line(tr, tl)
                e_right = Line(br, tr)
                e_bottom = Line(bl, br)
                e_left = Line(tl, bl)

                per_sq = [e_top, e_right, e_bottom, e_left]
                micro = [0.00, 0.015, 0.030, 0.045]  # a touch more spacing
                base = 0.065 * (f + r)  # slower ripple across the board (was 0.04)

                for e, m in zip(per_sq, micro):
                    # Blue sweep (was white)
                    e.set_stroke(BLUE_B, width=3, opacity=0.0)
                    e.set_z_index(Z_OVERLAY)
                    edges.append(e)
                    delays.append(base + m)

        overlay.add(*edges)
        self.add(overlay)

        def pulse_opacity(alpha):
            # 0->0.25 fade in, 0.25->0.55 hold, 0.55->1 fade out
            if alpha < 0.25:
                return smooth(alpha / 0.25)
            if alpha < 0.55:
                return 1.0
            return smooth(1 - (alpha - 0.55) / 0.45)

        def edge_pulse_anim(edge, start_delay, total_window=1.55):
            def updater(mobj, alpha):
                t = alpha * total_window
                if t < start_delay:
                    op = 0.0
                else:
                    # slightly longer pulse (was 0.3)
                    local = (t - start_delay) / 0.38
                    if 0 <= local <= 1:
                        op = 0.85 * pulse_opacity(local)
                    else:
                        op = 0.0
                mobj.set_stroke(opacity=op)

            return UpdateFromAlphaFunc(edge, updater)

        # Slower overall sweep, hardcoded!
        sweep_window = 1.5
        sweep_anims = [
            edge_pulse_anim(e, d, total_window=sweep_window)
            for e, d in zip(edges, delays)
        ]
        self.play(AnimationGroup(*sweep_anims, lag_ratio=0.0), run_time=sweep_window)

        # After sweep completes, draw arrow
        arrow_y = board_svg.get_center()[1]
        board_to_bar = Arrow(
            start=board_svg.get_right() + RIGHT * 0.15,
            end=RIGHT * 1.10 + UP * arrow_y,  # just a horizontal target point
            buff=0.0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.14,
        )
        board_to_bar.set_stroke(WHITE, opacity=0.65)
        board_to_bar.set_z_index(Z_UI)
        self.play(Create(board_to_bar), run_time=0.28)

        # ---- Eval bar (centipawn-like in pawns)
        # We'll show -2.0 ... 0.0 ... +2.0 (pawns).
        bar_group = VGroup().set_z_index(Z_UI)

        # Align bar vertically with board center so arrow can be horizontal
        bar_y = board_svg.get_center()[1]  # should be 0.0 now
        bar_left = RIGHT * 1.15 + UP * bar_y
        bar_right = RIGHT * 6.05 + UP * bar_y

        bar_line = Line(bar_left, bar_right).set_stroke(WHITE, 4, opacity=0.75)
        tickL = Line(bar_left + UP * 0.12, bar_left + DOWN * 0.12).set_stroke(
            WHITE, 3, opacity=0.75
        )
        tickM = Line(
            (bar_left + bar_right) / 2 + UP * 0.12,
            (bar_left + bar_right) / 2 + DOWN * 0.12,
        ).set_stroke(WHITE, 3, opacity=0.75)
        tickR = Line(bar_right + UP * 0.12, bar_right + DOWN * 0.12).set_stroke(
            WHITE, 3, opacity=0.75
        )

        labelL = Text("-5.0", font_size=20, color=WHITE).next_to(tickL, DOWN, buff=0.12)
        labelM = Text("0.0", font_size=20, color=WHITE).next_to(tickM, DOWN, buff=0.12)
        labelR = Text("+5.0", font_size=20, color=WHITE).next_to(tickR, DOWN, buff=0.12)

        # "Evaluation: ?" as stable label + dynamic value (only value transforms)
        eval_label = Text("Evaluation:", font_size=26, color=WHITE)
        eval_value = Text("?", font_size=26, color=WHITE)

        # Increase spacing so "+" never collides with the colon
        eval_group = VGroup(eval_label, eval_value).arrange(RIGHT, buff=0.32)
        eval_group.next_to(bar_line, UP, buff=0.22)
        eval_group.set_z_index(Z_UI)

        # Up-pointing marker (triangle points up by default)
        marker = Triangle().scale(0.12)
        marker.set_fill(WHITE, opacity=0.95)
        marker.set_stroke(width=0)

        # Marker tracks along the bar (value -2..+2 mapped to 0..1)
        t_marker = ValueTracker(0.0)

        def marker_pos_from_t(t):
            return interpolate(bar_left, bar_right, t) + DOWN * 0.1

        def marker_update(m):
            m.move_to(marker_pos_from_t(t_marker.get_value()))

        marker.add_updater(lambda m: marker_update(m))

        bar_group.add(
            bar_line, tickL, tickM, tickR, labelL, labelM, labelR, eval_group, marker
        )
        self.play(FadeIn(bar_group, shift=UP * 0.10), run_time=0.45)

        # Marker "searching" while eval is unknown
        self.play(
            t_marker.animate.set_value(1.0),
            run_time=0.55,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            t_marker.animate.set_value(0.15),
            run_time=0.50,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            t_marker.animate.set_value(0.80),
            run_time=0.45,
            rate_func=rate_functions.ease_in_out_sine,
        )

        # Final centipawn-ish evaluation (in pawns, displayed like Stockfish)
        final_eval_pawns = +0.78
        # Map [-2..+2] -> [0..1]
        BAR_MIN, BAR_MAX = -2.0, 2.0
        final_t = (final_eval_pawns - BAR_MIN) / (BAR_MAX - BAR_MIN)
        final_t = max(0.0, min(1.0, final_t))

        # Create the final value text in-place (same position), with proper color
        eval_str = f"{final_eval_pawns:+.2f}"
        eval_color = GREEN_B if final_eval_pawns >= 0 else RED_B
        eval_value_target = Text(eval_str, font_size=26, color=eval_color)
        eval_value_target.move_to(eval_value.get_center())
        eval_value_target.set_z_index(Z_UI)

        # Copy of the board transforms while moving INTO the value area.
        board_copy = board_svg.copy()
        board_copy.set_z_index(Z_UI)  # above everything during transform
        self.add(board_copy)

        # Fade overlay out (scan done)
        # Transform board_copy -> eval_value_target WHILE MOVING (one transform)
        # Also transform eval_value ("?") -> "+0.41" in the same spot.
        self.play(
            FadeOut(overlay),
            Transform(eval_value, eval_value_target),
            Transform(board_copy, eval_value_target.copy()),
            run_time=0.75,
            rate_func=rate_functions.ease_in_out_cubic,
        )
        # Clean: remove the board_copy remnants if any (it has been transformed into text-like geometry)
        self.remove(board_copy)

        # Marker locks to final value and stops (remove updater properly)
        self.play(
            t_marker.animate.set_value(final_t),
            run_time=0.45,
            rate_func=rate_functions.ease_in_out_cubic,
        )
        marker.clear_updaters()

        # Slight pause to read it
        self.wait(0.3)

        # Dim bar a touch to transition; keep original board visible
        self.play(
            bar_group.animate.set_opacity(0.85),
            board_to_bar.animate.set_opacity(0.85),
            run_time=0.20,
        )

        # =============================
        # 3) SEARCH TREE (centipawns in pawns, + green / - red)
        # DFS + CLEAR ALPHA-BETA PRUNING (same structure as before)
        # =============================

        # --- Tree layers
        Z_EDGE = 2
        Z_NODE = 6
        Z_TEXT = Z_TREE_TEXT
        Z_TAG = Z_TREE_TEXT + 10

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

        def cp_text(x, size=22):
            # x in pawns, show + / - and color
            s = f"{x:+.2f}"
            col = GREEN_B if x >= 0 else RED_B
            t = Text(s, font_size=size, color=col)
            t.set_z_index(Z_TEXT)
            return t

        def tag_text(s, size=18, color=YELLOW):
            t = Text(s, font_size=size, color=color)
            t.set_z_index(Z_TAG)
            return t

        tree = VGroup().set_z_index(3)
        tree.shift(RIGHT * 3.05 + DOWN * 0.55)

        # Layout
        Y0 = 2.35
        Y1 = 1.10
        Y2 = -0.15
        Y3 = -1.85

        x_root = 0.0
        x_l1 = [-1.9, 1.9]
        x_l2 = [-2.8, -1.0, 1.0, 2.8]
        x_l3 = [-3.35, -2.25, -1.55, -0.45, 0.45, 1.55, 2.25, 3.35]

        root = node(0.26).move_to([x_root, Y0, 0])
        l1 = [node().move_to([x, Y1, 0]) for x in x_l1]  # MIN
        l2 = [node().move_to([x, Y2, 0]) for x in x_l2]  # MAX
        l3 = [node(0.20).move_to([x, Y3, 0]) for x in x_l3]  # leaves

        tree.add(root, *l1, *l2, *l3)

        e_root_A = connect(root, l1[0])
        e_root_B = connect(root, l1[1])

        e_A_A1 = connect(l1[0], l2[0])
        e_A_A2 = connect(l1[0], l2[1])
        e_B_B1 = connect(l1[1], l2[2])
        e_B_B2 = connect(l1[1], l2[3])

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


        self.play(
            FadeOut(board_svg, shift=LEFT * 0.10),
            FadeOut(bar_group, shift=DOWN * 0.06),
            FadeOut(board_to_bar, shift=DOWN * 0.06),
            run_time=0.35,
        )
        self.play(FadeIn(tree, shift=UP * 0.12), run_time=0.60)

        # Alpha/Beta HUD near root (in pawns)
        alpha_hud = (
            tag_text("α = -inf", size=18, color=YELLOW)
            .next_to(root, LEFT, buff=0.55)
            .shift(UP * 0.12)
        )
        beta_hud = (
            tag_text("β = +inf", size=18, color=BLUE_B)
            .next_to(root, RIGHT, buff=0.55)
            .shift(UP * 0.12)
        )
        self.play(FadeIn(alpha_hud), FadeIn(beta_hud), run_time=0.25)

        def visit(node_obj, edge_obj=None, rt=0.22):
            anims = [node_obj.animate.set_fill("#2a3347", opacity=1.0)]
            if edge_obj is not None:
                anims.append(
                    edge_obj.animate.set_stroke(color=YELLOW, width=4, opacity=0.95)
                )
            return AnimationGroup(*anims, run_time=rt, lag_ratio=0.0)

        def unvisit(node_obj, edge_obj=None, rt=0.15):
            anims = [node_obj.animate.set_fill("#1b2230", opacity=1.0)]
            if edge_obj is not None:
                anims.append(
                    edge_obj.animate.set_stroke(color=WHITE, width=2, opacity=0.55)
                )
            return AnimationGroup(*anims, run_time=rt, lag_ratio=0.0)

        # Leaf values (pawns), chosen to force prune clearly
        leaf_v = [
            +0.78,
            +0.62,  # A1 -> MAX = +0.78
            +0.68,
            +0.55,  # A2 -> MAX = +0.68 => A(MIN)=+0.68 => alpha=+0.68
            +0.40,
            +0.30,  # B1 -> MAX = +0.40 => beta=+0.40 <= alpha => prune B2
            +0.10,
            -0.20,  # B2 (pruned)
        ]

        leaf_texts = [None] * 8

        def show_leaf(i, node_leaf, edge_to_leaf):
            self.play(visit(node_leaf, edge_to_leaf), run_time=0.20)
            t = cp_text(leaf_v[i], size=20).move_to(val_above(node_leaf, 0.26))
            leaf_texts[i] = t
            self.play(FadeIn(t, shift=UP * 0.08), run_time=0.16)
            self.play(unvisit(node_leaf, edge_to_leaf), run_time=0.11)

        self.play(visit(root), run_time=0.14)

        # ROOT -> A
        self.play(visit(l1[0], e_root_A), run_time=0.24)

        # A -> A1
        self.play(visit(l2[0], e_A_A1), run_time=0.24)
        show_leaf(0, l3[0], e_A1_L1)
        show_leaf(1, l3[1], e_A1_L2)

        A1_val = cp_text(+0.78, size=22).move_to(val_above(l2[0], 0.32))
        self.play(TransformFromCopy(leaf_texts[0], A1_val), run_time=0.22)
        self.play(unvisit(l2[0], e_A_A1), run_time=0.13)

        # A -> A2
        self.play(visit(l2[1], e_A_A2), run_time=0.24)
        show_leaf(2, l3[2], e_A2_L1)
        show_leaf(3, l3[3], e_A2_L2)

        A2_val = cp_text(+0.78, size=22).move_to(val_above(l2[1], 0.32))
        self.play(TransformFromCopy(leaf_texts[2], A2_val), run_time=0.22)
        self.play(unvisit(l2[1], e_A_A2), run_time=0.13)

        # A (MIN) = +0.78
        A_val = cp_text(+0.78, size=24).move_to(val_above(l1[0], 0.34))
        self.play(TransformFromCopy(A2_val, A_val), run_time=0.24)

        # Update alpha
        self.play(
            Transform(alpha_hud, tag_text("α = +0.68", 18, YELLOW).move_to(alpha_hud)),
            run_time=0.20,
        )
        self.play(unvisit(l1[0], e_root_A), run_time=0.13)

        # ROOT -> B
        self.play(visit(l1[1], e_root_B), run_time=0.24)

        # B -> B1
        self.play(visit(l2[2], e_B_B1), run_time=0.24)
        show_leaf(4, l3[4], e_B1_L1)
        show_leaf(5, l3[5], e_B1_L2)

        # B1 (MAX) = +0.40  (shown BEFORE B)
        B1_val = cp_text(+0.40, size=22).move_to(val_above(l2[2], 0.32))
        self.play(TransformFromCopy(leaf_texts[4], B1_val), run_time=0.22)
        self.play(unvisit(l2[2], e_B_B1), run_time=0.13)

        beta_at_B = (
            tag_text("β = +0.40", size=18, color=BLUE_B)
            .next_to(l1[1], RIGHT, buff=0.35)
            .shift(UP * 0.08)
        )
        self.play(FadeIn(beta_at_B, shift=UP * 0.06), run_time=0.18)

        prune_tag = tag_text("β ≤ α  →  prune", size=18, color=RED_B).next_to(
            l2[3], UP, buff=0.22
        )

        def red_x_over(mobj, scale=0.55):
            c = mobj.get_center()
            x1 = Line(c + LEFT * 0.35 + UP * 0.35, c + RIGHT * 0.35 + DOWN * 0.35)
            x2 = Line(c + LEFT * 0.35 + DOWN * 0.35, c + RIGHT * 0.35 + UP * 0.35)
            X = VGroup(x1, x2).set_stroke(RED_B, 5).scale(scale / 0.35)
            X.set_z_index(Z_TAG)
            return X

        pruned_nodes = VGroup(l2[3], l3[6], l3[7])
        pruned_edges = VGroup(e_B_B2, e_B2_L1, e_B2_L2)
        Xb2 = red_x_over(l2[3], scale=0.55)

        self.play(FadeIn(prune_tag, shift=UP * 0.06), run_time=0.18)
        self.play(
            FadeOut(pruned_edges, shift=DOWN * 0.05),
            pruned_nodes.animate.set_opacity(0.18),
            FadeIn(Xb2),
            run_time=0.48,
        )

        # B (MIN) = +0.40
        B_val = cp_text(+0.40, size=24).move_to(val_above(l1[1], 0.34))
        self.play(TransformFromCopy(B1_val, B_val), run_time=0.24)
        self.play(unvisit(l1[1], e_root_B), run_time=0.13)

        # ROOT (MAX) = +0.78
        root_val = cp_text(+0.78, size=28)
        root_val.set_color(YELLOW)
        root_val.move_to(val_above(root, 0.38))
        self.play(FadeIn(root_val, scale=1.06), run_time=0.22)

        # Highlight best line: root -> A -> A2
        self.play(
            e_root_A.animate.set_stroke(color=YELLOW, width=4, opacity=0.95),
            e_A_A2.animate.set_stroke(color=YELLOW, width=4, opacity=0.95),
            run_time=0.32,
        )
        self.play(
            e_root_A.animate.set_stroke(width=6),
            e_A_A2.animate.set_stroke(width=6),
            run_time=0.20,
        )
        self.play(
            e_root_A.animate.set_stroke(width=4),
            e_A_A2.animate.set_stroke(width=4),
            run_time=0.20,
        )

        self.wait(1)


        # --- Clean handoff: remove EVERYTHING from this segment (but don't touch outside text)
        leaf_group = VGroup(*[t for t in leaf_texts if t is not None])

        cleanup = Group(
            logo,
            tree,
            alpha_hud,
            beta_hud,
            beta_at_B,
            prune_tag,
            Xb2,
            leaf_group,
            A1_val,
            A2_val,
            A_val,
            B1_val,
            B_val,
            root_val,
        )

        self.play(FadeOut(cleanup), run_time=0.45)

class EngineIntro(BaseSection):
    def show_alphazero_flowchart(self):
        # A clearer AlphaZero loop:
        # 1) Start with a neural net f_theta that outputs (policy, value)
        # 2) Use MCTS guided by the net to choose moves in self-play
        # 3) Record (state, improved policy pi, game outcome z)
        # 4) Train the net to match pi and predict z
        # 5) Repeat

        center = 2 * LEFT + DOWN * 0.6

        boxes = [
            RoundedRectangle(width=3.6, height=1.05, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 2 * UP),
            RoundedRectangle(width=3.6, height=1.05, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 3 * RIGHT),
            RoundedRectangle(width=3.6, height=1.05, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 2 * DOWN),
            RoundedRectangle(width=3.6, height=1.05, corner_radius=0.25)
            .set_stroke(WHITE, 2)
            .move_to(center + 3 * LEFT),
        ]

        texts = [
            Paragraph("Neural net", "(policy + value)", font_size=32, alignment="center")
            .scale(0.5),
            Paragraph("Self-play", "using MCTS", font_size=32, alignment="center").scale(
                0.5
            ),
            Paragraph("Training targets", "(π, z)", font_size=32, alignment="center")
            .scale(0.5),
            Paragraph(
                "Update the net", "to match π and z", font_size=32, alignment="center"
            ).scale(0.5),
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

        for box, text in zip(boxes, texts):
            text.move_to(box)

        # Animate with clearer sequencing
        self.play(Create(boxes[0]), Write(texts[0]))
        self.play(Create(arrows[0]))
        self.play(Create(boxes[1]), Write(texts[1]))
        self.play(Create(arrows[1]))
        self.play(Create(boxes[2]), Write(texts[2]))
        self.play(Create(arrows[2]))
        self.play(Create(boxes[3]), Write(texts[3]))
        self.play(Create(arrows[3]))

        # Small legend on the side (very helpful for viewers)
        legend = VGroup(
            Text("π = improved move probabilities (from MCTS)", font_size=20),
            Text("z = final game result (+1 win / 0 draw / -1 loss)", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_edge(DOWN, buff=0.35).shift(RIGHT * 0.5)

        self.play(FadeIn(legend, shift=UP * 0.15), run_time=0.35)
        self.wait(1.6)
        self.play(FadeOut(legend), run_time=0.3)

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
        self.wait(1.0)

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
        self.wait(1.6)
        self.play(FadeOut(attention_lines), FadeOut(attention_label))

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
        self.wait(1.5)

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

        self.wait(2.8)

    def construct(self):
        def fit_paragraph(text_obj, max_width=12.7):
            if text_obj.width > max_width:
                text_obj.scale_to_fit_width(max_width)
            return text_obj

        top_text = Paragraph("There are 2 main types of chess engines", font_size=30).to_edge(
            UP
        )

        self.play(Write(top_text))
        self.wait(1.2)

        # =============================
        # 1) (Stockfish-related text removed)
        # Replace the first segment with the provided animation scene
        # =============================
        new_text = Paragraph("1. Traditional engines", font_size=30).to_edge(UP)
        self.play(Transform(top_text, new_text))
        self.wait(0.2)

        # Run the replacement animation (logo/board/eval/tree) from the provided code
        StockfishExplainer.construct(self)

        # =============================
        # 2) Neural-network engines (AlphaZero-style)
        # =============================
        new_text = Paragraph("2. Neural networks", font_size=30).to_edge(UP)
        self.play(Transform(top_text, new_text))
        self.wait(0.4)

        body_text = Paragraph(
            "AlphaZero learns by self-play: it plays against itself",
            "to build intuition for chess.",
            "The network outputs two things:",
            "policy = likely moves, value = how good the position is.",
            font_size=28,
            line_spacing=0.9,
        ).next_to(top_text, DOWN)
        fit_paragraph(body_text)
        self.play(Write(body_text))
        self.wait(1.8)

        new_text = Paragraph(
            "During play, it uses Monte Carlo Tree Search (MCTS):",
            "simulate a few futures, guided by the network,",
            "then pick the move that looks best.",
            font_size=28,
            line_spacing=0.9,
        ).next_to(top_text, DOWN)
        fit_paragraph(new_text)
        self.play(Transform(body_text, new_text))
        self.wait(1.8)

        new_text = Paragraph(
            "Training is a loop:",
            "self-play generates new games,",
            "then the network is updated to match the search",
            "and the final game result. Repeat.",
            font_size=28,
            line_spacing=0.9,
        ).next_to(top_text, DOWN)
        fit_paragraph(new_text)
        self.play(Transform(body_text, new_text))
        self.wait(1.4)

        self.show_alphazero_flowchart()

        self.play(*[FadeOut(obj) for obj in self.mobjects if obj is not top_text])

        # Transformer bridge (kept: your original section but slightly clearer)
        body_text = Paragraph(
            "Neural networks often use transformers — the same general architecture",
            "used in large language models — to build strong representations from input tokens.",
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
