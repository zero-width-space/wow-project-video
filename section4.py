from __future__ import annotations

import math
from typing import List

from manim import *


# ============================================================
# Global styling
# ============================================================
BG = "#0b0f14"
WHITE_SOFT = "#e9eef5"

ACCENT = BLUE_C
ACCENT2 = TEAL_C
HI = YELLOW
GOOD = GREEN_C
WARN = ORANGE
BAD = RED_C

# Fixed safe frame bounds (no auto-fit tricks)
# Manim default 16:9 frame is about width=14.22, height=8.0
X_L, X_R = -6.6, 6.6
Y_B, Y_T = -3.6, 3.6

TITLE_Y = 3.22
SUB_Y = 2.78

# Timing tuned for ~5 minutes total
T = {
    "fade": 0.8,
    "move": 1.0,
    "reveal": 1.2,
    "slow": 2.2,
    "hold_short": 1.2,
    "hold": 2.4,
    "hold_long": 4.0,
}


def title(text: str) -> Text:
    t = Text(text, font_size=60, weight=BOLD).set_color(WHITE)
    if t.width > (X_R - X_L):
        t.scale_to_fit_width(X_R - X_L)
    t.move_to(UP * TITLE_Y)
    return t


def subtitle(text: str) -> Text:
    t = Text(text, font_size=30).set_color(WHITE).set_opacity(0.92)
    if t.width > (X_R - X_L):
        t.scale_to_fit_width(X_R - X_L)
    t.move_to(UP * SUB_Y)
    return t


def chip(text: str, color=ACCENT, size=22) -> VGroup:
    tx = Text(text, font_size=size, weight=BOLD).set_color(WHITE)
    r = RoundedRectangle(
        corner_radius=0.18, width=tx.width + 0.55, height=tx.height + 0.30
    )
    r.set_fill(color, opacity=1.0)
    r.set_stroke(width=0)
    tx.move_to(r.get_center())
    return VGroup(r, tx)


def small_label(text: str, size=22, opacity=0.9) -> Text:
    return Text(text, font_size=size).set_opacity(opacity).set_color(WHITE)


def box(text: str, w=3.8, h=1.0, color=GREY_D, size=24, line_spacing=0.9) -> VGroup:
    r = RoundedRectangle(corner_radius=0.22, width=w, height=h)
    r.set_fill(color, opacity=1.0)
    r.set_stroke(width=0)
    tx = Text(text, font_size=size, weight=BOLD, line_spacing=line_spacing).set_color(
        WHITE
    )
    tx.move_to(r.get_center())
    return VGroup(r, tx)


def arrow_lr(a: Mobject, b: Mobject, buff=0.18) -> Arrow:
    # left-to-right arrow
    return Arrow(
        a.get_right(),
        b.get_left(),
        buff=buff,
        stroke_width=4,
        max_tip_length_to_length_ratio=0.12,
    ).set_color(WHITE)


def arrow_ud(a: Mobject, b: Mobject, buff=0.18) -> Arrow:
    # up-to-down arrow (a above b)
    return Arrow(
        a.get_bottom(),
        b.get_top(),
        buff=buff,
        stroke_width=4,
        max_tip_length_to_length_ratio=0.12,
    ).set_color(WHITE)


def chessboard(size=2.9) -> VGroup:
    cell = size / 8
    sqs = VGroup()
    for r in range(8):
        for c in range(8):
            sq = Square(side_length=cell, stroke_width=0)
            sq.set_fill(GREY_E if (r + c) % 2 == 0 else GREY_B, opacity=1.0)
            sq.move_to((c - 3.5) * cell * RIGHT + (3.5 - r) * cell * UP)
            sqs.add(sq)
    frame = RoundedRectangle(corner_radius=0.14, width=size + 0.18, height=size + 0.18)
    frame.set_stroke(WHITE, opacity=0.55, width=3)
    return VGroup(sqs, frame)


def board_cell_size(board: VGroup) -> float:
    # board created by chessboard(): board[0] is 64 squares in row-major r,c
    # each square is Square with side_length = size/8
    return board[0][0].width


def square_center(board: VGroup, file: int, rank: int) -> np.ndarray:
    # file: 0..7 for a..h, rank: 0..7 for 1..8
    # our chessboard builds squares by (r=0 top rank 8) to (r=7 bottom rank 1),
    # (c=0 left file a) to (c=7 right file h)
    r = 7 - rank  # rank 0 (1) -> r=7; rank 7 (8) -> r=0
    c = file
    idx = r * 8 + c
    return board[0][idx].get_center()


class VectorBar(VGroup):
    """
    Visual stand-in for a high-dimensional vector (e.g. 512-d).
    A vertical capsule with internal ticks.
    """

    def __init__(self, label_text: str, height=2.6, width=0.72, color=ACCENT):
        super().__init__()
        body = RoundedRectangle(corner_radius=0.12, width=width, height=height)
        body.set_stroke(WHITE, opacity=0.28, width=2)
        body.set_fill(BLACK, opacity=0.0)

        ticks = VGroup()
        n = 18
        for i in range(n):
            y = interpolate(-height / 2 + 0.18, height / 2 - 0.18, i / (n - 1))
            ln = Line(
                body.get_center() + LEFT * (width / 2 - 0.10) + UP * y,
                body.get_center() + RIGHT * (width / 2 - 0.10) + UP * y,
            )
            ln.set_stroke(color, opacity=0.35 if i % 3 else 0.75, width=2)
            ticks.add(ln)

        brace = Brace(body, RIGHT, buff=0.10)
        lab = Text(label_text, font_size=26, weight=BOLD).set_color(color)
        lab.next_to(brace, RIGHT, buff=0.08)

        self.add(body, ticks, brace, lab)
        self.body = body
        self.lab = lab


def token_tile(text: str, w=0.55, h=0.42, size=20) -> VGroup:
    r = RoundedRectangle(corner_radius=0.10, width=w, height=h)
    r.set_stroke(WHITE, opacity=0.35, width=2)
    r.set_fill(BLACK, opacity=0.0)
    tx = Text(text, font_size=size, weight=BOLD).set_color(WHITE)
    tx.move_to(r.get_center())
    return VGroup(r, tx)


def token_row(tokens: List[str], w=0.55, h=0.42, size=20, buff=0.08) -> VGroup:
    row = VGroup(*[token_tile(t, w=w, h=h, size=size) for t in tokens])
    row.arrange(RIGHT, buff=buff)
    return row


class Heatmap5(VGroup):
    """
    Small 5x5 attention heatmap.
    """

    def __init__(self, cell=0.34):
        super().__init__()
        self.cell = cell
        squares = VGroup()
        for r in range(5):
            for c in range(5):
                sq = Square(side_length=cell)
                sq.set_stroke(WHITE, opacity=0.22, width=1)
                sq.set_fill(BLACK, opacity=0.0)
                sq.move_to((c - 2) * cell * RIGHT + (2 - r) * cell * UP)
                squares.add(sq)
        frame = RoundedRectangle(
            corner_radius=0.12, width=5 * cell + 0.18, height=5 * cell + 0.18
        )
        frame.set_stroke(WHITE, opacity=0.45, width=2)
        frame.set_fill(BLACK, opacity=0.0)
        self.add(squares, frame)
        self.squares = squares

    def set_weights(self, w: List[List[float]], color=ACCENT):
        idx = 0
        for r in range(5):
            for c in range(5):
                a = max(0.0, min(1.0, float(w[r][c])))
                self.squares[idx].set_fill(color, opacity=0.06 + 0.88 * a)
                idx += 1
        return self

    def animate_weights(self, w: List[List[float]], color=ACCENT):
        anims = []
        idx = 0
        for r in range(5):
            for c in range(5):
                a = max(0.0, min(1.0, float(w[r][c])))
                anims.append(
                    self.squares[idx].animate.set_fill(color, opacity=0.06 + 0.88 * a)
                )
                idx += 1
        return AnimationGroup(*anims, lag_ratio=0.02)


def startpos_64() -> str:
    # 8 ranks top->bottom, concatenated ('.' for empty)
    # rnbqkbnr / pppppppp / 8 / 8 / 8 / 8 / PPPPPPPP / RNBQKBNR
    return (
        "rnbqkbnr"
        "pppppppp"
        "........"
        "........"
        "........"
        "........"
        "PPPPPPPP"
        "RNBQKBNR"
    )


def marks_for_board(grid_squares: VGroup, chars64: str) -> VGroup:
    marks = VGroup()
    for i, ch in enumerate(chars64):
        # show ALL 64: pieces as HI, empties as subtle dots
        if ch == ".":
            tx = Text(".", font_size=16, weight=BOLD).set_color(GREY_A).set_opacity(0.55)
        else:
            tx = Text(ch, font_size=18, weight=BOLD).set_color(HI).set_opacity(0.95)
        tx.move_to(grid_squares[i].get_center())
        marks.add(tx)
    return marks


def queen_rays(board: VGroup, origin_file: int, origin_rank: int) -> VGroup:
    # draw rays from origin to the last square in each of 8 directions
    rays = VGroup()
    of, ork = origin_file, origin_rank
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    o = square_center(board, of, ork)
    for df, dr in dirs:
        f, r = of, ork
        while 0 <= f + df <= 7 and 0 <= r + dr <= 7:
            f += df
            r += dr
        end = square_center(board, f, r)
        ln = Line(o, end).set_stroke(HI, opacity=0.55, width=4)
        rays.add(ln)
    return rays


def knight_dots(board: VGroup, origin_file: int, origin_rank: int) -> VGroup:
    dots = VGroup()
    of, ork = origin_file, origin_rank
    deltas = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    for df, dr in deltas:
        f, r = of + df, ork + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            d = Dot(radius=0.055).set_color(ACCENT).set_opacity(0.95)
            d.move_to(square_center(board, f, r))  # centered in the square
            dots.add(d)
    return dots


def curved_attn_arrow(a: Mobject, b: Mobject, angle: float) -> CurvedArrow:
    ar = CurvedArrow(
        a.get_bottom() + DOWN * 0.02,
        b.get_bottom() + DOWN * 0.02,
        angle=angle,
    )
    ar.set_stroke(ACCENT, opacity=0.70, width=4)
    return ar


# ============================================================
# Main scene
# ============================================================
class ChessTransformer(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ------------------------------------------------------------
        # 0) Hook: Search tree vs one forward pass
        # (unchanged)
        # ------------------------------------------------------------
        t0 = title("A chess engine without search")
        self.play(Write(t0), run_time=T["reveal"])

        board = chessboard(2.9).move_to(LEFT * 4.2 + DOWN * 0.2)

        root = Dot(radius=0.06).set_color(WHITE)
        lvl1 = (
            VGroup(*[Dot(radius=0.05).set_color(WHITE) for _ in range(3)])
            .arrange(DOWN, buff=0.28)
            .next_to(root, RIGHT, buff=0.75)
        )
        lvl2 = VGroup()
        for d in lvl1:
            kids = (
                VGroup(*[Dot(radius=0.04).set_color(WHITE) for _ in range(2)])
                .arrange(DOWN, buff=0.18)
                .next_to(d, RIGHT, buff=0.60)
            )
            lvl2.add(kids)

        edges = VGroup()
        for d in lvl1:
            edges.add(
                Line(root.get_center(), d.get_center()).set_stroke(
                    WHITE, opacity=0.28, width=2
                )
            )
        for i, d in enumerate(lvl1):
            for kid in lvl2[i]:
                edges.add(
                    Line(d.get_center(), kid.get_center()).set_stroke(
                        WHITE, opacity=0.22, width=2
                    )
                )

        tree = VGroup(edges, root, lvl1, lvl2).move_to(RIGHT * 3.1 + UP * 0.6)
        tree_label = (
            small_label("Traditional engines:\nsearch many futures", 26)
            .next_to(tree, UP, buff=0.25)
            .align_to(tree, LEFT)
        )

        net = box("Neural net", w=2.6, h=1.0, color=ACCENT2, size=26).move_to(
            RIGHT * 3.1 + DOWN * 2.0
        )
        net_label = (
            small_label("This project:\none forward pass", 26)
            .next_to(net, UP, buff=0.25)
            .align_to(net, LEFT)
        )

        xmark = Cross(tree, stroke_width=10).set_color(BAD).set_opacity(0.85)

        self.play(FadeIn(board, scale=0.98), run_time=T["fade"])
        self.play(
            FadeIn(tree_label, shift=DOWN),
            FadeIn(tree, shift=LEFT),
            run_time=T["reveal"],
        )
        self.wait(T["hold_short"])
        self.play(Create(xmark), run_time=T["fade"])
        self.play(
            FadeIn(net_label, shift=DOWN), FadeIn(net, shift=UP), run_time=T["reveal"]
        )
        self.wait(T["hold_long"])

        self.play(
            FadeOut(board),
            FadeOut(tree),
            FadeOut(tree_label),
            FadeOut(xmark),
            FadeOut(net),
            FadeOut(net_label),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 1) GPT -> Transformer (unchanged)
        # ------------------------------------------------------------
        t1 = title("Transformers (the “T” in GPT)")
        self.play(Transform(t0, t1), run_time=T["fade"])

        gpt = Text("GPT", font_size=120, weight=BOLD).set_color(WHITE)
        t_overlay = Text("T", font_size=120, weight=BOLD).set_color(HI)
        t_overlay.move_to(gpt[-1].get_center())
        g = VGroup(gpt, t_overlay).move_to(UP * 1.2)

        trans = Text("Transformer", font_size=72, weight=BOLD).set_color(HI)
        trans.next_to(g, DOWN, buff=0.65)
        arr = Arrow(
            g.get_bottom(), trans.get_top(), buff=0.18, stroke_width=6
        ).set_color(WHITE)

        expl = Text(
            "Idea: let every part of the position interact with every other part",
            font_size=32,
        ).set_opacity(0.92)
        expl.next_to(trans, DOWN, buff=0.40)

        self.play(FadeIn(g, scale=0.95), run_time=T["fade"])
        self.play(Create(arr), FadeIn(trans, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(expl, shift=UP), run_time=T["reveal"])
        self.wait(T["hold"])
        self.play(
            FadeOut(g), FadeOut(arr), FadeOut(trans), FadeOut(expl), run_time=T["fade"]
        )

        # ------------------------------------------------------------
        # 2) Tokenization: FIXED sequencing and overlap
        # ------------------------------------------------------------
        t2 = title("Step 1: Tokenize a chess position (FEN → 77 tokens)")
        self.play(Transform(t0, t2), run_time=T["fade"])

        fen_hdr = small_label("FEN is a compact text description of the board", 30)
        fen_hdr.move_to(UP * 2.25)

        fen = Text(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            font_size=26,
        ).set_opacity(0.95)
        fen.move_to(UP * 1.75)

        # digit expansion: "8" => "........"
        d_before = Text("8", font_size=88, weight=BOLD).set_color(WHITE)
        d_after = Text("........", font_size=74, weight=BOLD).set_color(WHITE)
        d_before.move_to(LEFT * 2.0 + UP * 0.6)
        d_after.move_to(RIGHT * 2.0 + UP * 0.6)
        exp_arr = Arrow(
            d_before.get_right(), d_after.get_left(), buff=0.35, stroke_width=6
        ).set_color(HI)
        exp_txt = small_label(
            "Digits are empty squares which expand into '.' tokens", 28
        ).move_to(DOWN * 0.15)

        # 77 token layout blocks (raised so nothing is clipped at bottom)
        parts = VGroup(
            chip("side: 1", ACCENT2),
            chip("board squares: 64", ACCENT),
            chip("castling: 4", ACCENT2),
            chip("en-passant: 2", ACCENT),
            chip("halfmove: 3", ACCENT2),
            chip("fullmove: 3", ACCENT),
        ).arrange(DOWN, buff=0.18)

        parts.move_to(LEFT * 4.2 + DOWN * 1.55)
        # extra safety: nudge up if too low
        if parts.get_bottom()[1] < (Y_B + 0.25):
            parts.shift(UP * ((Y_B + 0.25) - parts.get_bottom()[1]))

        sum77 = MathTex(r"1+64+4+2+3+3=77", font_size=46).set_color(HI)
        sum77.move_to(DOWN * 2.65)  # shown later as requested

        # board grid representing the 64 board tokens
        grid = VGroup()
        cell = 0.33
        for r in range(8):
            for c in range(8):
                sq = Square(side_length=cell)
                sq.set_stroke(WHITE, opacity=0.20, width=1.3)
                sq.set_fill(BLACK, opacity=0.0)
                sq.move_to((c - 3.5) * cell * RIGHT + (3.5 - r) * cell * UP)
                grid.add(sq)
        grid_frame = RoundedRectangle(
            corner_radius=0.12, width=8 * cell + 0.18, height=8 * cell + 0.18
        )
        grid_frame.set_stroke(WHITE, opacity=0.40, width=2)
        grid_frame.set_fill(BLACK, opacity=0.0)
        grid_g = VGroup(grid, grid_frame).move_to(RIGHT * 3.85 + DOWN * 0.95)
        grid_lab = (
            small_label("64 board-square tokens (all 64 shown)", 24)
            .next_to(grid_g, UP, buff=0.18)
            .align_to(grid_g, LEFT)
        )

        # ALL 64 marks for the starting position
        all_marks = marks_for_board(grid, startpos_64())

        self.play(
            FadeIn(fen_hdr, shift=DOWN),
            FadeIn(fen, shift=DOWN),
            run_time=T["reveal"],
        )
        self.play(FadeIn(d_before, shift=UP), run_time=T["fade"])
        self.play(
            Create(exp_arr),
            TransformFromCopy(d_before, d_after),
            run_time=T["reveal"],
        )
        self.play(FadeIn(exp_txt, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])

        # IMPORTANT: Fade OUT the 8->........ demo, KEEP the FEN on screen.
        self.play(
            FadeOut(d_before),
            FadeOut(d_after),
            FadeOut(exp_arr),
            FadeOut(exp_txt),
            run_time=T["fade"],
        )

        # Now show the 77 breakdown (with safe positioning)
        self.play(FadeIn(parts, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])

        # Then show the 64-square board tokens (all squares + all pieces)
        self.play(
            FadeIn(grid_g, shift=LEFT),
            FadeIn(grid_lab, shift=DOWN),
            run_time=T["reveal"],
        )
        self.play(
            LaggedStart(*[FadeIn(m, shift=UP) for m in all_marks], lag_ratio=0.008),
            run_time=2.4,
        )
        self.wait(T["hold_short"])

        # Then fade out ONLY the board + contents
        self.play(
            FadeOut(grid_lab),
            FadeOut(grid_g),
            FadeOut(all_marks),
            run_time=T["fade"],
        )

        # Then write the 1+64+... equation
        self.play(Write(sum77), run_time=T["reveal"])
        self.wait(T["hold_short"])

        self.play(
            FadeOut(fen_hdr),
            FadeOut(fen),
            FadeOut(parts),
            FadeOut(sum77),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 3) Action-value input sequence: 77 + action + CLS = 79
        # FIX: queen/knight geometry; text not clipped; no overlaps
        # ------------------------------------------------------------
        t3 = title("Action-value input: 77 FEN tokens + action + CLS")
        self.play(Transform(t0, t3), run_time=T["fade"])

        seq = VGroup(
            chip("FEN tokens (77)", ACCENT2, 24),
            chip("action token (1 of 1968)", ACCENT, 24),
            chip("CLS token (summary)", GREY_D, 24),
        ).arrange(RIGHT, buff=0.32)
        seq.move_to(UP * 2.15)

        eq79 = MathTex(r"77 + 1 + 1 = 79", font_size=50).set_color(HI)
        eq79.next_to(seq, DOWN, buff=0.28)

        # Proper action-space visualization
        mini = chessboard(2.55).move_to(LEFT * 4.25 + DOWN * 1.45)

        # choose a visually-central square: e4 (file=4, rank=3 if rank 1=0)
        # ranks: 1..8 -> 0..7; e4 => file=4, rank=3
        of, orank = 4, 3
        origin = Dot(radius=0.07).set_color(HI).set_opacity(0.95)
        origin.move_to(square_center(mini, of, orank))

        rays = queen_rays(mini, of, orank)
        kdots = knight_dots(mini, of, orank)

        # labels: moved right, wrapped to avoid clipping
        act_txt = small_label(
            "Fixed 1968-action space:\nqueen-like moves + knight moves + promotions",
            26,
        )
        act_txt.next_to(mini, RIGHT, buff=0.55).align_to(mini, UP).shift(DOWN * 0.10)

        prom_txt = small_label("… plus pawn promotions", 24, opacity=0.90)
        prom_txt.next_to(act_txt, DOWN, buff=0.22).align_to(act_txt, LEFT)

        self.play(FadeIn(seq, shift=DOWN), run_time=T["reveal"])
        self.play(FadeIn(eq79, shift=UP), run_time=T["fade"])

        self.play(FadeIn(mini, scale=0.98), FadeIn(origin, scale=0.95), run_time=T["fade"])
        self.play(
            LaggedStart(*[Create(l) for l in rays], lag_ratio=0.06),
            run_time=T["reveal"],
        )
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.9) for d in kdots], lag_ratio=0.05),
            run_time=T["reveal"],
        )
        self.play(FadeIn(act_txt, shift=LEFT), FadeIn(prom_txt, shift=LEFT), run_time=T["reveal"])
        self.wait(T["hold_long"])

        self.play(
            FadeOut(seq),
            FadeOut(eq79),
            FadeOut(mini),
            FadeOut(origin),
            FadeOut(rays),
            FadeOut(kdots),
            FadeOut(act_txt),
            FadeOut(prom_txt),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 4) Token + positional embeddings (match your actual model.py)
        # FIX: wording and layout (no “embedding table” vibe)
        # ------------------------------------------------------------
        t4 = title("Step 2: Token + positional embeddings (d_model = 512)")
        self.play(Transform(t0, t4), run_time=T["fade"])

        # token ids -> tok_emb + pos_emb -> (T x 512)
        tok_ids = VGroup(
            chip("token ids", GREY_D, 24),
            chip("pos ids", GREY_D, 24),
        ).arrange(DOWN, buff=0.18)
        tok_ids.move_to(LEFT * 5.2 + UP * 0.3)

        emb = box(
            "Lookups:\n tok_emb + pos_emb",
            w=3.4,
            h=1.35,
            color=ACCENT2,
            size=22,
        ).move_to(LEFT * 2.2 + UP * 0.3)

        vec = VectorBar("512", height=2.7, color=ACCENT).move_to(RIGHT * 1.1 + UP * 0.3)

        a1 = arrow_lr(tok_ids, emb, buff=0.28)
        a2 = arrow_lr(emb, vec, buff=0.28)

        mat = RoundedRectangle(corner_radius=0.16, width=5.3, height=2.55)
        mat.set_stroke(WHITE, opacity=0.35, width=2)
        mat.set_fill(BLACK, opacity=0.0)
        mat_lab = Text("79 tokens × 512 dims", font_size=28, weight=BOLD).set_color(HI)
        mat_lab.move_to(mat.get_center())
        mat_g = VGroup(mat, mat_lab).move_to(RIGHT * 3.7 + DOWN * 1.75)

        down = Arrow(vec.get_bottom(), mat_g.get_top(), buff=0.25, stroke_width=5).set_color(WHITE)

        self.play(FadeIn(tok_ids, shift=RIGHT), run_time=T["fade"])
        self.play(FadeIn(emb, shift=RIGHT), Create(a1), run_time=T["reveal"])
        self.play(Create(a2), FadeIn(vec, shift=LEFT), run_time=T["reveal"])
        self.wait(T["hold_short"])
        self.play(Create(down), FadeIn(mat_g, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_long"])

        self.play(
            FadeOut(tok_ids),
            FadeOut(emb),
            FadeOut(a1),
            FadeOut(a2),
            FadeOut(vec),
            FadeOut(down),
            FadeOut(mat_g),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 5) Inside one transformer layer: FIX OVERLAPS + non-horizontal attn arrows
        # ------------------------------------------------------------
        t5 = title("Inside ONE transformer layer")
        self.play(Transform(t0, t5), run_time=T["fade"])

        # Left: 5 tokens as vectors
        tok_names = ["t1", "t2", "t3", "t4", "t5"]
        tok_chips = VGroup(*[chip(n, GREY_D, 20) for n in tok_names]).arrange(
            RIGHT, buff=0.18
        )
        vecs = VGroup(*[VectorBar("512", height=2.05, color=ACCENT) for _ in tok_names]).arrange(
            RIGHT, buff=0.42
        )
        for i in range(5):
            tok_chips[i].next_to(vecs[i], UP, buff=0.14)
        seq_in = VGroup(tok_chips, vecs).move_to(LEFT * 3.65 + DOWN * 0.10)

        ln = box("LayerNorm", w=2.3, h=0.75, color=GREY_D, size=22).move_to(
            LEFT * 0.35 + UP * 1.40
        )
        qkv = box("QKV projection\n(512 → 3×512)", w=3.2, h=1.15, color=ACCENT2, size=22).move_to(
            LEFT * 0.35 + UP * 0.20
        )

        a_ln = arrow_lr(seq_in, ln, buff=0.25)
        a_qkv = arrow_ud(ln, qkv, buff=0.20)

        # Phase A: show LN + QKV briefly (then clear before attention visuals)
        self.play(FadeIn(seq_in, shift=UP), run_time=T["reveal"])
        self.play(Create(a_ln), FadeIn(ln, shift=LEFT), run_time=T["reveal"])
        self.play(Create(a_qkv), FadeIn(qkv, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])

        # Phase B: attention visuals (right side), with curved arrows
        self.play(
            FadeOut(ln),
            FadeOut(qkv),
            FadeOut(a_ln),
            FadeOut(a_qkv),
            run_time=T["fade"],
        )

        attn_title = Text(
            "Self-attention: one token mixes info from all tokens",
            font_size=28,
            weight=BOLD,
        ).set_color(WHITE).set_opacity(0.95)
        attn_title.move_to(RIGHT * 3.25 + UP * 2.35)

        focus = SurroundingRectangle(tok_chips[2], buff=0.08).set_stroke(HI, width=4)

        # Curved arrows (varied angles so they are NOT all horizontal)
        arrows = VGroup(
            curved_attn_arrow(tok_chips[2], tok_chips[0], angle=0.45),
            curved_attn_arrow(tok_chips[2], tok_chips[1], angle=0.30),
            curved_attn_arrow(tok_chips[2], tok_chips[3], angle=-0.28),
            curved_attn_arrow(tok_chips[2], tok_chips[4], angle=-0.42),
        )

        # Heatmap and equation: keep separated (no overlaps)
        hm = Heatmap5(cell=0.34).set_weights([[0.05] * 5 for _ in range(5)], color=ACCENT)
        hm_lab = small_label("attention weights (softmax)", 22).next_to(hm, UP, buff=0.15).align_to(hm, LEFT)
        hm_group = VGroup(hm_lab, hm).move_to(RIGHT * 3.35 + UP * 0.85)

        eq = MathTex(r"\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V", font_size=40).set_color(WHITE)
        eq.move_to(RIGHT * 3.35 + DOWN * 0.55)

        out_vecs = VGroup(*[VectorBar("512", height=1.75, color=ACCENT) for _ in range(5)]).arrange(
            RIGHT, buff=0.32
        )
        out_vecs.move_to(RIGHT * 3.35 + DOWN * 2.20)
        out_lab = small_label("output token vectors (mixed info)", 22).next_to(out_vecs, UP, buff=0.18).align_to(
            out_vecs, LEFT
        )

        self.play(FadeIn(attn_title, shift=DOWN), run_time=T["fade"])
        self.play(Create(focus), run_time=T["fade"])

        # Animate appearance + simple “strength” by opacity/width
        weights1 = [0.10, 0.18, 0.42, 0.25]  # corresponds to arrows order (t1,t2,t4,t5)
        for ar, w in zip(arrows, weights1):
            ar.set_stroke(ACCENT, opacity=0.18 + 0.60 * w, width=2 + 10 * w)
        self.play(LaggedStart(*[FadeIn(ar) for ar in arrows], lag_ratio=0.06), run_time=T["reveal"])

        self.play(FadeIn(hm_group, shift=LEFT), FadeIn(eq, shift=UP), run_time=T["reveal"])
        wmat1 = [
            [0.05] * 5,
            [0.05] * 5,
            [0.10, 0.18, 0.05, 0.42, 0.25],
            [0.05] * 5,
            [0.05] * 5,
        ]
        self.play(hm.animate_weights(wmat1, color=ACCENT), run_time=T["slow"])

        # Change pattern
        weights2 = [0.28, 0.10, 0.18, 0.38]
        for ar, w in zip(arrows, weights2):
            ar.set_stroke(ACCENT, opacity=0.18 + 0.60 * w, width=2 + 10 * w)
        wmat2 = [
            [0.05] * 5,
            [0.05] * 5,
            [0.28, 0.10, 0.06, 0.18, 0.38],
            [0.05] * 5,
            [0.05] * 5,
        ]
        self.play(hm.animate_weights(wmat2, color=ACCENT), run_time=T["slow"])

        self.play(FadeIn(out_vecs, shift=UP), FadeIn(out_lab, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])

        # Clear attention visuals before next sub-part
        self.play(
            FadeOut(attn_title),
            FadeOut(arrows),
            FadeOut(focus),
            FadeOut(hm_group),
            FadeOut(eq),
            FadeOut(out_vecs),
            FadeOut(out_lab),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # Multi-head: FIX overlap (h1.. labels no longer cover equation)
        # ------------------------------------------------------------
        mh_title = Text("Multi-head: split 512 dims into 8 heads", font_size=34, weight=BOLD).set_color(WHITE).set_opacity(
            0.95
        )
        mh_title.move_to(UP * 3.05)
        mh_eq = MathTex(r"512 = 8 \times 64", font_size=56).set_color(HI)
        mh_eq.move_to(UP * 2.35)

        slices = VGroup()
        for i in range(8):
            rct = RoundedRectangle(corner_radius=0.10, width=0.55, height=1.55)
            rct.set_fill(ACCENT2 if i % 2 == 0 else ACCENT, opacity=0.85)
            rct.set_stroke(width=0)
            slices.add(rct)
        slices.arrange(RIGHT, buff=0.10).move_to(UP * 1.25)

        head_labs = VGroup(
            *[Text(f"h{i+1}", font_size=20, weight=BOLD).set_color(WHITE) for i in range(8)]
        ).arrange(RIGHT, buff=0.33)
        head_labs.next_to(slices, DOWN, buff=0.18)

        self.play(FadeIn(mh_title, shift=DOWN), FadeIn(mh_eq, shift=DOWN), run_time=T["reveal"])
        self.play(FadeIn(slices, shift=UP), FadeIn(head_labs, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])
        self.play(FadeOut(mh_title), FadeOut(mh_eq), FadeOut(slices), FadeOut(head_labs), run_time=T["fade"])

        # ------------------------------------------------------------
        # SwiGLU MLP: FIX title overlap with blocks
        # ------------------------------------------------------------
        mlp_title = Text(
            "MLP (SwiGLU): expand, split, gate, project back",
            font_size=34,
            weight=BOLD,
        ).set_color(WHITE).set_opacity(0.95)
        mlp_title.move_to(UP * 3.05)

        v512 = VectorBar("512", height=2.25, color=ACCENT).move_to(LEFT * 5.4 + DOWN * 0.20)
        v8192 = VectorBar("4096×2", height=2.95, color=ACCENT2).move_to(LEFT * 2.8 + DOWN * 0.20)
        u = VectorBar("4096 (u)", height=2.75, color=ACCENT2).move_to(RIGHT * 0.05 + DOWN * 0.20)
        vv = VectorBar("4096 (v)", height=2.75, color=ACCENT2).move_to(RIGHT * 2.05 + DOWN * 0.20)

        mul = Text("silu(u) ⊙ v", font_size=30, weight=BOLD).set_color(HI).move_to(RIGHT * 4.25 + UP * 0.35)
        v2048 = VectorBar("2048", height=2.50, color=ACCENT2).move_to(RIGHT * 4.25 + DOWN * 1.05)
        back = VectorBar("512", height=2.25, color=ACCENT).move_to(RIGHT * 5.85 + DOWN * 0.20)

        a_m1 = arrow_lr(v512, v8192, buff=0.25)
        a_m2 = arrow_lr(v8192, u, buff=0.25)
        a_m4 = arrow_ud(mul, v2048, buff=0.25)
        a_m5 = arrow_lr(v2048, back, buff=0.25)

        fc1 = small_label("Linear: 512 → 8192", 22).next_to(a_m1, UP, buff=0.12)
        split2 = small_label("Split into two halves", 22).next_to(a_m2, UP, buff=0.12)
        proj = small_label("Linear: 2048 → 512", 22).next_to(a_m5, UP, buff=0.12)

        self.play(FadeIn(mlp_title, shift=DOWN), run_time=T["fade"])
        self.play(FadeIn(v512, shift=RIGHT), run_time=T["fade"])
        self.play(Create(a_m1), FadeIn(v8192, shift=LEFT), FadeIn(fc1, shift=DOWN), run_time=T["reveal"])
        self.play(Create(a_m2), FadeIn(u, shift=LEFT), FadeIn(vv, shift=LEFT), FadeIn(split2, shift=DOWN), run_time=T["reveal"])
        self.play(FadeIn(mul, shift=UP), run_time=T["fade"])
        self.play(Create(a_m4), FadeIn(v2048, shift=UP), run_time=T["reveal"])
        self.play(Create(a_m5), FadeIn(back, shift=LEFT), FadeIn(proj, shift=DOWN), run_time=T["reveal"])
        self.wait(T["hold_short"])

        self.play(
            FadeOut(mlp_title),
            FadeOut(v512),
            FadeOut(v8192),
            FadeOut(u),
            FadeOut(vv),
            FadeOut(mul),
            FadeOut(v2048),
            FadeOut(back),
            FadeOut(a_m1),
            FadeOut(a_m2),
            FadeOut(a_m4),
            FadeOut(a_m5),
            FadeOut(fc1),
            FadeOut(split2),
            FadeOut(proj),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 6) Full trunk: rename to "Our model"; fix head text overflow
        # ------------------------------------------------------------
        t6 = title("Our model: 12 layers, 512 dims, 8 heads")
        self.play(Transform(t0, t6), run_time=T["fade"])

        cfg = VGroup(
            chip("d_model=512", ACCENT),
            chip("n_layers=12", ACCENT2),
            chip("n_heads=8 (64 each)", ACCENT),
            chip("d_ff=2048 (SwiGLU)", ACCENT2),
            chip("non-causal attention", GREY_D),
        ).arrange(DOWN, buff=0.18)
        cfg.move_to(LEFT * 4.8 + DOWN * 0.1)

        blk = RoundedRectangle(corner_radius=0.16, width=4.9, height=0.55).set_fill(GREY_D, 1.0).set_stroke(width=0)
        blk_lab = Text("Transformer layer", font_size=24, weight=BOLD).set_color(WHITE).move_to(blk.get_center())
        one = VGroup(blk, blk_lab)
        stack = VGroup(*[one.copy() for _ in range(4)]).arrange(DOWN, buff=0.16)
        stack.move_to(RIGHT * 0.9 + UP * 0.95)

        br = Brace(stack, RIGHT, buff=0.18)
        times = Text("×12", font_size=34, weight=BOLD).set_color(HI)
        times.next_to(br, RIGHT, buff=0.14)

        cls = chip("CLS = last token", GREY_D, 24).next_to(stack, DOWN, buff=0.45).align_to(stack, LEFT)

        # wrap + slightly smaller font so it stays inside
        head = box(
            "Action-value head\n→ 128 bucket logits",
            w=5.8,
            h=1.15,
            color=ACCENT,
            size=26,
        )
        head.next_to(cls, DOWN, buff=0.30).align_to(cls, LEFT)

        self.play(FadeIn(cfg, shift=RIGHT), run_time=T["reveal"])
        self.play(FadeIn(stack, shift=UP), FadeIn(br), FadeIn(times, shift=LEFT), run_time=T["reveal"])
        self.play(FadeIn(cls, shift=UP), run_time=T["fade"])
        self.play(FadeIn(head, shift=UP), run_time=T["fade"])
        self.wait(T["hold_short"])

        self.play(FadeOut(cfg), FadeOut(stack), FadeOut(br), FadeOut(times), FadeOut(cls), FadeOut(head), run_time=T["fade"])

        # ------------------------------------------------------------
        # 7) Training loss: FIX overlap + pace + dot follows line properly
        # ------------------------------------------------------------
        t7 = title("Training: distill win% into 128 buckets (Gaussian-soft CE)")
        self.play(Transform(t0, t7), run_time=T["fade"])

        # Top pipeline row (kept high so it doesn't collide with plots)
        rec = box("Dataset record:\n(FEN, move, win_prob)", w=4.5, h=1.25, color=ACCENT2, size=24).move_to(
            LEFT * 4.9 + UP * 2.05
        )
        buck = box("bucketize win_prob\n→ bucket id", w=3.5, h=1.25, color=ACCENT, size=24).move_to(
            LEFT * 0.9 + UP * 2.05
        )
        logits = box("model outputs\n128 logits", w=3.1, h=1.25, color=GREY_D, size=24).move_to(
            RIGHT * 2.85 + UP * 2.05
        )
        a_rec = arrow_lr(rec, buck, buff=0.25)
        a_b = arrow_lr(buck, logits, buff=0.25)

        # Distributions (center)
        def bars(vals, color, width=3.6, height=1.6):
            n = len(vals)
            mx = max(vals) if max(vals) > 0 else 1.0
            g = VGroup()
            for v in vals:
                hh = 0.06 + (v / mx) * (height - 0.06)
                rct = Rectangle(width=width / n * 0.92, height=hh)
                rct.set_fill(color, opacity=0.85)
                rct.set_stroke(width=0)
                g.add(rct)
            g.arrange(RIGHT, buff=0.03, aligned_edge=DOWN)
            frame = RoundedRectangle(corner_radius=0.12, width=width + 0.25, height=height + 0.25)
            frame.set_stroke(WHITE, opacity=0.35, width=2)
            frame.set_fill(BLACK, opacity=0.0)
            g.move_to(frame.get_bottom() + UP * (0.12 + height / 2))
            return VGroup(frame, g)

        target_vals = [0.02, 0.03, 0.05, 0.10, 0.18, 0.28, 0.18, 0.10, 0.05, 0.03, 0.02]
        pred_vals_1 = [0.02, 0.03, 0.04, 0.08, 0.14, 0.20, 0.18, 0.12, 0.07, 0.06, 0.06]
        pred_vals_2 = [0.01, 0.02, 0.03, 0.06, 0.10, 0.26, 0.22, 0.12, 0.08, 0.05, 0.05]

        tgt = bars(target_vals, ACCENT2, width=3.4, height=1.35)
        prd = bars(pred_vals_1, ACCENT, width=3.4, height=1.35)
        tgt_lab = small_label("Gaussian target\n(sigma = 0.75)", 22).next_to(tgt, UP, buff=0.12).align_to(tgt, LEFT)
        prd_lab = small_label("model softmax", 22).next_to(prd, UP, buff=0.12).align_to(prd, LEFT)
        dists = VGroup(VGroup(tgt_lab, tgt), VGroup(prd_lab, prd)).arrange(RIGHT, buff=0.65)
        dists.move_to(LEFT * 1.05 + DOWN * 0.05)

        ce = MathTex(r"\mathcal{L}=-\sum_b p_{tgt}(b)\log p_\theta(b)", font_size=40).set_color(WHITE)
        ce.move_to(LEFT * 1.10 + DOWN * 2.55)

        # LR schedule visual (right side) — dot moves along the curve path
        ax = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=5.3,
            y_length=2.25,
            tips=False,
        )
        ax.set_stroke(opacity=0.40)
        xlab = small_label("training progress", 20).next_to(ax, DOWN, buff=0.12)
        ylab = small_label("learning rate", 20).next_to(ax, LEFT, buff=0.12).rotate(PI / 2)

        def lr_func(x):
            if x < 0.18:
                return x / 0.18
            t = (x - 0.18) / (1 - 0.18)
            return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * t))

        curve = ax.plot(lr_func, x_range=[0, 1], stroke_width=5).set_color(WHITE)
        dot = Dot(radius=0.07).set_color(HI)
        dot.move_to(curve.point_from_proportion(0.0))

        lr_title = small_label("warmup + cosine decay", 22)
        lr_g = VGroup(lr_title, VGroup(ax, xlab, ylab, curve, dot)).arrange(DOWN, buff=0.10)
        lr_g.move_to(RIGHT * 4.35 + DOWN * 1.05)

        runchips = VGroup(
            chip("AV stage: lr=6e-4 → min 6e-5", GREY_D, 20),
            chip("… cosine, warmup=2000, wd=0.05, grad_clip=8.0", GREY_D, 20),
        ).arrange(DOWN, buff=0.10)
        runchips.next_to(lr_g, UP, buff=0.25).align_to(lr_g, LEFT)

        self.play(FadeIn(rec, shift=RIGHT), run_time=T["fade"])
        self.play(FadeIn(buck, shift=RIGHT), Create(a_rec), run_time=T["reveal"])
        self.play(FadeIn(logits, shift=RIGHT), Create(a_b), run_time=T["reveal"])
        self.wait(0.3)

        self.play(FadeIn(dists, shift=UP), run_time=T["reveal"])
        self.play(Write(ce), run_time=T["reveal"])
        self.play(FadeIn(lr_g, shift=UP), FadeIn(runchips, shift=DOWN), run_time=T["reveal"])

        # move dot ALONG the curve (no off-curve drift)
        self.play(MoveAlongPath(dot, curve, rate_func=linear), run_time=3.6)

        # predicted distribution improves (animate bars)
        prd2 = bars(pred_vals_2, ACCENT, width=3.4, height=1.35)
        prd2.move_to(prd.get_center())
        self.play(Transform(prd, prd2), run_time=T["slow"])
        self.wait(T["hold_short"])

        self.play(
            FadeOut(rec),
            FadeOut(buck),
            FadeOut(logits),
            FadeOut(a_rec),
            FadeOut(a_b),
            FadeOut(dists),
            FadeOut(ce),
            FadeOut(lr_g),
            FadeOut(runchips),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 8) Liquid reasoning: keep as you had, but spaced & de-overlapped
        # (your original logic mostly fine; just layout safety)
        # ------------------------------------------------------------
        t8 = title("Liquid Reasoning: trunk once, then a 6-step loop")
        self.play(Transform(t0, t8), run_time=T["fade"])

        mem_title = small_label("memory = trunk outputs for non-CLS tokens", 26).move_to(UP * 2.55)
        mem = VGroup(*[VectorBar("512", height=1.55, color=ACCENT2) for _ in range(6)]).arrange(RIGHT, buff=0.25)
        mem.move_to(UP * 1.55)

        r = VectorBar("r (512)", height=2.75, color=ACCENT).move_to(LEFT * 5.25 + DOWN * 0.95)
        r0_lab = small_label("r₀ = CLS", 22).next_to(r, UP, buff=0.12).align_to(r, LEFT)

        attn_lines = VGroup()
        for m in mem:
            ln = Line(r.body.get_right(), m.body.get_left()).set_stroke(ACCENT, opacity=0.20, width=2)
            attn_lines.add(ln)

        step_counter = Text("step 1 / 6", font_size=40, weight=BOLD).set_color(HI)
        step_counter.move_to(RIGHT * 5.05 + DOWN * 3.05)

        # Discard gate
        dg_frame = RoundedRectangle(corner_radius=0.12, width=3.6, height=0.42).set_stroke(WHITE, opacity=0.35, width=2)
        dg_fill = Rectangle(width=1.2, height=0.32).set_fill(WARN, opacity=0.85).set_stroke(width=0)
        dg_fill.align_to(dg_frame, LEFT).move_to(dg_frame.get_left() + RIGHT * (dg_fill.width / 2))
        dg_lab = small_label("discard gate d (1=keep old)", 22).next_to(dg_frame, UP, buff=0.12)
        dg = VGroup(dg_lab, VGroup(dg_frame, dg_fill)).arrange(DOWN, buff=0.10)
        dg.move_to(RIGHT * 3.7 + DOWN * 1.10)

        # Stop gate
        sg_frame = RoundedRectangle(corner_radius=0.12, width=3.6, height=0.42).set_stroke(WHITE, opacity=0.35, width=2)
        sg_fill = Rectangle(width=0.6, height=0.32).set_fill(GOOD, opacity=0.85).set_stroke(width=0)
        sg_fill.align_to(sg_frame, LEFT).move_to(sg_frame.get_left() + RIGHT * (sg_fill.width / 2))
        sg_lab = small_label("stop prob s = sigmoid(stop_gate(r_new))", 22).next_to(sg_frame, UP, buff=0.12)
        thresh = Line(
            sg_frame.get_left() + RIGHT * (3.6 * 0.9),
            sg_frame.get_left() + RIGHT * (3.6 * 0.9) + UP * 0.42,
        ).set_stroke(HI, width=4, opacity=0.9)
        sg = VGroup(sg_lab, VGroup(sg_frame, sg_fill, thresh)).arrange(DOWN, buff=0.10)
        sg.move_to(RIGHT * 3.7 + DOWN * 2.20)

        params = VGroup(
            chip("lrt_steps=6", ACCENT, 20),
            chip("lrt_heads=4 → head_dim=128", ACCENT2, 20),
            chip("cross-attn uses scaled_dot_product_attention", GREY_D, 20),
        ).arrange(DOWN, buff=0.10)
        params.move_to(RIGHT * 3.95 + UP * 0.40)

        self.play(FadeIn(mem_title, shift=DOWN), run_time=T["fade"])
        self.play(FadeIn(mem, shift=DOWN), run_time=T["reveal"])
        self.play(FadeIn(r, shift=RIGHT), FadeIn(r0_lab, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(attn_lines), run_time=T["fade"])
        self.play(FadeIn(params, shift=LEFT), run_time=T["fade"])
        self.play(FadeIn(dg, shift=UP), FadeIn(sg, shift=UP), FadeIn(step_counter, shift=UP), run_time=T["reveal"])

        attn_patterns = [
            [0.10, 0.15, 0.42, 0.12, 0.14, 0.07],
            [0.08, 0.12, 0.22, 0.30, 0.18, 0.10],
            [0.06, 0.10, 0.15, 0.44, 0.18, 0.07],
            [0.05, 0.08, 0.12, 0.50, 0.18, 0.07],
            [0.05, 0.08, 0.10, 0.48, 0.22, 0.07],
            [0.05, 0.08, 0.10, 0.46, 0.24, 0.07],
        ]
        discard_vals = [0.55, 0.52, 0.48, 0.50, 0.52, 0.53]
        stop_vals = [0.30, 0.45, 0.60, 0.72, 0.79, 0.79]

        for i in range(6):
            new_counter = Text(f"step {i+1} / 6", font_size=40, weight=BOLD).set_color(HI)
            new_counter.move_to(step_counter.get_center())

            anims = [Transform(step_counter, new_counter)]
            for j, ln in enumerate(attn_lines):
                w = attn_patterns[i][j]
                anims.append(ln.animate.set_stroke(ACCENT, opacity=0.45, width=2 + 10 * w))

            dw = 3.6 * discard_vals[i]
            new_dfill = Rectangle(width=max(0.08, dw), height=0.32).set_fill(WARN, opacity=0.85).set_stroke(width=0)
            new_dfill.align_to(dg_frame, LEFT).move_to(dg_frame.get_left() + RIGHT * (new_dfill.width / 2))
            anims.append(Transform(dg_fill, new_dfill))

            sw = 3.6 * stop_vals[i]
            new_sfill = Rectangle(width=max(0.08, sw), height=0.32).set_fill(GOOD, opacity=0.85).set_stroke(width=0)
            new_sfill.align_to(sg_frame, LEFT).move_to(sg_frame.get_left() + RIGHT * (new_sfill.width / 2))
            anims.append(Transform(sg_fill, new_sfill))

            self.play(AnimationGroup(*anims, lag_ratio=0.02), run_time=1.15)
            self.wait(0.30)

        lrt_takeaway = Text("Goal: harder positions should “use more steps”.", font_size=30, weight=BOLD).set_color(WHITE).set_opacity(
            0.95
        )
        lrt_takeaway.move_to(DOWN * 3.25)
        self.play(FadeIn(lrt_takeaway, shift=UP), run_time=T["fade"])
        self.wait(T["hold_short"])

        self.play(
            FadeOut(mem_title),
            FadeOut(mem),
            FadeOut(r),
            FadeOut(r0_lab),
            FadeOut(attn_lines),
            FadeOut(params),
            FadeOut(dg),
            FadeOut(sg),
            FadeOut(step_counter),
            FadeOut(lrt_takeaway),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 9) Stop gate finetune (keep your original; just overlap fixes)
        # (your original section mostly OK; leaving as-is)
        # ------------------------------------------------------------
        t9 = title("Stop gate finetune (Lichess puzzles) — exactly your method")
        self.play(Transform(t0, t9), run_time=T["fade"])

        pz = box("Puzzle batch", w=3.2, h=1.0, color=ACCENT2, size=26).move_to(LEFT * 5.4 + UP * 1.8)
        runk = box("Run 6 steps\n(get r_hist, stop_logits)", w=4.3, h=1.2, color=GREY_D, size=24).move_to(
            LEFT * 1.65 + UP * 1.8
        )
        logp = box("Per-step log P(correct move)", w=4.6, h=1.0, color=ACCENT, size=24).move_to(RIGHT * 3.25 + UP * 1.8)

        a_p = arrow_lr(pz, runk, buff=0.25)
        a_r = arrow_lr(runk, logp, buff=0.25)

        steps = VGroup(*[Text(str(i + 1), font_size=22, weight=BOLD).set_color(WHITE) for i in range(6)]).arrange(
            RIGHT, buff=0.55
        )
        steps.move_to(LEFT * 1.65 + DOWN * 0.40)

        vals = [0.35, 0.55, 0.68, 0.72, 0.71, 0.71]
        bars_v = VGroup()
        for i, v in enumerate(vals):
            rct = Rectangle(width=0.30, height=0.20 + 1.55 * v).set_fill(ACCENT, 0.80).set_stroke(width=0)
            rct.align_to(steps[i], DOWN).shift(UP * 0.20)
            bars_v.add(rct)

        chart = VGroup(steps, bars_v).move_to(DOWN * 0.65 + LEFT * 1.65)
        chart_lab = small_label("log P(correct move) at each reasoning step", 22).next_to(chart, UP, buff=0.16).align_to(chart, LEFT)

        pick = SurroundingRectangle(VGroup(steps[2], bars_v[2]), buff=0.18).set_stroke(HI, width=4)
        tstar = Text("t* = 3", font_size=34, weight=BOLD).set_color(HI).next_to(pick, RIGHT, buff=0.25)

        rule = Text(
            "choose_t_star:\n• take best step\n• allow step 1 only if near-best AND confident (p0 ≥ 0.75)\n• clamp to min_steps=1",
            font_size=24,
        ).set_color(WHITE).set_opacity(0.92)
        rule.move_to(RIGHT * 3.55 + DOWN * 0.55)

        params2 = VGroup(
            chip("margin=0.15", GREY_D, 20),
            chip("p0_thresh=0.75", GREY_D, 20),
            chip("min_steps=1", GREY_D, 20),
            chip("lr=3e-3 (stop gate only)", GREY_D, 20),
        ).arrange(DOWN, buff=0.10)
        params2.move_to(RIGHT * 4.8 + DOWN * 2.55)

        yrow = token_row(["0", "0", "1", "1", "1", "1"], w=0.50, h=0.38, size=20, buff=0.10)
        ylab = small_label("stop targets y_t = 1 for t ≥ t*", 22).next_to(yrow, UP, buff=0.12).align_to(yrow, LEFT)
        y = VGroup(ylab, yrow).arrange(DOWN, buff=0.10).move_to(LEFT * 1.65 + DOWN * 2.60)

        bce = MathTex(
            r"\mathcal{L}=\mathrm{BCEWithLogits}(\text{stop\_logits}, y) + \lambda(1-\sigma(\text{stop\_logits}))",
            font_size=32,
        ).set_color(WHITE)
        bce.move_to(RIGHT * 2.05 + DOWN * 2.60)

        self.play(FadeIn(pz, shift=RIGHT), run_time=T["fade"])
        self.play(FadeIn(runk, shift=RIGHT), Create(a_p), run_time=T["reveal"])
        self.play(FadeIn(logp, shift=RIGHT), Create(a_r), run_time=T["reveal"])
        self.play(FadeIn(chart_lab, shift=DOWN), FadeIn(chart, shift=UP), run_time=T["reveal"])
        self.play(Create(pick), FadeIn(tstar, shift=LEFT), run_time=T["fade"])
        self.play(FadeIn(rule, shift=UP), FadeIn(params2, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(y, shift=UP), Write(bce), run_time=T["reveal"])
        self.wait(T["hold_short"])

        merge = Text(
            "Then you copied these stop-gate weights into your main checkpoint.",
            font_size=30,
            weight=BOLD,
        ).set_color(WHITE).set_opacity(0.95)
        merge.move_to(DOWN * 3.25)
        self.play(FadeIn(merge, shift=UP), run_time=T["fade"])
        self.wait(T["hold_short"])

        self.play(
            FadeOut(pz),
            FadeOut(runk),
            FadeOut(logp),
            FadeOut(a_p),
            FadeOut(a_r),
            FadeOut(chart_lab),
            FadeOut(chart),
            FadeOut(pick),
            FadeOut(tstar),
            FadeOut(rule),
            FadeOut(params2),
            FadeOut(y),
            FadeOut(bce),
            FadeOut(merge),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 10) What you observed (unchanged)
        # ------------------------------------------------------------
        t10 = title("What you observed: step 1 ≈ step 6")
        self.play(Transform(t0, t10), run_time=T["fade"])

        left = box("Use step 1", w=3.0, h=1.0, color=GREY_D, size=26).move_to(LEFT * 3.2 + UP * 1.0)
        right = box("Use step 6", w=3.0, h=1.0, color=GREY_D, size=26).move_to(RIGHT * 3.2 + UP * 1.0)
        approx = MathTex(r"\approx", font_size=90).set_color(HI).move_to(UP * 1.0)

        def strength_bar(x, y, wfill=2.8):
            frame = RoundedRectangle(corner_radius=0.12, width=3.4, height=0.5).set_stroke(WHITE, opacity=0.35, width=2)
            fill = Rectangle(width=wfill, height=0.36).set_fill(ACCENT, opacity=0.85).set_stroke(width=0)
            fill.align_to(frame, LEFT).move_to(frame.get_left() + RIGHT * (fill.width / 2))
            lab = small_label("measured strength", 22).next_to(frame, UP, buff=0.12)
            g = VGroup(lab, VGroup(frame, fill)).arrange(DOWN, buff=0.10)
            g.move_to(x * RIGHT + y * UP)
            return g

        barL = strength_bar(-3.2, -0.6, wfill=2.75)
        barR = strength_bar(3.2, -0.6, wfill=2.75)

        msg = Text(
            "In your testing, enabling LRT didn’t change results.\nSo the reasoning loop likely wasn’t taking effect.",
            font_size=30,
            weight=BOLD,
        ).set_opacity(0.95).move_to(DOWN * 2.65)

        self.play(FadeIn(left, shift=DOWN), FadeIn(right, shift=DOWN), FadeIn(approx), run_time=T["reveal"])
        self.play(FadeIn(barL, shift=UP), FadeIn(barR, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(msg, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_short"])

        self.play(FadeOut(left), FadeOut(right), FadeOut(approx), FadeOut(barL), FadeOut(barR), FadeOut(msg), run_time=T["fade"])

        # ------------------------------------------------------------
        # 11) Evaluation: REPLACE Elo curve with puzzle accuracy graph + Elo note
        # ------------------------------------------------------------
        t11 = title("Evaluation: final puzzle accuracy by rating")
        self.play(Transform(t0, t11), run_time=T["fade"])

        # Data provided by you
        ranges = [
            "200-399", "400-599", "600-799", "800-999", "1000-1199", "1200-1399",
            "1400-1599", "1600-1799", "1800-1999", "2000-2199", "2200-2399", "2400-2599",
            "2600-2799", "2800-2999", "3000-3199", "3200-3399",
        ]
        first = [0.9667, 0.9667, 0.8600, 0.7200, 0.8000, 0.7333, 0.6200, 0.6000, 0.4800, 0.5400, 0.4667, 0.5400, 0.5000, 0.4000, 0.4800, 0.5000]
        full =  [0.9667, 0.9667, 0.8000, 0.7000, 0.6800, 0.7000, 0.5200, 0.4000, 0.3400, 0.3000, 0.2333, 0.0800, 0.0400, 0.0000, 0.0200, 0.0833]

        ax = Axes(
            x_range=[0, len(ranges) - 1, 1],
            y_range=[0, 1.0, 0.2],
            x_length=11.3,
            y_length=3.1,
            tips=False,
        )
        ax.set_stroke(opacity=0.40)
        ax.move_to(DOWN * 0.35)

        # Plot two lines (FirstAcc vs FullAcc)
        def to_points(vals):
            return [ax.c2p(i, vals[i]) for i in range(len(vals))]

        first_pts = to_points(first)
        full_pts = to_points(full)

        first_line = VMobject().set_points_as_corners(first_pts).set_stroke(ACCENT, width=4, opacity=0.9)
        full_line = VMobject().set_points_as_corners(full_pts).set_stroke(ACCENT2, width=4, opacity=0.9)

        first_dots = VGroup(*[Dot(radius=0.045).set_color(ACCENT).move_to(p) for p in first_pts])
        full_dots = VGroup(*[Dot(radius=0.045).set_color(ACCENT2).move_to(p) for p in full_pts])

        # X labels (every other to keep readable)
        xlabels = VGroup()
        for i, r in enumerate(ranges):
            if i % 2 == 0:
                lab = Text(r, font_size=18).set_color(WHITE).set_opacity(0.92)
                lab.move_to(ax.c2p(i, 0) + DOWN * 0.35)
                xlabels.add(lab)

        ylab = small_label("accuracy", 22).next_to(ax, LEFT, buff=0.15).rotate(PI / 2)
        xlab = small_label("puzzle rating range (bucket size = 200)", 22).next_to(ax, DOWN, buff=0.70)

        leg = VGroup(
            VGroup(Dot(radius=0.06).set_color(ACCENT), small_label("FirstAcc", 22)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.06).set_color(ACCENT2), small_label("FullAcc", 22)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.12)
        leg.move_to(RIGHT * 5.1 + UP * 2.25)

        elo_note = Text(
            "Estimated playing strength: ~2000–2150 Elo\n(from fastchess + Ordo)",
            font_size=26,
            weight=BOLD,
        ).set_color(HI).set_opacity(0.95)
        elo_note.move_to(LEFT * 4.35 + UP * 2.45)

        self.play(FadeIn(ax, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(ylab), FadeIn(xlab), FadeIn(xlabels), run_time=T["fade"])
        self.play(Create(first_line), FadeIn(first_dots), run_time=T["reveal"])
        self.play(Create(full_line), FadeIn(full_dots), run_time=T["reveal"])
        self.play(FadeIn(leg, shift=DOWN), FadeIn(elo_note, shift=DOWN), run_time=T["reveal"])
        self.wait(T["hold_long"])

        self.play(
            FadeOut(ax),
            FadeOut(ylab),
            FadeOut(xlab),
            FadeOut(xlabels),
            FadeOut(first_line),
            FadeOut(full_line),
            FadeOut(first_dots),
            FadeOut(full_dots),
            FadeOut(leg),
            FadeOut(elo_note),
            run_time=T["fade"],
        )

        # ------------------------------------------------------------
        # 12) Failure modes (unchanged)
        # ------------------------------------------------------------
        t12 = title("Two failure modes you saw (common for searchless play)")
        self.play(Transform(t0, t12), run_time=T["fade"])

        f1 = box("1) Missed checkmates", w=5.4, h=1.0, color=WARN, size=28).move_to(UP * 1.6)
        f2 = box("2) Repeated moves (3-fold)", w=5.4, h=1.0, color=WARN, size=28).next_to(f1, DOWN, buff=0.40)

        b = chessboard(2.4).move_to(LEFT * 4.3 + DOWN * 2.0)
        glow = SurroundingRectangle(b, buff=0.10).set_stroke(HI, width=5).set_opacity(0.85)
        mate_txt = Text("Mate in 2 exists", font_size=28, weight=BOLD).set_color(HI).next_to(b, RIGHT, buff=0.30).shift(UP * 0.5)
        wrong = Text("model chooses\nnon-forcing line", font_size=24, weight=BOLD).set_color(BAD).next_to(mate_txt, DOWN, buff=0.20).align_to(mate_txt, LEFT)

        A = chip("Position A", GREY_D, 22)
        Bp = chip("Position B", GREY_D, 22)
        loop = VGroup(A, Bp).arrange(RIGHT, buff=0.7).move_to(RIGHT * 3.2 + DOWN * 2.0)
        c1 = CurvedArrow(A.get_bottom() + DOWN * 0.05, Bp.get_bottom() + DOWN * 0.05, angle=-TAU / 4)
        c2 = CurvedArrow(Bp.get_top() + UP * 0.05, A.get_top() + UP * 0.05, angle=-TAU / 4)
        c1.set_stroke(WHITE, opacity=0.55, width=4)
        c2.set_stroke(WHITE, opacity=0.55, width=4)
        draw = Text("… repeats …", font_size=24, weight=BOLD).set_color(WARN).next_to(loop, DOWN, buff=0.20)

        end = Text(
            "Without tree search, the model must be *exact* on tactics and endings.\nSmall underfit models often fail here.",
            font_size=30,
            weight=BOLD,
        ).set_opacity(0.95).move_to(DOWN * 3.25)

        self.play(FadeIn(f1, shift=DOWN), FadeIn(f2, shift=DOWN), run_time=T["reveal"])
        self.play(FadeIn(b, scale=0.98), Create(glow), run_time=T["reveal"])
        self.play(FadeIn(mate_txt, shift=LEFT), FadeIn(wrong, shift=LEFT), run_time=T["reveal"])
        self.play(FadeIn(loop, shift=UP), Create(c1), Create(c2), FadeIn(draw, shift=UP), run_time=T["reveal"])
        self.play(FadeIn(end, shift=UP), run_time=T["reveal"])
        self.wait(T["hold_long"])

        self.play(
            FadeOut(f1),
            FadeOut(f2),
            FadeOut(b),
            FadeOut(glow),
            FadeOut(mate_txt),
            FadeOut(wrong),
            FadeOut(loop),
            FadeOut(c1),
            FadeOut(c2),
            FadeOut(draw),
            FadeOut(end),
            run_time=T["fade"],
        )
        self.play(FadeOut(t0), run_time=T["fade"])
