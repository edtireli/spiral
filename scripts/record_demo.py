"""Record the REAL CLI into assets/demo.gif — a genuine terminal recording.

The command runs in a real PTY; the raw ANSI byte stream is captured with its
true timing, replayed through a terminal emulator (pyte), and each screen state
is rasterized into a GIF frame. Nothing shown is simulated — the banner, the
cascade, the spinners are the program's actual output. Only the window chrome
and the typed prompt line are drawn by this script.

    python scripts/record_demo.py --dir ~/code/pomo \
        --cmd 'build "a pomodoro TUI in python, with tests"' --max 24

Requires: pyte (pip install pyte), the Menlo / Apple Braille system fonts.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pyte
import pyte.graphics
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).parent.parent

# ---- palette -----------------------------------------------------------------
BG = (13, 13, 15)
CHROME = (26, 26, 30)
FG = (224, 226, 223)
CLAY = (217, 119, 87)
ANSI = {
    "black": (40, 42, 46), "red": (255, 99, 92), "green": (78, 201, 108),
    "brown": (255, 176, 0), "blue": (108, 153, 255), "magenta": (198, 120, 221),
    "cyan": (86, 182, 194), "white": (224, 226, 223),
    "brightblack": (110, 116, 124), "brightred": (255, 121, 116),
    "brightgreen": (110, 220, 138), "brightyellow": (255, 200, 80),
    "brightblue": (140, 178, 255), "brightmagenta": (216, 148, 235),
    "brightcyan": (120, 200, 210), "brightwhite": (238, 240, 236),
}

MENLO = "/System/Library/Fonts/Menlo.ttc"
BRAILLE_TTF = "/System/Library/Fonts/Apple Braille.ttf"
FALLBACKS = ("/System/Library/Fonts/Apple Symbols.ttf", "/Library/Fonts/Arial Unicode.ttf")
SIZE = 20
PAD, TITLE_H = 26, 42


class FontKit:
    def __init__(self) -> None:
        self.reg = ImageFont.truetype(MENLO, SIZE, index=0)
        self.bold = self.reg
        for i in range(1, 8):
            try:
                f = ImageFont.truetype(MENLO, SIZE, index=i)
                if f.getname()[1] == "Bold":
                    self.bold = f
                    break
            except Exception:
                break
        self.fb = [ImageFont.truetype(p, SIZE) for p in FALLBACKS if Path(p).is_file()]
        self._notdef: dict[int, bytes] = {}
        self._cache: dict[tuple[str, bool], ImageFont.FreeTypeFont] = {}
        self.cell = round(self.reg.getlength("M"))
        asc, desc = self.reg.getmetrics()
        self.line_h = asc + desc + 2

    @staticmethod
    def _bitmap(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
        im = Image.new("L", (SIZE * 2, SIZE * 2), 0)
        ImageDraw.Draw(im).text((0, 0), ch, font=font, fill=255)
        return im.tobytes()

    def _covers(self, font: ImageFont.FreeTypeFont, ch: str) -> bool:
        key = id(font)
        if key not in self._notdef:
            self._notdef[key] = self._bitmap(font, "￿")
        bm = self._bitmap(font, ch)
        return bm != self._notdef[key] and any(bm)

    def pick(self, ch: str, bold: bool) -> ImageFont.FreeTypeFont:
        got = self._cache.get((ch, bold))
        if got:
            return got
        primary = self.bold if bold else self.reg
        font = primary if (ch.isascii() or self._covers(primary, ch)) else next(
            (f for f in self.fb if self._covers(f, ch)), primary)
        self._cache[(ch, bold)] = font
        return font


FK = FontKit()

_BRAILLE_DOT = {0x01: (0, 0), 0x02: (0, 1), 0x04: (0, 2), 0x40: (0, 3),
                0x08: (1, 0), 0x10: (1, 1), 0x20: (1, 2), 0x80: (1, 3)}


def draw_char(d: ImageDraw.ImageDraw, cx: float, cy: float, ch: str, color, bold: bool) -> None:
    """Braille and rules are hand-drawn (crisp at terminal size); everything
    else renders through the font chain."""
    w, h = FK.cell, FK.line_h
    o = ord(ch)
    if 0x2800 <= o <= 0x28FF:
        bits = o - 0x2800
        r = max(1.6, w / 5.4)
        for bit, (bx, by) in _BRAILLE_DOT.items():
            if bits & bit:
                px = cx + w * (0.28 + 0.44 * bx)
                py = cy + h * (0.16 + 0.24 * by)
                d.ellipse((px - r, py - r, px + r, py + r), fill=color)
        return
    if ch in "─━":
        t = 3 if ch == "━" else 1
        my = cy + h * 0.52
        d.rectangle((cx, my - t / 2, cx + w, my + t / 2), fill=color)
        return
    d.text((cx, cy), ch, font=FK.pick(ch, bold), fill=color)


def _color(spec: str, default) -> tuple:
    if spec == "default":
        return default
    if spec in ANSI:
        return ANSI[spec]
    try:
        return tuple(int(spec[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _dim(c: tuple) -> tuple:
    return tuple(int(v * 0.55 + b * 0.45) for v, b in zip(c, BG))


# ---- recording ---------------------------------------------------------------
def record(argv: list[str], cwd: str, cols: int, rows: int, max_s: float,
           watch: str = "", reply: str = "", reply_delay: float = 1.4,
           tail: float = 7.0) -> list[tuple[float, bytes]]:
    """Run the command in a real PTY and capture its byte stream with timing.
    If `watch` text appears in the output, `reply` is typed into the pty after
    `reply_delay` seconds (the echo shows up in the recording, like a human
    answering), and capture ends `tail` seconds later."""
    master, slave = os.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor",
           "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(argv, cwd=cwd, stdin=slave, stdout=slave, stderr=slave,
                            env=env, close_fds=True)
    os.close(slave)
    chunks: list[tuple[float, bytes]] = []
    t0 = time.monotonic()
    seen_tail = b""
    reply_at: float | None = None
    replied_at: float | None = None
    try:
        while True:
            t = time.monotonic() - t0
            if t > max_s:
                break
            if reply_at is not None and replied_at is None and t >= reply_at:
                os.write(master, reply.encode() + b"\n")
                replied_at = t
            if replied_at is not None and t > replied_at + tail:
                break
            r, _, _ = select.select([master], [], [], 0.03)
            if r:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append((t, data))
                if watch and reply_at is None:
                    seen_tail = (seen_tail + data)[-4096:]
                    if watch.encode() in seen_tail:
                        reply_at = t + reply_delay
            elif proc.poll() is not None:
                break
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        os.close(master)
    return chunks


# ---- replay → frames ---------------------------------------------------------
Grid = tuple  # rows of (char, fg, bg, bold, dim)


def replay(chunks: list[tuple[float, bytes]], cols: int, rows: int) -> list[tuple[Grid, int]]:
    pyte.graphics.TEXT[2] = "+blink"  # carry SGR 2 (dim) through pyte's blink flag
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)

    def snap() -> Grid:
        out = []
        for y in range(rows):
            line = screen.buffer[y]
            out.append(tuple(
                (line[x].data or " ", line[x].fg, line[x].bg, line[x].bold, line[x].blink)
                for x in range(cols)))
        return tuple(out)

    frames: list[tuple[Grid, int]] = []
    last_t = 0.0
    for t, data in chunks:
        gap = int((t - last_t) * 1000)
        grid = snap()
        if frames and grid == frames[-1][0]:
            frames[-1] = (grid, frames[-1][1] + gap)
        elif gap > 25 or not frames:  # sub-25ms bursts are one repaint, not a frame
            frames.append((grid, gap))
        else:  # burst of writes — replace the barely-shown frame
            frames[-1] = (frames[-1][0], frames[-1][1] + gap)
        stream.feed(data)
        last_t = t
    frames.append((snap(), 600))
    frames = [(g, min(max(ms, 40), 600)) for g, ms in frames if ms >= 20]

    # thin status-line-only ticks (the long model-wait spinner): a change confined
    # to one row keeps ~4-5 fps; everything else keeps its real pacing
    thinned: list[tuple[Grid, int]] = []
    for g, ms in frames:
        if thinned:
            changed = [i for i, (a, b) in enumerate(zip(thinned[-1][0], g)) if a != b]
            if len(changed) <= 1 and thinned[-1][1] < 220:
                thinned[-1] = (g, thinned[-1][1] + ms)
                continue
        thinned.append((g, ms))
    return _squeeze(thinned)


def _squeeze(frames: list[tuple[Grid, int]], max_run: int = 10, fast_ms: int = 140) -> list[tuple[Grid, int]]:
    """Time-lapse the waits: a long run of status-only frames (minutes of model
    thinking) becomes a short fast-spin flourish instead of dead GIF time."""
    out: list[tuple[Grid, int]] = []
    run: list[tuple[Grid, int]] = []

    def flush() -> None:
        nonlocal run
        if len(run) > max_run:
            idx = [round(i * (len(run) - 1) / (max_run - 1)) for i in range(max_run)]
            run = [(run[i][0], fast_ms) for i in idx]
        out.extend(run)
        run = []

    prev: Grid | None = None
    for g, ms in frames:
        status_only = prev is not None and sum(1 for a, b in zip(prev, g) if a != b) <= 1
        if status_only:
            run.append((g, ms))
        else:
            flush()
            out.append((g, ms))
        prev = g
    flush()
    return out


# ---- rasterize ---------------------------------------------------------------
def render(grid: Grid, cols: int, rows: int, title: str, prompt: str) -> Image.Image:
    W = PAD * 2 + FK.cell * cols
    H = TITLE_H + PAD + FK.line_h * (rows + 1) + PAD
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=14, fill=BG)
    d.rounded_rectangle((0, 0, W - 1, TITLE_H), radius=14, fill=CHROME)
    d.rectangle((0, TITLE_H // 2, W - 1, TITLE_H), fill=CHROME)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse((PAD - 6 + i * 22, 15, PAD + 6 + i * 22, 27), fill=c)
    d.text(((W - FK.reg.getlength(title)) / 2, 10), title, font=FK.reg, fill=_dim(FG))

    y0 = TITLE_H + PAD
    if prompt:
        draw_char(d, PAD, y0, "❯", CLAY, True)
        for i, ch in enumerate(prompt):
            if ch != " ":
                draw_char(d, PAD + (i + 2) * FK.cell, y0, ch, FG, False)
    for ry, row in enumerate(grid):
        y = y0 + (ry + 1) * FK.line_h
        for rx, (ch, fg, bg, bold, dim) in enumerate(row):
            if ch == " " and bg == "default":
                continue
            x = PAD + rx * FK.cell
            color = _color(fg, FG)
            if bg != "default":
                d.rectangle((x, y, x + FK.cell, y + FK.line_h), fill=_color(bg, BG))
            draw_char(d, x, y, ch, _dim(color) if dim else color, bold)
    return img


def save_gif(imgs: list[Image.Image], durs: list[int], out: Path) -> None:
    """Delta-encoded GIF: one shared palette, and every frame after the first
    keeps only its changed pixels (the rest transparent, disposal=keep). On a
    terminal recording almost everything is static, so this is the difference
    between a bloated GIF and a small one."""
    from PIL import ImageChops

    w, h = imgs[0].size
    step = max(1, len(imgs) // 6)
    sample = Image.new("RGB", (w, h * min(6, len(imgs))))
    for i, f in enumerate(imgs[::step][:6]):
        sample.paste(f.convert("RGB"), (0, i * h))
    pal = sample.quantize(colors=127, method=Image.Quantize.MEDIANCUT)
    qs = [f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in imgs]

    first = qs[0].copy()
    outer = imgs[0].getchannel("A").point(lambda a: 255 if a < 96 else 0)
    first.paste(255, mask=outer)  # window corners stay transparent
    frames = [first]
    prev = qs[0].convert("RGB")
    for q in qs[1:]:
        cur = q.convert("RGB")
        same = ImageChops.difference(cur, prev).convert("L").point(lambda v: 255 if v == 0 else 0)
        fr = q.copy()
        fr.paste(255, mask=same)
        frames.append(fr)
        prev = cur
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs,
                   loop=0, transparency=255, disposal=1, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default='build --approve "a pomodoro TUI in python, with tests"',
                    help="spiral subcommand line to record")
    ap.add_argument("--dir", default=".", help="project directory to run in")
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--scale", type=float, default=0.70, help="output downscale factor")
    ap.add_argument("--max", type=float, default=480.0, help="hard cap in seconds")
    ap.add_argument("--watch", default="execute this plan?",
                    help="when this text appears, type --reply and finish (empty disables)")
    ap.add_argument("--reply", default="y")
    ap.add_argument("--tail", type=float, default=7.0, help="seconds kept after the reply")
    ap.add_argument("--out", default=str(REPO / "assets" / "demo.gif"))
    ap.add_argument("--dump", type=int, default=0, help="also write every Nth frame as PNG")
    ap.add_argument("--chunks", default="", help="save the raw recording here (pickle)")
    ap.add_argument("--from-chunks", default="", help="re-render from a saved recording instead of recording")
    a = ap.parse_args()

    import pickle
    import shlex
    argv = [sys.executable, "-m", "spiral.cli", *shlex.split(a.cmd)]
    prompt = f"spiral {a.cmd}"
    title = f"spiral — {Path(a.dir).resolve().name}"

    if a.from_chunks:
        chunks = pickle.loads(Path(a.from_chunks).read_bytes())
        print(f"re-rendering {len(chunks)} chunks from {a.from_chunks}")
    else:
        print(f"recording: {' '.join(argv)}  (cwd={a.dir}, {a.cols}x{a.rows}, cap {a.max}s)")
        chunks = record(argv, a.dir, a.cols, a.rows, a.max,
                        watch=a.watch, reply=a.reply, tail=a.tail)
        if a.chunks:
            Path(a.chunks).write_bytes(pickle.dumps(chunks))
            print(f"raw recording saved to {a.chunks} (re-render with --from-chunks)")
    # redact the recording machine's interpreter path — cosmetic, and no $HOME leak
    exe = sys.executable.encode()
    chunks = [(t, d.replace(exe, b"python3")) for t, d in chunks]
    if exe in b"".join(d for _, d in chunks):
        print("warning: executable path split across chunks — still visible")
    print(f"captured {len(chunks)} chunks · {sum(len(c) for _, c in chunks)} bytes")
    frames = replay(chunks, a.cols, a.rows)
    print(f"{len(frames)} frames after dedup")

    # typed-prompt intro (the one presentational flourish), then the recording
    imgs: list[Image.Image] = []
    durs: list[int] = []
    blank: Grid = tuple(tuple((" ", "default", "default", False, False) for _ in range(a.cols))
                        for _ in range(a.rows))
    for i in range(0, len(prompt) + 1, 3):
        imgs.append(render(blank, a.cols, a.rows, title, prompt[:i]))
        durs.append(55)
    imgs.append(render(blank, a.cols, a.rows, title, prompt))
    durs.append(450)
    for grid, ms in frames:
        imgs.append(render(grid, a.cols, a.rows, title, prompt))
        durs.append(ms)
    durs[-1] = 3200

    if a.scale != 1.0:
        size = (int(imgs[0].width * a.scale), int(imgs[0].height * a.scale))
        imgs = [f.resize(size, Image.LANCZOS) for f in imgs]
    if a.dump:
        for i in range(0, len(imgs), a.dump):
            imgs[i].convert("RGB").save(REPO / "assets" / f"_rec_{i:03d}.png")
    out = Path(a.out)
    save_gif(imgs, durs, out)
    print(f"wrote {out} · {len(imgs)} frames · {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
