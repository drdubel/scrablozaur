"""Train the cell reader: a small CNN that reads every square of a
straightened board -- empty, a blank, or which letter -- from the pixels.

It is the live scanner's reader, replacing the classical tile detector and
letter classifier, which need OpenCV and a glyph normaliser the phone does
not have. Its outputs are what the scan decoder takes: from one softmax over
(empty, blank, every letter), p_tile = 1 - P(empty), p_blank = P(blank) /
p_tile, and each letter's share of the rest.

Data: the boards `export_scan_dataset.py` straightened, cut into one crop per
square with half a square of context around it -- tiles stand proud of the
board, so a photo at an angle shifts them toward a neighbour. Crops are cut
on the GPU with a random affine jitter (imperfect corners, parallax), and
the lighting, blur, noise and glare of real photos are varied. No test photo
shows a blank, so some tiles have their ink painted out and become blanks.

Evaluation holds out whole layouts (the photos are a few arrangements shot
many times) and compares, square by square, with what the classical pipeline
read on the same boards.

From the repository root:

    uv run python board_reader/scripts/train_cell_reader.py            # cross-validate
    uv run python board_reader/scripts/train_cell_reader.py --final    # train on everything, save
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "board_reader" / "test" / "derived" / "scan_dataset"
GRID = 15
EXPORT_CELL, EXPORT_MARGIN = 64, 32  # export_scan_dataset.py's geometry
CELL = 32  # pixels per square the reader sees
CROP = 48  # crop side: one and a half squares
EMPTY, BLANK = 0, 1
# Every letter is four classes, one per quarter turn clockwise that would
# stand the tile upright: class 2 + TURNS * letter + turns. A board can be
# seen from any side, and which way up it is shows only in its letters --
# so one read of the board as it lies says both what each tile is and,
# summed over the tiles, how far to turn the board.
TURNS = 4


def alphabet(code: str = "pl") -> str:
    return json.loads((ROOT / "languages" / f"{code}.json").read_text(encoding="utf-8"))["alphabet"]


class Boards:
    """Every exported board, downscaled so a square is CELL pixels, with a
    margin of a square on every side for the context crops; and its labels."""

    def __init__(self, path: Path, letters: str, device):
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        self.meta = index
        cls = {ch: 2 + i for i, ch in enumerate(letters)}
        rng = random.Random(0)
        images, labels, self.blanked = [], [], []
        variants, variant_labels = [], []
        for entry in index:
            img = cv2.imread(str(path / f"board{entry['board']}.jpg"))
            rec = json.loads((path / f"board{entry['board']}.json").read_text(encoding="utf-8"))
            # '?' marks a blank tile in the ground truth (board 90 has one).
            label = [
                EMPTY if ch == "-" else BLANK if ch == "?" else cls[ch.lower()] for row in rec["truth"] for ch in row
            ]
            images.append(self.scaled(img))
            labels.append(label)
            # The same board with some of its tiles made blanks.
            variant, variant_label = img.copy(), list(label)
            for cell in range(GRID * GRID):
                if label[cell] >= 2 and rng.random() < 0.12 and paint_out(variant, *divmod(cell, GRID)):
                    variant_label[cell] = BLANK
            variants.append(self.scaled(variant))
            variant_labels.append(variant_label)
        # Board i's variant is board i + len(index): never train on one while holding out the other.
        self.images = torch.stack(images + variants).to(device)  # uint8, (2 * boards, 3, size, size)
        self.labels = torch.tensor(labels + variant_labels, device=device)  # (2 * boards, 225)
        self.classical = [json.loads((path / f"board{e['board']}.json").read_text(encoding="utf-8")) for e in index]

    @staticmethod
    def scaled(img):
        """The export has a half-square margin; widen it to a whole one,
        repeating the edge, then scale squares to CELL pixels. RGB, CHW."""
        pad = EXPORT_CELL - EXPORT_MARGIN
        size = (GRID + 2) * CELL
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)


def crops(images, board, cell, jitter: bool, gen=None):
    """CROP x CROP crops centred on squares `cell` (0..224) of `board`, as
    floats in [0, 1]. With `jitter`, each is shifted, scaled and turned a
    little, as imperfect corners and tile parallax would."""
    n = len(board)
    device = images.device
    size = images.shape[-1]
    r, c = (cell // GRID).float(), (cell % GRID).float()
    cy, cx = (r + 1.5) * CELL, (c + 1.5) * CELL  # centre, in board pixels
    scale = torch.ones(n, device=device)
    angle = torch.zeros(n, device=device)
    if jitter:
        u = lambda: torch.rand(n, device=device, generator=gen) * 2 - 1  # noqa: E731
        cx = cx + u() * 0.18 * CELL
        cy = cy + u() * 0.18 * CELL
        scale = 1 + u() * 0.1
        angle = u() * math.radians(4)
    half = CROP / 2 * scale
    cos, sin = torch.cos(angle), torch.sin(angle)
    # Affine from crop coordinates in [-1, 1] to board coordinates in [-1, 1].
    k = 2 / size
    theta = torch.stack(
        [
            torch.stack([cos * half * k, -sin * half * k, cx * k - 1], 1),
            torch.stack([sin * half * k, cos * half * k, cy * k - 1], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, (n, 3, CROP, CROP), align_corners=False)
    src = images[board].float() / 255
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)


def photometric(x, gen=None):
    """Lighting, white balance, blur, noise and glare, varied per crop."""
    n, device = x.shape[0], x.device

    def u(*shape):
        return torch.rand(*shape, device=device, generator=gen)

    gray = x.mean(1, keepdim=True)
    x = torch.lerp(gray, x, 0.6 + 0.8 * u(n, 1, 1, 1))  # saturation
    x = x * (0.85 + 0.3 * u(n, 3, 1, 1))  # white balance
    x = (x - 0.5) * (0.7 + 0.6 * u(n, 1, 1, 1)) + 0.5 + (u(n, 1, 1, 1) - 0.5) * 0.3  # contrast, brightness
    x = x.clamp(1e-4, 1) ** (0.7 + 0.6 * u(n, 1, 1, 1))  # gamma
    # Glare: a soft bright blob on some crops.
    glare = u(n) < 0.15
    if glare.any():
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, CROP, device=device), torch.linspace(-1, 1, CROP, device=device), indexing="ij"
        )
        gy, gx, rad = (u(n, 1, 1) * 2 - 1), (u(n, 1, 1) * 2 - 1), 0.3 + 0.7 * u(n, 1, 1)
        blob = torch.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / rad**2) * (0.3 + 0.6 * u(n, 1, 1))
        x = torch.where(glare[:, None, None, None], x + blob[:, None] * (1 - x), x)
    # Blur: a box blur of random width on some crops.
    blur = u(n) < 0.3
    if blur.any():
        k = 3
        weight = torch.ones(3, 1, k, k, device=device) / (k * k)
        x = torch.where(
            blur[:, None, None, None], F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=3), x
        )
    x = x + torch.randn(x.shape, device=device, generator=gen) * (0.03 * u(n, 1, 1, 1))
    return x.clamp(0, 1)


def paint_out(img, r, c):
    """Turn the tile on square (r, c) of a straightened board (export
    geometry, BGR) into a blank, in place: inpaint its glyph and digit. Only
    one test photo shows a real blank, so the reader learns them from these.
    False if no tile face could be found there.

    The tile's box is the run of rows, and of columns, through the square's
    middle that are mostly face -- bright and unsaturated, the board being
    teal, red and blue -- once the strokes are closed over. A run longer than
    a tile has met a neighbour, so it is cut to one tile around the middle."""
    cell, margin = EXPORT_CELL, EXPORT_MARGIN
    x0, y0 = margin + c * cell, margin + r * cell
    pad = int(cell * 0.4)  # tiles stand proud: parallax shifts them
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(img.shape[1], x0 + cell + pad), min(img.shape[0], y0 + cell + pad)
    region = img[Y0:Y1, X0:X1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    q = cell // 4
    cy, cx = y0 - Y0 + cell // 2, x0 - X0 + cell // 2
    bright = np.percentile(val[cy - q : cy + q, cx - q : cx + q], 80)
    face = ((val > bright - 45) & (sat < 110)).astype(np.uint8)
    solid = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((q, q), np.uint8))

    def run(profile, centre):
        ok = profile > 0.45
        if not ok[centre]:
            return None
        lo = hi = centre
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        size = int(cell * 0.92)
        if hi - lo + 1 > size + 4:
            lo, hi = max(lo, centre - size // 2), min(hi, centre + size // 2)
        return lo + 2, hi - 2

    rows = run(solid[:, cx - q : cx + q].mean(1), cy)
    cols = run(solid[cy - q : cy + q, :].mean(0), cx)
    if rows is None or cols is None:
        return False
    box = np.zeros(face.shape, np.uint8)
    box[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1] = 1
    ink = box & (val < bright - 28).astype(np.uint8)
    if ink.sum() < 20:
        return False
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8)) & box
    img[Y0:Y1, X0:X1] = cv2.inpaint(region, ink, 4, cv2.INPAINT_TELEA)
    return True


class CellReader(nn.Module):
    def __init__(self, classes: int, width: int = 16):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
            )

        w = width
        self.features = nn.Sequential(
            block(3, w), block(w, w), nn.MaxPool2d(2),  # 24
            block(w, 2 * w), block(2 * w, 2 * w), nn.MaxPool2d(2),  # 12
            block(2 * w, 4 * w), block(4 * w, 4 * w), nn.MaxPool2d(2),  # 6
            block(4 * w, 8 * w),
        )  # fmt: skip
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(8 * w, classes))
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.head(self.features((x - self.mean) * 4))


def train(boards: Boards, train_ids, epochs, device, seed, log=None):
    """A reader trained on boards `train_ids` and their blank variants."""
    train_ids = list(train_ids) + [i + len(boards.meta) for i in train_ids]
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    classes = 2 + TURNS * len(alphabet())
    model = CellReader(classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    labels = boards.labels[train_ids]
    tile_cells = (labels != EMPTY).nonzero()  # (k, 2): board position in train_ids, cell
    # Each letter's tiles weighted by 1 / sqrt(its count): 'ś' and 'ż', one
    # tile each in a bag, still come up often enough to tell from 's' and 'z'.
    tile_labels = labels[tile_cells[:, 0], tile_cells[:, 1]]
    tile_weight = torch.bincount(tile_labels).float().clamp(min=1)[tile_labels].rsqrt()
    all_cells = torch.stack(
        torch.meshgrid(
            torch.arange(len(train_ids), device=device), torch.arange(GRID * GRID, device=device), indexing="ij"
        ),
        -1,
    ).view(-1, 2)
    steps_per_epoch = max(1, len(all_cells) // 256)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=total, pct_start=0.15)
    ids = torch.tensor(train_ids, device=device)
    model.train()
    for step in range(total):
        # Half the batch tiles, rare letters more often; half any square:
        # tiles are 1 square in 6.
        t = tile_cells[torch.multinomial(tile_weight, 128, replacement=True, generator=gen)]
        a = all_cells[torch.randint(len(all_cells), (128,), device=device, generator=gen)]
        pick = torch.cat([t, a])
        y = labels[pick[:, 0], pick[:, 1]]
        x = crops(boards.images, ids[pick[:, 0]], pick[:, 1], jitter=True, gen=gen)
        # Each crop turned a random number of quarter turns counter-clockwise,
        # as a board seen from another side shows it: that many clockwise
        # turns stand the tile upright again.
        turns = torch.randint(0, TURNS, (len(y),), device=device, generator=gen)
        for k in range(1, TURNS):
            if (turns == k).any():
                x[turns == k] = torch.rot90(x[turns == k], k, dims=(2, 3))
        y = torch.where(y >= 2, 2 + (y - 2) * TURNS + turns, y)
        loss = F.cross_entropy(model(photometric(x, gen)), y, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % steps_per_epoch == 0:
            log(f"    epoch {(step + 1) // steps_per_epoch:3d}/{epochs} loss {loss.item():.3f}")
    return model


@torch.no_grad()
def predict(model, boards: Boards, ids, device, turn: int = 0):
    """Logits for every square of boards `ids`, each crop turned `turn`
    quarter turns counter-clockwise: (len(ids), 225, classes). Turning every
    crop about its centre is turning the whole board."""
    model.eval()
    out = []
    cells = torch.arange(GRID * GRID, device=device)
    for b in ids:
        x = crops(boards.images, torch.full_like(cells, b), cells, jitter=False)
        out.append(model(torch.rot90(x, turn, dims=(2, 3)) if turn else x))
    return torch.stack(out)


def upright(logits):
    """Logits over (empty, blank, every letter) for a board known to be
    upright: the letters' no-turn classes."""
    return torch.cat([logits[..., :2], logits[..., 2::TURNS]], -1)


def vote(logits) -> int:
    """How many quarter turns clockwise stand a board upright, from its
    logits as it lies: the turn whose letter classes hold the most
    probability on the squares it takes for lettered tiles."""
    p = F.softmax(logits, -1)  # (225, classes)
    tiles = (p[:, EMPTY] < 0.5) & (p[:, BLANK] < 0.5)
    if not tiles.any():
        return 0
    per_turn = p[tiles][:, 2:].reshape(-1, p.shape[-1] // TURNS, TURNS).sum(1)  # (tiles, turns)
    return int(per_turn.clamp(min=1e-9).log().sum(0).argmax())


def export(model: CellReader, letters: str, temperature: float, path: Path) -> None:
    """Write the SCRBCNN2 file the engine's reader loads (src/scan/reader.rs):
    the alphabet, the geometry, the input normalisation, the temperature and
    the turns each letter's classes span, then every layer, batch norm folded
    into its convolution."""
    out = bytearray(b"SCRBCNN2")
    out += struct.pack("<I", len(letters)) + b"".join(struct.pack("<I", ord(ch)) for ch in letters)
    # The model normalises 0-1 input as (x - 0.5) * 4; the engine gets 0-255.
    out += struct.pack("<IIfffI", CELL, CROP, 4 / 255, -2.0, temperature, TURNS)
    layers = []
    with torch.no_grad():
        for m in model.features:
            if isinstance(m, nn.MaxPool2d):
                layers.append(struct.pack("<I", 1))
                continue
            conv, bn = m[0], m[1]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = (conv.weight * scale[:, None, None, None]).permute(2, 3, 1, 0)  # [ky][kx][cin][cout]
            b = bn.bias - bn.running_mean * scale
            layers.append(
                struct.pack("<III", 0, conv.in_channels, conv.out_channels)
                + w.contiguous().cpu().numpy().astype("<f4").tobytes()
                + b.cpu().numpy().astype("<f4").tobytes()
            )
        layers.append(struct.pack("<I", 2))
        linear = model.head[-1]
        layers.append(
            struct.pack("<III", 3, linear.in_features, linear.out_features)
            + linear.weight.cpu().numpy().astype("<f4").tobytes()
            + linear.bias.cpu().numpy().astype("<f4").tobytes()
        )
    out += struct.pack("<I", len(layers)) + b"".join(layers)
    path.write_bytes(bytes(out))


def fit_temperature(logits, labels):
    """The softmax temperature that makes held-out probabilities honest."""
    best = (float("inf"), 1.0)
    for t in np.arange(0.5, 3.01, 0.05):
        nll = F.cross_entropy(logits / t, labels).item()
        best = min(best, (nll, float(t)))
    return best[1]


def report(boards: Boards, logits, letters: str):
    """Learned against classical, per difficulty, on the same squares. The
    app's camera scans (tier "c") have no classical readings."""
    pred = logits.argmax(-1).cpu()
    labels = boards.labels[: len(boards.meta)].cpu()
    stats = collections.defaultdict(collections.Counter)
    for i, entry in enumerate(boards.meta):
        rec = boards.classical[i]
        s = stats[entry["difficulty"]]
        for cell in range(GRID * GRID):
            r, c = divmod(cell, GRID)
            truth = labels[i, cell].item()
            p = pred[i, cell].item()
            # classical
            is_tile, _ = rec["tiles"][f"{r},{c}"] if "tiles" in rec else (False, 0)
            reading = rec["readings"].get(f"{r},{c}") if "tiles" in rec else None
            cl = (
                EMPTY
                if not is_tile
                else (
                    2 + letters.index(reading[0].lower())
                    if reading and reading[0] and reading[0].lower() in letters
                    else -1
                )
            )
            for name, guess in (("learned", p), ("classical", cl)) if "tiles" in rec else (("learned", p),):
                g_tile, t_tile = guess != EMPTY, truth != EMPTY
                s[name + " tp"] += g_tile and t_tile
                s[name + " fp"] += g_tile and not t_tile
                s[name + " fn"] += t_tile and not g_tile
                if t_tile and g_tile:
                    s[name + " letters"] += 1
                    s[name + " letters ok"] += guess == truth
                s[name + " cells ok"] += guess == truth
                s[name + " cells"] += 1
    lines = []
    for tier in "emhc":
        s = stats[tier]
        if not s:
            continue
        for name in ("classical", "learned") if s["classical cells"] else ("learned",):
            tp, fp, fn = s[name + " tp"], s[name + " fp"], s[name + " fn"]
            lines.append(
                f"  {tier} {name:9}: precision {tp / max(1, tp + fp):.4f} recall {tp / max(1, tp + fn):.4f} "
                f"letters {s[name + ' letters ok'] / max(1, s[name + ' letters']):.4f} "
                f"cells {s[name + ' cells ok'] / max(1, s[name + ' cells']):.4f} "
                f"(wrong squares {s[name + ' cells'] - s[name + ' cells ok']})"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--final", action="store_true", help="train on every board and save the model")
    ap.add_argument("--out", type=Path, default=ROOT / "board_reader" / "src" / "models" / "pl" / "cell_reader.pt")
    ap.add_argument("--dump", type=Path, help="write the held-out probabilities here (cross-validation)")
    ap.add_argument(
        "--save-folds",
        type=Path,
        help="cross-validation: export each fold's reader here (fold<k>.bin, folds.json naming the layouts it held"
        " out), so the engine's whole pipeline can be judged on boards its reader never saw",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        help="with --final: the softmax temperature cross-validation fitted (it prints it, and --dump keeps it)",
    )
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    letters = alphabet()
    boards = Boards(args.dataset, letters, device)
    print(f"{len(boards.meta)} boards, {len({e['layout'] for e in boards.meta})} layouts, on {device}", flush=True)

    if args.final:
        if args.temperature is None:
            sys.exit("--final needs --temperature: run the cross-validation first, it fits one")
        start = time.time()
        model = train(boards, list(range(len(boards.meta))), args.epochs, device, args.seed, log=print)
        model = model.cpu().eval()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": model.state_dict(),
                "alphabet": letters,
                "cell": CELL,
                "crop": CROP,
                "temperature": args.temperature,
                "turns": TURNS,
            },
            args.out,
        )
        export(model, letters, args.temperature, args.out.with_suffix(".bin"))
        print(f"saved {args.out} and its .bin after {time.time() - start:.0f} s")
        return

    # Whole layouts per fold, spread so folds have similar numbers of boards.
    layouts = collections.Counter(e["layout"] for e in boards.meta)
    folds: list[list[int]] = [[] for _ in range(args.folds)]
    load = [0] * args.folds
    for layout, n in sorted(layouts.items(), key=lambda kv: -kv[1]):
        f = load.index(min(load))
        folds[f].append(layout)
        load[f] += n
    n = len(boards.meta)
    # Held-out logits for every board, and for its blank variant, upright.
    logits = torch.zeros(2 * n, GRID * GRID, 2 + TURNS * len(letters), device=device)
    turned_right, turned_all = 0, 0
    start = time.time()
    for f, held in enumerate(folds):
        test = [i for i, e in enumerate(boards.meta) if e["layout"] in held]
        train_ids = [i for i in range(len(boards.meta)) if i not in test]
        model = train(boards, train_ids, args.epochs, device, args.seed + f)
        if args.save_folds:
            args.save_folds.mkdir(parents=True, exist_ok=True)
            # The temperature only calibrates; a fold's own is not fitted yet.
            export(model.cpu().eval(), letters, 1.0, args.save_folds / f"fold{f}.bin")
            model.to(device)
            (args.save_folds / "folds.json").write_text(json.dumps({"folds": folds}))
        both = test + [i + n for i in test]
        logits[both] = predict(model, boards, both, device)
        # Every held-out board turned every way: is the way back found?
        for k in range(TURNS):
            for one in predict(model, boards, test, device, turn=k):
                turned_right += vote(one) == k
                turned_all += 1
        print(
            f"fold {f + 1}/{args.folds}: layouts {held}, {len(test)} boards held out ({time.time() - start:.0f} s)",
            flush=True,
        )
    # Letters are judged with the board's orientation known: the no-turn
    # classes. The temperature is fitted over all of them.
    labels = boards.labels[:n].reshape(-1)
    full = torch.where(labels >= 2, 2 + (labels - 2) * TURNS, labels)
    t = fit_temperature(logits[:n].reshape(-1, logits.shape[-1]), full)
    print(f"held-out temperature {t:.2f}; orientation found for {turned_right} of {turned_all} turned boards")
    print(report(boards, upright(logits[:n]), letters))
    if args.dump:
        # Probabilities over (empty, blank, every letter) as an upright board
        # gives them -- what the decoder's evaluation draws from.
        probs = F.softmax(upright(logits) / t, -1)
        probs = (probs / probs.sum(-1, keepdim=True)).cpu().numpy()
        variant_blanks = (boards.labels[n:] == BLANK).cpu().numpy()
        args.dump.write_text(
            json.dumps(
                {
                    "alphabet": letters,
                    "classes": "empty, blank, then the alphabet",
                    "temperature": t,
                    "boards": [e["board"] for e in boards.meta],
                    "probs": np.round(probs[:n], 5).tolist(),
                    # What the reader made of the tiles made blanks, held out
                    # like the rest: the only blanks there are to draw from.
                    "blank_probs": np.round(probs[n:][variant_blanks], 5).tolist(),
                }
            ),
            encoding="utf-8",
        )
        print(f"held-out probabilities written to {args.dump}")


if __name__ == "__main__":
    main()
