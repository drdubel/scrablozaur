"""Train the grid finder: a small fully-convolutional net that finds the
four corners of the playing grid in a camera frame, for the live scanner.

It draws one heatmap at a quarter of its input's resolution, peaking at each
of the grid's outer corners. The corners are one class: which is which comes
from their arrangement in the frame, and which way up the board is from the
cell reader, which only reads letters upright in one of four rotations.

Training views are random homographies of the exported test photos -- the
grid anywhere in a square frame, at any rotation, scale and perspective --
with lighting, blur and noise varied. The labels are the grid corners the
classical pipeline found, moved by the residual offset the export measured
against the tiles where its mesh was off.

The net is a plain sequence -- 3x3 convolutions, some dilated, and max
pooling -- so that the engine can run it as it runs the cell reader.

From the repository root:

    uv run python board_reader/scripts/train_grid_finder.py            # cross-validate
    uv run python board_reader/scripts/train_grid_finder.py --final    # train on everything, save
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "board_reader" / "test" / "derived" / "scan_dataset"
PHOTOS = ROOT / "board_reader" / "test" / "in"
GRID = 15
SIDE = 256  # the net's input, square
STRIDE = 4  # input pixels per heatmap pixel
PHOTO_SIDE = 768  # photos are kept this big on the GPU


def corners_of(rec) -> np.ndarray:
    """The grid's four outer corners in photo pixels (TL, TR, BR, BL), the
    export's residual applied: its mesh was half a square off on two boards."""
    m = np.asarray(rec["mesh"], np.float64)
    rx, ry = rec.get("residual", [0.0, 0.0])

    def at(gx, gy):
        # Bilinear on the mesh, extrapolating past its edge with its last square.
        i = int(np.clip(np.floor(gy), 0, GRID - 1))
        j = int(np.clip(np.floor(gx), 0, GRID - 1))
        v, u = gy - i, gx - j
        return (
            (1 - u) * (1 - v) * m[i, j]
            + u * (1 - v) * m[i, j + 1]
            + (1 - u) * v * m[i + 1, j]
            + u * v * m[i + 1, j + 1]
        )

    return np.array([at(0 + rx, 0 + ry), at(GRID + rx, 0 + ry), at(GRID + rx, GRID + ry), at(0 + rx, GRID + ry)])


class Photos:
    """Every exported photo, longest side PHOTO_SIDE, padded square, on the
    GPU; and its grid corners in those pixels."""

    def __init__(self, path: Path, device):
        self.meta = json.loads((path / "index.json").read_text(encoding="utf-8"))
        images, corners = [], []
        for e in self.meta:
            rec = json.loads((path / f"board{e['board']}.json").read_text(encoding="utf-8"))
            img = cv2.imread(str(PHOTOS / rec["photo"]))
            scale = PHOTO_SIDE / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            canvas = np.zeros((PHOTO_SIDE, PHOTO_SIDE, 3), np.uint8)
            canvas[: img.shape[0], : img.shape[1]] = img
            images.append(torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
            corners.append(corners_of(rec) * scale)
            e["size"] = img.shape[:2]
        self.images = torch.stack(images).to(device)  # uint8 (n, 3, PHOTO_SIDE, PHOTO_SIDE)
        self.corners = torch.tensor(np.array(corners), dtype=torch.float32, device=device)  # (n, 4, 2) x, y


def homography(src, dst):
    """Batched 3x3 homographies taking 4 points `src` to `dst`: (n, 4, 2) each."""
    n = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack([x, y, one, zero, zero, zero, -u * x, -u * y], -1)
    rows_v = torch.stack([zero, zero, zero, x, y, one, -v * x, -v * y], -1)
    a = torch.cat([rows_u, rows_v], 1)  # (n, 8, 8)
    b = torch.cat([u, v], 1)[..., None]  # (n, 8, 1)
    h = torch.linalg.solve(a.cpu(), b.cpu()).to(src.device)[..., 0]  # MPS lacks a batched solve
    return torch.cat([h, torch.ones(n, 1, device=src.device)], 1).view(n, 3, 3)


def random_views(photos: Photos, ids, gen):
    """One random view per photo in `ids`: the grid placed anywhere in a
    SIDE-pixel square at any rotation, scale and perspective. Returns the
    images (n, 3, SIDE, SIDE) in [0, 1] and the corners (n, 4, 2)."""
    device = photos.images.device
    n = len(ids)

    def u(*shape):
        return torch.rand(*shape, device=device, generator=gen)

    size = SIDE * (0.3 + 0.65 * u(n, 1, 1))
    angle = 2 * math.pi * u(n, 1, 1)
    unit = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], device=device, dtype=torch.float32) / 2
    rot = torch.stack([torch.cat([angle.cos(), -angle.sin()], -1), torch.cat([angle.sin(), angle.cos()], -1)], -2)
    quad = (unit[None] @ rot.squeeze(1).transpose(1, 2)) * size + (
        torch.randn(n, 4, 2, device=device, generator=gen) * 0.05 * size
    )
    # Centre it, then keep every corner in the frame.
    lo, hi = quad.min(1, keepdim=True).values, quad.max(1, keepdim=True).values
    room_lo, room_hi = 6 - lo, SIDE - 6 - hi
    shift = room_lo + (room_hi - room_lo).clamp(min=0) * u(n, 1, 2)
    target = quad + shift
    src = photos.corners[ids]
    h = homography(target, src)  # output pixel -> photo pixel
    ys, xs = torch.meshgrid(
        torch.arange(SIDE, device=device) + 0.5, torch.arange(SIDE, device=device) + 0.5, indexing="ij"
    )
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3) @ h.transpose(1, 2)
    pts = pts[..., :2] / pts[..., 2:].clamp(min=1e-6)
    grid = (pts / PHOTO_SIDE * 2 - 1).view(n, SIDE, SIDE, 2)
    img = F.grid_sample(
        photos.images[ids].float() / 255, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return img, target


def photometric(x, gen):
    n, device = x.shape[0], x.device

    def u(*shape):
        return torch.rand(*shape, device=device, generator=gen)

    gray = x.mean(1, keepdim=True)
    x = torch.lerp(gray, x, 0.6 + 0.8 * u(n, 1, 1, 1))
    x = x * (0.8 + 0.4 * u(n, 3, 1, 1))
    x = (x - 0.5) * (0.6 + 0.8 * u(n, 1, 1, 1)) + 0.5 + (u(n, 1, 1, 1) - 0.5) * 0.4
    x = x.clamp(1e-4, 1) ** (0.7 + 0.6 * u(n, 1, 1, 1))
    blur = u(n) < 0.4
    if blur.any():
        k = torch.ones(3, 1, 3, 3, device=device) / 9
        x = torch.where(blur[:, None, None, None], F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=3), x)
    x = x + torch.randn(x.shape, device=device, generator=gen) * 0.03 * u(n, 1, 1, 1)
    return x.clamp(0, 1)


class GridFinder(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()

        def block(cin, cout, dilation=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        w = width
        self.features = nn.Sequential(
            block(3, 16), nn.MaxPool2d(2),  # 128
            block(16, w), nn.MaxPool2d(2),  # 64: the heatmap's resolution
            block(w, w),
            block(w, w, 2), block(w, w, 4), block(w, w, 8), block(w, w, 16),  # sees the whole frame
        )  # fmt: skip
        self.heat = nn.Conv2d(w, 1, 3, padding=1)
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.heat(self.features((x - self.mean) * 4))[:, 0]  # logits (n, SIDE / 4, SIDE / 4)


def target_heatmap(corners, size):
    """Gaussian peaks (sigma one heatmap pixel) at the corners, max over them."""
    ys, xs = torch.meshgrid(
        torch.arange(size, device=corners.device) + 0.5, torch.arange(size, device=corners.device) + 0.5, indexing="ij"
    )
    c = corners / STRIDE  # (n, 4, 2)
    d2 = (xs[None, None] - c[..., 0, None, None]) ** 2 + (ys[None, None] - c[..., 1, None, None]) ** 2
    return torch.exp(-d2 / 2).amax(1)  # (n, size, size)


def focal_loss(logits, target):
    """CenterNet's: penalties eased near each peak."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target > 0.99
    pos_loss = -((1 - p) ** 2) * torch.log(p) * pos
    neg_loss = -((1 - target) ** 4) * (p**2) * torch.log(1 - p) * ~pos
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def peaks(heat_logits, k=4):
    """The k strongest local maxima of one heatmap, each refined to sub-pixel
    by the centroid of its 3x3 neighbourhood: (k, 2) in input pixels, and
    their scores."""
    h = torch.sigmoid(heat_logits)
    local = F.max_pool2d(h[None, None], 3, 1, 1)[0, 0]
    cand = (h == local) & (h > 0.01)
    ys, xs = cand.nonzero(as_tuple=True)
    order = h[ys, xs].argsort(descending=True)[:k]
    out = []
    for y, x in zip(ys[order].tolist(), xs[order].tolist()):
        y0, y1, x0, x1 = max(0, y - 1), min(h.shape[0], y + 2), max(0, x - 1), min(h.shape[1], x + 2)
        w = h[y0:y1, x0:x1]
        gy, gx = torch.meshgrid(
            torch.arange(y0, y1, device=h.device), torch.arange(x0, x1, device=h.device), indexing="ij"
        )
        cy, cx = (w * gy).sum() / w.sum(), (w * gx).sum() / w.sum()
        out.append([(cx.item() + 0.5) * STRIDE, (cy.item() + 0.5) * STRIDE])
    return np.array(out), h[ys[order], xs[order]].cpu().numpy()


def train(photos: Photos, ids, steps, device, seed, log=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = GridFinder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    ids = torch.tensor(ids, device=device)
    model.train()
    for step in range(steps):
        pick = ids[torch.randint(len(ids), (24,), device=device, generator=gen)]
        img, corners = random_views(photos, pick, gen)
        img = photometric(img, gen)
        logits = model(img)
        loss = focal_loss(logits, target_heatmap(corners, logits.shape[-1]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % 200 == 0:
            log(f"    step {step + 1}/{steps} loss {loss.item():.3f}")
    return model.eval()


def export(model: GridFinder, path: Path) -> None:
    """Write the SCRBGRD1 file the engine's grid finder loads
    (src/scan/grid.rs): the input side and heatmap stride, the input
    normalisation, then the layers -- batch norm folded into each
    convolution, the heatmap's convolution last, with no ReLU."""
    out = bytearray(b"SCRBGRD1")
    out += struct.pack("<IIff", SIDE, STRIDE, 4 / 255, -2.0)
    layers = []

    def conv(c, scale=None, shift=None, relu=True):
        w, b = c.weight, c.bias if c.bias is not None else torch.zeros(c.out_channels)
        if scale is not None:
            w, b = w * scale[:, None, None, None], shift
        return (
            struct.pack("<IIIII", 4, c.in_channels, c.out_channels, c.dilation[0], int(relu))
            + w.permute(2, 3, 1, 0).contiguous().cpu().numpy().astype("<f4").tobytes()  # [ky][kx][cin][cout]
            + b.cpu().numpy().astype("<f4").tobytes()
        )

    with torch.no_grad():
        for m in model.features:
            if isinstance(m, nn.MaxPool2d):
                layers.append(struct.pack("<I", 1))
                continue
            c, bn = m[0], m[1]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            layers.append(conv(c, scale, bn.bias - bn.running_mean * scale))
        layers.append(conv(model.heat, relu=False))
    out += struct.pack("<I", len(layers)) + b"".join(layers)
    path.write_bytes(bytes(out))


def order(quad):
    """Clockwise, starting from the corner nearest the image's top left."""
    c = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    return np.roll(quad, -int(np.argmin(quad.sum(1))), 0)


def convex(quad) -> bool:
    def turn(a, b):
        return a[0] * b[1] - a[1] * b[0]

    turns = [turn(quad[(i + 1) % 4] - quad[i], quad[(i + 2) % 4] - quad[(i + 1) % 4]) for i in range(4)]
    return all(t > 0 for t in turns) or all(t < 0 for t in turns)


def best_quad(found, scores):
    """Of the strongest peaks, the four that make the likeliest grid: convex,
    no side under a third of the longest, the highest joint score. The
    strongest four alone sometimes include a table edge's corner."""
    best = None
    for combo in itertools.combinations(range(len(found)), 4):
        quad = order(found[list(combo)])
        sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if not convex(quad) or min(sides) < max(sides) / 3:
            continue
        score = float(np.log(scores[list(combo)]).sum())
        if best is None or score > best[0]:
            best = (score, quad)
    return None if best is None else best[1]


@torch.no_grad()
def find(model, image):
    """The grid's corners in an image (3, h, w) in [0, 1]: once on the whole
    image, then again on the region found, scaled up so a square spans more
    pixels. None if no grid-like quadrilateral shows."""
    device = image.device

    def once(region):
        scale = SIDE / max(region.shape[1:])
        size = (round(region.shape[1] * scale), round(region.shape[2] * scale))
        # MPS cannot area-resample to sizes that do not divide evenly.
        small = F.interpolate(region[None].cpu(), size=size, mode="area")[0].to(device)
        canvas = torch.zeros(1, 3, SIDE, SIDE, device=device)
        canvas[..., : size[0], : size[1]] = small
        found, scores = peaks(model(canvas)[0], k=8)
        quad = best_quad(found, scores) if len(found) >= 4 else None
        return None if quad is None else quad / scale

    h, w = image.shape[1:]
    first = once(image)
    if first is None:
        return None
    pad = 0.12 * (first.max(0) - first.min(0))
    x0, y0 = np.maximum(first.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(first.max(0) + pad, [w, h]).astype(int)
    second = once(image[:, y0:y1, x0:x1])
    return first if second is None else second + [x0, y0]


def evaluate(model, photos: Photos, ids):
    """Each held-out photo as the camera saw it: the error of each true
    corner, in squares of that photo's grid, against the nearest corner found."""
    errors = []
    for i in ids:
        h, w = photos.meta[i]["size"]
        quad = find(model, photos.images[i, :, :h, :w].float() / 255)
        truth = photos.corners[i].cpu().numpy()
        square = np.linalg.norm(truth[1] - truth[0]) / GRID
        if quad is None:
            errors.extend([float(GRID)] * 4)
            continue
        errors.extend((np.linalg.norm(quad[:, None] - truth[None], axis=-1).min(0) / square).tolist())
    return np.array(errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "board_reader" / "src" / "models" / "pl" / "grid_finder.pt")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    photos = Photos(args.dataset, device)
    print(f"{len(photos.meta)} photos on {device}", flush=True)
    if args.final:
        model = train(photos, list(range(len(photos.meta))), args.steps, device, args.seed, log=print).cpu()
        torch.save({"state": model.state_dict(), "side": SIDE, "stride": STRIDE}, args.out)
        export(model, args.out.with_suffix(".bin"))
        print(f"saved {args.out} and its .bin")
        return
    layouts = collections.Counter(e["layout"] for e in photos.meta)
    folds: list[list[int]] = [[] for _ in range(args.folds)]
    load = [0] * args.folds
    for layout, n in sorted(layouts.items(), key=lambda kv: -kv[1]):
        f = load.index(min(load))
        folds[f].append(layout)
        load[f] += n
    errors, by_tier = [], collections.defaultdict(list)
    start = time.time()
    for f, held in enumerate(folds):
        test = [i for i, e in enumerate(photos.meta) if e["layout"] in held]
        model = train(photos, [i for i in range(len(photos.meta)) if i not in test], args.steps, device, args.seed + f)
        for i in test:
            e = evaluate(model, photos, [i])
            errors.extend(e.tolist())
            by_tier[photos.meta[i]["difficulty"]].extend(e.tolist())
        fold = np.array(errors[-4 * len(test) :])
        print(
            f"fold {f + 1}/{args.folds}: {len(test)} photos held out, corner error median {np.median(fold):.3f} "
            f"p90 {np.percentile(fold, 90):.3f} ({time.time() - start:.0f} s)",
            flush=True,
        )
    e = np.array(errors)
    print(
        f"corner error, in squares, over {len(e)} corners: median {np.median(e):.3f} "
        f"p90 {np.percentile(e, 90):.3f} max {e.max():.2f}; over half a square: {(e > 0.5).mean():.1%}"
    )
    for tier in "emh":
        if by_tier[tier]:
            t = np.array(by_tier[tier])
            print(f"  {tier}: median {np.median(t):.3f} p90 {np.percentile(t, 90):.3f} max {t.max():.2f}")


if __name__ == "__main__":
    main()
