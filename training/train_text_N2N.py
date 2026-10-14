#!/usr/bin/env python3
"""Noise2Noise removal of burned-in TEXT annotations.

Same technique, architecture, loss and normalisation as `train_twin_N2N.py`
(calipers). The only thing that changes is what gets pasted: text stamps
harvested by `build_text_stamps.py` instead of caliper shapes. A separate
model is trained and shipped separately; the caliper weights are untouched.

THE NOISE2NOISE TRICK, AS IT APPLIES HERE
  Take a frame with no annotation on it. Paste a random set of stamps to make
  the INPUT. Paste a *different*, independently drawn set to make the TARGET.
  Ask the network to map input to target under L1.

  At a pixel the input stamp covers, the target is the bare frame most of the
  time -- the second draw rarely lands in the same place -- so the L1-optimal
  output is the median over target draws, which is the tissue underneath. The
  network learns to erase a stamp and paint back plausible tissue, and it
  never needs a clean/annotated pair of the same real frame, which does not
  exist for text the way caliper twins exist for calipers.

  Anything identical on both sides is preserved, because the loss punishes
  changing it. Scanner chrome is identical on both sides. That is the whole
  reason the sidebar survives: stamps are pasted only inside the ultrasound
  region, and crops are taken from the WHOLE frame, so the network sees
  chrome-with-no-edit and tissue-with-edit side by side and learns the
  difference.

VALIDATION
  `val_n2n` is the training loss on held-out frames: honest, but its floor is
  the variance of the second stamp draw, not zero, so it is hard to read.
  `val_clean` is L1 against the untouched crop -- what we actually want, since
  the clean frame IS the ground truth here (it was clean before we pasted).
  Checkpoints are selected on `val_clean`.

  python training/train_text_N2N.py --epochs 40
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch  # MUST precede numpy/PIL on Windows -- MKL/OpenMP DLL clash
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.train_twin_N2N import N2N_Original_Used_UNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data/text_inpaint"
MODELS = ROOT / "src/ML_processing/models"
FULL = Path("D:/DATA/CADBUSI/database_images")


class TextStampDataset(Dataset):
    """One item = (input, target, clean) crops, uint8 [1,S,S]. See `norm`.

    `clean` is only used by validation; training never looks at it.
    """

    def __init__(self, frames_csv, stamps_dir, manifest_csv, full_dir,
                 crop=512, val=False, max_stamps=4, hit_prob=0.75):
        f = pd.read_csv(frames_csv)
        self.rows = f[f.val == int(val)].reset_index(drop=True)
        self.full_dir = Path(full_dir)
        self.crop = crop
        self.val = val
        self.max_stamps = max_stamps
        self.hit_prob = hit_prob

        man = pd.read_csv(manifest_csv)
        self.stamps = [np.asarray(Image.open(Path(stamps_dir) / n).convert("RGBA"))
                       for n in man.file]
        if not self.stamps:
            raise SystemExit("no stamps -- run training/build_text_stamps.py first")

    def __len__(self):
        return len(self.rows)

    def _paste(self, img, rect, rng):
        """Composite 1..max_stamps stamps inside `rect`. Returns placed boxes."""
        h, w = img.shape
        rx, ry, rw, rh = rect
        placed = []
        for _ in range(rng.randint(1, self.max_stamps)):
            st = self.stamps[rng.randrange(len(self.stamps))]
            s = rng.uniform(0.7, 1.4)
            sh, sw = max(4, int(st.shape[0] * s)), max(8, int(st.shape[1] * s))
            if sh >= rh or sw >= rw:
                continue
            # NEAREST keeps the strokes hard-edged, as burned-in text is.
            st = np.asarray(Image.fromarray(st, "RGBA")
                            .resize((sw, sh), Image.NEAREST))
            a = (st[:, :, 3].astype(np.float32) / 255.0) ** rng.uniform(0.7, 1.3)
            v = st[:, :, 0].astype(np.float32) * rng.uniform(0.75, 1.10)

            x = rng.randint(rx, max(rx, rx + rw - sw))
            y = rng.randint(ry, max(ry, ry + rh - sh))
            x, y = min(x, w - sw), min(y, h - sh)
            if x < 0 or y < 0:
                continue
            win = img[y:y + sh, x:x + sw].astype(np.float32)
            img[y:y + sh, x:x + sw] = np.clip(win * (1 - a) + v * a, 0, 255
                                              ).astype(np.uint8)
            placed.append((x, y, sw, sh))
        return placed

    def _window(self, shape, placed, rng):
        """Crop box, biased to contain a stamp so most samples carry signal."""
        h, w = shape
        c = self.crop
        if placed and rng.random() < self.hit_prob:
            x, y, sw, sh = placed[rng.randrange(len(placed))]
            cx = x + sw // 2 + rng.randint(-c // 4, c // 4)
            cy = y + sh // 2 + rng.randint(-c // 4, c // 4)
            x0, y0 = cx - c // 2, cy - c // 2
        else:
            x0 = rng.randint(0, max(0, w - c))
            y0 = rng.randint(0, max(0, h - c))
        return (int(np.clip(x0, 0, max(0, w - c))),
                int(np.clip(y0, 0, max(0, h - c))))

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        # Validation must be the same pair every epoch or the metric wanders.
        rng = random.Random(i) if self.val else random.Random()

        img = np.asarray(Image.open(self.full_dir / f"{r.stem}.png").convert("L"))
        c = self.crop
        if img.shape[0] < c or img.shape[1] < c:
            img = np.pad(img, ((0, max(0, c - img.shape[0])),
                               (0, max(0, c - img.shape[1]))), mode="constant")
        rect = (int(r.us_x), int(r.us_y),
                min(int(r.us_w), img.shape[1] - int(r.us_x)),
                min(int(r.us_h), img.shape[0] - int(r.us_y)))

        a_img, b_img = img.copy(), img.copy()
        placed = self._paste(a_img, rect, rng)     # input stamps
        self._paste(b_img, rect, rng)              # target stamps, independent
        x0, y0 = self._window(img.shape, placed, rng)

        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        a, b, clean = a_img[sl], b_img[sl], img[sl]

        if not self.val:                            # same flip on all three
            if rng.random() < 0.5:
                a, b, clean = a[:, ::-1], b[:, ::-1], clean[:, ::-1]
            if rng.random() < 0.5:
                a, b, clean = a[::-1], b[::-1], clean[::-1]

        # uint8 across the worker boundary, normalised on the GPU by `norm()`.
        # float32 here costs 4x the shared memory and Windows backs those
        # mappings with the page file -- 8 workers of float crops exhausts the
        # commit limit and the run dies with "Couldn't open shared file
        # mapping ... error code 1455".
        def t(z):
            return torch.from_numpy(np.ascontiguousarray(z)).unsqueeze(0)
        return t(a), t(b), t(clean)


def norm(x, dev):
    """uint8 [B,1,H,W] on the worker -> [-1,1] float on the device.

    Same transform as `Normalize(mean=0.5, std=0.5)` in train_twin_N2N.py,
    just applied after the transfer instead of before it.
    """
    return x.to(dev, non_blocking=True).float().div_(127.5).sub_(1.0)


def run(args) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / args.out
    samples = DATA / "train_samples"
    samples.mkdir(exist_ok=True)

    kw = dict(frames_csv=DATA / "frames.csv", stamps_dir=DATA / "stamps",
              manifest_csv=DATA / "stamps_manifest.csv", full_dir=args.full_dir,
              crop=args.crop, max_stamps=args.max_stamps)
    tr = TextStampDataset(val=False, **kw)
    va = TextStampDataset(val=True, **kw)
    print(f"train {len(tr):,} frames | val {len(va):,} | "
          f"{len(tr.stamps):,} stamps | crop {args.crop} | {dev}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    model = N2N_Original_Used_UNet(in_channels=1, out_channels=1).to(dev)
    if args.resume and Path(args.resume).exists():
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"resumed from {args.resume}")

    crit = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")

    best = float("inf")
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for a, b, _ in tqdm(tl, desc=f"epoch {ep}/{args.epochs}", leave=False):
            a, b = norm(a, dev), norm(b, dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(model(a), b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * a.size(0)
            n += a.size(0)
        train = tot / max(n, 1)

        model.eval()
        vn, vc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for a, b, clean in vl:
                a, b, clean = norm(a, dev), norm(b, dev), norm(clean, dev)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    o = model(a)
                vn += crit(o, b).item() * a.size(0)
                vc += crit(o.float(), clean).item() * a.size(0)
                m += a.size(0)
        vn, vc = vn / max(m, 1), vc / max(m, 1)
        sched.step()
        hist.append(dict(epoch=ep, train=train, val_n2n=vn, val_clean=vc,
                         lr=opt.param_groups[0]["lr"],
                         minutes=round((time.time() - t0) / 60, 2)))
        pd.DataFrame(hist).to_csv(DATA / "train_log.csv", index=False)

        flag = ""
        if vc < best:
            best, flag = vc, "  <- best"
            torch.save(model.state_dict(), out)
        print(f"epoch {ep:>3}  train {train:.4f}  val_n2n {vn:.4f}  "
              f"val_clean {vc:.4f}  ({hist[-1]['minutes']:.1f} min){flag}",
              flush=True)

        if ep % args.sample_every == 0 or ep == args.epochs:
            dump_samples(model, va, dev, samples / f"epoch_{ep:03d}.png")

    print(f"\nbest val_clean {best:.4f} -> {out}")


@torch.no_grad()
def dump_samples(model, ds, dev, path, n=4) -> None:
    """input | output | clean, stacked, so removal is judged by eye not loss."""
    model.eval()
    rows = []
    for i in np.linspace(0, len(ds) - 1, n).astype(int):
        a, _, clean = ds[int(i)]
        a = norm(a.unsqueeze(0), dev)
        o = model(a)[0].cpu()
        rows.append(torch.cat([a[0].cpu(), o, norm(clean.unsqueeze(0), dev)[0].cpu()],
                              dim=2))
    grid = torch.cat(rows, dim=1)[0].numpy()
    Image.fromarray(((np.clip(grid, -1, 1) + 1) * 127.5).astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4,
                    help="Windows backs worker shared memory with the "
                         "page file; more than ~6 risks the commit limit")
    ap.add_argument("--max-stamps", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--full-dir", type=Path, default=FULL)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="N2N_text_1.pth")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
