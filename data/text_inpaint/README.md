# Text inpainting (N2N) — burned-in annotation removal

A second Noise2Noise inpainter, separate from the caliper one. Same
architecture, same loss, same normalisation, same runtime shape — different
weights, different trigger, different output column. The caliper model
(`N2N_7.pth`, `src/ML_processing/inpaint_N2N.py`) is untouched.

## Why

BUS_framework `experiments/text_leakage/TEXT_LEAKAGE.md` OCR'd 18,675 v7
training frames and measured what the malignancy classifier does with the
burned-in words that land inside its 256px crop. The short version:

- **2.44%** of v7 frames carry a sonographer annotation inside the crop.
- Clinical annotation is ~**2.2x** more common on malignant exams, which is
  the raw material for a shortcut.
- At **bag** level the effect is negligible — masking the word moves the exam
  probability by a median of 0.0003.
- At **frame** level it is real: masking the text box moves the frame's own
  score more than masking a same-size box next to it (p < 0.001).
- **`CLIP` is the one phrase that matters.** ~0.9 logits at the median per
  frame, and a tail: 4 of 25 tested exams move by more than 0.05 when the word
  alone is greyed out; the worst goes 0.67 → 0.24. `RIB`, `SCAR` and
  `IMPLANT` show smaller tails in the same direction on n = 9–22.

That audit's own recommendation was *not* to grey the text at inference —
doing so on `CLIP` bags made Brier worse, because the word is largely true
evidence of a prior biopsy. Removing it **upstream, before training**, is the
other option, and is what this model exists to enable: a text-blind corpus,
built once, that a future retrain can use.

**Standalone.** Nothing in `main.py` calls it; it is a training script plus an
inference module you invoke yourself. See *Wiring* if that ever changes.

## The three assets

| path | what | n |
|---|---|---|
| `stamps/` + `stamps_manifest.csv` | RGBA glyph cutouts of real burned-in annotation | 1,001 |
| `frames.csv` | full-res frames verified free of clinical annotation — the training pool | 13,506 (4,011 val) |
| `frames_annotated.csv` | frames carrying REAL annotation, excluded from training, held out for QC | 227 |
| `../../src/ML_processing/models/N2N_text_1.pth` | the trained model | — |

`source_detections.csv` is the slice of the audit's OCR output the stamps were
cut from, copied here so the library can be rebuilt without the research repo.

## How it works

**The stamps are the training signal.** `build_text_stamps.py` cuts each
clinical OCR box out of its full-resolution frame and builds an alpha channel
from a grayscale morphological top-hat — patch minus its opening, which
responds to thin bright strokes and not to tissue. The RGB is the untouched
patch, so a paste reproduces the scanner's own rendering. Without the alpha a
paste would carry a rectangle of donor tissue and teach the network to erase
rectangles.

After dedup on a coarse alpha signature (the same annotation recurs on every
frame of an exam and would otherwise swamp the library):

```
NIPPLE 328 | CLIP 211 | CLOCK 99 | DIST 92 | LAT 29 | SCAR 20 | ORIENT 13
RIB 11 | CALC 11 | PALP 8 | BREAST 8 | IMPLANT 8 | MASS 3 | AXILLA, RADIAL, ... 1-4
median stamp 86 x 23 px
```

`RIB` and `IMPLANT` need a note. The audit's classifier files them as `other`,
not `clinical`, so a plain `kind == "clinical"` harvest misses them — and both
are in its section-8 tail. The harvester carries a short allow-list
(`EXTRA_CLINICAL`) that pulls them, plus `CALC` and `BREAST`, back in. The rest
of the `other` bucket genuinely is chrome the classifier misfiled (`+`,
`TAC1`, `DYN R 65`, `P LOW`, `X4`, `58%`), which is why it is an allow-list and
not a wider net.

**The pairing is the Noise2Noise trick.** Take a frame with no annotation.
Paste a random set of stamps to make the INPUT; paste a *different*,
independently drawn set to make the TARGET; train with L1. At a pixel the
input stamp covers, the target is bare tissue most of the time — the second
draw rarely lands in the same place — so the L1-optimal output is the tissue
underneath. No clean/annotated pair of the same real frame is ever needed,
which matters because, unlike calipers, text has no twin frames.

**Anything identical on both sides is preserved**, because the loss punishes
changing it. That is the whole mechanism behind the next section.

## What it removes, and what it deliberately keeps

Removes: sonographer annotation inside the **ultrasound region** — NIPPLE,
CLIP, SCAR, PALP, clock positions, distances, laterality.

Keeps: **everything outside it.** The scanner sidebar, the banner, and the
annotation strip below the image survive. This is not an accident and not a
post-hoc mask:

- stamps are pasted only inside the ultrasound rectangle, so chrome is
  identical on both sides of every training pair;
- crops are taken from the **whole frame**, not just the tissue, so the network
  sees chrome-with-no-edit and tissue-with-edit side by side and learns the
  difference from context.

It matters twice over. The database pipeline reads laterality, clock position
and distance off that chrome — erasing it would destroy metadata extraction.
And the classifier never sees it: out-of-region text is outside the 256px crop,
so it was never part of the leakage in the first place.

`inpaint_text_N2N.py` additionally discards any edit outside the ultrasound
rectangle when the rectangle is known. Belt and braces: a single bad frame
cannot cost the pipeline its metadata.

## Rebuild

```bash
# 1. cut the stamp library out of the audit's OCR boxes      (~2 min)
python training/build_text_stamps.py --contact-sheet

# 2. pick the clean training frames + the held-out QC set    (~1 min)
python training/build_text_frames.py

# 3. train                                     (~3.5 min/epoch on an RTX 4080)
python -m training.train_text_N2N --epochs 40

# 4. measure removal on frames with REAL annotation
python -m training.eval_text_removal

# 5. eyeball it
python -m src.ML_processing.inpaint_text_N2N --preview 12 --on annotated
python -m src.ML_processing.inpaint_text_N2N --preview 12 --on clean
```

Training writes `train_log.csv` and a `train_samples/epoch_NNN.png` grid
(input | output | clean) every few epochs.

Checkpoints are selected on **`val_clean`**, L1 against the untouched crop, not
on the N2N loss. The clean frame is the real ground truth here — it was clean
before a stamp was pasted on it — whereas the N2N loss floors at the variance
of the second stamp draw and is hard to read.

## Results

Trained 40 epochs, 139 min on an RTX 4080. `val_clean` — L1 against the
untouched crop — fell from 0.0119 to **0.00054**, a 22x improvement, flat over
the last five epochs. Train and `val_n2n` finish within 0.00001 of each other
(0.004349 vs 0.004346), so there is no overfitting to report.

**Held-out test on REAL annotation.** 227 frames the model never trained on,
carrying genuine sonographer boxes. A box is graded by how much of the word's
own contrast survives -- box mean minus the mean of the ring around it, in grey
levels out of 255.

Two ratios appear in the artefacts here and they do not agree. The SIGNED ratio
(`eval_text_removal.csv`) lets an overshoot -- the model digging a hole where
the word was -- come out negative and sort next to a clean removal, which
flatters the median. The MAGNITUDE ratio `|after| / |before|`
(`eval_boxes_magnitude.csv`, and `presentation.html`) is monotone in "how wrong
is this" and is the one to quote:

| | signed | **magnitude** |
|---|---:|---:|
| removal targets | 249 | **172** (deduped per frame+phrase) |
| median remaining contrast | 0.106 | **0.306** |
| cut below half | 77.5% | **69.2%** |

On the magnitude metric:

| | |
|---|---:|
| median contrast, before -> after | **28.3 -> 8.6** grey levels |
| reduced below half | **69.2%** |
| reduced below a quarter | **42.4%** |
| left under 5 grey levels (invisible) | **35.5%** |
| **left completely untouched** | **7.0%** |
| preservation targets changed (outside the region) | **0.00** grey levels |

That last pair is the honest headline: the typical word loses about 70% of its
contrast, and roughly one in fourteen is not touched at all. Splitting on role
is not cosmetic -- 21% of the clinical boxes on these frames sit outside the
ultrasound region, and grading them together scores a correctly-preserved word
as a removal failure.

By phrase, median remaining contrast: `CLIP` 0.229 (n=44), `NIPPLE` 0.283
(n=82), distances 0.365 (n=15), clock positions 0.426 (n=10). `CLIP` is the
phrase the leakage audit singled out and the one the model clears best.

**Cost to everything else**, mean absolute change in grey levels (0-255):

| | |
|---|---:|
| tissue inside the ultrasound region, excluding every text box | **0.016** |
| whole ultrasound region, on clean frames with nothing to remove | **0.003** (p95 0.012) |

Both are far below one grey level, so the edit is surgical rather than a
whole-frame denoise. That is the failure mode the BUS_framework caliper
audit found in the raw N2N output, and it is avoided here by two things
together: the masked composite, and `us_rect` discarding edits outside the
ultrasound region.

`presentation.html` is the report page -- headline figures, the distribution,
and the best / typical / worst cases side by side. `preview_annotated.png` is
the raw version of the same thing: the 8x difference column is
black everywhere except the exact words removed, with the scanner sidebar and
the annotation strip below the image untouched. `preview_clean.png` is the
same on frames with nothing to remove. `eval_text_removal.csv` has one row per
box.

**What the residual does not say.** A median residual of 0.106 is measured on
the full-resolution frame. Whether 11% of a word's contrast is still legible to
the classifier at 256px is a separate question, and answering it means running
the BUS_framework occlusion test again on detexted crops. Nothing here has been
measured against the classifier.

## Inference

```python
from src.ML_processing.inpaint_text_N2N import load_model, remove_text
model = load_model()
out, n_changed = remove_text(pil_image, model, us_rect=(x, y, w, h))
```

`--mode masked` (default) keeps the original pixels wherever the network did
not meaningfully change them, so the output differs from the input only where
text was. `--mode full` returns the raw network output, which is also a
whole-frame denoise — the BUS_framework caliper-inference audit found the raw
N2N output changes pixels far outside the markers (the top 1% of changed
pixels carry only ~13% of the total change), and recommended the masked form.

## Wiring (deliberately not done)

Left standalone by request. If it is ever folded into the database build,
`Detext_Dataset_N2N` mirrors `Inpaint_Dataset_N2N` and slots into `main.py`
next to it:

```python
if not args.skip_inpaint:
    Inpaint_Dataset_N2N(f'{CONFIG["DATABASE_DIR"]}/images/')
    # Detext_Dataset_N2N(f'{CONFIG["DATABASE_DIR"]}/images/')   # <- here
```

It is left commented out on purpose. Turning it on changes what every
downstream crop contains, so it is a decision about the next corpus build, not
a bug fix — and the audit's verdict was that the bag-level effect does not
warrant disturbing the shipping model. The sequence that would justify it:
build a text-blind corpus, retrain, and compare, rather than inpainting at
inference time on a model trained with the words present.

Two other steps remain when that decision is made:

1. the `detexted_version` column must exist on the `Images` table;
2. `N2N_text_1.pth` should be uploaded to the HuggingFace repo and added to
   `src/ML_processing/download_models.py` alongside the `N2N` entry, so a
   fresh clone can fetch it.

## Limits

- **The model has only ever seen pasted stamps.** Real annotation may sit at
  systematic positions or carry a drop shadow the paste does not reproduce.
  The 227 held-out annotated frames are the test of that transfer; see
  `eval_text_removal.csv`.
- **The stamp library inherits the audit's OCR recall.** A phrase OCR never
  read is a phrase the model never learned to remove. The audit's 150-frame
  stress test found no missed clinical annotation, but that is a stress test,
  not an unbiased estimate.
- **The clean pool is the OCR'd sample**, 13.5k of ~558k v7 frames, because
  those are the only frames with a verdict. Frames outside it could not be
  used: an annotation already burned in appears identically in input and
  target, so the loss would teach the model to keep it.
- **`RIB` and `IMPLANT` are thin** — 11 and 8 stamps against NIPPLE's 328,
  because OCR only found 14 and 11 detections in the whole sample. They are
  represented, but a phrase seen 11 times is not learned the way one seen 328
  times is. Both were tail findings in the audit on n = 9–10 bags, so neither
  the leakage estimate nor the removal is well determined for them.
