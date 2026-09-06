# Automatic funscript generation — findings to date

A distillation of several months of measured experiments, written when the
long-form effort was paused (2026-08). Every claim below was measured, most of
them several times. Anyone resuming this work should read this before trying
anything, because the graveyard section has already consumed weeks.

Vocabulary note: class names follow the committed `content.example.json`
placeholders ("anchor" = the anatomy the ROI centers on, "contact classes" =
what touches it). Real names live in the git-ignored overlay. Truth data below
means the operator's hand-made funscripts.

## What shipped and works

**The short-clip lane is fully automated.** Every clip in that lane (~300)
carries a generated script the player loads automatically, device-approved
("good enough for jazz"). The recipe, end to end:

1. **Domain fine-tune of the detector from auto-labels.** Seed a video
   segmentation tracker (SAM2) with every detector hit, harvest its per-frame
   anchor masks as bounding-box labels, exclude frames where the mask is
   suspect rather than labeling them empty (an empty label trains a false
   negative), keep other classes as pseudo-labels so they aren't forgotten.
   This lifted anchor recall on held-out material from 4–38% to 54–91% in
   every domain tried. It is the single highest-leverage step found.
2. **ROI optical flow for the signal.** The detector aims a padded union ROI;
   DIS optical flow inside it, magnitude-weighted mean dy, mapped as
   `pos = 50 + gain*dy`. This is a *velocity* signal, phase-shifted a quarter
   stroke from true position — measured, and perceptually irrelevant for
   rhythmic content.
3. **Ultimate Autotune as the output stage** (imported from the FunGen
   checkout). Raw signals score ~0.10 lower against truth without it, and the
   device feel difference is larger than the number suggests.

Supporting machinery that also works and is worth keeping: per-item
checkpointing everywhere (crashes cost one item), duty-cycle GPU throttling
(sustained peak load rebooted the machine three times), audio-envelope
cross-correlation to locate a clip inside its source when frame hashes fail
across upscale lineages (0.93 peak where frames scored zero), and gap-masking
partial truth scripts (inter-action gaps over ~3s are unscripted blanks, not
slow strokes — one truth video was 20 minutes long with 4.5 scripted).

## The central negative result

**Cross-video generalization with detection-derived features plateaus at
r ≈ 0.15–0.25 against truth, in every formulation tried**: hand-crafted
signals, learned per-frame models, fusions of both, with and without motion
channels. Measured independently three times (library-wide eval, learned
fusion with leave-one-video-out, the clip campaign gate). Within-video
supervision is different in kind: ~180 sparse position clicks on one video
took a learned model to r = 0.61 cross-validated on held-out time blocks of
that video. The wall is not data volume or model class — features computed
from detections do not carry enough invariant meaning across cameras, crops,
and performers.

**But the correlation metric does not decide shippability.** The clip lane
shipped and satisfied at gate r = 0.273; the long-form lane was rejected on
device at gate r = 0.200. Numbers that close mean the metric cannot arbitrate
across content types — only device feel does. Related traps, all hit at least
once: a reference script *derived from* signal X structurally favors signal X
in any correlation comparison; blending a strong signal with weaker ones
dilutes it (measured three times — pick the best single signal instead); and
temporal stability beats per-frame correctness on device (a per-frame-accurate
signal that flickers between sources feels worse than a stable proxy).

## What the signal must measure, by content type

The operator's own definition: *the deepest point currently being stimulated,
projected onto the anchor's axis* — and licking counts. But what carries that
information differs by content:

- **Mouth-dominant action:** the occlusion front (where the visible anchor
  ends) and the face position track the stroke. Face detection is the most
  reliable class (~91% of frames) and face-y alone reached r = 0.55 on such
  content.
- **Hand-dominant action** (most of the clip lane): geometry goes blind — a
  hand slides along a mostly visible anchor without changing its outline, so
  occlusion-front, face, and box features all decorrelate (measured ≈ 0 per
  frame). Only *motion* (flow rhythm) carries the stroke. Truth scripts on
  such content encode rhythm, not per-frame depth.

Long-form videos mix both plus idle/positional segments, which is presumably
why the clip recipe transferred poorly (rejected on device at r = 0.200
median vs truth). Unresolved.

## The graveyard (do not retry without new information)

- Dense per-frame segmentation as a position source: drifts onto adjacent
  skin under occlusion cycles and undersegments unpredictably; r ≈ 0.26 after
  drift-guarding. (As a *label source* seeded from frequent detector hits it
  is excellent — that contradiction is real and both halves are measured.)
- Moving an occluded anchor's remembered box by camera/flow heuristics: worse
  than freezing it (89px median error frozen vs 151px compensated).
- Confidence-threshold lowering to fix recall: plateaus ~4 points above
  baseline; the misses are occlusion blindness, not calibration.
- Learned fusion of many weak features: ties or loses to the best single
  feature at every data scale tried (176 clicks to 150 minutes of truth).
- Direct per-frame depth regression from a few hundred sparse clicks:
  under-beats its own best input feature. Sparse clicks audit and calibrate;
  they do not train per-frame models.
- Off-the-shelf multi-object trackers' motion models (ByteTrack-class Kalman
  coasting): built for ~1s misses, useless for 2–60s occlusions.

## State at close, and the levers left

Shipped and live: the clip lane (~300 scripts, auto-maintained — the variant
rehome stage in the library manager carries scripts across upscale renames),
plus blank-except-clip skeleton scripts on 52 source videos via the sidecar
scene-offset mapping. Long-form: paused after device rejection; no fleet run
was performed. The fine-tuned detector weights and pipeline scripts live
outside the repo (private cache; see the session notes of the operator's
agent for exact locations).

Levers never pulled, in rough order of expected value:

1. Mainstream hand/pose tracking as a feature for hand-dominant content —
   hands are the one relevant class mainstream CV is superb at, and no
   in-domain hand tracker was ever integrated.
2. Per-video micro-supervision at scale: the ~15-minute click session that
   took one video to r = 0.61 was never industrialized (active-learning
   schedules existed in the GUI's label-session mode).
3. Segment classification (mouth/hand/idle) to route between signals instead
   of blending them.
4. VR content: untouched; FunGen itself is strongest there.
