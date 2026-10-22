# decisions.md — Locked Research Design (v2, 22 Oct 2026)

Single source of truth for the dissertation design. It matches the revised proposal (Sections 3–7).
Status tags: **LOCKED** · **PILOT** (the rule is fixed; the value is set by Module 1 output, never guessed) · **OPEN**.
Every change gets a row in Section 14, which becomes the thesis section "Deviations from the approved proposal".

## 1. Title
Multi-Resolution Spatiotemporal Deep Learning for Short-Term Active-Region Solar Flare Forecasting from SDO/HMI SHARP Magnetograms

## 2. Aim (LOCKED)
To determine, under one controlled protocol, whether temporal and multi-resolution temporal modelling of SHARP Br sequences
improves 6–24 h active-region flare forecasting over a static baseline. The primary measure is TSS for ≥M; cost is reported as
params, peak VRAM, train time and inference latency.

## 3. RQs / hypotheses (LOCKED)
- RQ1: E1 vs E2. H1: TSS≥M(E2) > TSS≥M(E1).
- RQ2: E2 vs E3 (H2a) and E3 vs E4 (H2b), on TSS≥M.
- RQ3: E4 vs E5. H3: Recall_M and Recall_X rise; precision may fall.
- RQ4: descriptive cost and Grad-CAM.

**Decision rule:** claim an improvement only if the 95% paired cluster-bootstrap CI (resampled by HARP, 2,000 resamples) of the
test-set difference excludes 0, with the direction consistent across seeds if several seeds are run.

## 4. Task (LOCKED)
- **Sample** = (HARP, t), with t on a 6-hourly grid (00/06/12/18). A SHARP record must exist at exactly t.
- **Input**: records with T_REC ≤ t only.
- **Label**: max GOES class among flares associated with the HARP whose **start time** is in **[t+6 h, t+24 h)**.
  Class 0 = none/A/B/C, 1 = M, 2 = X. Binary ≥M = Class 1 or 2 (same predictions).
- **Time scales**: T_REC is TAI (TAI − UTC = 37 s since 2017). The nominal T_REC is used as the observation time; the 37 s offset
  is ignored as negligible against 12-minute cadence and hour-scale windows. This must be stated in the thesis.

## 5. Association (LOCKED rules; counts are PILOT)
1. One catalogue: HEK flare events from GOES via SunPy, restricted to one `frm_name` (default "SWPC"). The pilot reports all
   frm_names seen before filtering. (Hu et al. 2025 document catalogue inconsistencies.)
2. **NOAA number normalisation**: SWPC/HEK often report 4-digit region numbers (e.g. 3664); SHARP `NOAA_ARS` uses 5-digit numbers
   (13664). Numbers < 10000 are mapped to n + 10000. This is valid for regions after the numbering passed 10000 (June 2002),
   which covers the whole SDO era.
3. A flare is associated with a HARP if its NOAA number appears in that HARP's `NOAA_ARS` in any record within ±1 h of the
   flare start.
4. A flare is **unassociated** if it has no NOAA number or matches no HARP. If an unassociated **≥M** flare starts in a sample's
   forecast window, the sample is excluded.
5. A flare is **ambiguous** if it matches more than one HARP. Samples of those HARPs whose window contains it are excluded.
6. One HARP with several NOAA regions: flares from any of them count for the HARP.
7. **Only HARPs with ≥1 NOAA region at t are sampled** (`require_noaa: true`). A HARP without a NOAA number can never receive
   an associated flare, so its negatives would be unverifiable. (New in v2; see Change Log.)
8. **Limb**: every input frame that exists must have |LON_FWT| ≤ 68°, otherwise the sample is excluded. Limb frames are not
   substituted.
9. **QUALITY**: frames with QUALITY outside the accepted mask are treated as missing. Default: QUALITY == 0 only. The final rule
   is PILOT (the pilot reports the value distribution).
10. **Missing frames**: the frame at t is mandatory. Any other missing or bad frame is replaced by the nearest earlier good record
    inside the input window. More than 3 replacements, or an unfillable gap, means the sample is excluded.
11. A flow table of counts after each rule is reported.

## 6. Inputs and preprocessing (LOCKED)
- Data: `hmi.sharp_cea_720s`, segment Br.
- Preprocessing: NaN → 0, clip ±1500 G, divide by 1500, zero-pad to a centred square, resize to 128×128 (64 if the memory test
  fails), store as float16.
- Storage: one memory-mapped array per HARP plus a timestamp index, on local disk.
- No augmentation.

## 7. Temporal design (LOCKED)

| Model | Window | Spacing | Frames |
|---|---|---|---|
| E1 | t only | — | 1 |
| E2 | 12 h | 24 min | 31 |
| Coarse branch | 12 h | 48 min | 16 |
| Fine branch | 6 h | 24 min | 16 |

Only records on the 24-min grid relative to t are ever needed.

## 8. Models (LOCKED)
- **Shared encoder**: 4 × [Conv3×3–BN–ReLU–MaxPool] → GAP → 128-d.
- **E0**: persistence, the max associated class in [t−24 h, t).
- **E1**: encoder → MLP.
- **E2**: encoder → LSTM(128) → last hidden state → MLP.
- **E3**: shared encoder, an LSTM per branch, concatenation of the last hidden states → MLP.
- **E4**: E3 with additive attention pooling per branch.
- **E5**: E4 trained with class-weighted cross-entropy, w_c = N/(K·N_c), computed on the training split only.

## 9. Training (LOCKED)
- AdamW (weight decay 1e-4), batch 16, AMP, max 30 epochs, early stopping (patience 5) on validation TSS≥M.
- Learning rate ∈ {1e-3, 3e-4}, with the same tuning budget for every model.
- Seeds: 3 if time allows, otherwise 1.
- Hardware: laptop GPU; AWS EC2 or Kaggle for the remaining seeds.

## 10. Split (LOCKED)
- HARP-grouped month blocks in the repeating pattern T, T, T, V, Te.
- Each HARP is assigned by the month of its first record.
- Observation period: PILOT.

## 11. Evaluation (LOCKED)
- Primary: TSS≥M.
- Secondary: per-class P/R/F1, one-vs-rest TSS, macro-F1, confusion matrix, HSS≥M.
- X class: if there are fewer than 10 test X samples, report descriptively only.
- Cost:
  - params
  - `torch.cuda.max_memory_allocated`
  - s/epoch and total training time
  - batch-1 latency (median of 200 runs after 20 warm-up runs)
- Grad-CAM: about 2 true positives, 2 false negatives and 2 false positives.

## 12. Scope exclusions (LOCKED)
Excluded: 3D CNN, solar-cycle input, focal loss, resampling, multi-wavelength input, other architectures.
Optional only if time remains: E1w.

## 13. Pilot outputs that will fill PILOT items
- Catalogue `frm_name`
- QUALITY rule
- MB per exported Br frame and download MB/s
- Frames needed per month
- Observation period
- Image size (from the memory test in Module 4)

## 14. Change log vs approved proposal

| # | Change | Reason | Status |
|---|---|---|---|
| 1 | Branches: 2 h/1 h over the same 12 h → coarse 12 h @ 48 min, fine 6 h @ 24 min | SHARP 12-min cadence; the fine branch now adds information | Approved 22 Oct |
| 2 | E2 = 12 h @ 24 min | Fair architecture control | Approved 22 Oct |
| 3 | Strict chronological split → HARP-grouped month blocks | X-class scarcity; solar-cycle-phase confound | Approved 22 Oct |
| 4 | Primary metric TSS≥M; HSS added | Standard practice; few X events | Approved 22 Oct |
| 5 | E0 persistence reference added | No-skill reference | Approved 22 Oct |
| 6 | Interval written as half-open [t+6, t+24) | Consistency | Approved 22 Oct |
| 7 | Only HARPs with a NOAA number are sampled | Negatives must be verifiable | v2: approved by proceeding with the pilot |
| 8 | 4-digit → 5-digit NOAA normalisation | Catalogue convention | v2: technical, no design impact |
