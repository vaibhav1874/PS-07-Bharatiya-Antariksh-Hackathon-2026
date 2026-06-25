# Autonomous Debug & Build Prompt — Use With The Technical Reference File

> **Pehle yeh karo:** `PS07_Technical_Math_Physics_Reference.md` file ko apne
> project ke root folder mein daal do (jahan `pipeline.py`, `detrend.py` etc.
> hain). Phir neeche wala prompt apne AI agent (Antigravity/Claude
> Code/Cursor) ko de do. Agent ab khud file padh ke kaam karega — tumhe
> manually excerpt nikal ke dene ki zaroorat nahi.

---

## THE PROMPT (copy everything below this line)

```
Before doing anything else, open and fully read the file
`PS07_Technical_Math_Physics_Reference.md` in this project's root directory.

This file is the AUTHORITATIVE technical specification for this pipeline. It
contains, for every stage: the exact math/physics formulas to use, the exact
algorithm steps, required test cases, and a troubleshooting table mapping
known symptoms to root causes and fixes. From this point forward:

- Any formula, threshold, or algorithm choice you make must match what's in
  this file. If you believe the file is wrong or inapplicable to the actual
  data you're seeing, STOP and tell me why — do not silently deviate from it.
- Every "bug fix" or "phase complete" claim you make must be backed by
  re-running the relevant test case from Part D of the file and showing me
  the actual passing output, not just an assertion that it works.

## STEP 1 — Fix the 3 known, confirmed bugs first (in this order)

Go through Part C ("Master Troubleshooting Table") of the file and resolve,
in priority order:

1. The transit-fit bug (Depth = 0.0, uncertainty = nan, seen in
   `KIC_11904151_transit_model.png`). Read Part B5's 5-cause checklist in the
   file and work through it in the exact order given (batman import check
   first). Show me the real terminal output of `pip list` inside the actual
   `.venv`, not a paraphrase.

2. The boxy/rectangular detrending artifact (seen in
   `KIC_11904151_detrending.png`, around the data gaps at day ~200 and
   ~400-440). Implement the gap-segmentation fix exactly as described in
   Part B3 of the file.

3. The rising BLS noise floor toward long periods (seen in
   `KIC_11904151_periodogram.png`). Apply Part B4's diagnostic order: first
   re-check after fixing bug #2 above, then cap the max searched period per
   Part A1's formula, and only flag red noise as the explanation if the rise
   persists after both of those.

After each of the 3 fixes, regenerate the affected plot(s) for KIC 11904151
and run the matching Part D test case before moving to the next bug.

## STEP 2 — Validate against Part A before trusting any result

Confirm the Part A4 data-volume calculation in the file (baseline,
N_transits_observed, points-in-transit) actually matches what your code is
computing internally for this target. If your code's internal numbers
disagree with the file's worked example, that itself is a bug — find out why.

## STEP 3 — Continue building the remaining phases using the file's exact specs

For every phase not yet implemented (vetting: Part B6; statistical
significance: Part B7; classification: Part B8), implement it using the exact
formulas and algorithms given in those sections — do not substitute a
simplified version without telling me. After implementing each phase, run its
Part D test case(s) and show me real pass/fail output.

## REPORTING FORMAT (use this for every update you give me)

For each change you make, report:
1. Which Part C row / Part B section this addresses
2. What you changed (file + function)
3. The actual output/number/plot that proves it now works
4. Which Part D test case(s) now pass because of this change

## HARD RULES

- Never silently catch an exception and substitute a placeholder value (0,
  nan, or otherwise). If something fails, surface the real error.
- Never report a metric you have not actually computed by running the code.
- If real data forces you to deviate from a formula/threshold in the
  reference file (e.g. a gap_threshold that doesn't fit this star's cadence),
  say so explicitly and justify the new value — don't change it silently.
- Work through Step 1 fully (all 3 bugs verified fixed via their test cases)
  before starting Step 3.
```
