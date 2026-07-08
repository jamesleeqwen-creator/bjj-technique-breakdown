---
name: bjj-review-sheet
description: Turn a BJJ instructional video + subtitles into an illustrated student review sheet. Use when the user asks to create a review sheet, breakdown, or study guide from a BJJ video file.
---

# BJJ Review Sheet

You (the agent) drive a 5-stage pipeline. The script does mechanical work; YOU do two
judgment stages by reading and writing JSON files. Requires a vision-capable model
(you must be able to look at JPEG contact sheets).

**Prerequisites check (run once):** `python3 -c "import cv2, numpy"` and `ffmpeg -version`.
If missing, see README.md. Set SKILL_DIR to this skill's folder.
**Optional tools** (only check when the situation below arises): `yt-dlp` for Stage 0,
`whisper` for Stage 1.5.

## Stage 0 — obtain the video (optional; only when given a URL)
If the user supplies a YouTube or Bilibili URL instead of a local file, verify
`yt-dlp --version` works (install: `pip install yt-dlp`, or the standalone binary — see
README). Download the video AND its subtitles into the folder where the review sheet
should live:
```
yt-dlp -f "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" \
  --write-subs --write-auto-subs --sub-langs "en.*,zh.*" -P "<destination folder>" "<url>"
```
The format string prefers H.264 (`avc1`) video: YouTube often serves AV1 otherwise, and
common OpenCV builds cannot decode AV1, which breaks the candidates stage. If you must
work from an AV1 file, transcode a same-timeline proxy first
(`ffmpeg -i <video> -vf scale=1280:720 -c:v libx264 -preset ultrafast -an <proxy>.mp4`),
run candidates against the proxy, and render against the original.
yt-dlp's default naming (`Title [id].mp4` + `Title [id].en.vtt`) is exactly what Stage 1
auto-discovers — do not rename the files. If no subtitle file was produced (common on
Bilibili), continue to Stage 1.5 after noting the video path. If yt-dlp errors out on one
subtitle language (e.g. a 429 on `zh-Hans`) before the video itself downloads, retry with
a narrower `--sub-langs` (just the language you actually need) rather than requesting
several at once.
Keep the URL: Stage 1 takes it via `--url`, and the rendered sheet then links the title
and every step's timestamp straight to the video.

## Stage 1 — prepare (script)
```
python3 "$SKILL_DIR/scripts/bjj_breakdown.py" prepare --video "<video file>" [--url "<source url>"]
```
Creates `<video>_review/` next to the video with `transcript.json`, `scenes.json`, `meta.json`.
Pass `--url` whenever you know the video's online source — from Stage 0, from the user,
or reconstructed from a yt-dlp-style filename ending in `[<id>]` (an 11-character id →
`https://www.youtube.com/watch?v=<id>`; a `BV...` id → `https://www.bilibili.com/video/<id>` —
confirm a reconstructed URL with the user before using it). The sheet then gets a
"Watch the source video" link plus click-to-watch timestamps on every step and frame.
Re-running `prepare` just to add a URL is instant (all caches hit); `--url ""` removes
a stored URL.

## Stage 1.5 — generate a transcript (optional; only when prepare finds no subtitles)
If no `.vtt`/`.txt` subtitle file exists next to the video, transcribe the audio:
```
whisper "<video file>" --model small --output_format vtt --output_dir "<video folder>"
```
Add `--language zh` (or the actual language) when the instruction is not in English —
do not let Whisper guess. This writes `<video_base>.vtt`, which Stage 1 auto-discovers;
re-run `prepare` afterwards. Whisper segments are short (2–10 s), which gives you finer
step-boundary timing than YouTube captions. Expect minutes of runtime on long videos;
stay with `--model small` unless the transcript comes out unusable.
**Burned-in subtitles:** if the video has subtitles rendered into the picture, this
stage is still the answer — the speech is in the audio, so transcribe it. Do NOT try
to OCR the burned-in text; you will instead use it to verify terminology in Stage 4.

## Stage 2 — Segmentation (YOU)
Read `transcript.json` (timestamped speech) and `scenes.json` (visual cut times, use as
guardrails). Then work in this order:

**First, answer the One Thing question.** After reading the WHOLE transcript, answer:
*"If a student remembers only one thing from this video, what should it be?"* Write the
answer (1–2 sentences) as the top-level `one_thing`. Instructors usually say it
themselves — check the intro and the recap first. It must be grounded in what the
instructor actually says. Keep it in mind as the lens for every step you write below,
but do NOT force a reference to it into every step — that reads as repetitive padding.

**Then decide the technique grouping.** Does the video teach ONE technique, or
SEVERAL distinct ones? (Several is typical past ~8–10 minutes; instructors usually
announce each — "the first escape…", "now the second option…".) A technique is a
named, self-contained skill a student could drill on its own. If the video teaches
several: write a top-level `techniques` list — ids `tech_01`, `tech_02`, … in teaching
order, each with a short `title` and an optional 1–2 sentence `summary` grounded in
the transcript — and tag EVERY step with the `technique` it belongs to; one
technique's steps must be consecutive. Opening/closing chatter about the whole video
joins the first/last technique, or becomes its own "Introduction" part if substantial.
If the video teaches one technique, omit `techniques` and the per-step `technique`
field entirely — do not write a one-entry list. Each technique renders as its own tab
in the final sheet.

**Then segment into steps** and write `<work_dir>/steps.json`:

- **Ground every step in the transcript — never invent content.** Each title,
  key_point, and explanation must restate something the instructor actually says
  inside that step's time range. Work transcript-first: read all segments, group them
  into steps, then write. `check` prints a WARNING when a step's wording shares almost
  no vocabulary with its transcript window; treat any such warning as a bug in your
  steps.json.
- A **step** is one coherent action or concept the instructor teaches, typically 15–60s.
  Split on imperative instructions ("push", "swing your leg", "place your hand"),
  transitions ("now", "then", "next", "one more time", "other side"), and scene cuts.
  Avoid steps spanning a scene cut unless the speech clearly continues across it.
- **Seminar videos alternate demonstration and class drilling.** Drilling starts at
  spoken cues like "Okay, let's do it", "One, two", "Let's go" and the camera then
  films random student pairs. End demonstration steps AT the drill cue rather than
  spanning the drilling; when a window must include drilling time, warn your future
  judging self in `visual_cues` ("class drilling after ~MM:SS — avoid those frames").
- Cover the whole video; opening/closing chatter becomes category `intro` / `recap`.
- `title`: imperative, ≤80 chars, describes the action (NOT a transcript quote).
- `key_points`: 1–4 clean, complete sentences (aim for 2–4 on demonstration/drill
  steps) — the imperative ACTIONS: what to do, in what order. Fix speech-recognition
  errors, delete filler. These are what a student re-reads before class.
- `explanation`: 2–4 sentences of connected prose — WHY the technique works, what it
  should feel like, how the opponent reacts. Written for a student who watched the
  video once and needs the concept explained back clearly. Do NOT restate the
  key_points in paragraph form — if a sentence just repeats a bullet, cut it.
  Required for concept/demonstration/drill steps; optional for intro/recap.
- `visual_cues`: one sentence describing what the key moment LOOKS like on screen —
  you will use this yourself later to pick frames. Say how to recognize the
  demonstrators by appearance (e.g. "instructor in black rash guard, partner in pink
  shirt") — seminar cameras also film students drilling, and Stage 4 must be able to
  tell the two apart. The demo partner may change mid-video; the instructor is the
  constant.
- `category`: one of intro | concept | demonstration | drill | recap.
- **Language:** write everything student-facing (`one_thing`, titles, key_points,
  explanations) in the language of instruction, unless the user asks for another.

Schema (exact; `techniques` + per-step `technique` only for multi-technique videos —
omit both otherwise):
```json
{"schema_version": 2, "video_base": "<video_base>",
 "one_thing": "...",
 "techniques": [{"id": "tech_01", "title": "...", "summary": "..."}],
 "steps": [{"id": "step_01", "title": "...", "start": 12.5, "end": 41.0,
            "category": "demonstration", "key_points": ["..."],
            "explanation": "...", "visual_cues": "...", "technique": "tech_01"}]}
```
Then validate: `... bjj_breakdown.py check --video "<video>" --what steps`
Fix any reported error and re-run until PASS.

## Stage 3 — candidates (script)
```
python3 "$SKILL_DIR/scripts/bjj_breakdown.py" candidates --video "<video file>"
```
Creates one labeled contact sheet per step in `<work_dir>/contact_sheets/`.

## Stage 4 — Judging (YOU)
For EACH step: view `contact_sheets/contact_step_NN.jpg`. Each cell is labeled with a
letter and timestamp; for a closer look at one cell open `candidates/step_NN/<LETTER>.jpg`.

**Before anything else, identify the demonstrating pair.** Find the instructor in the
earliest sheets (they appear in nearly every step) and note how they are dressed; the
demo partner may change mid-video, the instructor does not. Approve ONLY frames of the
instructor's own demonstration. In seminar footage the camera also films the class
drilling — a frame of two random students practicing looks like a demonstration but is
NOT one. When the people, clothing, or mat area don't match the demo pair's, reject
the cell.

Pick cells that tell the technique's story **in order**: setup → key grip/contact →
movement → finish. Use the step's `visual_cues` and `key_points` as your criteria. A
student must be able to answer from your picks: what position it starts from, which
grips/frames matter, which direction to move, what the final position looks like.

- Frames per step: intro/recap 0–1, concept 1–2, demonstration/drill 2–4 — and on
  demonstration/drill steps treat **3–4 as the target and 2 as the floor**. `check`
  prints a WARNING when a demo/drill step has fewer than 2 picks; if the sheet truly
  lacks distinct usable moments, request a resample instead of settling.
- The contact sheet is for **triage only**. Before approving a cell, open its
  full-size thumb `candidates/step_NN/<LETTER>.jpg` and confirm it (a) shows the demo
  pair and (b) matches the step's `visual_cues`. Approving straight off the sheet is
  how wrong frames slip through.
- REJECT: both people kneeling/standing idle talking to camera; resets/walking back;
  title cards or big text overlays; motion-blurred limbs; near-identical frames;
  students drilling (see above).
- `notes` is REQUIRED whenever you pick frames — `check` fails without it: one short
  phrase per chosen letter saying who is in frame and what moment it shows. If you
  cannot describe how two picks differ, you picked duplicates — you must actually
  look at each cell before choosing it.
- If the frames show burned-in subtitles, read them and use them to verify or correct
  technical terminology in your steps.json (they are ground truth for jargon that ASR
  mishears). They are normal video content, not title cards — do not reject a frame
  just because subtitle text is visible.
- If NO cell shows the technique, request a narrower resample instead of settling.

Write `<work_dir>/approved_frames.json`:
```json
{"schema_version": 1, "steps": {
  "step_01": {"frames": ["C", "F"], "notes": "C: setup grip; F: finish position"},
  "step_02": {"needs_resample": true, "window": [95.0, 110.0], "notes": "why"}}}
```
Validate: `... check --video "<video>" --what approved` (fix until PASS).
For each `needs_resample` step, run
`... candidates --video "<video>" --resample step_NN <start> <end>`, judge the NEW
contact sheet, replace that step's entry with a `frames` list, and re-validate.
After a resample the letters refer to the NEW sheet — any letters you chose from the
old sheet are meaningless and must be re-judged, even if `check` still passes.

## Stage 5 — render (script)
```
python3 "$SKILL_DIR/scripts/bjj_breakdown.py" render --video "<video file>" --embed-images
```
Render refuses to run while any `needs_resample` step is unresolved (it prints the
exact resample commands to run). Open the printed HTML path and visually confirm every
image shows the technique it sits next to. On a multi-technique video, click through
every tab and confirm each one holds its own technique's steps. If a source URL was
set in Stage 1, confirm the header shows the "▶ Watch the source video" link, every
step card shows a "▶ Watch this step" button, and frame captions show "▶ MM:SS · watch" —
spot-check one step button: its `t=` value must equal that step's
start time in whole seconds. If a frame is wrong,
edit `approved_frames.json` and re-run render — it always re-extracts, never reuses old
images. `--embed-images` makes a single self-contained HTML file you can share directly.
