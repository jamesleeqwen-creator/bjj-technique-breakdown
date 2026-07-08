import io
import json
import re
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

import bjj_breakdown as breakdown


class TranscriptHelpersTest(unittest.TestCase):
    def test_reconstruct_vtt_merges_scrolling_duplicate_lines(self):
        vtt = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.000 --> 00:00:04.000
world again
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.vtt"
            path.write_text(vtt, encoding="utf-8")
            segments = breakdown.reconstruct_vtt(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][2], "Hello world again")

    def test_parse_transcript_txt_bracket_format(self):
        # Clean-transcript .txt: "[start --> end] text" per line (M8 review fix - the
        # v1 branch was lost in the M1 copy and .txt files parsed to zero segments).
        txt = (
            "[00:00:07.160 --> 00:00:42.950] Welcome to the side control challenge.\n"
            "[00:00:42.960 --> 00:02:20.270] Lesson number one.\n"
            "not a transcript line\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.en.txt"
            path.write_text(txt, encoding="utf-8")
            segments = breakdown.parse_transcript(path)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0][0], 7.16)
        self.assertAlmostEqual(segments[0][1], 42.95)
        self.assertEqual(segments[0][2], "Welcome to the side control challenge.")
        self.assertEqual(segments[1][2], "Lesson number one.")

    def test_parse_transcript_routes_vtt_through_scroll_merge(self):
        # .vtt suffix goes through reconstruct_vtt; .txt takes the bracket-line parser.
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello world\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.en.vtt"
            path.write_text(vtt, encoding="utf-8")
            segments = breakdown.parse_transcript(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][2], "Hello world")

    def test_parse_transcript_unparseable_txt_yields_zero_segments(self):
        # prepare turns this into a clean exit-1 error rather than writing an
        # empty transcript.json (M8 review fix).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.en.txt"
            path.write_text("no timestamps here at all\n", encoding="utf-8")
            segments = breakdown.parse_transcript(path)
        self.assertEqual(segments, [])

    def test_reconstruct_vtt_skips_numeric_cue_identifiers(self):
        # SRT->VTT converters keep numeric cue ids on their own line above each
        # timestamp; they are not caption text and must not get glued onto the
        # previous cue.
        vtt = """WEBVTT

1
00:00:00.000 --> 00:00:02.000
Hello world

2
00:00:02.000 --> 00:00:04.000
world again
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "numbered.vtt"
            path.write_text(vtt, encoding="utf-8")
            segments = breakdown.reconstruct_vtt(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][2], "Hello world again")

    def test_parse_transcript_txt_skips_malformed_timestamp_lines(self):
        # The loose [0-9:]+ pattern can match a stamp parse_time_str cannot read;
        # such a line is skipped instead of crashing prepare with a ValueError.
        txt = (
            "[00:00:01.000 --> 00:00:02.000] Good line.\n"
            "[1:2:3:4.5 --> 00:00:09.000] Bad stamp line.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.en.txt"
            path.write_text(txt, encoding="utf-8")
            segments = breakdown.parse_transcript(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][2], "Good line.")


class ValidationTest(unittest.TestCase):
    def _write_json(self, tmpdir, name, data):
        path = Path(tmpdir) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_steps_validation_reports_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = self._write_json(
                tmp,
                "meta.json",
                {
                    "schema_version": 1,
                    "algo_version": 2,
                    "video_path": "/abs/video.mp4",
                    "video_base": "video",
                    "subtitles_path": "/abs/video.vtt",
                    "duration": 100.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                },
            )
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Push the elbow and swing the leg across to spin free.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                            "explanation": "Explanation prose for step one.",
                            "visual_cues": "cue",
                        },
                        {
                            "id": "step_02",
                            "title": "Two",
                            "start": 19.8,
                            "end": 18.0,
                            "category": "bad",
                            "key_points": ["a"],
                            "visual_cues": "",
                        },
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            msg = str(ctx.exception)
            self.assertIn("steps[1].end", msg)

    def test_steps_validation_rejects_nonsequential_ids_and_bad_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = self._write_json(
                tmp,
                "meta.json",
                {
                    "schema_version": 1,
                    "algo_version": 2,
                    "video_path": "/abs/video.mp4",
                    "video_base": "video",
                    "subtitles_path": "/abs/video.vtt",
                    "duration": 100.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                },
            )
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Push the elbow and swing the leg across to spin free.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                            "explanation": "Explanation prose for step one.",
                            "visual_cues": "cue",
                        },
                        {
                            "id": "step_03",
                            "title": "Two",
                            "start": 21.0,
                            "end": 25.0,
                            "category": "bad",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        },
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[1].id", str(ctx.exception))

    def test_steps_validation_rejects_missing_visual_cues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = self._write_json(
                tmp,
                "meta.json",
                {
                    "schema_version": 1,
                    "algo_version": 2,
                    "video_path": "/abs/video.mp4",
                    "video_base": "video",
                    "subtitles_path": "/abs/video.vtt",
                    "duration": 100.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                },
            )
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Push the elbow and swing the leg across to spin free.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[0].visual_cues", str(ctx.exception))

    def test_approved_validation_reports_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = self._write_json(
                tmp,
                "meta.json",
                {
                    "schema_version": 1,
                    "algo_version": 2,
                    "video_path": "/abs/video.mp4",
                    "video_base": "video",
                    "subtitles_path": "/abs/video.vtt",
                    "duration": 100.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                },
            )
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Push the elbow and swing the leg across to spin free.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                            "explanation": "Explanation prose for step one.",
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            candidates = self._write_json(
                tmp,
                "candidates.json",
                {
                    "schema_version": 1,
                    "algo_version": 3,
                    "steps": {
                        "step_01": {
                            "contact_sheet": "contact_sheets/contact_step_01.jpg",
                            "window": [10.0, 20.0],
                            "params": {"per_step": 8, "ocr_filter": True},
                            "cells": {
                                "A": {"time": 11.0, "sharpness": 1.0, "source": "uniform", "thumb": "candidates/step_01/A.jpg"}
                            },
                        }
                    },
                },
            )
            approved_unknown_step = self._write_json(
                tmp,
                "approved_frames_unknown_step.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": ["A"]},
                        "step_02": {"frames": ["A"]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_unknown_step, steps, candidates, meta)
            self.assertIn("steps[step_02]", str(ctx.exception))

            approved_bad_letter = self._write_json(
                tmp,
                "approved_frames_bad_letter.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": ["Z"]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_bad_letter, steps, candidates, meta)
            self.assertIn("steps[step_01].frames[0]", str(ctx.exception))

            approved_both = self._write_json(
                tmp,
                "approved_frames_both.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": ["A"], "needs_resample": True, "window": [0.0, 1.0]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_both, steps, candidates, meta)
            self.assertIn("either frames or needs_resample, not both", str(ctx.exception))

            approved_too_many = self._write_json(
                tmp,
                "approved_frames_too_many.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": ["A", "B", "C", "D", "E"]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_too_many, steps, candidates, meta)
            self.assertIn("steps[step_01].frames", str(ctx.exception))

            approved_duplicates = self._write_json(
                tmp,
                "approved_frames_duplicates.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": ["A", "A"]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_duplicates, steps, candidates, meta)
            self.assertIn("steps[step_01].frames[1]", str(ctx.exception))

            approved_empty = self._write_json(
                tmp,
                "approved_frames_empty.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"frames": []},
                    },
                },
            )
            approved = breakdown.validate_approved_file(approved_empty, steps, candidates, meta)
            self.assertEqual(approved["step_01"]["frames"], [])

            approved_resample_window = self._write_json(
                tmp,
                "approved_frames_resample_window.json",
                {
                    "schema_version": 1,
                    "steps": {
                        "step_01": {"needs_resample": True, "window": [0.0, 1.0]},
                    },
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_approved_file(approved_resample_window, steps, candidates, meta)
            self.assertIn("steps[step_01].window", str(ctx.exception))


def _noise_frame(seed, size=(90, 160)):
    return np.asarray(
        np.random.default_rng(seed).integers(0, 255, size=(size[0], size[1], 3)), dtype=np.uint8
    )


class HelperLogicTest(unittest.TestCase):
    def test_candidate_entry_letters_chronological_and_stale_thumbs_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            step = {"id": "step_01", "start": 0.0, "end": 30.0}
            stale = work_dir / "candidates" / "step_01" / "Z.jpg"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            samples = [
                breakdown.Sample(step_id="step_01", time=t, source="uniform", frame=_noise_frame(int(t)), sharpness=1.0)
                for t in (20.0, 5.0, 12.0)
            ]
            entry = breakdown.candidate_entry_for_step(step, samples, work_dir, ocr_filter_enabled=False, per_step=8)
            self.assertFalse(stale.exists())
            letters = list(entry["cells"].keys())
            self.assertEqual(letters, ["A", "B", "C"])
            times = [entry["cells"][letter]["time"] for letter in letters]
            self.assertEqual(times, sorted(times))
            for letter in letters:
                self.assertTrue((work_dir / entry["cells"][letter]["thumb"]).exists())
            self.assertTrue((work_dir / entry["contact_sheet"]).exists())

    def test_finalize_drops_blur_and_near_cut_keeps_chronological_order(self):
        step = {"id": "step_01", "start": 0.0, "end": 100.0}
        samples = [
            breakdown.Sample(step_id="step_01", time=float(i * 10 + 5), source="uniform", frame=_noise_frame(i), sharpness=0.0)
            for i in range(5)
        ]
        for sample in samples:
            sample.sharpness = breakdown.frame_sharpness(sample.frame)
        blurred_frame = cv2.GaussianBlur(_noise_frame(99), (15, 15), 0)
        blurred = breakdown.Sample(
            step_id="step_01", time=75.0, source="uniform",
            frame=blurred_frame, sharpness=breakdown.frame_sharpness(blurred_frame),
        )
        near_cut_frame = _noise_frame(98)
        near_cut = breakdown.Sample(
            step_id="step_01", time=60.5, source="uniform",
            frame=near_cut_frame, sharpness=breakdown.frame_sharpness(near_cut_frame),
        )
        scenes = [(60.9, 0.5)]
        kept, _ = breakdown.finalize_step_samples(step, samples + [blurred, near_cut], scenes, None, False)
        kept_times = [sample.time for sample in kept]
        self.assertNotIn(75.0, kept_times)
        self.assertNotIn(60.5, kept_times)
        self.assertEqual(kept_times, sorted(kept_times))
        self.assertGreaterEqual(len(kept), breakdown.MIN_CANDIDATES_PER_STEP)

    def test_finalize_returns_empty_when_everything_filtered(self):
        step = {"id": "step_01", "start": 0.0, "end": 2.0}
        frame = _noise_frame(1)
        sample = breakdown.Sample(step_id="step_01", time=1.0, source="uniform", frame=frame, sharpness=breakdown.frame_sharpness(frame))
        kept, _ = breakdown.finalize_step_samples(step, [sample], [(1.2, 0.5)], None, False)
        self.assertEqual(kept, [])

    def test_dhash_distance(self):
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2 = img1.copy()
        img3 = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        hash1 = breakdown.dhash_image(img1)
        hash2 = breakdown.dhash_image(img2)
        hash3 = breakdown.dhash_image(img3)
        self.assertEqual(breakdown.hamming_distance(hash1, hash2), 0)
        self.assertGreater(breakdown.hamming_distance(hash1, hash3), breakdown.DHASH_HAMMING_DUP)

    def test_uniform_sample_times_are_inside_window(self):
        times = breakdown.build_uniform_sample_times(10.0, 20.0, 8)
        self.assertTrue(all(10.0 < t < 20.0 for t in times))

    def test_format_time(self):
        self.assertEqual(breakdown.format_time(83), "00:01:23")

    def test_cache_entry_validity(self):
        entry = {
            "algo_version": 2,
            "window": [10.0, 20.0],
            "params": {"per_step": 8, "ocr_filter": True},
        }
        self.assertTrue(breakdown.cache_entry_valid(entry, [10.0, 20.0], {"per_step": 8, "ocr_filter": True}, 2))
        self.assertFalse(breakdown.cache_entry_valid(entry, [10.0, 21.0], {"per_step": 8, "ocr_filter": True}, 2))
        self.assertFalse(breakdown.cache_entry_valid(entry, [10.0, 20.0], {"per_step": 4, "ocr_filter": True}, 2))
        self.assertFalse(breakdown.cache_entry_valid(entry, [10.0, 20.0], {"per_step": 8, "ocr_filter": False}, 2))
        self.assertFalse(breakdown.cache_entry_valid(entry, [10.0, 20.0], {"per_step": 8, "ocr_filter": True}, 3))


class ErrorContractTest(unittest.TestCase):
    META = {
        "schema_version": 1,
        "algo_version": 2,
        "video_path": "/abs/video.mp4",
        "video_base": "video",
        "subtitles_path": "/abs/video.vtt",
        "duration": 100.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
    }

    def test_load_json_missing_file_is_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.load_json(Path(tmp) / "steps.json")
            message = str(ctx.exception)
            self.assertIn("steps.json", message)
            self.assertIn("SKILL.md", message)

    def test_check_missing_steps_json_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            work = Path(tmp) / "video_review"
            work.mkdir()
            (work / "meta.json").write_text(json.dumps(self.META), encoding="utf-8")
            code = breakdown.main(["check", "--video", str(video), "--what", "steps"])
            self.assertEqual(code, 2)

    def test_check_valid_steps_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            work = Path(tmp) / "video_review"
            work.mkdir()
            (work / "meta.json").write_text(json.dumps(self.META), encoding="utf-8")
            steps = {
                "schema_version": 2,
                "video_base": "video",
                "one_thing": "Push the elbow and swing the leg across to spin free.",
                "steps": [
                    {
                        "id": "step_01",
                        "title": "One",
                        "start": 10.0,
                        "end": 20.0,
                        "category": "concept",
                        "key_points": ["a"],
                        "explanation": "Explanation prose for step one.",
                        "visual_cues": "cue",
                    }
                ],
            }
            (work / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
            code = breakdown.main(["check", "--video", str(video), "--what", "steps"])
            self.assertEqual(code, 0)


class GroundingLintTest(unittest.TestCase):
    def _transcript(self, tmp):
        path = Path(tmp) / "transcript.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 10.0,
                            "text": "push his elbow and swing your leg around the waist then spin to face your partner",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_warns_on_fabricated_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            fabricated = [
                {
                    "id": "step_01",
                    "title": "Build the hip escape frame",
                    "start": 0.0,
                    "end": 10.0,
                    "category": "concept",
                    "key_points": ["Connect foot hip shoulder into a single solid unit."],
                    "visual_cues": "x",
                }
            ]
            warnings = breakdown.step_grounding_warnings(fabricated, transcript)
            self.assertTrue(warnings)
            self.assertIn("step_01", warnings[0])
            self.assertTrue(any(w.startswith("overall:") for w in warnings))

    def test_no_warning_for_grounded_paraphrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            grounded = [
                {
                    "id": "step_01",
                    "title": "Swing the leg around the waist and spin out",
                    "start": 0.0,
                    "end": 10.0,
                    "category": "demonstration",
                    "key_points": ["Push his elbow, swing your leg around his waist, and spin to face your partner."],
                    "visual_cues": "x",
                }
            ]
            self.assertEqual(breakdown.step_grounding_warnings(grounded, transcript), [])


# --- M6 additions (spec §12, tests 10-15) ---------------------------------------------


class StepsSchemaV2Test(unittest.TestCase):
    META = {
        "schema_version": 1,
        "algo_version": 3,
        "video_path": "/abs/video.mp4",
        "video_base": "video",
        "subtitles_path": "/abs/video.vtt",
        "duration": 100.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
    }

    def _write_json(self, tmpdir, name, data):
        path = Path(tmpdir) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_schema_v1_file_fails_with_upgrade_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 1,
                    "video_base": "video",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            message = str(ctx.exception)
            self.assertIn("schema_version 1", message)
            self.assertIn("schema_version 2", message)
            self.assertIn("one_thing", message)
            self.assertIn("explanation", message)
            self.assertIn("SKILL.md", message)

    def test_missing_one_thing_error_names_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "intro",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("one_thing", str(ctx.exception))

    def test_missing_explanation_on_demonstration_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Spin to face your partner to twist his wrist.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "demonstration",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[0].explanation", str(ctx.exception))

    def test_intro_step_without_explanation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            steps = self._write_json(
                tmp,
                "steps.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Spin to face your partner to twist his wrist.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "intro",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            result = breakdown.validate_steps_file(steps, meta)
            self.assertEqual(len(result), 1)

    def test_over_length_one_thing_and_explanation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            too_long_one_thing = self._write_json(
                tmp,
                "steps_long_one_thing.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "x" * (breakdown.ONE_THING_MAX_CHARS + 1),
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "intro",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(too_long_one_thing, meta)
            self.assertIn("one_thing", str(ctx.exception))

            too_long_explanation = self._write_json(
                tmp,
                "steps_long_explanation.json",
                {
                    "schema_version": 2,
                    "video_base": "video",
                    "one_thing": "Spin to face your partner to twist his wrist.",
                    "steps": [
                        {
                            "id": "step_01",
                            "title": "One",
                            "start": 10.0,
                            "end": 20.0,
                            "category": "concept",
                            "key_points": ["a"],
                            "visual_cues": "cue",
                            "explanation": "x" * (breakdown.EXPLANATION_MAX_CHARS + 1),
                        }
                    ],
                },
            )
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(too_long_explanation, meta)
            self.assertIn("steps[0].explanation", str(ctx.exception))


class GroundingLintV2Test(unittest.TestCase):
    def _transcript(self, tmp):
        path = Path(tmp) / "transcript.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 10.0,
                            "text": "push his elbow and swing your leg around the waist then spin to face your partner",
                        },
                        {
                            "start": 10.0,
                            "end": 20.0,
                            "text": "the spin itself is what twists his wrist and breaks the grip so commit to it",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_explanation_grounds_step_with_weak_title_and_key_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            step = {
                "id": "step_01",
                "title": "Do the move correctly",
                "start": 0.0,
                "end": 10.0,
                "category": "demonstration",
                "key_points": ["Execute the technique with good form."],
                "explanation": "Push his elbow and swing your leg around his waist as you spin to face your partner.",
                "visual_cues": "x",
            }
            warnings = breakdown.step_grounding_warnings([step], transcript)
            self.assertEqual(warnings, [])

    def test_fabricated_one_thing_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            warning = breakdown.one_thing_grounding_warning(
                "Always maintain perfect posture and breathe from your diaphragm.", transcript
            )
            self.assertIsNotNone(warning)
            self.assertIn("one_thing", warning)

    def test_grounded_one_thing_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            warning = breakdown.one_thing_grounding_warning(
                "Spin to face your partner - the spin itself twists his wrist and breaks the grip.", transcript
            )
            self.assertIsNone(warning)


class TranscriptAssignmentTest(unittest.TestCase):
    def test_long_segment_spans_two_adjacent_steps(self):
        segments = [{"start": 10.0, "end": 40.0, "text": "long segment spanning two steps"}]
        first = breakdown.transcript_segments_for_step(segments, 0.0, 25.0)
        second = breakdown.transcript_segments_for_step(segments, 25.0, 50.0)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_weak_overlap_included_only_via_safety_net(self):
        # overlap = 1s; segment duration 11s (33% needs 3.63s); step duration 30s (50% needs 15s).
        # Neither threshold is met, but it is the only segment that overlaps at all.
        segments = [{"start": 90.0, "end": 101.0, "text": "brief trailing word"}]
        result = breakdown.transcript_segments_for_step(segments, 100.0, 130.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "brief trailing word")

    def test_no_overlap_yields_empty_transcript(self):
        segments = [{"start": 0.0, "end": 5.0, "text": "unrelated intro chatter"}]
        result = breakdown.transcript_segments_for_step(segments, 100.0, 130.0)
        self.assertEqual(result, [])

    def test_v1_regression_step_04_gets_transcript(self):
        # The exact v1 regression case from the spec: segment [96.0, 125.3] against step
        # [93.8, 106.5] must yield a non-empty transcript.
        segments = [{"start": 96.0, "end": 125.3, "text": "his shoulder, you get the other leg across"}]
        result = breakdown.transcript_segments_for_step(segments, 93.8, 106.5)
        self.assertTrue(result)


class CaptionMarkerTest(unittest.TestCase):
    def test_speaker_change_markers_stripped_and_spaces_collapsed(self):
        raw = ">> See, he was styling to keep holding the trouser."
        cleaned = breakdown.clean_text(raw)
        self.assertNotIn(">>", cleaned)
        self.assertNotIn("  ", cleaned)
        self.assertEqual(cleaned, "See, he was styling to keep holding the trouser.")


class MidWordRepairTest(unittest.TestCase):
    def test_repairs_single_letter_split(self):
        segments = [(0.0, 2.0, "Okay, t"), (2.0, 4.0, "his shoulder")]
        result = breakdown.repair_midword_splits(segments)
        self.assertEqual(result[0][2], "Okay,")
        self.assertEqual(result[1][2], "this shoulder")

    def test_two_letter_fragment_left_alone(self):
        segments = [(0.0, 2.0, "went to"), (2.0, 4.0, "the mount")]
        result = breakdown.repair_midword_splits(segments)
        self.assertEqual(result[0][2], "went to")
        self.assertEqual(result[1][2], "the mount")

    def test_standalone_a_left_alone(self):
        segments = [(0.0, 2.0, "grab a"), (2.0, 4.0, "handle now")]
        result = breakdown.repair_midword_splits(segments)
        self.assertEqual(result[0][2], "grab a")
        self.assertEqual(result[1][2], "handle now")

    def test_standalone_capital_i_left_alone(self):
        segments = [(0.0, 2.0, "and I"), (2.0, 4.0, "will show you")]
        result = breakdown.repair_midword_splits(segments)
        self.assertEqual(result[0][2], "and I")
        self.assertEqual(result[1][2], "will show you")

    def test_no_repair_when_next_segment_starts_uppercase(self):
        segments = [(0.0, 2.0, "Okay, t"), (2.0, 4.0, "His shoulder")]
        result = breakdown.repair_midword_splits(segments)
        self.assertEqual(result[0][2], "Okay, t")
        self.assertEqual(result[1][2], "His shoulder")


class RenderStructureTest(unittest.TestCase):
    @staticmethod
    def _stub_run_command(cmd):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"stub-jpeg-bytes")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    def _build(self, tmp):
        meta = {"duration": 100.0}
        steps = [
            {
                "id": "step_01",
                "title": "Push the Frame",
                "start": 0.0,
                "end": 20.0,
                "category": "concept",
                "key_points": ["Push the elbow into space."],
                "explanation": "This creates room because it moves his weight off your hip.",
                "visual_cues": "x",
            },
        ]
        candidates = {
            "step_01": {"cells": {"A": {"time": 5.0, "sharpness": 1.0, "source": "uniform", "thumb": "x"}}},
        }
        approved = {"step_01": {"frames": ["A"]}}
        transcript_segments = [{"start": 0.0, "end": 20.0, "text": "push the elbow into space to create room"}]
        images_dir = Path(tmp) / "video_review" / "images"
        images_dir.mkdir(parents=True)
        video_path = Path(tmp) / "video.mp4"
        original = breakdown.run_command
        breakdown.run_command = self._stub_run_command
        try:
            result = breakdown.build_render_html(
                video_path,
                meta,
                steps,
                "Spin to face your partner to twist his wrist and break the grip.",
                candidates,
                approved,
                transcript_segments,
                images_dir,
                False,
                "ffmpeg",
            )
        finally:
            breakdown.run_command = original
        return result

    def test_one_thing_hero_before_toc(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp)
        html = result["html"]
        self.assertLess(html.index('class="one-thing"'), html.index("Table of Contents"))

    def test_card_order_key_points_then_images_then_explanation_then_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp)
        html = result["html"]
        kp_idx = html.index('class="key-points"')
        strip_idx = html.index('class="image-strip"')
        expl_idx = html.index('class="explanation"')
        details_idx = html.index("Verbatim transcript (may overlap adjacent steps)")
        self.assertLess(kp_idx, strip_idx)
        self.assertLess(strip_idx, expl_idx)
        self.assertLess(expl_idx, details_idx)

    def test_details_summary_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp)
        self.assertIn("<summary>Verbatim transcript (may overlap adjacent steps)</summary>", result["html"])

    def test_markdown_one_thing_blockquote_before_toc(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp)
        md = result["md"]
        self.assertLess(md.index("> **If you remember one thing:**"), md.index("## Table of Contents"))


class SubtitleDiscoveryTest(unittest.TestCase):
    """M7 test 16 (§12). video_base deliberately contains `[`, `]`, and spaces - these are
    glob metacharacters, so a correct implementation must iterate the directory rather
    than glob (§15)."""

    def _touch(self, tmp, name):
        path = Path(tmp) / name
        path.touch()
        return path

    def test_explicit_subtitles_wins_even_with_better_priority_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Seated Guard Recovery [abc123]"
            self._touch(tmp, f"{base}.en.txt")
            explicit = self._touch(tmp, "custom_subs.vtt")
            video_path = Path(tmp) / f"{base}.mp4"
            result = breakdown.subtitle_path_from_video(video_path, str(explicit))
            self.assertEqual(result, explicit.resolve())

    def test_priority_order_en_txt_beats_en_vtt_beats_en_us_beats_bare_beats_other_lang(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Seated Guard Recovery [abc123]"
            en_txt = self._touch(tmp, f"{base}.en.txt")
            self._touch(tmp, f"{base}.en.vtt")
            self._touch(tmp, f"{base}.en-US.vtt")
            self._touch(tmp, f"{base}.vtt")
            self._touch(tmp, f"{base}.zh-Hans.vtt")
            video_path = Path(tmp) / f"{base}.mp4"
            chosen = breakdown.subtitle_path_from_video(video_path, None)
            self.assertEqual(chosen, en_txt.resolve())

    def test_bare_vtt_found_when_no_en_variant_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Whisper Output [xyz789]"
            bare = self._touch(tmp, f"{base}.vtt")
            self._touch(tmp, f"{base}.zh-Hans.vtt")
            video_path = Path(tmp) / f"{base}.mp4"
            chosen = breakdown.subtitle_path_from_video(video_path, None)
            self.assertEqual(chosen, bare.resolve())

    def test_same_priority_language_tags_pick_alphabetically_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Multi Lang [q1]"
            fr = self._touch(tmp, f"{base}.fr.vtt")
            self._touch(tmp, f"{base}.zh-Hans.vtt")
            video_path = Path(tmp) / f"{base}.mp4"
            chosen = breakdown.subtitle_path_from_video(video_path, None)
            self.assertEqual(chosen, fr.resolve())

    def test_unrelated_prefix_and_other_extensions_never_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Solo Video [q2]"
            self._touch(tmp, f"{base} extra.vtt")
            self._touch(tmp, f"{base}.srt")
            self._touch(tmp, f"{base}.json")
            video_path = Path(tmp) / f"{base}.mp4"
            with self.assertRaises(FileNotFoundError):
                breakdown.subtitle_path_from_video(video_path, None)

    def test_not_found_error_names_directory_patterns_and_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Missing Subs [q3]"
            video_path = Path(tmp) / f"{base}.mp4"
            with self.assertRaises(FileNotFoundError) as ctx:
                breakdown.subtitle_path_from_video(video_path, None)
            message = str(ctx.exception)
            self.assertIn(tmp, message)
            self.assertIn(f"{base}.en.txt", message)
            self.assertIn(f"{base}.vtt", message)
            self.assertIn("Stage 0", message)
            self.assertIn("Stage 1.5", message)

    def test_multi_candidate_prints_chosen_and_alternatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = "Announce Me [q4]"
            self._touch(tmp, f"{base}.en.vtt")
            self._touch(tmp, f"{base}.zh-Hans.vtt")
            video_path = Path(tmp) / f"{base}.mp4"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                breakdown.subtitle_path_from_video(video_path, None)
            output = buffer.getvalue()
            self.assertIn(f"{base}.en.vtt", output)
            self.assertIn(f"{base}.zh-Hans.vtt", output)


class CJKGroundingTokensTest(unittest.TestCase):
    """M7 test 17 (§12)."""

    def test_english_text_below_threshold_tokenizes_exactly_as_before(self):
        text = "Push the elbow and swing your leg around the waist then spin to face your partner."
        expected = {
            tok
            for tok in re.findall(r"[a-z']+", text.lower())
            if len(tok) > 2 and tok not in breakdown.GROUNDING_STOPWORDS
        }
        self.assertLessEqual(breakdown._cjk_text_fraction(text), breakdown.CJK_TEXT_FRAC)
        self.assertEqual(breakdown._grounding_tokens(text), expected)

    def test_chinese_text_tokenizes_to_character_bigrams(self):
        text = "把手往前推"
        self.assertGreater(breakdown._cjk_text_fraction(text), breakdown.CJK_TEXT_FRAC)
        tokens = breakdown._cjk_aware_tokens(text)
        self.assertEqual(tokens, {"把手", "手往", "往前", "前推"})
        self.assertEqual(breakdown._grounding_tokens(text), tokens)

    def _chinese_transcript(self, tmp):
        path = Path(tmp) / "transcript.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segments": [
                        {"start": 0.0, "end": 10.0, "text": "把对方的手臂往上推然后转身面向对手"},
                        {"start": 10.0, "end": 20.0, "text": "转身的力量能把他的手腕别断所以要坚持转身"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_grounded_chinese_steps_fixture_produces_no_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._chinese_transcript(tmp)
            step = {
                "id": "step_01",
                "title": "把对方手臂往上推",
                "start": 0.0,
                "end": 10.0,
                "category": "demonstration",
                "key_points": ["把对方的手臂往上推", "转身面向对手"],
                "explanation": "转身面向对手之后完成动作。",
                "visual_cues": "x",
            }
            warnings = breakdown.step_grounding_warnings([step], transcript)
            self.assertEqual(warnings, [])

    def test_fabricated_chinese_steps_fixture_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._chinese_transcript(tmp)
            step = {
                "id": "step_01",
                "title": "今天天气很好",
                "start": 0.0,
                "end": 10.0,
                "category": "demonstration",
                "key_points": ["适合户外运动"],
                "explanation": "今天天气很好适合户外运动。",
                "visual_cues": "x",
            }
            warnings = breakdown.step_grounding_warnings([step], transcript)
            self.assertTrue(any("step_01" in w for w in warnings))

    def test_mixed_chinese_with_latin_jargon_produces_bigrams_and_words(self):
        text = "这是 half guard 的一个动作"
        tokens = breakdown._cjk_aware_tokens(text)
        self.assertEqual(breakdown._grounding_tokens(text), tokens)
        self.assertIn("half", tokens)
        self.assertIn("guard", tokens)
        bigram_tokens = {tok for tok in tokens if breakdown._is_bigram_token(tok)}
        self.assertTrue(bigram_tokens)
        self.assertEqual(tokens, bigram_tokens | {"half", "guard"})


# --- M8 additions (spec §12, tests 18-19) ----------------------------------------------


class TechniquesValidationTest(unittest.TestCase):
    """M8 test 18 (§12): steps.json techniques + per-step technique field, §6.4 rules."""

    META = {
        "schema_version": 1,
        "algo_version": 3,
        "video_path": "/abs/video.mp4",
        "video_base": "video",
        "subtitles_path": "/abs/video.vtt",
        "duration": 200.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
    }

    def _write_json(self, tmpdir, name, data):
        path = Path(tmpdir) / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _step(self, n, start, end, technique=None, category="concept"):
        step = {
            "id": f"step_{n:02d}",
            "title": f"Step {n}",
            "start": start,
            "end": end,
            "category": category,
            "key_points": ["A key point."],
            "explanation": "Explanation prose for this step.",
            "visual_cues": "cue",
        }
        if technique is not None:
            step["technique"] = technique
        return step

    def _grouped_doc(self):
        return {
            "schema_version": 2,
            "video_base": "video",
            "one_thing": "Spin to face your partner to twist his wrist and break the grip.",
            "techniques": [
                {"id": "tech_01", "title": "Leg-swing grip escape", "summary": "Free the sleeve and spin to face him."},
                {"id": "tech_02", "title": "Second option"},
            ],
            "steps": [
                self._step(1, 0.0, 10.0, "tech_01"),
                self._step(2, 10.0, 20.0, "tech_01"),
                self._step(3, 20.0, 30.0, "tech_02"),
            ],
        }

    def test_grouped_fixture_with_contiguous_membership_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            steps = self._write_json(tmp, "steps.json", self._grouped_doc())
            result = breakdown.validate_steps_file(steps, meta)
            self.assertEqual(len(result), 3)

    def test_step_missing_technique_errors_on_step_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            del doc["steps"][1]["technique"]
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[1].technique", str(ctx.exception))

    def test_unknown_technique_id_errors_on_step_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            doc["steps"][0]["technique"] = "tech_99"
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[0].technique", str(ctx.exception))

    def test_noncontiguous_membership_errors_on_first_out_of_order_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            # tech_01, tech_02, tech_01 - the third step is the first one out of order.
            doc["steps"][1]["technique"] = "tech_02"
            doc["steps"][2]["technique"] = "tech_01"
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[2].technique", str(ctx.exception))

    def test_technique_owning_zero_steps_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            doc["techniques"].append({"id": "tech_03", "title": "Unused technique"})
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("techniques[2]", str(ctx.exception))

    def test_step_with_technique_while_techniques_absent_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = {
                "schema_version": 2,
                "video_base": "video",
                "one_thing": "Spin to face your partner to twist his wrist and break the grip.",
                "steps": [self._step(1, 0.0, 10.0, "tech_01")],
            }
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[0].technique", str(ctx.exception))

    def test_non_string_technique_rejected_cleanly(self):
        # An unhashable value (e.g. a list) must surface as a ValidationError naming
        # the JSON path, never as a raw TypeError from the id-lookup dict.
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            doc["steps"][0]["technique"] = ["tech_01"]
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("steps[0].technique", str(ctx.exception))

    def test_nonsequential_technique_ids_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            doc["techniques"][1]["id"] = "tech_03"
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("techniques[1].id", str(ctx.exception))

    def test_overlength_title_and_summary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc_title = self._grouped_doc()
            doc_title["techniques"][0]["title"] = "x" * (breakdown.TECHNIQUE_TITLE_MAX_CHARS + 1)
            steps_title = self._write_json(tmp, "steps_title.json", doc_title)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps_title, meta)
            self.assertIn("techniques[0].title", str(ctx.exception))

            doc_summary = self._grouped_doc()
            doc_summary["techniques"][0]["summary"] = "x" * (breakdown.TECHNIQUE_SUMMARY_MAX_CHARS + 1)
            steps_summary = self._write_json(tmp, "steps_summary.json", doc_summary)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps_summary, meta)
            self.assertIn("techniques[0].summary", str(ctx.exception))

    def test_empty_string_summary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = self._grouped_doc()
            doc["techniques"][0]["summary"] = "   "
            steps = self._write_json(tmp, "steps.json", doc)
            with self.assertRaises(breakdown.ValidationError) as ctx:
                breakdown.validate_steps_file(steps, meta)
            self.assertIn("techniques[0].summary", str(ctx.exception))

    def test_existing_ungrouped_v2_fixture_still_passes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._write_json(tmp, "meta.json", self.META)
            doc = {
                "schema_version": 2,
                "video_base": "video",
                "one_thing": "Spin to face your partner to twist his wrist and break the grip.",
                "steps": [self._step(1, 0.0, 10.0)],
            }
            steps = self._write_json(tmp, "steps.json", doc)
            result = breakdown.validate_steps_file(steps, meta)
            self.assertEqual(len(result), 1)


class TechniqueGroundingLintTest(unittest.TestCase):
    """M8 technique grounding lint (§5): title+summary pooled, checked against the
    technique's window (earliest step start to latest step end)."""

    def _transcript(self, tmp):
        path = Path(tmp) / "transcript.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segments": [
                        {"start": 0.0, "end": 10.0, "text": "grip the sleeve tight and control the arm"},
                        {
                            "start": 10.0,
                            "end": 20.0,
                            "text": "swing your leg across to create the spin and break the grip",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _steps(self):
        return [
            {"id": "step_01", "start": 0.0, "end": 10.0, "technique": "tech_01"},
            {"id": "step_02", "start": 10.0, "end": 20.0, "technique": "tech_01"},
        ]

    def test_fabricated_technique_summary_warns_naming_the_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            steps_doc = {
                "techniques": [
                    {"id": "tech_01", "title": "Totally unrelated", "summary": "Discussing nutrition and sleep schedules."}
                ]
            }
            warnings = breakdown.technique_grounding_warnings(steps_doc, self._steps(), transcript)
            self.assertTrue(warnings)
            self.assertIn("tech_01", warnings[0])

    def test_grounded_technique_summary_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = self._transcript(tmp)
            steps_doc = {
                "techniques": [
                    {
                        "id": "tech_01",
                        "title": "Grip and swing escape",
                        "summary": "Grip the sleeve, then swing your leg across to break the grip.",
                    }
                ]
            }
            warnings = breakdown.technique_grounding_warnings(steps_doc, self._steps(), transcript)
            self.assertEqual(warnings, [])


class TabbedRenderStructureTest(unittest.TestCase):
    """M8 test 19 (§12): tab bar / panels / per-panel TOC / MD Part mirror; regression
    guard that 0-or-1 technique renders with no tab markup at all."""

    @staticmethod
    def _stub_run_command(cmd):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"stub-jpeg-bytes")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    def _steps(self):
        return [
            {
                "id": "step_01",
                "title": "Grip the sleeve",
                "start": 0.0,
                "end": 20.0,
                "category": "concept",
                "key_points": ["Grip the sleeve tight."],
                "explanation": "This grip controls the arm before the spin.",
                "visual_cues": "x",
                "technique": "tech_01",
            },
            {
                "id": "step_02",
                "title": "Swing the leg across",
                "start": 20.0,
                "end": 40.0,
                "category": "demonstration",
                "key_points": ["Swing the leg across the waist."],
                "explanation": "The swing creates the rotation that breaks the grip.",
                "visual_cues": "x",
                "technique": "tech_01",
            },
            {
                "id": "step_03",
                "title": "Set up the second option",
                "start": 100.0,
                "end": 120.0,
                "category": "concept",
                "key_points": ["Frame against the far hip."],
                "explanation": "This frame sets up the alternate escape.",
                "visual_cues": "x",
                "technique": "tech_02",
            },
        ]

    def _techniques(self):
        return [
            {"id": "tech_01", "title": "Leg-swing grip escape", "summary": "Free the sleeve and spin to face him."},
            {"id": "tech_02", "title": "Frame escape"},
        ]

    def _build(self, tmp, techniques):
        meta = {"duration": 200.0}
        steps = self._steps()
        candidates = {
            step["id"]: {"cells": {"A": {"time": step["start"] + 1.0, "sharpness": 1.0, "source": "uniform", "thumb": "x"}}}
            for step in steps
        }
        approved = {step["id"]: {"frames": ["A"]} for step in steps}
        transcript_segments = [
            {"start": step["start"], "end": step["end"], "text": step["key_points"][0]} for step in steps
        ]
        images_dir = Path(tmp) / "video_review" / "images"
        images_dir.mkdir(parents=True)
        video_path = Path(tmp) / "video.mp4"
        original = breakdown.run_command
        breakdown.run_command = self._stub_run_command
        try:
            result = breakdown.build_render_html(
                video_path,
                meta,
                steps,
                "Spin to face your partner to twist his wrist and break the grip.",
                candidates,
                approved,
                transcript_segments,
                images_dir,
                False,
                "ffmpeg",
                techniques=techniques,
            )
        finally:
            breakdown.run_command = original
        return result

    def test_tab_bar_after_one_thing_and_before_first_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, self._techniques())
        html = result["html"]
        one_thing_idx = html.index('class="one-thing"')
        tabs_idx = html.index('class="tech-tabs"')
        panel_idx = html.index('class="tech-panel"')
        self.assertLess(one_thing_idx, tabs_idx)
        self.assertLess(tabs_idx, panel_idx)

    def test_one_panel_per_technique_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, self._techniques())
        html = result["html"]
        self.assertEqual(html.count('class="tech-panel"'), 2)
        idx1 = html.index('id="tech_01"')
        idx2 = html.index('id="tech_02"')
        self.assertLess(idx1, idx2)

    def test_step_card_sits_inside_its_own_technique_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, self._techniques())
        html = result["html"]
        panel1_start = html.index('id="tech_01"')
        panel2_start = html.index('id="tech_02"')
        section01 = html.index('id="section-01"')
        section02 = html.index('id="section-02"')
        section03 = html.index('id="section-03"')
        self.assertTrue(panel1_start < section01 < panel2_start)
        self.assertTrue(panel1_start < section02 < panel2_start)
        self.assertGreater(section03, panel2_start)

    def test_panel_summary_precedes_its_first_step_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, self._techniques())
        html = result["html"]
        summary_idx = html.index('class="tech-summary"')
        section01_idx = html.index('id="section-01"')
        self.assertLess(summary_idx, section01_idx)

    def test_markdown_has_part_headers_demoted_steps_and_nested_toc(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, self._techniques())
        md = result["md"]
        self.assertIn("## Part 1:", md)
        self.assertIn("### 01.", md)
        self.assertIn("  - ", md)  # nested TOC: technique line, then indented step links

    def test_ungrouped_fixture_has_no_tab_markup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, None)
        self.assertNotIn("tech-tabs", result["html"])
        self.assertNotIn("tech-panel", result["html"])
        self.assertNotIn("tech-tabs", result["md"])
        self.assertNotIn("tech-panel", result["md"])

    def test_one_entry_techniques_list_renders_without_tabs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, [self._techniques()[0]])
        self.assertNotIn("tech-tabs", result["html"])
        self.assertNotIn("tech-panel", result["html"])
        self.assertNotIn("tech-tabs", result["md"])
        self.assertNotIn("tech-panel", result["md"])


class ApprovedJudgingRobustnessTest(unittest.TestCase):
    """Spec test 21 (M9): enforced notes + the shallow-judging warn lint."""

    def _steps(self, category="demonstration"):
        return [
            {
                "id": "step_01",
                "title": "One",
                "start": 10.0,
                "end": 20.0,
                "category": category,
                "key_points": ["a"],
                "explanation": "Explanation prose for step one.",
                "visual_cues": "cue",
            }
        ]

    def _candidates(self, letters=("A", "B", "C")):
        return {
            "step_01": {
                "contact_sheet": "contact_sheets/contact_step_01.jpg",
                "window": [10.0, 20.0],
                "params": {"per_step": 8, "ocr_filter": True},
                "cells": {
                    letter: {
                        "time": 11.0 + i,
                        "sharpness": 1.0,
                        "source": "uniform",
                        "thumb": f"candidates/step_01/{letter}.jpg",
                    }
                    for i, letter in enumerate(letters)
                },
            }
        }

    def _validate(self, entry):
        data = {"schema_version": 1, "steps": {"step_01": entry}}
        return breakdown.validate_approved_data(data, self._steps(), self._candidates())

    def test_frames_without_notes_rejected(self):
        for bad_notes in (None, "", "   "):
            entry = {"frames": ["A"]}
            if bad_notes is not None:
                entry["notes"] = bad_notes
            with self.assertRaises(breakdown.ValidationError) as ctx:
                self._validate(entry)
            self.assertIn("steps[step_01].notes", str(ctx.exception))

    def test_frames_with_notes_pass(self):
        self._validate({"frames": ["A", "B"], "notes": "A: setup grip; B: finish position"})

    def test_non_string_letter_rejected_cleanly(self):
        # A nested list is unhashable - must surface as a ValidationError naming the
        # JSON path, never as a raw TypeError from the duplicate-letter set.
        with self.assertRaises(breakdown.ValidationError) as ctx:
            self._validate({"frames": [["A"]], "notes": "A: setup"})
        self.assertIn("steps[step_01].frames[0]", str(ctx.exception))

    def test_empty_frames_without_notes_pass(self):
        self._validate({"frames": []})

    def test_needs_resample_without_notes_passes(self):
        self._validate({"needs_resample": True, "window": [10.0, 15.0]})

    def _warnings(self, category, entry):
        doc = {"schema_version": 1, "steps": {"step_01": entry}}
        return breakdown.approved_frames_warnings(self._steps(category), doc)

    def test_demonstration_with_one_frame_warns(self):
        warnings = self._warnings("demonstration", {"frames": ["A"], "notes": "A: setup"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("step_01", warnings[0])
        self.assertIn("demonstration", warnings[0])
        self.assertIn("3-4", warnings[0])

    def test_drill_with_zero_frames_warns(self):
        warnings = self._warnings("drill", {"frames": []})
        self.assertEqual(len(warnings), 1)
        self.assertIn("drill", warnings[0])

    def test_drill_with_three_frames_no_warning(self):
        warnings = self._warnings("drill", {"frames": ["A", "B", "C"], "notes": "A: a; B: b; C: c"})
        self.assertEqual(warnings, [])

    def test_concept_with_one_frame_no_warning(self):
        warnings = self._warnings("concept", {"frames": ["A"], "notes": "A: the position"})
        self.assertEqual(warnings, [])

    def test_needs_resample_demonstration_no_warning(self):
        warnings = self._warnings("demonstration", {"needs_resample": True, "window": [10.0, 15.0]})
        self.assertEqual(warnings, [])

    def _multi_step_warnings(self, counts):
        letters = ["A", "B", "C", "D"]
        steps = []
        entries = {}
        for i, count in enumerate(counts, start=1):
            sid = f"step_{i:02d}"
            steps.append(
                {
                    "id": sid,
                    "title": f"Step {i}",
                    "start": float(i * 10),
                    "end": float(i * 10 + 9),
                    "category": "demonstration",
                    "key_points": ["a"],
                    "explanation": "Explanation prose.",
                    "visual_cues": "cue",
                }
            )
            picked = letters[:count]
            entries[sid] = {"frames": picked, "notes": "; ".join(f"{l}: moment" for l in picked) or None}
        return breakdown.approved_frames_warnings(steps, {"schema_version": 1, "steps": entries})

    def test_median_of_two_everywhere_warns_once_at_file_level(self):
        warnings = self._multi_step_warnings([2, 2, 2, 2, 2])
        file_level = [w for w in warnings if "median" in w]
        self.assertEqual(len(file_level), 1)
        self.assertIn("shallow judging pass", file_level[0])

    def test_median_three_no_file_warning(self):
        warnings = self._multi_step_warnings([2, 3, 3])
        self.assertEqual([w for w in warnings if "median" in w], [])

    def test_single_demo_step_no_median_warning(self):
        warnings = self._multi_step_warnings([2])
        self.assertEqual([w for w in warnings if "median" in w], [])


class ResampleArgsTest(unittest.TestCase):
    """`--resample STEP_ID START END` is validated before any video work; bad values
    used to reach an unguarded float() and crash with a raw traceback."""

    STEPS = [{"id": "step_01", "start": 10.0, "end": 20.0}]

    def test_absent_returns_none(self):
        self.assertIsNone(breakdown.parse_resample_args(None, self.STEPS))

    def test_valid_args_parse(self):
        step_id, window = breakdown.parse_resample_args(["step_01", "12", "18.5"], self.STEPS)
        self.assertEqual(step_id, "step_01")
        self.assertEqual(window, (12.0, 18.5))

    def test_non_numeric_raises_clean_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            breakdown.parse_resample_args(["step_01", "abc", "18"], self.STEPS)
        self.assertIn("must be numbers", str(ctx.exception))

    def test_inverted_window_rejected(self):
        with self.assertRaises(ValueError):
            breakdown.parse_resample_args(["step_01", "18", "12"], self.STEPS)

    def test_negative_start_rejected(self):
        with self.assertRaises(ValueError):
            breakdown.parse_resample_args(["step_01", "-3", "12"], self.STEPS)

    def test_unknown_step_id_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            breakdown.parse_resample_args(["step_99", "12", "18"], self.STEPS)
        self.assertIn("step_99", str(ctx.exception))


class TimestampedUrlTest(unittest.TestCase):
    def test_youtube_watch_url_keeps_params_and_gains_t(self):
        url = breakdown.timestamped_url("https://www.youtube.com/watch?v=abc123&foo=bar", 83.4)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.assertEqual(query["v"], "abc123")
        self.assertEqual(query["foo"], "bar")
        self.assertEqual(query["t"], "83")

    def test_existing_t_is_replaced_not_duplicated(self):
        url = breakdown.timestamped_url("https://www.youtube.com/watch?v=abc123&t=999", 83.4)
        pairs = urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)
        t_values = [value for key, value in pairs if key == "t"]
        self.assertEqual(t_values, ["83"])

    def test_supported_hosts_all_produce_t(self):
        for source in (
            "https://youtu.be/abc123",
            "https://m.youtube.com/watch?v=abc123",
            "https://www.bilibili.com/video/BV1xx411c7XX",
        ):
            url = breakdown.timestamped_url(source, 61.0)
            self.assertIsNotNone(url, source)
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
            self.assertEqual(query["t"], "61", source)

    def test_unknown_host_returns_none(self):
        self.assertIsNone(breakdown.timestamped_url("https://example.com/v/1", 10.0))

    def test_negative_seconds_clamp_to_zero(self):
        url = breakdown.timestamped_url("https://youtu.be/abc123", -4.2)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.assertEqual(query["t"], "0")

    def test_fragment_and_no_source_url(self):
        url = breakdown.timestamped_url("https://www.youtube.com/watch?v=abc123#frag", 5.0)
        parts = urllib.parse.urlsplit(url)
        self.assertEqual(parts.fragment, "frag")
        self.assertEqual(dict(urllib.parse.parse_qsl(parts.query))["t"], "5")
        self.assertIsNone(breakdown.timestamped_url(None, 5.0))
        self.assertIsNone(breakdown.timestamped_url("", 5.0))


class SourceUrlResolutionTest(unittest.TestCase):
    def test_omitted_flag_preserves_stored_value(self):
        stored = "https://www.youtube.com/watch?v=abc123"
        self.assertEqual(breakdown.resolve_source_url(stored, None), stored)

    def test_explicit_empty_string_clears(self):
        self.assertIsNone(breakdown.resolve_source_url("https://youtu.be/abc123", ""))

    def test_new_value_replaces(self):
        self.assertEqual(
            breakdown.resolve_source_url("https://youtu.be/old", "https://youtu.be/new"),
            "https://youtu.be/new",
        )

    def test_non_http_value_rejected(self):
        with self.assertRaises(ValueError):
            breakdown.resolve_source_url(None, "ftp://example.com/video")
        with self.assertRaises(ValueError):
            breakdown.resolve_source_url(None, "youtube.com/watch?v=abc123")


class SourceLinkRenderTest(unittest.TestCase):
    @staticmethod
    def _stub_run_command(cmd):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"stub-jpeg-bytes")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    def _build(self, tmp, source_url=None):
        meta = {"duration": 100.0}
        if source_url:
            meta["source_url"] = source_url
        steps = [
            {
                "id": "step_01",
                "title": "Push the Frame",
                "start": 12.5,
                "end": 34.0,
                "category": "concept",
                "key_points": ["Push the elbow into space."],
                "explanation": "This creates room because it moves his weight off your hip.",
                "visual_cues": "x",
            },
        ]
        candidates = {
            "step_01": {"cells": {"A": {"time": 15.0, "sharpness": 1.0, "source": "uniform", "thumb": "x"}}},
        }
        approved = {"step_01": {"frames": ["A"], "notes": "A: setup"}}
        transcript_segments = [{"start": 12.5, "end": 34.0, "text": "push the elbow into space to create room"}]
        images_dir = Path(tmp) / "video_review" / "images"
        images_dir.mkdir(parents=True)
        video_path = Path(tmp) / "video.mp4"
        original = breakdown.run_command
        breakdown.run_command = self._stub_run_command
        try:
            result = breakdown.build_render_html(
                video_path,
                meta,
                steps,
                "Spin to face your partner to twist his wrist and break the grip.",
                candidates,
                approved,
                transcript_segments,
                images_dir,
                False,
                "ffmpeg",
            )
        finally:
            breakdown.run_command = original
        return result

    def test_youtube_source_url_renders_all_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, source_url="https://www.youtube.com/watch?v=abc123")
        html = result["html"]
        md = result["md"]
        self.assertIn('class="video-link"', html)
        self.assertIn("Watch the source video", html)
        self.assertIn("t=12", html)  # watch-step pill at int(12.5)
        self.assertIn('class="watch-step"', html)
        self.assertIn("▶ Watch this step", html)
        self.assertIn('title="Watch this step in the video"', html)
        self.assertIn('title="Watch this moment in the video"', html)
        self.assertIn('<span class="play-glyph">▶</span>', html)
        self.assertIn("[▶ Watch the source video](", md)
        self.assertIn("[watch](", md)
        self.assertIn("t=12", md)

    def test_unknown_host_renders_header_link_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp, source_url="https://example.com/v/1")
        html = result["html"]
        md = result["md"]
        self.assertIn('class="video-link"', html)
        self.assertIn("Watch the source video", html)
        self.assertNotIn('class="watch-step"', html)
        self.assertNotIn("Watch this step", html)
        self.assertNotIn('title="Watch this moment in the video"', html)
        self.assertIn("[▶ Watch the source video](", md)
        self.assertNotIn("[watch](", md)

    def test_without_source_url_output_is_pre_m10(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._build(tmp)
        for text in (result["html"], result["md"]):
            self.assertNotIn("video-link", text)
            self.assertNotIn("Watch the source video", text)
            self.assertNotIn("watch-step", text)
            self.assertNotIn("Watch this step", text)
            self.assertNotIn("[watch](", text)


if __name__ == "__main__":
    unittest.main()
