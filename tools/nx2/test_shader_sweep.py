#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import shader_sweep as sweep
from playthrough_hints import export_hints, import_hints, restore_pipeline


def environment(version, stage):
    """Create a cache environment with version-specific metadata tails."""
    code = bytes(88)
    counts = [len(code), 1, 1, 1]
    if version >= 47:
        counts.append(1)
    counts.append(1)
    header = struct.pack("<" + "Q" * len(counts) + "7I", *counts, 0, 2, 0, 0, 80, 1, stage)
    context = bytes(8 + 8 + 12 + (9 if version >= 47 else 0) + 12)
    tail = bytes(16 if stage == 5 else 80 + (32 if stage == 3 else 0))
    if version >= 53 and stage in (7, 8):
        tail += struct.pack("<3I", 32, 64, 128) + bytes(range(80))
    return header + code + context + tail


def pipeline_cache(version):
    data = b"yuzucach" + struct.pack("<I", version)
    data += struct.pack("<I", 3)
    data += environment(version, 0) + environment(version, 3) + environment(version, 4)
    data += bytes(sweep.GRAPHICS_KEY_SIZE)
    data += struct.pack("<I", 1) + environment(version, 5) + bytes(24)
    return data


class CacheTests(unittest.TestCase):
    def test_supported_versions_preserve_records_and_stage(self):
        """Verify ordinary record layouts stay compatible across supported versions."""
        for version in range(46, 55):
            records = list(sweep.parse_cache(pipeline_cache(version)))
            self.assertEqual([r["stage"] for r in records], ["vertex", "geometry", "fragment", "compute"])
            for record, stage in zip(records, (0, 3, 4, 5)):
                self.assertEqual(record["bytes"], environment(version, stage))
                self.assertEqual(record["version"], version)
                self.assertEqual(len(record["key"]), 24 if stage == 5 else 1056)

    def test_mesh_context_keeps_following_stages_and_pipeline_keys_aligned(self):
        """Verify mesh metadata leaves following cache fields aligned."""
        for version in (53, 54):
            data = b"yuzucach" + struct.pack("<II", version, 3)
            data += environment(version, 7) + environment(version, 8) + environment(version, 4)
            key = bytes(range(256)) * 4 + bytes(32)
            data += key
            data += struct.pack("<I", 1) + environment(version, 5) + bytes(24)
            records = list(sweep.parse_cache(data))
            self.assertEqual([r["stage"] for r in records], ["task", "mesh", "fragment", "compute"])
            for record, stage in zip(records, (7, 8, 4, 5)):
                self.assertEqual(record["bytes"], environment(version, stage))
                self.assertEqual(record["key"], bytes(24) if stage == 5 else key)
        truncated = b"yuzucach" + struct.pack("<II", 53, 1) + environment(53, 8)[:-1]
        with self.assertRaises(sweep.CacheError):
            list(sweep.parse_cache(truncated))

    def test_legacy_mesh_records_without_context_are_rejected(self):
        """Reject legacy mesh records that cannot supply mesh context."""
        data = b"yuzucach" + struct.pack("<II", 52, 1) + environment(52, 8) + bytes(1056)
        with self.assertRaisesRegex(sweep.CacheError, "require cache version 53"):
            list(sweep.parse_cache(data))

    def test_records_cover_the_file_and_keep_their_offsets(self):
        """Pipeline records split the file at the boundaries the runtime reads them at."""
        data = pipeline_cache(54)
        records = list(sweep.parse_records(data))
        self.assertEqual([r["offset"] for r in records], [12, 12 + len(records[0]["bytes"])])
        self.assertEqual(b"".join(r["bytes"] for r in records), data[12:])
        self.assertEqual([len(r["environments"]) for r in records], [3, 1])
        self.assertEqual(records[1]["environments"][0]["offset"], records[1]["offset"] + 4)

    def test_truncation_is_not_an_empty_success(self):
        data = pipeline_cache(47)
        for length in (0, 7, 11, 13, 50, len(data) - 1):
            with self.subTest(length=length), self.assertRaises(sweep.CacheError):
                list(sweep.parse_cache(data[:length]))

    def test_unknown_version_and_environment_count_rejected(self):
        """Reject unsupported versions and out-of-range stage counts."""
        for data in (b"yuzucach" + struct.pack("<I", 55),
                     b"yuzucach" + struct.pack("<II", 47, 7)):
            with self.assertRaises(sweep.CacheError):
                list(sweep.parse_cache(data))

    def test_playthrough_hints_round_trip_requires_local_code(self):
        """Rebuild an exact task, mesh and fragment pipeline using only local code."""
        key = bytes(range(256)) * 4 + bytes(32)
        payload = struct.pack("<I", 3)
        payload += environment(53, 7) + environment(53, 8) + environment(53, 4) + key
        cache = b"yuzucach" + struct.pack("<I", 53) + payload
        hint = export_hints(cache, "0400486024dec000", "131072")
        record, = hint["pipelines"]
        programs = {}
        for stage, env in zip(record["stages"], sweep.parse_cache(cache), strict=True):
            self.assertEqual(stage["shader_sha256"], sweep.digest(env["code"]))
            self.assertEqual(
                bytes.fromhex(stage["prefix"]) + env["code"] + bytes.fromhex(stage["context"]),
                env["bytes"])
            self.assertEqual(set(stage), {"stage", "shader_sha256", "header_sha256",
                                          "prefix", "context"})
            self.assertIsNone(restore_pipeline(record, 53, programs))
            programs[env["stage"], stage["shader_sha256"]] = env["code"]
        self.assertEqual(restore_pipeline(record, 53, programs), (key, payload))
        record["stages"][1]["context"] = record["stages"][1]["context"][:-2]
        with self.assertRaises(sweep.CacheError):
            restore_pipeline(record, 53, programs)

    def test_import_skips_malformed_records_and_restores_valid_ones(self):
        """Check malformed metadata before and after shader lookup."""
        data = pipeline_cache(53)
        hint = export_hints(data, "0400486024dec000", "131072")
        compute = hint["pipelines"][1]
        invalid = {**compute, "stages": [{**compute["stages"][0], "context": ""}]}
        hint["pipelines"] = [None, invalid, compute]
        code = list(sweep.parse_cache(data))[-1]["code"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"programs": [{
                "title_id": hint["title_id"], "stage": "compute",
                "format": None, "code_sha256": sweep.digest(code),
            }]}), encoding="utf-8")
            args = SimpleNamespace(manifest=[manifest], title=hint["title_id"],
                                   output=root / "out", compiler=root / "compiler", jobs=1)
            with patch("playthrough_hints.pack_archive_shader", return_value=("compute", code)):
                summary = import_hints(args, hint)
            self.assertEqual(summary["invalid_record"], 2)
            self.assertEqual(summary["restored"], 1)
            output = args.output / args.title / "vulkan.fork-v53.bin"
            record, = sweep.parse_cache(output.read_bytes())
            self.assertEqual(record["code"], code)

    def test_native_code_matching_excludes_header_scheduling_padding_and_map(self):
        words = [0] * 10 + [11, 12, 0, 13, 14, 15, 0, 16, 0, 80, 88, 2, sweep.MAP_MAGIC]
        code = struct.pack("<" + "Q" * len(words), *words)
        self.assertEqual(sweep.unpack_native_code(code), struct.pack("<6Q", 11, 12, 13, 14, 15, 16))

    def test_maxwell_without_address_map_does_not_match_native(self):
        self.assertIsNone(sweep.unpack_native_code(bytes(128)))


FAKE_COMPILER = '''
import hashlib, json, struct, sys
from pathlib import Path
sys.path.insert(0, {tools!r})
import shader_sweep
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
data = Path(args["--pipeline"]).read_bytes()
version, path = int(args["--cache-version"]), args["--path"]
record, = shader_sweep.parse_records(b"yuzucach" + struct.pack("<I", version) + data)
entry = int(Path(args["--pipeline"]).stem.split("-")[1])
result = {{"schema_version": 1, "status": "ok", "phase": "output", "path": path, "stages": []}}
if path == "fail" or (path == "fail-second" and entry == 1):
    result.update(status="error", phase="translate", failed_slot=0,
                  error={{"code": "compilation_failed", "message": "synthetic failure"}})
out = Path(args["--output-dir"])
out.mkdir(parents=True, exist_ok=True)
for slot, env in enumerate(record["environments"]):
    module = hashlib.sha256(env["bytes"]).digest() * 2
    if path == "flip" and entry == 1:
        module = module[:20] + bytes([module[20] ^ 1]) + module[21:]
    key = "same/{{}}".format(slot) if path == "alias" else hashlib.sha256(data).hexdigest()[:16] + "/{{}}".format(slot)
    stage = {{"slot": slot, "stage": env["stage"], "key": key, "spirv": None, "spirv_words": 0}}
    if result["status"] == "ok":
        stage["spirv"] = "{{}}.spv".format(slot)
        (out / stage["spirv"]).write_bytes(module)
    result["stages"].append(stage)
print(json.dumps(result))
sys.exit(0 if result["status"] == "ok" else 1)
'''


@unittest.skipIf(os.name == "nt", "the scripted compiler needs a POSIX shebang")
class ReplayTests(unittest.TestCase):
    """Comparison, failure and key collision handling, with a scripted compiler."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.compiler = self.root / "fake-shader-sweep"
        self.compiler.write_text(f"#!{sys.executable}\n" + FAKE_COMPILER.format(
            tools=str(Path(sweep.__file__).resolve().parent)))
        self.compiler.chmod(0o755)
        self.cache = self.root / "0400d4201c6ea000" / "vulkan.fork-v54.bin"
        self.cache.parent.mkdir()
        data = b"yuzucach" + struct.pack("<II", 54, 1) + environment(54, 5) + bytes(24)
        data += struct.pack("<I", 2) + environment(54, 0) + environment(54, 4)
        data += struct.pack("<6Q", 0, 7, 0, 0, 0, 8) + bytes(1008)
        self.cache.write_bytes(data)

    def tearDown(self):
        self.directory.cleanup()

    def replay(self, path_a, path_b, *extra):
        output = self.root / f"out-{path_a}-{path_b}"
        with patch.object(sys, "argv", ["shader_sweep.py", "replay", str(self.cache.parent),
                                        "--compiler-a", str(self.compiler), "--path-a", path_a,
                                        "--path-b", path_b, "--output", str(output), *extra]), \
                patch("builtins.print"):
            code = sweep.main()
        return code, json.loads((output / "replay.json").read_text()), output

    def test_identical_paths_match(self):
        for jobs in ("1", "3"):
            code, report, output = self.replay("legacy", "legacy", "--jobs", jobs,
                                               "--timeout", "60")
            self.assertEqual(code, 0)
            self.assertEqual([row["verdict"] for row in report["stages"]], ["match"] * 3)
            self.assertEqual(list((output / "records").iterdir()), [])
            shutil.rmtree(output)

    def test_mismatch_reports_record_identity_and_first_word(self):
        code, report, output = self.replay("legacy", "flip")
        self.assertEqual(code, 1)
        bad = [row for row in report["stages"] if row["verdict"] != "match"]
        self.assertEqual([(row["record"], row["stage"]) for row in bad],
                         [("000-000001", "vertex"), ("000-000001", "fragment")])
        self.assertEqual(bad[0]["first_difference_word"], 5)
        self.assertEqual(bad[0]["record_offset"], 12 + 4 + len(environment(54, 5)) + 24)
        self.assertEqual(bad[0]["env_offset"], bad[0]["record_offset"] + 4)
        self.assertTrue((output / "records" / "000-000001.record").is_file())
        self.assertIn("first difference at word 5", (output / "summary.md").read_text())

    def test_one_sided_failure_is_fatal_and_shared_failure_is_not(self):
        code, report, _ = self.replay("legacy", "fail-second")
        self.assertEqual(code, 1)
        self.assertEqual([row["verdict"] for row in report["stages"]],
                         ["match", "one-sided", "one-sided"])
        self.assertEqual(report["stages"][1]["error_b"]["message"], "synthetic failure")
        code, report, _ = self.replay("fail", "fail")
        self.assertEqual(code, 0)
        self.assertEqual({row["verdict"] for row in report["stages"]}, {"unreplayable"})

    def test_summary_lists_a_record_only_one_side_has(self):
        sides = ({"label": "A", "path": "v2"}, {"label": "B", "path": "v2"})
        record = {"kind": "graphics", "record_key": "k", "entry": 1, "offset": 12,
                  "environments": [{"offset": 16, "stage": "vertex"}]}
        _, rows = sweep.pair_records(0, "cache", [record], [], sides)
        summary = sweep.replay_summary(rows, [], [], sides)
        self.assertIn("| cache | 000-a000001 | 16 | vertex | None |  |  | one-sided | "
                      "only side A has this record |", summary)

    def test_key_collision_in_path_b_is_fatal(self):
        code, report, _ = self.replay("legacy", "alias")
        self.assertEqual(code, 1)
        self.assertEqual({row["verdict"] for row in report["stages"]}, {"match"})
        collision, = report["collisions"]
        self.assertEqual((collision["key_b"], collision["first"], collision["second"]),
                         ("same/0", "000-000000", "000-000001"))

    def test_unreadable_cache_is_fatal(self):
        self.cache.write_bytes(self.cache.read_bytes()[:-1])
        code, report, _ = self.replay("legacy", "legacy")
        self.assertEqual(code, 1)
        self.assertEqual(len(report["input_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
