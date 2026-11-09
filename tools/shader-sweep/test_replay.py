#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replay synthetic v54 caches built from the captured fixtures through shader-sweep.

With --v2 (shader-sweep built in the tree), also replay them against the shader cache v2 path.
"""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_captured import serialize_environment

BINARY = Path(sys.argv.pop(1)).resolve()
V2 = "--v2" in sys.argv
if V2:
    sys.argv.remove("--v2")
ROOT = Path(__file__).resolve().parent
SWEEP = ROOT.parent / "nx2" / "shader_sweep.py"
MAXWELL_EXIT = 0xE30000000007000F
# FixedPipelineState::driver_id in a graphics key, after the six program hashes
DRIVER_OFFSET = 48 + 16
# A bit of the first instruction of a graphics record's first environment: after the environment
# count, the environment's sizes and parameters and the 80-byte shader program header
PLANT_OFFSET, PLANT_MASK = 4 + 6 * 8 + 7 * 4 + 80, 0x10


def fixture(name):
    data = json.loads((ROOT / "fixtures" / f"{name}.json").read_text())
    return serialize_environment(data["environment"]), data


def fixture_hash(data, slot):
    return struct.unpack_from("<Q", bytes.fromhex(data["pipeline_key"]), slot * 8)[0]


def graphics_key(state_from, hashes):
    """A fixture's pipeline state with only the given program slots' hashes."""
    key = bytearray(bytes.fromhex(state_from["pipeline_key"]))
    struct.pack_into("<6Q", key, 0, *(hashes.get(slot, 0) for slot in range(6)))
    return bytes(key)


def synthetic(stage):
    """A minimal environment: a header, one EXIT and the stage's metadata tail."""
    sph = bytearray(80)
    struct.pack_into("<5I", sph, 0, ((min(stage, 4) + 1) << 10) | 1, 0, 1 << 24, 1 << 24, 1)
    if stage == 4:
        struct.pack_into("<I", sph, 72, 15)
    code = bytes(sph) + struct.pack("<Q", MAXWELL_EXIT)
    data = struct.pack("<6Q", len(code), 0, 0, 0, 0, 0)
    data += struct.pack("<7I", 0, 2, 0, 0, len(code) - 8, 1, stage) + code + bytes(sph)
    if stage == 3:
        data += bytes(32)
    if stage in (7, 8):
        data += struct.pack("<3I", 32, 64, 128) + bytes(sph)
    return data


def record(envs, key):
    return struct.pack("<I", len(envs)) + b"".join(envs) + key


def fixture_cache():
    """Compute, single-stage and linked graphics pipelines, and a duplicate record."""
    byte_store, byte_store_data = fixture("bananza-byte-store")
    aliasing, aliasing_data = fixture("global-memory-without-aliasing")
    passthrough, passthrough_data = fixture("tennis-geometry-passthrough")
    atomic, atomic_data = fixture("bananza-unmapped-atomic-counter")
    records = [
        record([byte_store], graphics_key(byte_store_data, {1: fixture_hash(byte_store_data, 1)})),
        record([aliasing], graphics_key(aliasing_data, {1: fixture_hash(aliasing_data, 1)})),
        record([passthrough], graphics_key(passthrough_data,
                                           {4: fixture_hash(passthrough_data, 4)})),
        record([aliasing, passthrough], graphics_key(passthrough_data, {
            1: fixture_hash(aliasing_data, 1), 4: fixture_hash(passthrough_data, 4)})),
        record([atomic], bytes.fromhex(atomic_data["compute_pipeline_key"])),
    ]
    records.append(records[0])
    return b"yuzucach" + struct.pack("<I", 54) + b"".join(records)


class ReplayCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.runs = 0

    def tearDown(self):
        self.directory.cleanup()

    def replay(self, cache, *extra, code=0):
        title = self.root / "0400d4201c6ea000"
        title.mkdir(exist_ok=True)
        (title / "vulkan.fork-v54.bin").write_bytes(cache)
        self.runs += 1
        output = self.root / f"replay-{self.runs}"
        result = subprocess.run([sys.executable, str(SWEEP), "replay", str(title),
                                 "--compiler-a", str(BINARY), "--output", str(output), *extra],
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return json.loads((output / "replay.json").read_text())

    def compile(self, data, *extra):
        path = self.root / "pipeline.record"
        path.write_bytes(data)
        result = subprocess.run([str(BINARY), "--pipeline", str(path), "--cache-version", "54",
                                 "--output-dir", str(self.root / "modules"), *extra],
                                capture_output=True, text=True, timeout=60)
        return result.returncode, json.loads(result.stdout)


class ReplayTests(ReplayCase):
    def test_same_compiler_is_deterministic_serial_and_parallel(self):
        """Paths A and B on one binary agree word for word on every fixture stage."""
        cache = fixture_cache()
        reports = [self.replay(cache, "--jobs", jobs) for jobs in ("1", "4")]
        for report in reports:
            self.assertEqual(len(report["stages"]), 7)
            self.assertEqual({row["verdict"] for row in report["stages"]}, {"match"})
            self.assertEqual(report["collisions"], [])
            for row in report["stages"]:
                self.assertEqual(row["key_a"], row["key_b"])
                self.assertGreater(row["words_b"], 0)
        serial, parallel = (sorted((row["record"], row["slot"], row["spirv_sha256_b"])
                                   for row in report["stages"]) for report in reports)
        self.assertEqual(serial, parallel)

    def test_linked_stage_is_emitted_with_the_previous_stage(self):
        """A geometry stage after a vertex stage differs from the same stage alone."""
        report = self.replay(fixture_cache())
        geometry = {row["record"]: row for row in report["stages"] if row["stage"] == "geometry"}
        alone, linked = geometry["000-000002"], geometry["000-000003"]
        self.assertNotEqual(alone["spirv_sha256_b"], linked["spirv_sha256_b"])
        self.assertNotEqual(alone["key_b"], linked["key_b"])

    def test_duplicate_record_shares_its_key_without_collision(self):
        report = self.replay(fixture_cache())
        first, duplicate = (next(row for row in report["stages"] if row["record"] == name)
                            for name in ("000-000000", "000-000005"))
        self.assertEqual(first["key_b"], duplicate["key_b"])
        self.assertEqual(first["spirv_sha256_b"], duplicate["spirv_sha256_b"])

    def test_pipeline_hash_covers_only_the_compared_key_bytes(self):
        """Bytes past FixedPipelineState::Size() don't change the legacy pipeline hash."""
        byte_store, data = fixture("bananza-byte-store")
        key = graphics_key(data, {1: fixture_hash(data, 1)})
        hashes = []
        for offset in (None, 1000, 100):
            changed = bytearray(key)
            if offset is not None:
                changed[offset] ^= 1
            code, result = self.compile(record([byte_store], bytes(changed)))
            self.assertEqual(code, 0, result)
            hashes.append(result["pipeline_hash"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_unknown_path_and_inconsistent_records_are_input_errors(self):
        byte_store, data = fixture("bananza-byte-store")
        for key, extra in ((graphics_key(data, {1: fixture_hash(data, 1)}), ["--path", "v3"]),
                           (graphics_key(data, {1: 1, 5: 2}), []),
                           (graphics_key(data, {1: 1}) + b"x", [])):
            code, result = self.compile(record([byte_store], key), *extra)
            self.assertEqual(code, 2, result)
            self.assertEqual(result["phase"], "input")

    def test_task_mesh_and_vertex_a_records(self):
        """Synthetic task/mesh and vertex A pipelines replay identically or fail on both paths."""
        key = bytearray(1056)
        struct.pack_into("<6Q", key, 0, 0, 1, 2, 0, 0, 3)
        mesh = record([synthetic(7), synthetic(8), synthetic(4)], bytes(key))
        struct.pack_into("<6Q", key, 0, 4, 5, 0, 0, 0, 6)
        dual = record([synthetic(6), synthetic(0), synthetic(4)], bytes(key))
        report = self.replay(b"yuzucach" + struct.pack("<I", 54) + mesh + dual)
        for row in report["stages"]:
            self.assertIn(row["verdict"], ("match", "unreplayable"), row)
        print(json.dumps([{k: row.get(k) for k in ("record", "stage", "verdict", "words_b",
                                                   "error_b")}
                          for row in report["stages"]]), file=sys.stderr)


# Forwards to shader-sweep, then flips one byte of one record a --list call wrote
PLANTING_COMPILER = '''
import json, subprocess, sys
from pathlib import Path
result = subprocess.run([{binary!r}, *sys.argv[1:]], capture_output=True, text=True)
if "--list" in sys.argv:
    listed = json.loads(result.stdout)
    directory = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    path = directory / listed["records"][{record}]["file"]
    data = bytearray(path.read_bytes())
    data[{offset}] ^= {mask}
    path.write_bytes(data)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
'''


@unittest.skipUnless(V2, "needs shader-sweep built in the tree (--v2)")
class V2ReplayTests(ReplayCase):
    def test_migrated_cache_replays_without_mismatch(self):
        """The production migration and reader give the same SPIR-V as the legacy records."""
        report = self.replay(fixture_cache(), "--path-b", "v2")
        verdicts = [row["verdict"] for row in report["stages"]]
        self.assertEqual(sorted(verdicts), ["collapsed"] + ["match"] * 6, report["stages"])
        self.assertEqual(report["collisions"], [])
        migration = report["listings"][0]["b"]["migration"]
        self.assertEqual(migration, {"source_version": 54, "graphics": 4, "compute": 1,
                                     "collapsed": 1})
        collapsed, = (row for row in report["stages"] if row["verdict"] == "collapsed")
        self.assertEqual(collapsed["entry"], 0)
        for row in report["stages"]:
            if row["verdict"] == "match":
                self.assertEqual(row["key_b"].split("/")[0], row["record_key"])

    def test_graphics_record_key_leaves_the_driver_out(self):
        """Records differing only in the driver fields pair, and the earlier one collapses."""
        byte_store, data = fixture("bananza-byte-store")
        key = graphics_key(data, {1: fixture_hash(data, 1)})
        other_driver = bytearray(key)
        struct.pack_into("<II", other_driver, DRIVER_OFFSET, 1, 2)
        cache = (b"yuzucach" + struct.pack("<I", 54) + record([byte_store], key) +
                 record([byte_store], bytes(other_driver)))
        report = self.replay(cache, "--path-b", "v2")
        self.assertEqual(sorted(row["verdict"] for row in report["stages"]),
                         ["collapsed", "match"])

    @unittest.skipIf(os.name == "nt", "the planting compiler needs a POSIX shebang")
    def test_planted_byte_in_a_migrated_record_is_one_mismatch(self):
        planting = self.root / "planting-shader-sweep"
        planting.write_text(f"#!{sys.executable}\n" + PLANTING_COMPILER.format(
            binary=str(BINARY), record=1, offset=PLANT_OFFSET, mask=PLANT_MASK))
        planting.chmod(0o755)
        report = self.replay(fixture_cache(), "--path-b", "v2", "--compiler-b", str(planting),
                             code=1)
        bad = [row for row in report["stages"] if row["verdict"] not in ("match", "collapsed")]
        self.assertEqual([(row["verdict"], row["record"]) for row in bad],
                         [("mismatch", "000-000001")], bad)


if __name__ == "__main__":
    unittest.main()
