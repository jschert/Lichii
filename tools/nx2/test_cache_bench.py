#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_bench  # noqa: E402

TITLE = "0400486024DEC000"


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def tree(root):
    root = Path(root)
    if not root.exists():
        return None
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class BenchTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.user = self.root / "user"
        self.shader = cache_bench.title_dir(self.user, TITLE)
        write(self.shader / "vulkan.fork-v54.bin", b"yuzucach" + b"\x36\x00\x00\x00" + b"r" * 100)
        write(self.shader / "vulkan_pipelines.fork-v54.bin", b"driver blob")
        write(self.shader / "nested" / "precache-graphics.bin", b"precache")
        write(self.user / "nand" / "user" / "save" / "0000" / "save.bin", b"the owner's save")
        write(self.user / "sdmc" / "file.bin", b"sd card")
        self.original = tree(self.shader)
        self.session = self.root / "session"
        self.running = False
        originals = (cache_bench.lichii_running, cache_bench.port_open)
        cache_bench.lichii_running = lambda name: self.running
        cache_bench.port_open = lambda address: self.running
        self.addCleanup(lambda: (setattr(cache_bench, "lichii_running", originals[0]),
                                 setattr(cache_bench, "port_open", originals[1])))
        self.config = {
            "user_dir": str(self.user), "output_dir": str(self.root / "out"),
            "paths": {"A": {"package": str(self.root / "pkg-a")},
                      "B": {"package": str(self.root / "pkg-b"), "env": {"EDEN_X": "1"}}},
            "games": {"oot": {"title_id": TITLE, "game": "C:/games/oot.nsp"},
                      "mkw": {"title_id": "0400C3F00006E000", "game": "C:/games/mkw.nsp"}},
        }

    def cache_dir(self):
        return cache_bench.cache_dirs(self.config, {"game": "oot"}, self.session / "backup")[0]

    def saves_untouched(self):
        self.assertEqual(tree(self.user / "nand"), {"user/save/0000/save.bin": b"the owner's save"})
        self.assertEqual(tree(self.user / "sdmc"), {"file.bin": b"sd card"})

    def test_title_dir_layout(self):
        self.assertEqual(self.shader, self.user / "cache" / "shader" / "0400486024dec000")

    def test_backup_manifest_matches(self):
        directory = self.cache_dir()
        directory.back_up()
        self.assertEqual(tree(directory.backup), self.original)
        expected = directory.expected()
        self.assertEqual(set(expected), set(self.original))
        self.assertEqual(expected["vulkan_pipelines.fork-v54.bin"]["size"], len(b"driver blob"))
        self.assertEqual(cache_bench.differences(self.shader, expected), [])
        # A second backup only checks: a changed original stops the bench.
        directory.back_up()
        write(self.shader / "vulkan.fork-v54.bin", b"changed")
        with self.assertRaises(cache_bench.BenchError):
            directory.back_up()

    def test_differences_reports_each_kind(self):
        expected = cache_bench.manifest(self.shader)
        write(self.shader / "vulkan.fork-v54.bin", b"other")
        (self.shader / "nested" / "precache-graphics.bin").unlink()
        write(self.shader / "extra.bin", b"x")
        problems = cache_bench.differences(self.shader, expected)
        self.assertIn("changed vulkan.fork-v54.bin", problems)
        self.assertIn("missing nested/precache-graphics.bin", problems)
        self.assertIn("unexpected extra.bin", problems)

    def run_once(self, state, emulator_writes):
        directory = self.cache_dir()
        directory.back_up()
        directory.check()
        directory.prepare(state)
        during = tree(self.shader)
        emulator_writes()
        left = directory.restore(self.session / "runs" / "x" / "after")
        self.assertEqual(tree(self.shader), self.original)
        self.assertFalse(directory.aside.exists())
        self.saves_untouched()
        return during, left

    def test_cold_run_starts_empty_and_restores(self):
        def emulator():
            write(self.shader / "vulkan.fork-v54.bin", b"cold cache")
        during, left = self.run_once("cold", emulator)
        self.assertIsNone(during)
        self.assertEqual(left, {"vulkan.fork-v54.bin": len(b"cold cache")})
        kept = self.session / "runs" / "x" / "after" / "shader-0400486024dec000"
        self.assertEqual(tree(kept), {"vulkan.fork-v54.bin": b"cold cache"})

    def test_warm_run_uses_a_copy(self):
        def emulator():
            with open(self.shader / "vulkan.fork-v54.bin", "ab") as stream:
                stream.write(b"appended")
            (self.shader / "vulkan_pipelines.fork-v54.bin").write_bytes(b"rewritten")
        during, left = self.run_once("warm", emulator)
        self.assertEqual(during, self.original)
        self.assertEqual(left["vulkan_pipelines.fork-v54.bin"], len(b"rewritten"))

    def test_absent_cache_stays_absent(self):
        directory = cache_bench.cache_dirs(self.config, {"game": "mkw"}, self.session / "backup")[0]
        directory.back_up()
        self.assertIsNone(directory.expected())
        for state in ("cold", "warm"):
            directory.check()
            directory.prepare(state)
            self.assertFalse(directory.path.exists())
            write(directory.path / "vulkan.fork-v54.bin", b"new")
            directory.restore(self.session / state)
            self.assertFalse(directory.path.exists())
            self.assertEqual(tree(self.session / state / directory.key), {"vulkan.fork-v54.bin": b"new"})

    def test_failed_check_moves_nothing(self):
        directory = self.cache_dir()
        directory.back_up()
        write(self.shader / "vulkan.fork-v54.bin", b"changed by the owner")
        with self.assertRaises(cache_bench.BenchError):
            directory.check()
        # The original is not the run's output, so a restore leaves it where it is.
        with self.assertRaises(cache_bench.BenchError):
            directory.restore(self.session / "keep")
        self.assertEqual(tree(self.shader)["vulkan.fork-v54.bin"], b"changed by the owner")
        self.assertFalse((self.session / "keep").exists())

    def test_restore_after_a_stop_halfway(self):
        directory = self.cache_dir()
        directory.back_up()
        directory.prepare("warm")
        write(self.shader / "vulkan.fork-v54.bin", b"half-written")
        report = cache_bench.restore_all(self.config, self.session)
        self.assertTrue(any("renamed back" in line for line in report))
        self.assertEqual(tree(self.shader), self.original)
        report = cache_bench.restore_all(self.config, self.session)
        self.assertTrue(any("matches its backup" in line for line in report))

    def test_restore_waits_for_the_emulator(self):
        directory = self.cache_dir()
        directory.back_up()
        directory.prepare("warm")
        self.running = True
        with self.assertRaises(cache_bench.BenchError):
            cache_bench.restore_all(self.config, self.session)
        self.assertTrue(directory.aside.exists())
        self.running = False
        cache_bench.restore_all(self.config, self.session)
        self.assertEqual(tree(self.shader), self.original)

    def test_restore_removes_a_cache_that_was_absent(self):
        directory = cache_bench.cache_dirs(self.config, {"game": "mkw"}, self.session / "backup")[0]
        directory.back_up()
        directory.prepare("cold")
        write(directory.path / "vulkan.fork-v54.bin", b"half-written")
        report = cache_bench.restore_all(self.config, self.session)
        self.assertTrue(any("absent before the bench" in line for line in report))
        self.assertFalse(directory.path.exists())
        recovered = list((self.session / "recovered").rglob("vulkan.fork-v54.bin"))
        self.assertEqual([p.read_bytes() for p in recovered], [b"half-written"])

    def test_guard_refuses_saves_and_outside_paths(self):
        roots = [cache_bench.shader_root(self.user)]
        cache_bench.guard(self.shader, roots)
        for path in (self.user / "nand" / "user" / "save", self.user / "sdmc",
                     self.user / "cache" / "shader", self.root / "elsewhere",
                     cache_bench.shader_root(self.user) / "save" / "x"):
            with self.assertRaises(cache_bench.BenchError, msg=str(path)):
                cache_bench.guard(path, roots)

    def test_plan_alternates_paths(self):
        plan = cache_bench.make_plan(self.config, games=["oot"], repeats=2)
        self.assertEqual(len(plan), 2 * 2 * 2 * 2)
        first = [run["path"] for run in plan if run["repeat"] == 0][:2]
        second = [run["path"] for run in plan if run["repeat"] == 1][:2]
        self.assertEqual(first, ["A", "B"])
        self.assertEqual(second, ["B", "A"])
        self.assertEqual({run["cache"] for run in plan}, {"cold", "warm"})
        self.assertEqual({run["mesh"] for run in plan}, {"mesh", "no_mesh"})
        with self.assertRaises(cache_bench.BenchError):
            cache_bench.make_plan(self.config, games=["zelda"])
        with self.assertRaises(cache_bench.BenchError):
            cache_bench.make_plan(self.config, caches=["second", "cold"])

    def test_run_env(self):
        run = {"game": "oot", "path": "B", "mesh": "no_mesh"}
        env = cache_bench.run_env(self.config, run, base={})
        self.assertEqual(env["EDEN_NX2_STUTTER_TRACE"], "1")
        self.assertEqual(env["EDEN_NX2_DEBUG_SERVER"], "127.0.0.1:6545")
        self.assertEqual(env["EDEN_NX2_FORCE_MESH_FALLBACK"], "1")
        self.assertEqual(env["EDEN_X"], "1")
        cmd, package = cache_bench.command(self.config, run)
        self.assertEqual(cmd[1:], ["-g", "C:/games/oot.nsp"])
        self.assertEqual(package, self.root / "pkg-b")

    def test_dry_run_touches_nothing(self):
        config_path = self.root / "bench.json"
        config_path.write_text(json.dumps(self.config))
        before = tree(self.root)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cache_bench.main(["run", "--config", str(config_path), "--dry-run",
                                               "--repeats", "1"]), 0)
        self.assertEqual(tree(self.root), before)
        text = output.getvalue()
        self.assertIn("16 runs:", text)
        self.assertIn("r0-oot-cold-no_mesh-B", text)
        self.assertIn("EDEN_NX2_FORCE_MESH_FALLBACK=1", text)
        self.assertIn("3 files", text)

    def test_steps_drive_the_debug_server(self):
        class Fake:
            def __init__(self):
                self.calls = []
                self.presents = 0

            def call(self, method, params, timeout):
                self.calls.append((method, params))
                if method == "status":
                    self.presents += 500
                    return {"presents": self.presents}
                if method == "log_tail":
                    return {"lines": [f"[{100 + len(self.calls):4d}.000000] HW.GPU <Info> x"],
                            "next": 1}
                if method == "screenshot":
                    return {"png_base64": "iVBORw=="}
                if method == "frame_times":
                    return {"frames": 60}
                return {}

        fake = Fake()
        steps = [{"wait_presents": 900}, {"input_sequence": [{"repeat": 3, "steps": [{"frames": 1}]}]},
                 {"screenshot": "spot"}, {"measure": 90}]
        run_dir = self.root / "run"
        run_dir.mkdir()
        record = cache_bench.run_steps(fake, steps, run_dir, sleep=lambda s: None)
        methods = [method for method, _ in fake.calls]
        self.assertEqual(methods.count("status"), 2)
        self.assertEqual(fake.calls[2], ("input_sequence", {"steps": [{"frames": 1}] * 3}))
        self.assertEqual(record["screenshots"], ["screenshot-spot.png"])
        self.assertTrue((run_dir / "screenshot-spot.png").exists())
        self.assertEqual([p["seconds"] for m, p in fake.calls if m == "frame_times"], [60.0, 30.0])
        self.assertLess(record["window"]["start_s"], record["window"]["end_s"])

    def test_bench_restores_every_run(self):
        seen = []

        def fake_boot(config, run, run_dir):
            seen.append((run["cache"], tree(self.shader)))
            write(self.shader / "vulkan.fork-v54.bin", b"written by " + run["id"].encode())
            (Path(run_dir) / "eden_log.txt").write_text(
                f"[   1.000000] Frontend <Info> m.cpp:1:x: Booting game: {TITLE} | Ocarina | 1\n"
                f"[   2.000000] Render.Vulkan <Info> x:1:y: Vulkan driver pipelines cached at: "
                f"{self.shader / 'vulkan_pipelines.fork-v54.bin'}\n"
                f"[   3.000000] HW.GPU <Info> x:1:y: Nx2Boot event=boot_start mono_s=1.000\n"
                f"[   9.000000] HW.GPU <Info> x:1:y: Nx2Boot event=first_frame mono_s=7.000\n")
            return {"window": {"start_s": 9.0, "end_s": 20.0}}

        original_boot = cache_bench.boot_and_measure
        cache_bench.boot_and_measure = fake_boot
        self.addCleanup(lambda: setattr(cache_bench, "boot_and_measure", original_boot))
        plan = cache_bench.make_plan(self.config, games=["oot"], caches=["cold", "warm", "second"],
                                     meshes=["mesh"], paths=["A"], repeats=1)
        with redirect_stdout(io.StringIO()):
            summary = cache_bench.run_bench(self.config, plan, self.session)
        self.assertEqual([state for state, _ in seen], ["cold", "warm", "second"])
        self.assertIsNone(seen[0][1])
        self.assertEqual(seen[1][1], self.original)
        self.assertEqual(seen[2][1], {"vulkan.fork-v54.bin": b"written by r0-oot-cold-mesh-A"})
        self.assertEqual(tree(self.shader), self.original)
        self.saves_untouched()
        for run in plan:
            data = json.loads((self.session / "runs" / run["id"] / "run.json").read_text())
            self.assertNotIn("error", data)
            self.assertTrue(data["cache_dir_used"])
            self.assertEqual(data["metrics"]["boot_to_first_frame_s"], 6.0)
            kept = self.session / "runs" / run["id"] / "after" / "shader-0400486024dec000"
            self.assertEqual((kept / "vulkan.fork-v54.bin").read_bytes(),
                             b"written by " + run["id"].encode())
        self.assertIn("## oot cold mesh", summary)
        self.assertIn("| boot_to_first_frame_s | 6 | n/a |", summary)

    def test_failed_runs_stay_out_of_the_tables(self):
        plan = cache_bench.make_plan(self.config, games=["oot"], caches=["cold"], meshes=["mesh"],
                                     repeats=1)
        metrics = {"boot_to_first_frame_s": 6.0}
        for run, extra in zip(plan, ({}, {"error": "boot timed out"})):
            run_dir = self.session / "runs" / run["id"]
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps({"metrics": metrics, **extra}))
        summary = cache_bench.summarize(self.session, plan)
        self.assertIn("| boot_to_first_frame_s | 6 | n/a |", summary)
        self.assertIn(f"- {plan[1]['id']}: boot timed out", summary)

    def test_cache_dir_check_reads_the_log(self):
        log = self.root / "log.txt"
        log.write_text(f"[   2.000000] Render.Vulkan <Info> x:1:y: Vulkan driver pipelines cached at: "
                       f"{self.shader / 'vulkan_pipelines.fork-v54.bin'}\n")
        self.assertTrue(cache_bench.check_cache_dir(log, self.shader))
        self.assertFalse(cache_bench.check_cache_dir(log, self.root / "other"))
        log.write_text("[   2.000000] nothing\n")
        self.assertIsNone(cache_bench.check_cache_dir(log, self.shader))


if __name__ == "__main__":
    unittest.main()
