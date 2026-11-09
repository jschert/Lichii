#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shader cache bench for the owner's PC: boots games cold and warm, A against B.

  cache_bench.py run --config bench.json [--dry-run] [--resume DIR] [--games oot,mkw]
                     [--cache cold,warm,second] [--mesh mesh,no_mesh] [--paths A,B] [--repeats N]
  cache_bench.py restore --config bench.json --session DIR

Every run boots one game through lichii.exe -g with EDEN_NX2_DEBUG_SERVER and
EDEN_NX2_STUTTER_TRACE=1, drives it to a fixed spot through the debug server's input methods,
takes screenshots, measures, kills it, copies its log and parses it with cache_metrics.py.

Shader caches: before the first run every cache directory a run will touch (the game's
<user>/cache/shader/<title id>, and the driver cache directories the config names) is copied
with a sha256 manifest. Each run then moves the real directory aside (a rename in the same
folder), so the emulator never opens the originals: a cold run starts with no directory, a warm
run with a fresh copy of the recorded one, and a second run (opt-in) with a copy of what its
cold run left, which measures the boot after one session of that build. After the run the directory the run wrote is moved
into the run's folder, the original is renamed back and checked against the manifest. Nothing
is deleted, and nothing outside the cache directories is touched: saves (nand, sdmc) never.
If the bench stops halfway, 'restore' puts every original back.

Windows Python 3.14 (stdlib only); see the example config cache_bench.example.json.
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_metrics  # noqa: E402
from debug_mcp import Emulator, EmulatorError, default_timeout  # noqa: E402

ASIDE_SUFFIX = ".bench-aside"
FORBIDDEN_PARTS = {"nand", "sdmc", "save", "saves", "keys"}
MESH_ENV = {"mesh": {}, "no_mesh": {"EDEN_NX2_FORCE_MESH_FALLBACK": "1"}}
CACHE_STATES = ("cold", "warm", "second")
DEFAULT_CACHE_STATES = ("cold", "warm")
TIME = re.compile(r"^\[\s*(\d+\.\d+)\]")
CACHE_PATH_LINE = re.compile(r"(?:Vulkan driver pipelines cached at|Loaded Vulkan driver pipeline "
                             r"cache): (.+)$")


class BenchError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Cache directories: manifest, backup, aside and restore.


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(root):
    """{relative path: {size, sha256}} of every file under root; None when root is absent."""
    root = Path(root)
    if not root.exists():
        return None
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files[path.relative_to(root).as_posix()] = {"size": path.stat().st_size,
                                                    "sha256": sha256(path)}
    return files


def differences(root, expected):
    """What differs between root and a manifest: [] when identical."""
    actual = manifest(root)
    if expected is None or actual is None:
        return [] if expected == actual else [f"{root}: exists={actual is not None}, "
                                              f"expected exists={expected is not None}"]
    problems = [f"missing {name}" for name in expected if name not in actual]
    problems += [f"unexpected {name}" for name in actual if name not in expected]
    problems += [f"changed {name}" for name in expected
                 if name in actual and actual[name] != expected[name]]
    return problems


class CacheDir:
    """One directory a run touches, its backup copy and its manifest."""

    def __init__(self, key, path, backup_root, allowed_roots):
        self.key = key
        self.path = Path(path)
        self.aside = self.path.with_name(self.path.name + ASIDE_SUFFIX)
        self.backup = Path(backup_root) / key
        self.manifest_file = Path(backup_root) / f"{key}.manifest.json"
        guard(self.path, allowed_roots)

    def expected(self):
        with open(self.manifest_file, encoding="utf-8") as stream:
            return json.load(stream)["files"]

    def back_up(self):
        """Copies the directory once and records its manifest; a second call only checks it."""
        if self.manifest_file.exists():
            problems = differences(self.path, self.expected())
            if problems:
                raise BenchError(f"{self.path} changed since its backup: {problems[:5]}")
            return
        if self.aside.exists():
            raise BenchError(f"{self.aside} exists: an earlier bench stopped halfway, run restore")
        files = manifest(self.path)
        if files is not None:
            shutil.copytree(self.path, self.backup)
            if differences(self.backup, files):
                raise BenchError(f"the backup of {self.path} does not match it")
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_file, "w", encoding="utf-8") as stream:
            json.dump({"path": str(self.path), "files": files}, stream, indent=1)

    def check(self):
        """Before a run touches anything: the original matches its backup, nothing is aside."""
        problems = differences(self.path, self.expected())
        if problems:
            raise BenchError(f"{self.path} differs from its backup before the run: {problems[:5]}")
        if self.aside.exists():
            raise BenchError(f"{self.aside} exists: run restore first")

    def prepare(self, state, cold_output=None):
        """Moves the original aside; a warm run gets a copy of the recorded directory, a second
        run a copy of cold_output, what its cold run left of this directory."""
        if self.path.exists():
            os.replace(self.path, self.aside)
        source = {"warm": self.backup, "second": cold_output}.get(state)
        if source is not None and Path(source).exists():
            shutil.copytree(source, self.path)

    def holds_run_output(self):
        """Whether the directory at path is one a run wrote, not the original."""
        return self.aside.exists() or self.expected() is None

    def restore(self, keep_dir):
        """Moves whatever the run left into keep_dir, renames the original back and checks it.
        Returns the size of each file the run left."""
        left = {}
        if self.path.exists() and self.holds_run_output():
            left = {p.relative_to(self.path).as_posix(): p.stat().st_size
                    for p in self.path.rglob("*") if p.is_file()}
            destination = Path(keep_dir) / self.key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.path), str(destination))
        if self.aside.exists():
            os.replace(self.aside, self.path)
        problems = differences(self.path, self.expected())
        if problems:
            raise BenchError(f"{self.path} does not match its backup after the run: {problems[:5]}")
        return left


def guard(path, allowed_roots):
    """Refuses any directory outside the shader cache roots, and anything that looks like saves."""
    resolved = Path(os.path.abspath(path))
    parts = {part.lower() for part in resolved.parts}
    if parts & FORBIDDEN_PARTS:
        raise BenchError(f"refusing {resolved}: it looks like save or key data")
    for root in allowed_roots:
        root = Path(os.path.abspath(root))
        if resolved != root and root in resolved.parents:
            return
    raise BenchError(f"refusing {resolved}: not inside {[str(r) for r in allowed_roots]}")


def shader_root(user_dir):
    """Common::FS::EdenPath::ShaderDir: <user>/cache/shader (path_util.cpp)."""
    return Path(user_dir) / "cache" / "shader"


def title_dir(user_dir, title_id):
    """vk_pipeline_cache.cpp LoadDiskResources: <shader dir>/<title id, 16 lower-case hex>."""
    return shader_root(user_dir) / f"{int(title_id, 16):016x}"


# ---------------------------------------------------------------------------------------------
# Plan.


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("user_dir", "output_dir", "paths", "games"):
        if key not in config:
            raise BenchError(f"config: missing '{key}'")
    for name, game in config["games"].items():
        if not re.fullmatch(r"[0-9A-Fa-f]{16}", game.get("title_id", "")):
            raise BenchError(f"config: game {name} needs a 16-digit title_id")
        if "game" not in game:
            raise BenchError(f"config: game {name} needs the 'game' path for lichii.exe -g")
    for name, path in config["paths"].items():
        if "package" not in path and "package" not in config:
            raise BenchError(f"config: path {name} needs a 'package' directory")
    return config


def make_plan(config, games=None, caches=None, meshes=None, paths=None, repeats=None):
    """Runs in order: per repeat and game, every cache state and mesh mode, with A and B
    alternating which goes first from one repeat to the next."""
    games = games or list(config["games"])
    caches = caches or list(config.get("cache_states", DEFAULT_CACHE_STATES))
    meshes = meshes or list(config.get("mesh_modes", MESH_ENV))
    paths = paths or list(config["paths"])
    repeats = repeats if repeats is not None else int(config.get("repeats", 3))
    for value, known, what in ((games, config["games"], "game"), (caches, CACHE_STATES, "cache state"),
                               (meshes, MESH_ENV, "mesh mode"), (paths, config["paths"], "path")):
        unknown = [v for v in value if v not in known]
        if unknown:
            raise BenchError(f"unknown {what}: {unknown}")
    if "second" in caches and ("cold" not in caches or caches.index("cold") > caches.index("second")):
        raise BenchError("a second run boots from what its cold run left: list cold before second")
    plan = []
    for repeat in range(repeats):
        for game in games:
            for cache in caches:
                for mesh in meshes:
                    order = paths if repeat % 2 == 0 else list(reversed(paths))
                    for path in order:
                        plan.append({"id": f"r{repeat}-{game}-{cache}-{mesh}-{path}", "repeat": repeat,
                                     "game": game, "cache": cache, "mesh": mesh, "path": path})
    return plan


def run_env(config, run, base=None):
    env = dict(os.environ if base is None else base)
    env.update({"EDEN_NX2_STUTTER_TRACE": "1",
                "EDEN_NX2_DEBUG_SERVER": config.get("debug_server", "127.0.0.1:6545")})
    env.update(config.get("env", {}))
    env.update(MESH_ENV[run["mesh"]])
    env.update(config["paths"][run["path"]].get("env", {}))
    return env


def command(config, run):
    package = Path(config["paths"][run["path"]].get("package", config.get("package", "")))
    exe = package / config.get("exe", "lichii.exe")
    return [str(exe), "-g", config["games"][run["game"]]["game"]], package


def cache_dirs(config, run, backup_root):
    """The directories a run touches: its game's shader cache and the configured driver caches."""
    user_dir = config["user_dir"]
    roots = [shader_root(user_dir)] + [Path(d).parent for d in config.get("driver_cache_dirs", [])]
    title_id = config["games"][run["game"]]["title_id"]
    dirs = [CacheDir(f"shader-{title_id.lower()}", title_dir(user_dir, title_id), backup_root, roots)]
    for index, directory in enumerate(config.get("driver_cache_dirs", [])):
        dirs.append(CacheDir(f"driver-{index}", directory, backup_root, roots))
    return dirs


# ---------------------------------------------------------------------------------------------
# Driving the game.


def log_time(emulator):
    """Log seconds of the newest log line, to bound the measured window in the log."""
    lines = emulator.call("log_tail", {"limit": 1}, 10).get("lines", [])
    match = TIME.match(lines[-1]) if lines else None
    return float(match.group(1)) if match else None


def call(emulator, method, params):
    timeout = max(int(params.get("timeout_s", default_timeout(method, params))), 1) + 30
    return emulator.call(method, params, timeout)


def expand(sequence):
    """Input steps with {"repeat": n, "steps": [...]} items written out."""
    out = []
    for item in sequence:
        if "repeat" in item:
            out += expand(item["steps"]) * int(item["repeat"])
        else:
            out.append(item)
    return out


def run_steps(emulator, steps, run_dir, measure_s=60, sleep=time.sleep):
    """Runs the game's steps and returns what the run recorded. Steps:
    {"wait_s": s}, {"wait_presents": n} (since launch), {"input": {...}}, {"input_sequence": [...]},
    ({"repeat": n, "steps": [...]} items repeat), {"screenshot": name}, {"speed_limit": bool}, {"wait_log": text, "timeout_s": s} and
    {"measure": s, "input": {...}?}; without a measure step one of measure_s ends the steps."""
    record = {"screenshots": [], "frame_times": [], "window": None}
    if not any("measure" in step for step in steps):
        steps = list(steps) + [{"measure": measure_s}]
    for step in steps:
        if "wait_s" in step:
            sleep(float(step["wait_s"]))
        elif "wait_presents" in step:
            deadline = time.monotonic() + float(step.get("timeout_s", 600))
            while call(emulator, "status", {"fps_window_ms": 0})["presents"] < step["wait_presents"]:
                if time.monotonic() > deadline:
                    raise BenchError(f"fewer than {step['wait_presents']} presents in time")
                sleep(1.0)
        elif "input" in step and "measure" not in step:
            call(emulator, "input", step["input"])
        elif "input_sequence" in step:
            call(emulator, "input_sequence", {"steps": expand(step["input_sequence"])})
        elif "screenshot" in step:
            result = call(emulator, "screenshot", {"height": int(step.get("height", 720))})
            png = result.pop("png_base64", None)
            if png:
                path = Path(run_dir) / f"screenshot-{step['screenshot']}.png"
                path.write_bytes(base64.b64decode(png))
                record["screenshots"].append(path.name)
        elif "speed_limit" in step:
            call(emulator, "speed_limit", {"on": bool(step["speed_limit"])})
        elif "wait_log" in step:
            deadline = time.monotonic() + float(step.get("timeout_s", 300))
            after = 0
            while True:
                found = emulator.call("log_tail", {"after": after, "contains": step["wait_log"],
                                                   "limit": 1}, 10)
                if found.get("lines"):
                    break
                after = found.get("next", after)
                if time.monotonic() > deadline:
                    raise BenchError(f"'{step['wait_log']}' was not logged in time")
                sleep(1.0)
        elif "measure" in step:
            seconds = float(step["measure"])
            start = log_time(emulator)
            if "input" in step:
                frames = int(seconds * 60)
                call(emulator, "input_sequence", {"steps": [dict(step["input"], frames=frames)]})
            else:
                remaining = seconds
                while remaining > 0:
                    chunk = min(remaining, 60.0)
                    record["frame_times"].append(call(emulator, "frame_times", {"seconds": chunk}))
                    remaining -= chunk
            record["window"] = {"start_s": start, "end_s": log_time(emulator)}
        else:
            raise BenchError(f"unknown step {step}")
    return record


def port_open(address):
    host, _, port = address.rpartition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=1):
            return True
    except OSError:
        return False


def wait_closed(address, seconds=30):
    """The previous run's emulator has let go of the debug port."""
    deadline = time.monotonic() + seconds
    while port_open(address):
        if time.monotonic() > deadline:
            raise BenchError(f"{address} still answers: another Lichii is running")
        time.sleep(1.0)


def lichii_running(exe_name):
    if os.name != "nt":
        return False
    listing = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                             capture_output=True, text=True).stdout
    return exe_name.lower() in listing.lower()


def check_cache_dir(log_path, expected):
    """Whether the log shows the emulator used the expected cache directory (None: not logged)."""
    seen = None
    with open(log_path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = CACHE_PATH_LINE.search(line.rstrip())
            if match:
                used = Path(match.group(1).strip()).parent
                seen = os.path.normcase(os.path.abspath(used)) == os.path.normcase(
                    os.path.abspath(expected))
                if not seen:
                    return False
    return seen


def boot_and_measure(config, run, run_dir):
    game = config["games"][run["game"]]
    cmd, package = command(config, run)
    address = config.get("debug_server", "127.0.0.1:6545")
    log_file = Path(config.get("log_file", Path(config["user_dir"]) / "log" / "eden_log.txt"))
    launched = time.time()
    process = subprocess.Popen(cmd, cwd=package, env=run_env(config, run),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    emulator = Emulator(address)
    record = {"command": cmd, "launched": launched}
    try:
        deadline = time.monotonic() + float(game.get("boot_timeout_s", 900))
        while True:
            if process.poll() is not None:
                raise BenchError(f"lichii exited with {process.returncode} while booting")
            try:
                if emulator.call("status", {"fps_window_ms": 0}, 10)["presents"] > 0:
                    break
            except EmulatorError:
                pass
            if time.monotonic() > deadline:
                raise BenchError("no frame was presented before boot_timeout_s")
            time.sleep(1.0)
        record["boot_wall_s"] = round(time.time() - launched, 1)
        record.update(run_steps(emulator, game.get("steps", []), run_dir,
                                float(game.get("measure_s", 60))))
    finally:
        emulator.close()
        process.kill()
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            record["kill"] = "did not exit within 60 s"
        time.sleep(1.0)
        if log_file.exists() and log_file.stat().st_mtime >= launched:
            shutil.copy2(log_file, Path(run_dir) / "eden_log.txt")
        else:
            record["log"] = f"no log written at {log_file}"
    return record


# ---------------------------------------------------------------------------------------------
# Session.


def describe_dir(directory):
    directory = Path(directory)
    if not directory.exists():
        return "absent (a warm run starts empty too)"
    files = [p for p in directory.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.1f} MiB"


def print_plan(config, plan, session, out=None):
    out = out or sys.stdout
    backup_root = Path(session) / "backup"
    out.write(f"Session: {session}\nUser data: {config['user_dir']} (saves are never touched)\n\n")
    out.write("Backed up before the first run, with a sha256 manifest:\n")
    seen = set()
    for run in plan:
        for directory in cache_dirs(config, run, backup_root):
            if directory.key not in seen:
                seen.add(directory.key)
                out.write(f"  {directory.path} -> {directory.backup}: "
                          f"{describe_dir(directory.path)}\n")
    out.write("\nEach run: rename the originals to <dir>" + ASIDE_SUFFIX + "; cold starts with no "
              "directory, warm with a copy of the backup, second with a copy of what its cold run left; boot, steps, measure, kill, copy the "
              "log; move what the run wrote into its folder; rename the originals back; verify "
              "the sha256 manifest.\n\n")
    for name, game in config["games"].items():
        if any(run["game"] == name for run in plan):
            out.write(f"Game {name} ({game['title_id']}): -g {game['game']}; "
                      f"{len(game.get('steps', []))} steps, measure {game.get('measure_s', 60)} s\n")
    out.write(f"\n{len(plan)} runs:\n")
    for index, run in enumerate(plan):
        cmd, package = command(config, run)
        env = run_env(config, run, base={})
        out.write(f"  {index + 1:3d}. {run['id']}: {' '.join(cmd)} (in {package}) "
                  f"env {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}\n")


def summarize(session, plan):
    """summary.md: A against B for every game, cache state and mesh mode; failed runs apart."""
    runs = {}
    failed = []
    for run in plan:
        path = Path(session) / "runs" / run["id"] / "run.json"
        if path.exists():
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            if "error" in data:
                failed.append(f"- {run['id']}: {data['error']}")
            elif data.get("metrics"):
                runs.setdefault((run["game"], run["cache"], run["mesh"]), {}).setdefault(
                    run["path"], []).append(data["metrics"])
    sections = []
    for (game, cache, mesh), by_path in sorted(runs.items()):
        names = sorted(by_path)
        other = names[1] if len(names) > 1 else "-"
        table = cache_metrics.compare(by_path[names[0]], by_path.get(other, []), names[0], other)
        sections.append(f"## {game} {cache} {mesh}\n\n{table}")
    text = "# Shader cache bench\n\n" + "\n".join(sections)
    if failed:
        text += "\n## Failed runs (not in the tables)\n\n" + "\n".join(failed) + "\n"
    (Path(session) / "summary.md").write_text(text, encoding="utf-8")
    return text


def ensure_closed(config):
    """Nothing may write the caches while the bench moves them: no emulator, no debug server."""
    exe_name = config.get("exe", "lichii.exe")
    address = config.get("debug_server", "127.0.0.1:6545")
    if lichii_running(exe_name) or port_open(address):
        raise BenchError(f"{exe_name} is running or {address} answers: close it first")


def run_bench(config, plan, session):
    session = Path(session)
    backup_root = session / "backup"
    (session / "runs").mkdir(parents=True, exist_ok=True)
    with open(session / "plan.json", "w", encoding="utf-8") as stream:
        json.dump({"config": config, "plan": plan}, stream, indent=1)
    ensure_closed(config)
    address = config.get("debug_server", "127.0.0.1:6545")
    for run in plan:
        for directory in cache_dirs(config, run, backup_root):
            directory.back_up()
    for index, run in enumerate(plan):
        run_dir = session / "runs" / run["id"]
        if (run_dir / "run.json").exists():
            with open(run_dir / "run.json", encoding="utf-8") as stream:
                if "error" not in json.load(stream):
                    continue
            attempt = 1
            while run_dir.with_name(f"{run['id']}.failed-{attempt}").exists():
                attempt += 1
            os.replace(run_dir, run_dir.with_name(f"{run['id']}.failed-{attempt}"))
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index + 1}/{len(plan)}] {run['id']}", flush=True)
        dirs = cache_dirs(config, run, backup_root)
        for directory in dirs:
            directory.check()
        wait_closed(address)
        result = dict(run)
        prepared = []
        cold_dir = session / "runs" / f"r{run['repeat']}-{run['game']}-cold-{run['mesh']}-{run['path']}"
        try:
            if run["cache"] == "second":
                with open(cold_dir / "run.json", encoding="utf-8") as stream:
                    if "error" in json.load(stream):
                        raise BenchError(f"its cold run {cold_dir.name} failed")
            for directory in dirs:
                prepared.append(directory)
                directory.prepare(run["cache"], cold_dir / "after" / directory.key)
            result.update(boot_and_measure(config, run, run_dir))
        except (BenchError, EmulatorError, OSError) as error:
            result["error"] = str(error)
        finally:
            result["cache_left"] = {d.key: d.restore(run_dir / "after") for d in prepared}
        log = run_dir / "eden_log.txt"
        if log.exists():
            window = result.get("window") or {}
            result["metrics"] = cache_metrics.parse_log(log, start_s=window.get("start_s"),
                                                        end_s=window.get("end_s"))
            result["cache_dir_used"] = check_cache_dir(log, dirs[0].path)
            if result["cache_dir_used"] is False:
                result["error"] = (f"the emulator used another shader cache directory than "
                                   f"{dirs[0].path}: fix user_dir")
        else:
            result.setdefault("error", f"no log: {result.get('log', 'not copied')}")
        with open(run_dir / "run.json", "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=1)
        # Without the log or the right cache directory the remaining runs would measure nothing.
        if result.get("cache_dir_used") is False or not log.exists():
            raise BenchError(result["error"])
    return summarize(session, plan)


def restore_all(config, session):
    """Puts every original cache directory back after a bench that stopped halfway."""
    ensure_closed(config)
    backup_root = Path(session) / "backup"
    keep = Path(session) / "recovered" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plan = make_plan(config, repeats=1)
    done = set()
    report = []
    for run in plan:
        for directory in cache_dirs(config, run, backup_root):
            if directory.key in done or not directory.manifest_file.exists():
                continue
            done.add(directory.key)
            if directory.aside.exists():
                directory.restore(keep)
                report.append(f"{directory.path}: original renamed back, run leftovers in {keep}")
            elif directory.expected() is None and directory.path.exists():
                directory.restore(keep)
                report.append(f"{directory.path}: absent before the bench, run leftovers in {keep}")
            else:
                problems = differences(directory.path, directory.expected())
                report.append(f"{directory.path}: " + ("matches its backup" if not problems else
                              f"DIFFERS from its backup ({problems[:5]}); copy it back from "
                              f"{directory.backup} by hand after checking"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true", help="print the plan and touch nothing")
    run.add_argument("--resume", help="session directory to continue")
    for name in ("games", "cache", "mesh", "paths"):
        run.add_argument(f"--{name}", type=lambda s: [v for v in s.split(",") if v])
    run.add_argument("--repeats", type=int)
    restore = commands.add_parser("restore")
    restore.add_argument("--config", required=True)
    restore.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "restore":
            print("\n".join(restore_all(config, args.session)))
            return 0
        plan = make_plan(config, args.games, args.cache, args.mesh, args.paths, args.repeats)
        session = args.resume or str(Path(config["output_dir"]) /
                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if args.dry_run:
            print_plan(config, plan, session)
            return 0
        print(run_bench(config, plan, session))
        return 0
    except BenchError as error:
        print(f"cache_bench: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
