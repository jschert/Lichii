#!/usr/bin/env python3
"""Replay native shader captures and v46-v54 pipeline caches without a GPU.

`shader_sweep.py replay` compiles every pipeline record of recorded caches through two
compilation paths and compares their SPIR-V; see tools/nx2/shader-sweep.md.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


STAGES = ("vertex", "tess-control", "tess-eval", "geometry", "fragment", "compute",
          "vertex-a", "task", "mesh")
GAMES = {
    "040074a01bf12000": "Donkey Kong Bananza",
    "040077201be50000": "Kirby Air Riders",
    "04007c6028aa8000": "Yoshi demo",
    "04007ef01f8ae000": "Fire Emblem: Fortune's Weave",
    "0400c3f00006e000": "Mario Kart World",
    "04008e001f098000": "Nintendo Switch Sports Resort",
    "0400a3a01ee4c000": "PokÃ©mon Pokopia",
    "0400a4802437e000": "Drag x Drive Global Jam",
    "040041e0212ae000": "Fast Fusion",
    "04005b902517c000": "Kirby Air Riders Global Test Ride",
    "040057c02159a000": "Nintendo Switch 2 Welcome Tour",
    "040004f027560000": "Orbitals Global Friends Pass",
    "0400380020056000": "Raiders",
    "0400d1d02399a000": "The Duskbloods network test",
    "0400d4201c6ea000": "Mario Tennis Fever",
    "0400e88025b1a000": "Star Fox",
}
MAP_MAGIC = 0x50414D4E4544454E
GRAPHICS_KEY_SIZE = 1056
# Versions 47 through 54 share one record layout; 46 lacks storage-pointer classifications.
CACHE_VERSIONS = range(46, 55)
CACHE_NAME = re.compile(r"vulkan(?:\.fork-v\d+)?\.bin")
# The shader cache v2 log of a title, beside its Vulkan pipeline cache
V2_LOG_NAME = "lichii-cache.v2.bin"


class CacheError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_records(data):
    """Yield pipeline records: their byte span, shader environments and pipeline key."""
    if len(data) < 12 or data[:8] != b"yuzucach":
        raise CacheError("not a serialized shader-environment cache")
    version, = struct.unpack_from("<I", data, 8)
    if version not in CACHE_VERSIONS:
        raise CacheError(f"unsupported cache version {version}")
    cursor = 12
    entry = 0
    while cursor < len(data):
        record_begin = cursor
        if cursor + 4 > len(data):
            raise CacheError(f"truncated pipeline count at {cursor}")
        count, = struct.unpack_from("<I", data, cursor)
        cursor += 4
        if not 1 <= count <= 6:
            raise CacheError(f"invalid environment count {count} at {cursor - 4}")
        environments = []
        for index in range(count):
            begin = cursor
            header = struct.Struct("<" + "Q" * (6 if version >= 47 else 5) + "7I")
            if cursor + header.size > len(data):
                raise CacheError(f"truncated environment header at {cursor}")
            fields = header.unpack_from(data, cursor)
            sizes, params = fields[:-7], fields[-7:]
            code_size, textures, formats, constants = sizes[:4]
            pointers, replacements = sizes[4:] if version >= 47 else (0, sizes[4])
            local, bound, start, low, high, viewport, stage = params
            if stage >= len(STAGES) or code_size > 64 * 1024 * 1024 or code_size % 8:
                raise CacheError(f"invalid stage/code size at {cursor}")
            if any(n > 1_000_000 for n in sizes[1:]):
                raise CacheError(f"excessive context count at {cursor}")
            cursor += header.size
            code = data[cursor:cursor + code_size]
            cursor += code_size + textures * 8 + formats * 8 + constants * 12
            cursor += pointers * 9 + replacements * 12
            if stage in (7, 8) and version < 53:
                raise CacheError("task/mesh environments require cache version 53")
            cursor += 16 if stage == 5 else 80 + (32 if stage == 3 else 0)
            if stage in (7, 8):
                cursor += 92  # Three mesh registers and the primitive program header.
            if cursor > len(data):
                raise CacheError(f"truncated environment at {begin}")
            environments.append({"version": version, "entry": entry, "index": index,
                                 "stage": STAGES[stage], "offset": begin,
                                 "bytes": data[begin:cursor], "code": code})
        compute = environments[0]["stage"] == "compute"
        if compute and count != 1:
            raise CacheError(f"compute pipeline has {count} environments")
        key_size = 24 if compute else GRAPHICS_KEY_SIZE
        if cursor + key_size > len(data):
            raise CacheError(f"truncated pipeline key at {cursor}")
        key = data[cursor:cursor + key_size]
        cursor += key_size
        for env in environments:
            env["key"] = key
        yield {"version": version, "entry": entry, "offset": record_begin,
               "bytes": data[record_begin:cursor], "environments": environments, "key": key}
        entry += 1


def parse_cache(data):
    """Yield shader environments with their original pipeline keys."""
    for record in parse_records(data):
        yield from record["environments"]


def unpack_native_code(code):
    if len(code) < 96 or len(code) % 8:
        return None
    words = struct.unpack("<" + "Q" * (len(code) // 8), code)
    if words[-1] != MAP_MAGIC:
        return None
    count = words[-2]
    if not 0 < count <= 65536 or count + 2 >= len(words):
        raise CacheError("invalid native address map")
    end = len(words) - count - 3
    return b"".join(struct.pack("<Q", words[i]) for i in range(10, end) if i % 4)


def invoke(command, timeout):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "phase": "timeout", "error": {"message": "compiler timeout"}}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"status": "error", "phase": "process", "error": {
            "message": f"compiler exited {proc.returncode}: {proc.stderr[-2000:]}"}}
    result["returncode"] = proc.returncode
    if proc.returncode and result.get("status") == "ok":
        result.update(status="error", phase="process", error={"message": "nonzero compiler exit"})
    return result


def failure(phase, message):
    return {"status": "error", "phase": phase, "error": {"message": message}}


def summarize(results):
    counts = Counter()
    for row in results:
        message = row.get("error", {}).get("message", "ok")
        message = re.sub(r"%\d+", "%id", message)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", message)
        counts[(row["game"], row.get("stage") or "unknown", row["phase"],
                row.get("opcode", ""), message)] += 1
    lines = ["| Game | Stage | Phase | Opcode | Cause | Count |", "|---|---|---|---|---|---:|"]
    for (game, stage, phase, opcode, cause), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        cause = cause.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {game} | {stage} | {phase} | {opcode} | {cause} | {count} |")
    return "\n".join(lines) + "\n"


def first_difference(a, b):
    """Index of the first differing 32-bit word, or None when the modules are equal."""
    if a == b:
        return None
    words = min(len(a), len(b)) // 4
    for index in range(words):
        if a[index * 4:index * 4 + 4] != b[index * 4:index * 4 + 4]:
            return index
    return words


def replay_record(pair, sides, root, timeout, keep_spirv):
    """Compile one record on each side (A's and B's copy of it) and compare each stage's module."""
    outputs = []
    for side, record in zip(sides, pair["records"]):
        directory = root / "spirv" / side["name"] / pair["id"]
        command = [side["compiler"], "--pipeline", record["path"], "--cache-version",
                   str(record["version"]), "--path", side["path"], "--output-dir", str(directory)]
        result = invoke(command, timeout)
        stages = result.get("stages") if result["status"] == "ok" else None
        if stages is not None and len(stages) != len(record["environments"]):
            result = failure("process", "compiler reported a different stage count")
            stages = None
        outputs.append((directory, result, stages))
    (dir_a, a, stages_a), (dir_b, b, stages_b) = outputs
    record_a, record_b = pair["records"]
    rows = []
    for index, env in enumerate(record_a["environments"]):
        reported = [result.get("stages") or [] for result in (a, b)]
        known = [stages[index] if index < len(stages) else {} for stages in reported]
        row = {"source": pair["source"], "record": pair["id"], "entry": record_a["entry"],
               "record_offset": record_a["offset"], "env_offset": env["offset"],
               "stage": env["stage"], "slot": known[0].get("slot", known[1].get("slot")),
               "unique_hash": known[0].get("unique_hash", known[1].get("unique_hash")),
               "key_a": known[0].get("key"), "key_b": known[1].get("key"),
               "pipeline_hash_a": a.get("pipeline_hash"), "pipeline_hash_b": b.get("pipeline_hash")}
        if record_b is not record_a:
            row.update(record_key=pair["record_key"], entry_b=record_b["entry"],
                       record_offset_b=record_b["offset"], relinked=record_b.get("relinked", False))
        if len(record_b["environments"]) != len(record_a["environments"]):
            row.update(verdict="mismatch", detail="the records have different stage counts")
            rows.append(row)
            continue
        if stages_a is None or stages_b is None:
            row["verdict"] = "unreplayable" if stages_a is None and stages_b is None else "one-sided"
            for name, result in (("a", a), ("b", b)):
                if result["status"] != "ok":
                    row[f"error_{name}"] = {"phase": result.get("phase"),
                                            "slot": result.get("failed_slot"),
                                            **result.get("error", {})}
            rows.append(row)
            continue
        modules = [(directory / stages[index]["spirv"]).read_bytes() if stages[index]["spirv"] else None
                   for directory, stages in ((dir_a, stages_a), (dir_b, stages_b))]
        row["words_a"], row["words_b"] = (len(m) // 4 if m is not None else 0 for m in modules)
        row["spirv_sha256_b"] = digest(modules[1]) if modules[1] is not None else None
        if modules[0] is None or modules[1] is None:
            row["verdict"] = "match" if modules[0] is modules[1] else "mismatch"
        else:
            difference = first_difference(*modules)
            row["verdict"] = "match" if difference is None else "mismatch"
            if difference is not None:
                row["first_difference_word"] = difference
        rows.append(row)
    if not keep_spirv and all(row["verdict"] == "match" for row in rows):
        for directory in (dir_a, dir_b):
            shutil.rmtree(directory, ignore_errors=True)
    return rows


def list_records(side, cache, directory, timeout):
    """The records of a cache as the side's compiler reads them on its path (shader-sweep --list).
    A v2 side takes the Program records of the v2 log beside the cache, as the boot does."""
    command = [side["compiler"], "--list", str(cache), "--path", side["path"],
               "--output-dir", str(directory)]
    programs = cache.parent / V2_LOG_NAME
    if side["path"] == "v2" and programs.is_file():
        command += ["--programs-log", str(programs)]
    result = invoke(command, timeout)
    if result["status"] != "ok":
        raise CacheError(f"{side['label']} can't list the cache: "
                         f"{result.get('error', {}).get('message', 'unknown error')}")
    records = [{"entry": entry, "offset": listed["offset"], "kind": listed["kind"],
                "path": str(directory / listed["file"]), "version": result["record_version"],
                "record_key": listed.get("record_key"), "relinked": listed.get("relinked", False),
                "environments": listed["stages"]}
               for entry, listed in enumerate(result["records"])]
    if any(record["record_key"] is None for record in records):
        raise CacheError(f"{side['label']} has no v2 record keys; build shader-sweep in the tree")
    info = {key: result[key] for key in ("path", "cache_version", "damaged", "migration", "log",
                                         "programs") if key in result}
    return info, records


def pair_records(index, source, records_a, records_b, sides):
    """Pairs records by the v2 record key, which leaves the driver out. Within a key the last
    record pairs, as the migration keeps the last; the others are collapsed. A key only a legacy
    side has was dropped by the migration; one only a v2 side has is one-sided."""
    keys = defaultdict(lambda: ([], []))
    for side, records in enumerate((records_a, records_b)):
        for record in records:
            keys[(record["kind"], record["record_key"])][side].append(record)
    pairs, left = [], []
    for (kind, record_key), (on_a, on_b) in keys.items():
        if on_a and on_b:
            pairs.append({"id": f"{index:03d}-{on_a[-1]['entry']:06d}", "source": source,
                          "record_key": record_key, "records": (on_a[-1], on_b[-1])})
            left += [(name, record, "collapsed") for name, records in (("a", on_a), ("b", on_b))
                     for record in records[:-1]]
            continue
        name, records = ("a", on_a) if on_a else ("b", on_b)
        other = sides[1 if name == "a" else 0]
        verdict = ("dropped" if other["path"] == "v2" and sides[0 if name == "a" else 1]["path"]
                   != "v2" else "one-sided")
        left += [(name, record, verdict) for record in records]
    rows = []
    for name, record, verdict in left:
        for env in record["environments"]:
            rows.append({"source": source, "record": f"{index:03d}-{name}{record['entry']:06d}",
                         "entry": record["entry"], "side": name, "record_offset": record["offset"],
                         "env_offset": env["offset"], "stage": env["stage"], "slot": None,
                         "record_key": record["record_key"], "verdict": verdict,
                         "detail": f"only side {name.upper()} has this record"})
    return sorted(pairs, key=lambda pair: (pair["records"][0]["entry"], pair["records"][1]["entry"])), rows


def key_collisions(rows):
    """Records with different SPIR-V that path B files under the same key, per cache file."""
    seen, collisions = {}, []
    for row in rows:
        if row.get("spirv_sha256_b") is None or row.get("key_b") is None:
            continue
        first = seen.setdefault((row["source"], row["key_b"]), row)
        if first["spirv_sha256_b"] != row["spirv_sha256_b"]:
            collisions.append({"source": row["source"], "key_b": row["key_b"],
                               "stage": row["stage"], "first": first["record"],
                               "first_env_offset": first["env_offset"], "second": row["record"],
                               "second_env_offset": row["env_offset"]})
    return collisions


def replay_summary(rows, collisions, input_errors, sides, listings=()):
    counts = Counter(row["verdict"] for row in rows)
    lines = [f"# Replay: A = {sides[0]['label']}, B = {sides[1]['label']}", "",
             "| Verdict | Stages |", "|---|---:|"]
    for verdict in ("match", "mismatch", "one-sided", "unreplayable", "collapsed", "dropped"):
        lines.append(f"| {verdict} | {counts[verdict]} |")
    lines += [f"| key collisions in B | {len(collisions)} |",
              f"| unreadable caches | {len(input_errors)} |", ""]
    for error in input_errors:
        lines.append(f"- Unreadable cache {error['source']}: {error['error']['message']}")
    for listing in listings:
        for name in ("a", "b"):
            info = listing[name]
            if info.get("path") != "v2":
                continue
            migration, programs = info.get("migration", {}), info.get("programs", {})
            lines.append(
                f"- {listing['source']}, side {name.upper()}: migrated from version "
                f"{migration.get('source_version')}, {migration.get('graphics')} graphics and "
                f"{migration.get('compute')} compute records, {migration.get('collapsed')} "
                f"collapsed; Program records {programs.get('records')} ({programs.get('hits')} "
                f"of this build, {programs.get('relowered')} lowered again, "
                f"{programs.get('unusable')} unusable), {programs.get('relinked_records')} "
                f"pipeline records moved to their code")
    identity = "| Source | Record | Env offset | Stage | Slot | Key A | Key B |"
    bad = [row for row in rows if row["verdict"] in ("mismatch", "one-sided")]
    if bad:
        lines += ["", "## Mismatches", "", identity[:-1] + " Verdict | Detail |",
                  "|---|---|---:|---|---:|---|---|---|---|"]
        for row in bad:
            if "detail" in row:
                detail = row["detail"]
            elif row["verdict"] == "mismatch":
                detail = (f"words {row['words_a']} vs {row['words_b']}, first difference at "
                          f"word {row.get('first_difference_word')}")
            else:
                side = "A" if "error_a" in row else "B"
                detail = f"only {side} failed: {row.get('error_' + side.lower(), {}).get('message', '')}"
            detail = detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['source']} | {row['record']} | {row['env_offset']} | {row['stage']} | "
                         f"{row['slot']} | {row.get('key_a', '')} | {row.get('key_b', '')} | {row['verdict']} | "
                         f"{detail} |")
    if collisions:
        lines += ["", "## Key collisions in B", "",
                  "| Source | Key B | Stage | First record (env offset) | Second record (env offset) |",
                  "|---|---|---|---|---|"]
        for c in collisions:
            lines.append(f"| {c['source']} | {c['key_b']} | {c['stage']} | {c['first']} "
                         f"({c['first_env_offset']}) | {c['second']} ({c['second_env_offset']}) |")
    causes = Counter()
    for row in rows:
        if row["verdict"] == "unreplayable":
            error = row.get("error_b") or row.get("error_a") or {}
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", error.get("message", ""))
            causes[(error.get("phase"), message)] += 1
    if causes:
        lines += ["", "## Unreplayable (both paths fail)", "", "| Phase | Cause | Stages |",
                  "|---|---|---:|"]
        for (phase, cause), count in sorted(causes.items(), key=lambda item: -item[1]):
            cause = cause.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {phase} | {cause} | {count} |")
    return "\n".join(lines) + "\n"


def replay(argv):
    parser = argparse.ArgumentParser(
        prog="shader_sweep.py replay",
        description="Compile every pipeline record of recorded caches through compilation paths "
                    "A and B and compare their SPIR-V word for word.")
    parser.add_argument("caches", nargs="+", type=Path,
                        help="vulkan.fork-vNN.bin files, or directories searched for them")
    parser.add_argument("--compiler-a", required=True, type=Path)
    parser.add_argument("--compiler-b", type=Path, help="defaults to --compiler-a")
    parser.add_argument("--path-a", default="legacy", help="compilation path A (shader-sweep --path)")
    parser.add_argument("--path-b", default="legacy", help="compilation path B (shader-sweep --path)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-spirv", action="store_true",
                        help="keep the modules of matching records as well")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    compilers = [args.compiler_a, args.compiler_b or args.compiler_a]
    if not all(compiler.is_file() for compiler in compilers):
        parser.error("--compiler-a and --compiler-b must name built shader-sweep executables")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    sides = [{"name": name, "compiler": str(compiler.resolve()), "path": path,
              "label": f"{compiler.name} --path {path}"}
             for name, compiler, path in zip("ab", compilers, (args.path_a, args.path_b))]
    files = []
    for path in args.caches:
        if path.is_dir():
            files += sorted(f for f in path.rglob("*.bin") if CACHE_NAME.fullmatch(f.name))
        elif path.is_file():
            files.append(path)
        else:
            parser.error(f"input does not exist: {path}")
    if not files:
        parser.error("no Vulkan pipeline caches found")
    records_dir = args.output / "records"
    records_dir.mkdir(parents=True)
    pairs, rows, input_errors, listings = [], [], [], []
    for index, path in enumerate(files):
        if "v2" in (args.path_a, args.path_b):
            # Each side reads the cache its own way; v2 migrates it into a log of its own.
            try:
                listed = [list_records(side, path, records_dir / side["name"] / f"{index:03d}",
                                       args.timeout) for side in sides]
            except CacheError as exc:
                input_errors.append({"source": str(path), **failure("input", str(exc))})
                continue
            for side in sides:
                if not args.keep_spirv:
                    shutil.rmtree(records_dir / side["name"] / f"{index:03d}" / "migration",
                                  ignore_errors=True)
            listings.append({"source": str(path), "a": listed[0][0], "b": listed[1][0]})
            paired, left = pair_records(index, str(path), listed[0][1], listed[1][1], sides)
            pairs += paired
            rows += left
            continue
        try:
            parsed = list(parse_records(path.read_bytes()))
        except (CacheError, OSError) as exc:
            input_errors.append({"source": str(path), **failure("input", str(exc))})
            continue
        for record in parsed:
            record_id = f"{index:03d}-{record['entry']:06d}"
            record_path = records_dir / f"{record_id}.record"
            record_path.write_bytes(record["bytes"])
            record = {**record, "path": str(record_path)}
            pairs.append({"id": record_id, "source": str(path), "records": (record, record)})
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        work = executor.map(lambda pair: replay_record(pair, sides, args.output, args.timeout,
                                                       args.keep_spirv), pairs)
        for done, record_rows in enumerate(work, 1):
            rows += record_rows
            if done % 100 == 0:
                print(f"replayed {done}/{len(pairs)} records", file=sys.stderr)
    collisions = key_collisions(rows)
    # Records that replay identically leave nothing behind; the others keep their record for
    # rerunning the compilers by hand.
    keep = {row["record"] for row in rows if row["verdict"] != "match"}
    keep |= {c[side] for c in collisions for side in ("first", "second")}
    if not args.keep_spirv:
        for pair in pairs:
            if pair["id"] not in keep:
                for record in pair["records"]:
                    Path(record["path"]).unlink(missing_ok=True)
    report = {"a": sides[0], "b": sides[1], "inputs": [str(f) for f in files],
              "input_errors": input_errors, "listings": listings, "collisions": collisions,
              "stages": rows}
    (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = replay_summary(rows, collisions, input_errors, sides, listings)
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    counts = Counter(row["verdict"] for row in rows)
    print(f"{len(files)} caches; {len(pairs)} record pairs; {len(rows)} stages; "
          f"{counts['mismatch']} mismatches; {counts['one-sided']} one-sided; "
          f"{counts['collapsed']} collapsed; {counts['dropped']} dropped; "
          f"{len(collisions)} key collisions")
    return int(bool(input_errors or collisions or counts["mismatch"] or counts["one-sided"]))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        return replay(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dump-game", default="Mario Tennis Fever")
    parser.add_argument("--native-stage", action="append", default=[], metavar="KEY=STAGE",
                        help="stage established by a capture log for a native filename hash")
    parser.add_argument("--contexts-from", type=Path,
                        help="prior results.json preserving native/cache matches across lowering changes")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    if not args.compiler.is_file():
        parser.error("--compiler must name the built shader-sweep executable")
    for path in args.inputs:
        if not path.exists():
            parser.error(f"input does not exist: {path}")
    native_stages = {}
    for mapping in args.native_stage:
        key, separator, stage = mapping.partition("=")
        if not separator or not re.fullmatch(r"[0-9a-f]{16}", key) or stage not in STAGES[:6]:
            parser.error("--native-stage requires a 16-digit lowercase hash and a graphics/compute stage")
        native_stages[key] = stage
    compiler = str(args.compiler.resolve())
    validator = shutil.which(args.spirv_val)
    if not validator:
        parser.error("spirv-val is required, install SPIRV-Tools or pass --spirv-val")
    args.output.mkdir(parents=True, exist_ok=True)
    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)
    files = sorted({f.resolve() for p in args.inputs for f in
                    (p.rglob("*.bin") if p.is_dir() else [p])})
    rows, jobs, cache_index = [], {}, defaultdict(list)
    prior_contexts = defaultdict(list)
    if args.contexts_from:
        for row in json.loads(args.contexts_from.read_text()):
            if row.get("kind") == "native" and "env" in row:
                prior_contexts[row["native_id"]].append(row)
    for path in files:
        if not CACHE_NAME.fullmatch(path.name):
            continue
        game = GAMES.get(path.parent.name.lower(), path.parent.name)
        try:
            envs = list(parse_cache(path.read_bytes()))
            for env in envs:
                identity = digest(game.encode() + env["bytes"] + env["key"] + bytes([env["version"]]))
                if identity not in jobs:
                    env_path = records_dir / f"{identity}.env"
                    key_path = records_dir / f"{identity}.key"
                    env_path.write_bytes(env["bytes"])
                    key_path.write_bytes(env["key"])
                    jobs[identity] = {"id": identity, "game": game, "stage": env["stage"],
                                      "source": str(path), "offset": env["offset"],
                                      "kind": "cache", "env": str(env_path),
                                      "key": str(key_path), "version": env["version"]}
                lowered = unpack_native_code(env["code"])
                if lowered is not None:
                    match = cache_index[digest(lowered)]
                    if identity not in match:
                        match.append(identity)
        except (CacheError, OSError) as exc:
            rows.append({"game": game, "source": str(path), **failure("input", str(exc))})
    header_stages = {}
    for path in files:
        rejected = re.fullmatch(r"rejected-stage-([1-5])-[0-9a-f]+\.bin", path.name)
        if rejected:
            header_stages[digest(path.read_bytes()[128:])] = STAGES[int(rejected[1]) - 1]
    raw_seen = set()
    for path in files:
        match = re.fullmatch(r"(?:[0-9a-f]+-(gfx|cs)-[0-9a-f]+|rejected-stage-(\d+)-[0-9a-f]+)\.bin", path.name)
        if not match:
            continue
        source = path.read_bytes()
        identity = digest(source)
        if identity in raw_seen:
            continue
        raw_seen.add(identity)
        rejected_stage = int(match[2]) if match[2] is not None else None
        stage = STAGES[rejected_stage - 1] if rejected_stage is not None and 1 <= rejected_stage <= 5 else None
        if match[1] == "cs":
            stage = "compute"
        if stage is None:
            stage = header_stages.get(identity)
        explicit_stage = native_stages.get(path.name.split("-")[0])
        if explicit_stage:
            if stage is not None and explicit_stage != stage:
                parser.error(f"explicit stage conflicts with capture metadata: {path}")
            stage = explicit_stage
        fmt = "native-header" if rejected_stage is not None else "instructions"
        lower_path = records_dir / f"{identity}.lowered"
        cmd = [compiler, str(path), "--format", fmt, "--lower-only", str(lower_path)]
        if stage:
            cmd += ["--stage", stage]
        result = invoke(cmd, args.timeout)
        if (stage is None and result["status"] != "ok" and
                "outside geometry stage" in result.get("error", {}).get("message", "")):
            stage = "geometry"
            result = invoke(cmd + ["--stage", stage], args.timeout)
        base = {"game": args.dump_game, "id": identity, "source": str(path),
                "native_id": identity, "stage": stage or "unknown", "kind": "native"}
        if result["status"] != "ok":
            offset = result.get("error", {}).get("offset")
            if result.get("phase") == "lower" and isinstance(offset, int):
                offset += 128 if fmt == "native-header" else 0
                if 0 <= offset <= len(source) - 16:
                    words = struct.unpack_from("<4I", source, offset)
                    base["opcode"] = f"0x{words[0] & 0xfff:03x}"
                    base["instruction"] = [f"{word:08x}" for word in words]
            rows.append({**base, **result})
            continue
        matches = [jobs[target] for target in cache_index.get(digest(lower_path.read_bytes()), [])
                   if jobs[target]["game"] == args.dump_game]
        if not matches:
            matches = prior_contexts.get(identity, [])
        if matches:
            compatible = False
            for cached in matches:
                if stage is not None and cached["stage"] != stage:
                    continue
                compatible = True
                new_id = digest((identity + cached["env"] + cached["key"]).encode())
                jobs[new_id] = {**cached, **base, "id": new_id, "stage": cached["stage"], "format": fmt}
            if not compatible:
                rows.append({**base, **failure("context", "cached stage disagrees with captured stage")})
        elif stage is not None:
            jobs[identity] = {**base, "format": fmt}
        else:
            rows.append({**base, **failure("context", "instruction-only dump has no matching cached stage/header")})

    def compile_one(job):
        output = records_dir / f"{job['id']}.spv"
        command = [compiler, "--output", str(output)]
        if job["kind"] == "native":
            command += [job["source"], "--stage", job["stage"], "--format", job["format"]]
        if "env" in job:
            command += ["--environment", job["env"], "--cache-version", str(job["version"])]
            if job["stage"] != "compute":
                command += ["--pipeline-key", job["key"]]
        result = invoke(command, args.timeout)
        if result["status"] == "ok":
            try:
                validation = subprocess.run([validator, "--target-env", "vulkan1.2", str(output)],
                                            capture_output=True, text=True, timeout=args.timeout)
                if validation.returncode:
                    result.update(failure("validate", validation.stderr.strip() or validation.stdout.strip()))
                else:
                    result["phase"] = "validated"
            except subprocess.TimeoutExpired:
                result.update(failure("timeout", "spirv-val timeout"))
        return {**job, **result, "command": command}

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for i, row in enumerate(executor.map(compile_one, jobs.values()), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"compiled {i}/{len(jobs)}", file=sys.stderr)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = summarize(rows)
    (args.output / "summary.md").write_text(report)
    print(report)
    print(f"{len(files)} input files; {len(raw_seen)} unique native captures; {len(jobs)} compile jobs")
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
