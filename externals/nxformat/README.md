# NXFormat reader

Header-only reader for NXFormat (`.nx`) archives, used by the NX loader
(`src/core/loader/nx.cpp`). The Qt frontend's NX converter uses nxformat's prebuilt C library
instead, downloaded by `CMakeModules/NxFormat.cmake`.

- Upstream: https://github.com/fab918/nxformat by fab918
- Version: tag `V1.0.0`, commit `c70daf6f35d3223d83603dee293f92df7a8f815c`
- License: MIT (`LICENSE`)

The tag first pointed at `ddf67861b58ec2d9611e80930670601a65c4a9c4`. On 2026-11-08 it was moved to
`c70daf6f`, which changes only `packer/src/repack.rs` and `packer/src/writer.rs`, and the release
archives were rebuilt. The headers below are the same at both commits.

| File | Upstream path | SHA-256 |
|---|---|---|
| `nx_reader.hpp` | `dist/nx_reader.hpp` | `db87154f6f1849782f510826db3bdfa16fa92e3f697f89922dfb8e83851c481a` |
| `nx_session.hpp` | `dist/nx_session.hpp` | `fdd9fabec4264fc7b180061fb60122f938b1d4851bbff08b7bc8d0d8c35c9fc8` |
| `nx_disk_source.hpp` | `dist/nx_disk_source.hpp` | `35ce56e0c1316b55353d56910751869befccced0236502ec9743629ff1fa23f2` |
| `LICENSE` | `LICENSE` | `c803c9420a16a056c0ed7459c30c804ebe35ddee989f068ef6db2031ca24bf1a` |

They are identical to the `include/` copies in the release archives (the Windows zip stores them
with CRLF line endings). No local patches.

Release archives pinned in `CMakeModules/NxFormat.cmake`. Each sum matches a fresh download, the
release's `SHA256SUMS` and the GitHub asset digest:

| Archive | SHA-256 |
|---|---|
| `nxformat-1.0.0-linux-x64.tar.gz` | `201481663a54f1d38f4a2dc4b1bf1084d995a899ff4237a78506646b1bd144e1` |
| `nxformat-1.0.0-linux-arm64.tar.gz` | `c48952e4f3bcd92fd43e0317c3a534085c4be8bb32a8e27ed7880649549dd66c` |
| `nxformat-1.0.0-macos-universal.tar.gz` | `efe81f6a28b285af11be4ea35b7097d94134f9a81c604c3fc7764876575aa458` |
| `nxformat-1.0.0-windows-x64.zip` | `04da098da72d302816c1d8a39da926f1c5310505f9eda17fb557be44042cb173` |

Provenance, checked 2026-11-08:

- `github-actions[bot]` created the release and uploaded every asset (18:34:42Z). They came from
  Release run 42509964340 (run 4, a push of tag `V1.0.0` by fab918, head `c70daf6f`, success).
  Its `build (linux-x64, ubuntu-22.04, …)` job ran 18:27:02Z–18:30:24Z.
- The earlier assets came from run 42501937265 (run 3, workflow_dispatch on `main` at `ddf67861`,
  success). `ddf67861` is no longer on any branch or tag: `c70daf6f` replaced it on `main` with
  the same author date.
- Rebuilt from source at `c70daf6f` the way `release.yml` does (Ubuntu 22.04, Rust 1.99.1 as in
  the library's `.comment`, `cargo build --release --locked --target x86_64-unknown-linux-gnu
  --manifest-path ffi/Cargo.toml` under `/home/runner/work/nxformat/nxformat`):
  `libnxformat.so` and `libnxformat_static.a` are byte-identical to the linux-x64 archive's.
  Lichii uses only `libnxformat.so`; the archive's `nxconvert` app was not reproduced.
