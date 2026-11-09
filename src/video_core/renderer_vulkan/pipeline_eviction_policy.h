// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

/// Built graphics pipelines kept on desktop before unused ones are destroyed. Destroying them
/// sooner costs far more than the 2-3 ms per rebuild once assumed: an 80-minute Ocarina of Time
/// session evicted 8,567 and rebuilt 8,158 of them on the GPU thread (35.5 s of builds, 66.7 s of
/// draws held), while no more than ~2,000 were alive at once.
inline constexpr size_t DefaultKeptPipelines = 4096;

/// Host memory one built graphics pipeline holds, rounded up: building OoT's 12,900 cached
/// pipelines at boot added 3.4 GiB of committed memory (~0.27 MiB each, NVIDIA).
inline constexpr u64 PipelineBytes = 300 * 1024;

/// The most built pipelines kept by default on any machine (~3.5 GiB).
inline constexpr size_t MaxKeptPipelines = 12288;

/// The most built pipelines kept by default on a machine with `total_memory` bytes of RAM: as many
/// as fit in a quarter of it, and never fewer than MaxKeptPipelines. A disk cache grows with every
/// new area (OoT: 13,023 pipelines after ~6 hours); past the cap, the pipelines of a screen used
/// now and then, like the pause menu, are evicted and built again each time it opens.
constexpr size_t MaxKeptPipelinesForMemory(u64 total_memory) {
    const u64 fit = total_memory / 4 / PipelineBytes;
    return fit < MaxKeptPipelines ? MaxKeptPipelines : static_cast<size_t>(fit);
}

/// Default limit for a session that loaded `cached_at_boot` pipelines from the disk cache: the
/// whole cache plus room for new content, so the cache's own pipelines are not evicted and built
/// again on the GPU thread (OoT, 7,316 cached with a limit of 4096: 3,925 evicted, 1,395 rebuilt),
/// up to `max_kept` (MaxKeptPipelinesForMemory).
constexpr size_t KeptPipelinesForCache(size_t cached_at_boot,
                                       size_t max_kept = MaxKeptPipelines) {
    const size_t wanted = cached_at_boot + DefaultKeptPipelines / 4;
    return wanted < DefaultKeptPipelines ? DefaultKeptPipelines
                                         : (wanted > max_kept ? max_kept : wanted);
}

/// Picks which unused pipelines to destroy so that at most `kept` stay built.
/// @param last_uses Frame each candidate was last used (0 for a prebuilt one no draw has used).
///                  Candidates are built pipelines old enough to evict and not the current one.
/// @param built     Built graphics pipelines in the cache, candidates included.
/// @param kept      How many built pipelines to keep, non-zero (a limit of 0 is the cache's old
///                  eviction, which does not use this).
/// @return Indices into `last_uses`, least recently used first (ties in input order): only as
///         many as it takes to get back to `kept`, or every candidate when that is not enough.
inline std::vector<size_t> PipelinesToEvict(std::span<const u64> last_uses, size_t built,
                                            size_t kept) {
    if (built <= kept) {
        return {};
    }
    std::vector<size_t> order(last_uses.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const size_t count = std::min(built - kept, order.size());
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return last_uses[a] < last_uses[b]; });
    order.resize(count);
    return order;
}

} // namespace Vulkan
