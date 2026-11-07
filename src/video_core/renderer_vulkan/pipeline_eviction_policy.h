// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

/// Built graphics pipelines kept on desktop before unused ones are destroyed: ~1.8 GB of driver
/// memory at ~0.45 MB each. Destroying them sooner costs far more than the 2-3 ms per rebuild once
/// assumed: an 80-minute Ocarina of Time session evicted 8,567 and rebuilt 8,158 of them on the GPU
/// thread (35.5 s of builds, 66.7 s of draws held), while no more than ~2,000 were alive at once.
inline constexpr size_t DefaultKeptPipelines = 4096;

/// Picks which unused pipelines to destroy so that at most `kept` stay built.
/// @param last_uses Frame each candidate was last used (0 for a prebuilt one no draw has used).
///                  Candidates are built pipelines old enough to evict and not the current one.
/// @param built     Built graphics pipelines in the cache, candidates included.
/// @param kept      How many built pipelines to keep; 0 destroys every candidate.
/// @return Indices into `last_uses`: with `kept` 0, every candidate in input order; otherwise the
///         least recently used first (ties in input order), only as many as it takes to get back
///         to `kept`, or every candidate when that is not enough.
inline std::vector<size_t> PipelinesToEvict(std::span<const u64> last_uses, size_t built,
                                            size_t kept) {
    std::vector<size_t> order(last_uses.size());
    std::iota(order.begin(), order.end(), size_t{0});
    if (kept == 0) {
        return order;
    }
    if (built <= kept) {
        return {};
    }
    const size_t count = std::min(built - kept, order.size());
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return last_uses[a] < last_uses[b]; });
    order.resize(count);
    return order;
}

} // namespace Vulkan
