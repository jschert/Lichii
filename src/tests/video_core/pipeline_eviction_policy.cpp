// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/pipeline_eviction_policy.h"

using Vulkan::DefaultKeptPipelines;
using Vulkan::KeptPipelinesForCache;
using Vulkan::MaxKeptPipelines;
using Vulkan::PipelinesToEvict;

namespace {
/// Candidates last used on frames 1, 2, ..., count.
std::vector<u64> Ascending(size_t count) {
    std::vector<u64> last_uses(count);
    for (size_t i = 0; i < count; ++i) {
        last_uses[i] = i + 1;
    }
    return last_uses;
}
} // Anonymous namespace

TEST_CASE("Nothing is evicted at or under the kept limit", "[video_core][pipeline_eviction]") {
    const auto last_uses = Ascending(1900);
    REQUIRE(PipelinesToEvict(last_uses, 1900, DefaultKeptPipelines).empty());
    REQUIRE(PipelinesToEvict(last_uses, DefaultKeptPipelines, DefaultKeptPipelines).empty());
    REQUIRE(PipelinesToEvict({}, 0, DefaultKeptPipelines).empty());
}

TEST_CASE("Past the limit only the oldest go, back down to the limit",
          "[video_core][pipeline_eviction]") {
    // 5,000 built, all evictable, listed newest first: the 904 oldest go.
    std::vector<u64> last_uses(5000);
    for (size_t i = 0; i < last_uses.size(); ++i) {
        last_uses[i] = 10'000 - i;
    }
    const auto evicted = PipelinesToEvict(last_uses, 5000, DefaultKeptPipelines);
    REQUIRE(evicted.size() == 5000 - DefaultKeptPipelines);
    for (size_t i = 0; i < evicted.size(); ++i) {
        REQUIRE(evicted[i] == last_uses.size() - 1 - i);
    }
}

TEST_CASE("Never-used prebuilt pipelines (frame 0) go first, ties in order",
          "[video_core][pipeline_eviction]") {
    const std::vector<u64> last_uses{500, 0, 200, 0, 900};
    REQUIRE(PipelinesToEvict(last_uses, 7, 4) == std::vector<size_t>{1, 3, 2});
}

TEST_CASE("Recently used pipelines count toward the limit but are never picked",
          "[video_core][pipeline_eviction]") {
    // 4,100 built but only 3 are old enough to evict: all 3 go, the limit is left exceeded.
    const auto last_uses = Ascending(3);
    REQUIRE(PipelinesToEvict(last_uses, 4100, DefaultKeptPipelines) ==
            std::vector<size_t>{0, 1, 2});
}

TEST_CASE("A limit of 0 evicts every candidate (old behaviour)",
          "[video_core][pipeline_eviction]") {
    const std::vector<u64> last_uses{30, 10, 20};
    REQUIRE(PipelinesToEvict(last_uses, 3, 0) == std::vector<size_t>{0, 1, 2});
    REQUIRE(PipelinesToEvict(last_uses, 1'000'000, 0).size() == 3);
}

TEST_CASE("The default limit fits the disk cache loaded at boot",
          "[video_core][pipeline_eviction]") {
    // A small or empty cache keeps the base limit.
    REQUIRE(KeptPipelinesForCache(0) == DefaultKeptPipelines);
    REQUIRE(KeptPipelinesForCache(1600) == DefaultKeptPipelines);
    // OoT's 7,316 cached pipelines all fit, with room for 1,024 new ones.
    REQUIRE(KeptPipelinesForCache(7316) == 7316 + DefaultKeptPipelines / 4);
    // A huge cache is capped, so it can't hold its driver memory without bound.
    REQUIRE(KeptPipelinesForCache(48'000) == MaxKeptPipelines);
    static_assert(KeptPipelinesForCache(MaxKeptPipelines) == MaxKeptPipelines);
}
