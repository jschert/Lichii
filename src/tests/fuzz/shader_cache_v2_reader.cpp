// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// libFuzzer target for the shader cache v2 reader: any bytes parse without a fault, and the live
// records it finds write back into a log that reads as the same records with nothing dead.

#include <algorithm>
#include <cstdlib>
#include <span>
#include <vector>

#include "video_core/shader_cache_v2/cache_format.h"

namespace V2 = VideoCommon::ShaderCacheV2;

extern "C" int LLVMFuzzerTestOneInput(const u8* data, size_t size) {
    const V2::ParsedLog parsed = V2::ParseLog(std::span<const u8>{data, size});
    if (parsed.stats.live != parsed.live.size() || parsed.stats.records < parsed.live.size() ||
        parsed.stats.dead_bytes > size) {
        std::abort();
    }
    const auto header = V2::EncodeFileHeader(parsed.created_unix);
    std::vector<u8> rewritten(header.begin(), header.end());
    for (const V2::LogRecord& record : parsed.live) {
        if (record.payload.data() < data || record.payload.data() + record.payload.size() > data + size) {
            std::abort();
        }
        V2::EncodeRecord(rewritten, record.kind, record.key, record.payload, record.flags);
    }
    const V2::ParsedLog again = V2::ParseLog(rewritten);
    if (again.header != V2::HeaderStatus::Valid || again.live.size() != parsed.live.size() ||
        again.stats.dead_bytes != 0) {
        std::abort();
    }
    for (size_t i = 0; i < parsed.live.size(); ++i) {
        const V2::LogRecord& before = parsed.live[i];
        const V2::LogRecord& after = again.live[i];
        if (before.kind != after.kind || before.key != after.key || before.flags != after.flags ||
            !std::ranges::equal(before.payload, after.payload)) {
            std::abort();
        }
    }
    return 0;
}
