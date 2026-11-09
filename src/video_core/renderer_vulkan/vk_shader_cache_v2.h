// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/records.h"

namespace Vulkan {

struct GraphicsPipelineCacheKey;

/// The pipeline key a graphics or compute record (SerializePipelinePayload) ends with, read
/// without the rest of it. Nullopt when the record is too short to hold one.
template <typename Key>
[[nodiscard]] std::optional<Key> PipelineRecordKey(std::span<const u8> payload) {
    if (payload.size() < sizeof(Key)) {
        return std::nullopt;
    }
    Key key;
    std::memcpy(&key, payload.data() + payload.size() - sizeof(Key), sizeof(Key));
    return key;
}

/// Clears the driver of a graphics key, which the v2 cache keeps out of its keys: a driver update
/// would leave every recorded pipeline unmatched.
void ClearDriverFields(GraphicsPipelineCacheKey& key);

/// The v2 record key of a compute pipeline record (SerializePipelinePayload): a hash of all of it.
/// The translations rebuilt for one pipeline key stay beside each other, as in the old cache.
[[nodiscard]] u64 ComputeRecordKey(std::span<const u8> payload);

/// The v2 record key of a graphics pipeline record: its key's hash, driver cleared
[[nodiscard]] u64 GraphicsRecordKey(const GraphicsPipelineCacheKey& key);

struct LegacyCacheMigration {
    std::optional<u32> version; ///< The cache version migrated from
    u32 graphics{};
    u32 compute{};
    u32 collapsed{}; ///< Records a later one of the same record key replaced
    u32 used{};      ///< Used-list entries carried over
    bool failed{};   ///< There was a cache to migrate, but the log couldn't be written
};

/// Starts the v2 `log` from the newest Vulkan disk cache in `directory` (vulkan.fork-v<version>,
/// the current version first, then the compatible ones) and the used list beside it, unless the log
/// exists. Graphics keys lose their driver fields; records sharing a record key collapse into the
/// last one. A Migrated record names the source. The older files are only read.
LegacyCacheMigration MigrateLegacyPipelineCache(const std::filesystem::path& directory,
                                                const std::filesystem::path& log, u8 max_used_age);

/// A program this build lowers to other code than the build that recorded it
struct RelinkedProgram {
    u64 old_unique_hash{};
    /// What the old lowering made the environments' texture bound buffer (~0U: the engine's)
    u32 old_texture_bound{~0U};
    VideoCommon::ShaderCacheV2::ProgramRecord program; ///< Lowered by this build
};

/// Programs by the unique hash their pipeline records use
using RelinkedPrograms = std::unordered_map<u64, RelinkedProgram>;

/// Programs of another build's lowering, lowered by this one
struct RevalidatedPrograms {
    /// Same code and texture bound buffer: only the stamp is new
    std::vector<VideoCommon::ShaderCacheV2::ProgramRecord> unchanged;
    /// Other code or another texture bound buffer
    RelinkedPrograms relinked;
};

/// Lowers `programs` again, through `for_each` when given. A program this build can't lower is
/// in neither list: its pipeline records still match the build that recorded it.
[[nodiscard]] RevalidatedPrograms RevalidateProgramRecords(
    std::vector<VideoCommon::ShaderCacheV2::ProgramRecord> programs,
    const VideoCommon::ShaderCacheV2::ForEach& for_each = {});

struct RekeyedRecord {
    u64 key;
    std::vector<u8> payload;
};

/// A graphics or compute record with the code and texture bound buffer of the relinked programs it
/// uses replaced and its pipeline key moved to their new unique hashes, recorded values kept.
/// Nullopt when it uses none of them, or a stage's texture bound buffer came from the engine and
/// can't be told now. Throws std::ios_base::failure for a damaged record, as ReadPipelinePayload
/// does.
[[nodiscard]] std::optional<RekeyedRecord> RekeyPipelineRecord(
    VideoCommon::ShaderCacheV2::RecordKind kind, std::span<const u8> payload,
    const RelinkedPrograms& programs);

/// The records that move the graphics and compute records among `live` to the relinked
/// programs they use: each one's new record, then a tombstone for it when its key moved. Damaged
/// records are left alone. Stops early once `stop` is true.
[[nodiscard]] std::vector<VideoCommon::ShaderCacheV2::LogAppend> RelinkPipelineRecords(
    std::span<const VideoCommon::ShaderCacheV2::LogRecord> live, const RelinkedPrograms& programs,
    const std::function<bool()>& stop = {});

/// The used-list session key and time of a used list carried over: `age` sessions old
[[nodiscard]] u64 MigratedUsedSessionKey(u8 age);
[[nodiscard]] u64 MigratedUsedSessionTime(u8 age);

/// Graphics key hashes the newest `max_age` sessions of the log's Used records used, with how many
/// sessions ago each was last used (as the old used list holds them). `expired` gets the record
/// keys of older sessions.
[[nodiscard]] std::unordered_map<u64, u8> UsedPipelinesFromLog(
    std::span<const VideoCommon::ShaderCacheV2::LogRecord> live, u8 max_age,
    std::vector<u64>& expired);

/// The Used records of the newest `max_age` sessions that name a graphics pipeline `appends`
/// (RelinkPipelineRecords) moved to another key, with the pipeline under its new key. `used`
/// (UsedPipelinesFromLog) moves the same way.
[[nodiscard]] std::vector<VideoCommon::ShaderCacheV2::LogAppend> RelinkUsedPipelines(
    std::span<const VideoCommon::ShaderCacheV2::LogRecord> live,
    std::span<const VideoCommon::ShaderCacheV2::LogAppend> appends, u8 max_age,
    std::unordered_map<u64, u8>& used);

} // namespace Vulkan
