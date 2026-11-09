// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

/// Shader cache v2, the Vulkan pipeline cache: one append-only log of framed records per title.
///
/// File header (32 bytes): "LXSHCACH", u32 format version, u32 header size, u64 creation time
/// (Unix seconds), u32 CRC-32 of the bytes before it, u32 zero.
/// Record, at an 8-byte aligned offset: u32 magic 'LXCR', u16 kind, u16 flags, u32 payload size,
/// u64 key, u32 CRC-32 of the bytes before it; the payload; the payload's XXH3-64 seeded with the
/// kind and key; zeros up to the next 8-byte boundary.
///
/// The latest record of a (kind, key) wins. A tombstone record removes the earlier record of its
/// key and of the kind its payload names. Kinds this build doesn't know are kept as they are.
namespace VideoCommon::ShaderCacheV2 {

inline constexpr std::array<char, 8> FileMagic{'L', 'X', 'S', 'H', 'C', 'A', 'C', 'H'};
inline constexpr u32 FormatVersion = 2;
inline constexpr u32 FileHeaderSize = 32;
inline constexpr u32 RecordMagic = 0x5243584C; // "LXCR"
inline constexpr u32 RecordHeaderSize = 24;
inline constexpr u32 RecordHashSize = 8;
inline constexpr u32 RecordAlignment = 8;
/// A larger payload is taken for a damaged size
inline constexpr u32 MaxPayloadSize = 256u << 20;

enum class RecordKind : u16 {
    Program = 1,
    Graphics = 2,
    Compute = 3,
    Tombstone = 4,
    Used = 5,
    Migrated = 6,
    /// Driver blob files (driver_blobs.h): the device the file is for, and the binaries of the
    /// pipeline with the record's key
    DeviceIdentity = 7,
    GraphicsBinary = 8,
    ComputeBinary = 9,
};

/// Bytes a record with a payload of this size takes in the file
[[nodiscard]] constexpr u64 RecordSize(u64 payload_size) {
    return (RecordHeaderSize + payload_size + RecordHashSize + RecordAlignment - 1) /
           RecordAlignment * RecordAlignment;
}

/// XXH3-64 of `bytes`
[[nodiscard]] u64 Hash64(std::span<const u8> bytes, u64 seed = 0);

[[nodiscard]] std::array<u8, FileHeaderSize> EncodeFileHeader(u64 created_unix);

/// Appends one framed record to `out`, which must end at an 8-byte boundary of the file.
void EncodeRecord(std::vector<u8>& out, RecordKind kind, u64 key, std::span<const u8> payload,
                  u16 flags = 0);

/// The payload of a tombstone for the record of this kind (its key is the record's key)
[[nodiscard]] std::vector<u8> TombstonePayload(RecordKind kind);

enum class HeaderStatus {
    Valid,
    Empty,        ///< No bytes at all
    Damaged,      ///< Cut short, or not a valid header: nothing in the file can be trusted
    OtherVersion, ///< Another format version's file, never changed by this build
};

/// A record to append to a log
struct LogAppend {
    RecordKind kind;
    u64 key;
    std::vector<u8> payload;

    bool operator==(const LogAppend&) const = default;
};

struct LogRecord {
    RecordKind kind;
    u16 flags;
    u64 key;
    u64 offset;                  ///< Where the record begins in the file
    std::span<const u8> payload; ///< Into the bytes that were parsed
};

struct LogStats {
    u64 file_size{};
    u64 records{};      ///< Well-formed records, tombstones included
    u64 live{};         ///< Records no later record replaces or removes
    u64 superseded{};   ///< Replaced by a later record of their kind and key, or removed
    u64 tombstones{};
    u64 bad_headers{};  ///< Damaged record headers, each followed by a scan for the next record
    u64 bad_payloads{}; ///< Records dropped because their payload doesn't match its hash
    u64 skipped_bytes{};    ///< Bytes passed over while scanning for the next record
    u64 torn_tail_bytes{};  ///< A record cut off at the end of the file
    u64 dead_bytes{};   ///< Bytes compaction would remove
};

struct ParsedLog {
    HeaderStatus header{};
    u64 created_unix{};
    /// The live records in the order of the file
    std::vector<LogRecord> live;
    LogStats stats;
};

/// Calls its task with every index below its count, on any threads, and returns once every call
/// returned. A call left out is made by the caller of the task instead.
using ForEach = std::function<void(size_t count, const std::function<void(size_t)>& task)>;
/// The record at the start of `bytes` (read from the file at `offset`), when its header and
/// payload are intact. Its payload points into `bytes`.
[[nodiscard]] std::optional<LogRecord> ParseRecord(std::span<const u8> bytes, u64 offset = 0);

/// Reads a whole log held in memory. Never throws on any input: a damaged record header is
/// skipped by scanning for the next valid one, a record whose payload doesn't match its hash is
/// dropped alone, and a record cut off at the end is ignored. With `for_each`, the payload hashes
/// are checked on its threads; the result is the same.
[[nodiscard]] ParsedLog ParseLog(std::span<const u8> bytes, const ForEach& for_each = {});

} // namespace VideoCommon::ShaderCacheV2
