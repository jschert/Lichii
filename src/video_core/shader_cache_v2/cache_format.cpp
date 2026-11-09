// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_map>

#include <boost/crc.hpp>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "video_core/shader_cache_v2/cache_format.h"

namespace VideoCommon::ShaderCacheV2 {
namespace {

template <typename T>
T Load(std::span<const u8> bytes, u64 offset) {
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void Store(u8* out, T value) {
    std::memcpy(out, &value, sizeof(T));
}

u32 Crc32(const u8* data, size_t size) {
    boost::crc_32_type crc;
    crc.process_bytes(data, size);
    return crc.checksum();
}

u64 PayloadHash(u16 kind, u64 key, std::span<const u8> payload) {
    return XXH3_64bits_withSeed(payload.data(), payload.size(), key ^ (u64{kind} << 48));
}

struct RecordHeader {
    u16 kind;
    u16 flags;
    u32 payload_size;
    u64 key;
};

/// The record header at `offset` (which leaves room for one), when it is intact
std::optional<RecordHeader> ReadRecordHeader(std::span<const u8> bytes, u64 offset) {
    if (Load<u32>(bytes, offset) != RecordMagic ||
        Load<u32>(bytes, offset + 20) != Crc32(bytes.data() + offset, 20)) {
        return std::nullopt;
    }
    const RecordHeader header{
        .kind = Load<u16>(bytes, offset + 4),
        .flags = Load<u16>(bytes, offset + 6),
        .payload_size = Load<u32>(bytes, offset + 8),
        .key = Load<u64>(bytes, offset + 12),
    };
    if (header.kind == 0 || header.payload_size > MaxPayloadSize) {
        return std::nullopt;
    }
    return header;
}

/// The next 8-byte aligned offset from `offset` holding an intact record header, else the size
u64 FindRecord(std::span<const u8> bytes, u64 offset) {
    for (; offset + RecordHeaderSize <= bytes.size(); offset += RecordAlignment) {
        if (ReadRecordHeader(bytes, offset)) {
            return offset;
        }
    }
    return bytes.size();
}

bool PayloadIntact(std::span<const u8> bytes, u64 offset, u16 kind, u64 key,
                   std::span<const u8> payload) {
    return Load<u64>(bytes, offset + RecordHeaderSize + payload.size()) ==
           PayloadHash(kind, key, payload);
}

/// Walks the records after the file header, passing each one whose payload `payload_intact` takes
/// for intact to `on_record` with the bytes it takes in the file, and counts what it passes over
template <typename PayloadIntactFn, typename OnRecord>
void WalkRecords(std::span<const u8> bytes, LogStats& stats, PayloadIntactFn&& payload_intact,
                 OnRecord&& on_record) {
    u64 offset = FileHeaderSize;
    while (offset < bytes.size()) {
        const u64 left = bytes.size() - offset;
        if (left < RecordHeaderSize) {
            stats.torn_tail_bytes = left;
            break;
        }
        const std::optional<RecordHeader> header = ReadRecordHeader(bytes, offset);
        if (!header) {
            ++stats.bad_headers;
            const u64 next = FindRecord(bytes, offset + RecordAlignment);
            stats.skipped_bytes += next - offset;
            offset = next;
            continue;
        }
        // The padding of the last record may be cut off.
        const u64 framed = u64{RecordHeaderSize} + header->payload_size + RecordHashSize;
        if (framed > left) {
            const u64 next = FindRecord(bytes, offset + RecordAlignment);
            if (next == bytes.size()) {
                stats.torn_tail_bytes = left;
                break;
            }
            // Cut off by a crash, then appended to
            ++stats.bad_payloads;
            stats.skipped_bytes += next - offset;
            offset = next;
            continue;
        }
        const auto payload = bytes.subspan(offset + RecordHeaderSize, header->payload_size);
        if (!payload_intact(offset, *header, payload)) {
            ++stats.bad_payloads;
            const u64 next = FindRecord(bytes, offset + RecordAlignment);
            stats.skipped_bytes += next - offset;
            offset = next;
            continue;
        }
        const u64 size = std::min(RecordSize(header->payload_size), left);
        on_record(offset, *header, payload, size);
        offset += size;
    }
}

struct KindKey {
    u16 kind;
    u64 key;
    bool operator==(const KindKey&) const = default;
};

struct KindKeyHash {
    size_t operator()(const KindKey& value) const noexcept {
        return static_cast<size_t>(value.key ^ (u64{value.kind} * 0x9e3779b97f4a7c15ULL));
    }
};

} // Anonymous namespace

u64 Hash64(std::span<const u8> bytes, u64 seed) {
    return XXH3_64bits_withSeed(bytes.data(), bytes.size(), seed);
}

std::array<u8, FileHeaderSize> EncodeFileHeader(u64 created_unix) {
    std::array<u8, FileHeaderSize> header{};
    std::memcpy(header.data(), FileMagic.data(), FileMagic.size());
    Store<u32>(header.data() + 8, FormatVersion);
    Store<u32>(header.data() + 12, FileHeaderSize);
    Store<u64>(header.data() + 16, created_unix);
    Store<u32>(header.data() + 24, Crc32(header.data(), 24));
    return header;
}

void EncodeRecord(std::vector<u8>& out, RecordKind kind, u64 key, std::span<const u8> payload,
                  u16 flags) {
    std::array<u8, RecordHeaderSize> header{};
    Store<u32>(header.data(), RecordMagic);
    Store<u16>(header.data() + 4, static_cast<u16>(kind));
    Store<u16>(header.data() + 6, flags);
    Store<u32>(header.data() + 8, static_cast<u32>(payload.size()));
    Store<u64>(header.data() + 12, key);
    Store<u32>(header.data() + 20, Crc32(header.data(), 20));
    std::array<u8, RecordHashSize> hash{};
    Store<u64>(hash.data(), PayloadHash(static_cast<u16>(kind), key, payload));
    const size_t end = out.size() + RecordSize(payload.size());
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), payload.begin(), payload.end());
    out.insert(out.end(), hash.begin(), hash.end());
    out.resize(end);
}

std::vector<u8> TombstonePayload(RecordKind kind) {
    std::vector<u8> payload(sizeof(u16));
    Store<u16>(payload.data(), static_cast<u16>(kind));
    return payload;
}

std::optional<LogRecord> ParseRecord(std::span<const u8> bytes, u64 offset) {
    if (bytes.size() < RecordHeaderSize) {
        return std::nullopt;
    }
    const std::optional<RecordHeader> header = ReadRecordHeader(bytes, 0);
    if (!header || u64{RecordHeaderSize} + header->payload_size + RecordHashSize > bytes.size()) {
        return std::nullopt;
    }
    const auto payload = bytes.subspan(RecordHeaderSize, header->payload_size);
    if (Load<u64>(bytes, RecordHeaderSize + header->payload_size) !=
        PayloadHash(header->kind, header->key, payload)) {
        return std::nullopt;
    }
    return LogRecord{.kind = static_cast<RecordKind>(header->kind),
                     .flags = header->flags,
                     .key = header->key,
                     .offset = offset,
                     .payload = payload};
}

ParsedLog ParseLog(std::span<const u8> bytes, const ForEach& for_each) {
    ParsedLog log;
    LogStats& stats = log.stats;
    stats.file_size = bytes.size();
    if (bytes.empty()) {
        log.header = HeaderStatus::Empty;
        return log;
    }
    if (bytes.size() < FileHeaderSize) {
        log.header = HeaderStatus::Damaged;
        stats.torn_tail_bytes = bytes.size();
        stats.dead_bytes = bytes.size();
        return log;
    }
    if (std::memcmp(bytes.data(), FileMagic.data(), FileMagic.size()) == 0 &&
        Load<u32>(bytes, 24) == Crc32(bytes.data(), 24)) {
        if (Load<u32>(bytes, 8) != FormatVersion) {
            log.header = HeaderStatus::OtherVersion;
            return log;
        }
        log.header = Load<u32>(bytes, 12) == FileHeaderSize ? HeaderStatus::Valid
                                                             : HeaderStatus::Damaged;
        log.created_unix = Load<u64>(bytes, 16);
    } else {
        // The records check themselves: a damaged file header loses none of them.
        log.header = HeaderStatus::Damaged;
    }

    struct Entry {
        LogRecord record;
        u64 size;
        bool live;
    };
    std::vector<Entry> entries;
    std::unordered_map<KindKey, size_t, KindKeyHash> latest;
    const auto remove = [&](const KindKey& target) {
        if (const auto it = latest.find(target); it != latest.end()) {
            entries[it->second].live = false;
            latest.erase(it);
        }
    };

    // Hashes checked on the threads of for_each, for the records found where every payload is
    // taken for intact. The walk below checks those it finds anywhere else itself.
    struct Checked {
        u64 offset;
        std::span<const u8> payload;
        u16 kind;
        u64 key;
        u8 intact; ///< 0 not checked, 1 intact, 2 damaged
    };
    std::vector<Checked> checked;
    if (for_each) {
        LogStats ignored;
        WalkRecords(
            bytes, ignored, [](u64, const RecordHeader&, std::span<const u8>) { return true; },
            [&](u64 offset, const RecordHeader& header, std::span<const u8> payload, u64) {
                checked.push_back({offset, payload, header.kind, header.key, 0});
            });
        const size_t chunks = std::min<size_t>(checked.size(), 256);
        for_each(chunks, [&](size_t chunk) {
            const size_t end = (chunk + 1) * checked.size() / chunks;
            for (size_t index = chunk * checked.size() / chunks; index < end; ++index) {
                Checked& record = checked[index];
                record.intact = PayloadIntact(bytes, record.offset, record.kind, record.key,
                                              record.payload)
                                    ? 1
                                    : 2;
            }
        });
    }
    const auto payload_intact = [&](u64 offset, const RecordHeader& header,
                                    std::span<const u8> payload) {
        const auto it = std::ranges::lower_bound(checked, offset, {}, &Checked::offset);
        if (it != checked.end() && it->offset == offset && it->intact != 0) {
            return it->intact == 1;
        }
        return PayloadIntact(bytes, offset, header.kind, header.key, payload);
    };
    WalkRecords(bytes, stats, payload_intact,
                [&](u64 offset, const RecordHeader& header, std::span<const u8> payload,
                    u64 size) {
                    ++stats.records;
                    const KindKey id{header.kind, header.key};
                    if (static_cast<RecordKind>(header.kind) == RecordKind::Tombstone) {
                        ++stats.tombstones;
                        stats.dead_bytes += size;
                        if (payload.size() == sizeof(u16)) {
                            remove({Load<u16>(payload, 0), header.key});
                        }
                    } else {
                        remove(id);
                        latest.emplace(id, entries.size());
                        entries.push_back({
                            .record = {.kind = static_cast<RecordKind>(header.kind),
                                       .flags = header.flags,
                                       .key = header.key,
                                       .offset = offset,
                                       .payload = payload},
                            .size = size,
                            .live = true,
                        });
                    }
                });

    for (const Entry& entry : entries) {
        if (entry.live) {
            log.live.push_back(entry.record);
        } else {
            ++stats.superseded;
            stats.dead_bytes += entry.size;
        }
    }
    stats.live = log.live.size();
    stats.dead_bytes += stats.skipped_bytes + stats.torn_tail_bytes;
    return log;
}

} // namespace VideoCommon::ShaderCacheV2
