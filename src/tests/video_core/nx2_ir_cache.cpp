// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "shader_recompiler/stage.h"
#include "video_core/nx2_ir_cache.h"

namespace {

using namespace VideoCommon::Nx2IrCache;

// Mirrors the constants in nx2_ir_cache.cpp; the malformed record below is crafted by hand.
constexpr std::array<char, 8> FileMagic{'S', '2', 'I', 'R', 'C', 'A', 'C', 'H'};
constexpr u32 RecordMagic = 0x52493253; // "S2IR"
constexpr u32 FileVersion = 3;
constexpr u64 BuildTag = 0x1234567890abcdefull;

static_assert(sizeof(Shader::Stage) == sizeof(u32));

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-nx2-ir-cache-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

Record MakeRecord(u64 hash, size_t answer_bytes) {
    Record record;
    record.unique_hash = hash;
    record.stage = Shader::Stage::VertexB;
    record.texture_bound_used = true;
    record.texture_bound = 2;
    record.answers.resize(answer_bytes);
    for (size_t i = 0; i < answer_bytes; ++i) {
        record.answers[i] = static_cast<u8>(i * 7 + 1);
    }
    // LoadProgram decompresses this only on demand; Find must return the record without it.
    record.program = {'p', 'r', 'o', 'g', 'r', 'a', 'm'};
    return record;
}

void PutU32(std::vector<u8>& out, u32 value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

void PutU64(std::vector<u8>& out, u64 value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}

} // Anonymous namespace

// Find reads a HeaderReadSize (2048) prefix first and re-reads the whole record only when parsing
// runs past it. The answers count guard is evaluated against the full record, so a record whose
// answers straddle the prefix takes the full-record retry instead of being dropped as one this
// build did not write.
TEST_CASE("StageCache::Find loads answers past the header read", "[video_core][nx2_ir_cache]") {
    TempDir dir;
    const auto path = dir.path / "vulkan_ir.bin";

    StageCache writer;
    REQUIRE(writer.Open(path, BuildTag, true));
    const auto small = MakeRecord(0x1111, 64);
    writer.Append(small);
    // 3000 answer bytes start at body offset 40 and end at 3040: the 2048-byte prefix read cuts
    // inside them, while the count itself stays inside the prefix.
    const auto large = MakeRecord(0x2222, 3000);
    writer.Append(large);

    StageCache cache;
    REQUIRE(cache.Open(path, BuildTag, false));
    REQUIRE(cache.Find(0x2222).size() == 1);
    const auto found = cache.Find(0x2222);
    REQUIRE(found.size() == 1);
    CHECK(found[0].unique_hash == 0x2222);
    CHECK(found[0].stage == Shader::Stage::VertexB);
    CHECK(found[0].texture_bound == 2);
    CHECK(found[0].answers == large.answers);
    // Control: a record that fits the prefix takes the same path without the retry.
    const auto found_small = cache.Find(0x1111);
    REQUIRE(found_small.size() == 1);
    CHECK(found_small[0].answers == small.answers);
}

// A count that exceeds even the full record's remaining bytes must be rejected before the answers
// resize allocates: answers is a byte vector, so the bound keeps the allocation within the bytes
// the record actually holds.
TEST_CASE("StageCache::Find rejects answer counts larger than the record",
          "[video_core][nx2_ir_cache]") {
    TempDir dir;
    const auto path = dir.path / "vulkan_ir.bin";

    std::vector<u8> file;
    file.insert(file.end(), FileMagic.begin(), FileMagic.end());
    PutU32(file, FileVersion);
    PutU64(file, BuildTag);
    std::vector<u8> record;
    PutU32(record, RecordMagic);
    PutU32(record, u32{}); // record size, patched below
    PutU64(record, 0x900d);
    PutU32(record, static_cast<u32>(Shader::Stage::VertexB));
    PutU32(record, 0);           // texture bound used / viewport / local memory flags
    PutU32(record, 0);           // texture bound
    PutU32(record, 1);           // viewport transform state
    PutU32(record, 0);           // local memory size
    PutU32(record, 0);           // storage pointer count
    PutU32(record, 0);           // texture check count
    PutU32(record, 0x80000000u); // answers count: 2 GiB from a 48-byte record
    file.insert(file.end(), record.begin(), record.end());
    const u32 record_size = static_cast<u32>(record.size());
    std::memcpy(file.data() + file.size() - record.size() + sizeof(u32), &record_size,
                sizeof(record_size));
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(file.data()),
                  static_cast<std::streamsize>(file.size()));
    }

    StageCache cache;
    REQUIRE(cache.Open(path, BuildTag, false));
    CHECK(cache.Contains(0x900d)); // indexed by the magic and size, then rejected at parse
    CHECK(cache.Find(0x900d).empty());
}
