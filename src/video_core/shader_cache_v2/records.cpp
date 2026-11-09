// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstddef>
#include <cstring>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/stage.h"
#include "video_core/native_lowering.h"
#include "video_core/shader_cache_v2/cache_format.h"
#include "video_core/shader_cache_v2/lowering_stamp.h"
#include "video_core/shader_cache_v2/payload_io.h"
#include "video_core/shader_cache_v2/records.h"

namespace VideoCommon::ShaderCacheV2 {
namespace {

constexpr u32 ProgramPayloadVersion = 1;

} // Anonymous namespace

u64 LoweringStamp() {
    return GeneratedLoweringStamp;
}

ProgramKeyInputs MakeProgramKeyInputs(Shader::Stage stage, u32 header_size,
                                      const Shader::ProgramHeader& sph,
                                      std::span<const Shader::Ampere::Instruction> native,
                                      const NativeLoweringInputs& inputs,
                                      std::span<const u32> verified_traps) {
    return {
        .source = stage == Shader::Stage::Compute ? ProgramSource::NativeCompute
                                                  : ProgramSource::NativeGraphics,
        .stage = static_cast<u32>(stage),
        .header_size = header_size,
        .sph = {reinterpret_cast<const u8*>(&sph), sizeof(sph)},
        .geometry = inputs.geometry,
        .tessellation_stage = static_cast<u32>(inputs.tessellation_stage),
        .native_code_address =
            LoadsProgramCounter(native, inputs.geometry) ? inputs.native_code_address : 0,
        .verified_traps = verified_traps,
        .native = native,
    };
}

ProgramKeyInputs ProgramKeyInputsOf(const ProgramRecord& record) {
    return {
        .source = record.source,
        .stage = record.stage,
        .header_size = record.header_size,
        .sph = record.sph,
        .geometry = record.geometry,
        .tessellation_stage = record.tessellation_stage,
        .native_code_address = record.native_code_address,
        .verified_traps = record.verified_traps,
        .native = record.native,
    };
}

ProgramRecord MakeProgramRecord(Shader::Stage stage, u32 header_size,
                                const Shader::ProgramHeader& sph,
                                std::span<const Shader::Ampere::Instruction> native,
                                const NativeLoweringInputs& inputs,
                                const LoweredNativeCode& lowered) {
    const std::vector<u32> verified_traps = NativeTrapOffsets(native);
    const ProgramKeyInputs key =
        MakeProgramKeyInputs(stage, header_size, sph, native, inputs, verified_traps);
    ProgramRecord record;
    record.source = key.source;
    record.stage = key.stage;
    record.header_size = header_size;
    std::memcpy(record.sph.data(), &sph, sizeof(sph));
    record.native.assign(native.begin(), native.end());
    record.geometry = key.geometry;
    record.tessellation_stage = key.tessellation_stage;
    record.verified_traps = verified_traps;
    record.native_code_address = key.native_code_address;
    record.lowering_stamp = LoweringStamp();
    record.lowered_code = lowered.code;
    record.texture_bound = lowered.texture_bound;
    record.source_size = lowered.source_size;
    record.unique_hash = NativeShaderHash(lowered.code);
    return record;
}

std::shared_ptr<LoweredNativeCode> RelowerProgram(const ProgramRecord& record) {
    const NativeLoweringInputs inputs{
        .geometry = record.geometry,
        .tessellation_stage =
            static_cast<Shader::Ampere::TessellationStage>(record.tessellation_stage),
        .native_code_address = record.native_code_address,
    };
    auto lowered = LowerNativeProgram(record.native, inputs);
    if (!lowered) {
        return nullptr;
    }
    Shader::ProgramHeader sph;
    std::memcpy(&sph, record.sph.data(), sizeof(sph));
    return MakeLoweredNativeCode(record.native, sph, record.header_size,
                                 NativeTrapOffsets(record.native), lowered);
}

std::optional<ProgramRecord> RelowerProgramRecord(const ProgramRecord& record) {
    if (record.native.empty()) {
        return std::nullopt;
    }
    const auto lowered = RelowerProgram(record);
    if (!lowered) {
        return std::nullopt;
    }
    ProgramRecord relowered = record;
    relowered.lowering_stamp = LoweringStamp();
    relowered.lowered_code = lowered->code;
    relowered.texture_bound = lowered->texture_bound;
    relowered.source_size = lowered->source_size;
    relowered.unique_hash = NativeShaderHash(lowered->code);
    return relowered;
}

std::shared_ptr<LoweredNativeCode> TakeProgramCode(ProgramRecord& record) {
    const size_t trap_count =
        record.verified_traps.empty()
            ? 0
            : record.verified_traps.back() / sizeof(Shader::Ampere::Instruction) + 1;
    auto code = std::make_shared<LoweredNativeCode>();
    code->native_program_key = Shader::Ampere::NativeProgramKey(record.native);
    code->code = std::move(record.lowered_code);
    code->trap_prefix.assign(record.native.begin(),
                             record.native.begin() +
                                 static_cast<std::ptrdiff_t>(
                                     std::min(trap_count, record.native.size())));
    code->source_size = static_cast<size_t>(record.source_size);
    code->texture_bound = record.texture_bound;
    return code;
}

u64 ProgramKey(const ProgramKeyInputs& inputs) {
    // Streamed: the hash of the fields laid out one after another, without copying the program
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    const auto put = [&state](const void* data, size_t size) {
        XXH3_64bits_update(&state, data, size);
    };
    const u64 trap_count = inputs.verified_traps.size();
    const u64 native_count = inputs.native.size();
    put(&inputs.source, sizeof(inputs.source));
    put(&inputs.stage, sizeof(inputs.stage));
    put(&inputs.header_size, sizeof(inputs.header_size));
    put(inputs.sph.data(), inputs.sph.size());
    put(&inputs.geometry, sizeof(inputs.geometry));
    put(&inputs.tessellation_stage, sizeof(inputs.tessellation_stage));
    put(&inputs.native_code_address, sizeof(inputs.native_code_address));
    put(&trap_count, sizeof(trap_count));
    put(inputs.verified_traps.data(), inputs.verified_traps.size_bytes());
    put(&native_count, sizeof(native_count));
    put(inputs.native.data(), inputs.native.size_bytes());
    return XXH3_64bits_digest(&state);
}

u64 ProgramKey(const ProgramRecord& record) {
    return ProgramKey(ProgramKeyInputsOf(record));
}

std::vector<u8> EncodeProgram(const ProgramRecord& record) {
    PayloadWriter out;
    out.Put(ProgramPayloadVersion);
    out.Put(record.source);
    out.Put(record.geometry);
    out.Put(u16{});
    out.Put(record.stage);
    out.Put(record.header_size);
    out.Put(record.tessellation_stage);
    out.Put(record.texture_bound);
    out.Put(static_cast<u32>(record.native.size()));
    out.Put(static_cast<u32>(record.verified_traps.size()));
    out.Put(static_cast<u32>(record.lowered_code.size()));
    out.Put(record.native_code_address);
    out.Put(record.lowering_stamp);
    out.Put(record.source_size);
    out.Put(record.unique_hash);
    out.Put(record.sph);
    out.PutSpan(std::span<const Shader::Ampere::Instruction>{record.native});
    out.PutSpan(std::span<const u32>{record.verified_traps});
    out.PutSpan(std::span<const u64>{record.lowered_code});
    return out.Take();
}

std::optional<ProgramRecord> DecodeProgram(std::span<const u8> payload) {
    PayloadReader in{payload};
    ProgramRecord record;
    u32 version{};
    u16 reserved{};
    u32 native_count{};
    u32 trap_count{};
    u32 code_count{};
    if (!in.Get(version) || version != ProgramPayloadVersion || !in.Get(record.source) ||
        !in.Get(record.geometry) || !in.Get(reserved) || !in.Get(record.stage) ||
        !in.Get(record.header_size) || !in.Get(record.tessellation_stage) ||
        !in.Get(record.texture_bound) || !in.Get(native_count) || !in.Get(trap_count) ||
        !in.Get(code_count) || !in.Get(record.native_code_address) ||
        !in.Get(record.lowering_stamp) || !in.Get(record.source_size) ||
        !in.Get(record.unique_hash) || !in.Get(record.sph) ||
        !in.GetVector(record.native, native_count) ||
        !in.GetVector(record.verified_traps, trap_count) ||
        !in.GetVector(record.lowered_code, code_count) || !in.AtEnd()) {
        return std::nullopt;
    }
    return record;
}

std::optional<u64> ProgramStamp(std::span<const u8> payload) {
    // Version, source, geometry, reserved, seven u32 fields, then the code address before it
    constexpr size_t StampOffset = 4 + 1 + 1 + 2 + 7 * 4 + 8;
    u32 version{};
    u64 stamp{};
    if (payload.size() < StampOffset + sizeof(stamp)) {
        return std::nullopt;
    }
    std::memcpy(&version, payload.data(), sizeof(version));
    std::memcpy(&stamp, payload.data() + StampOffset, sizeof(stamp));
    if (version != ProgramPayloadVersion) {
        return std::nullopt;
    }
    return stamp;
}

std::vector<u8> EncodeUsed(const UsedRecord& record) {
    PayloadWriter out;
    out.Put(record.session_time);
    out.Put(static_cast<u64>(record.pipelines.size()));
    out.PutSpan(std::span<const u64>{record.pipelines});
    return out.Take();
}

std::optional<UsedRecord> DecodeUsed(std::span<const u8> payload) {
    PayloadReader in{payload};
    UsedRecord record;
    u64 count{};
    if (!in.Get(record.session_time) || !in.Get(count) || !in.GetVector(record.pipelines, count) ||
        !in.AtEnd()) {
        return std::nullopt;
    }
    return record;
}

std::vector<u8> EncodeMigrated(const MigratedRecord& record) {
    PayloadWriter out;
    out.Put(record.source_size);
    out.Put(record.source_mtime);
    out.Put(record.source_version);
    out.Put(record.graphics);
    out.Put(record.compute);
    out.Put(record.used);
    out.Put(static_cast<u64>(record.source.size()));
    out.PutSpan(std::span<const char>{record.source});
    return out.Take();
}

std::optional<MigratedRecord> DecodeMigrated(std::span<const u8> payload) {
    PayloadReader in{payload};
    MigratedRecord record;
    u64 length{};
    std::vector<char> name;
    if (!in.Get(record.source_size) || !in.Get(record.source_mtime) ||
        !in.Get(record.source_version) || !in.Get(record.graphics) || !in.Get(record.compute) ||
        !in.Get(record.used) || !in.Get(length) || !in.GetVector(name, length) || !in.AtEnd()) {
        return std::nullopt;
    }
    record.source.assign(name.begin(), name.end());
    return record;
}

} // namespace VideoCommon::ShaderCacheV2
