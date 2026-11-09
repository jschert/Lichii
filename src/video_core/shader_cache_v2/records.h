// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader {
enum class Stage : u32;
struct ProgramHeader;
} // namespace Shader

namespace VideoCommon {
struct LoweredNativeCode;
struct NativeLoweringInputs;
} // namespace VideoCommon

namespace VideoCommon::ShaderCacheV2 {

/// A hash of this build's lowering sources (src/shader_recompiler/frontend/ampere, the native
/// lowering and its fix-ups), made when it is built: lowered code with another stamp may differ.
[[nodiscard]] u64 LoweringStamp();

enum class ProgramSource : u8 {
    NativeGraphics = 0,
    NativeCompute = 1,
};

/// A native program as captured from guest memory, the inputs its lowering took that its bytes
/// don't hold, and what the lowering made of it (see LowerNativeProgram).
struct ProgramRecord {
    ProgramSource source{};
    u32 stage{};       ///< Shader::Stage
    u32 header_size{}; ///< Bytes of the native header before the code (128, or 0 for compute)
    std::array<u8, 80> sph{}; ///< The shader program header (zeros for compute)
    std::vector<Shader::Ampere::Instruction> native;

    bool geometry{};
    u32 tessellation_stage{}; ///< Shader::Ampere::TessellationStage
    std::vector<u32> verified_traps;
    /// Where the code was in guest memory, for programs that load their own address (else 0)
    u64 native_code_address{};

    u64 lowering_stamp{};
    /// The environment's code: header, lowered instructions, the native address map
    std::vector<u64> lowered_code;
    u32 texture_bound{~0U};
    u64 source_size{};
    /// The shader hash of the lowered code, which pipeline keys hold
    u64 unique_hash{};
};

/// What the key of a program's record covers (see ProgramKey), without copying it
struct ProgramKeyInputs {
    ProgramSource source{};
    u32 stage{};
    u32 header_size{};
    std::span<const u8> sph;
    bool geometry{};
    u32 tessellation_stage{};
    u64 native_code_address{}; ///< 0 unless the lowering puts it into the code
    std::span<const u32> verified_traps;
    std::span<const Shader::Ampere::Instruction> native;
};

/// The key inputs of a native program captured `header_size` bytes after the start of its header,
/// lowered with these inputs and traps (NativeTrapOffsets)
[[nodiscard]] ProgramKeyInputs MakeProgramKeyInputs(
    Shader::Stage stage, u32 header_size, const Shader::ProgramHeader& sph,
    std::span<const Shader::Ampere::Instruction> native, const NativeLoweringInputs& inputs,
    std::span<const u32> verified_traps);

/// The key inputs a record holds
[[nodiscard]] ProgramKeyInputs ProgramKeyInputsOf(const ProgramRecord& record);

/// The record of a native program captured `header_size` bytes after the start of its header and
/// lowered with these inputs into `lowered`
[[nodiscard]] ProgramRecord MakeProgramRecord(Shader::Stage stage, u32 header_size,
                                            const Shader::ProgramHeader& sph,
                                            std::span<const Shader::Ampere::Instruction> native,
                                            const NativeLoweringInputs& inputs,
                                            const LoweredNativeCode& lowered);

/// Lowers a recorded program again with this build's lowering, without guest memory. Null when
/// that fails.
[[nodiscard]] std::shared_ptr<LoweredNativeCode> RelowerProgram(const ProgramRecord& record);

/// The record of a program lowered again by this build (RelowerProgram), with its stamp. Nullopt
/// for a record without native instructions or one that doesn't lower.
[[nodiscard]] std::optional<ProgramRecord> RelowerProgramRecord(const ProgramRecord& record);
/// The code a record holds, laid out as MakeLoweredNativeCode lays out a lowering. Takes the
/// record's lowered code.
[[nodiscard]] std::shared_ptr<LoweredNativeCode> TakeProgramCode(ProgramRecord& record);

/// The record key of a program: a hash of everything its lowering depends on except the build.
/// The address of the code is part of it only where the lowering puts it into the code.
[[nodiscard]] u64 ProgramKey(const ProgramKeyInputs& inputs);
[[nodiscard]] u64 ProgramKey(const ProgramRecord& record);

[[nodiscard]] std::vector<u8> EncodeProgram(const ProgramRecord& record);
[[nodiscard]] std::optional<ProgramRecord> DecodeProgram(std::span<const u8> payload);
/// The lowering stamp of an encoded program, without decoding the rest
[[nodiscard]] std::optional<u64> ProgramStamp(std::span<const u8> payload);

/// The graphics pipelines a session used (key hashes), a Used record keyed by the session
struct UsedRecord {
    /// Orders sessions: the Unix time it started, in nanoseconds
    u64 session_time{};
    std::vector<u64> pipelines;

    bool operator==(const UsedRecord&) const = default;
};

[[nodiscard]] std::vector<u8> EncodeUsed(const UsedRecord& record);
[[nodiscard]] std::optional<UsedRecord> DecodeUsed(std::span<const u8> payload);

/// An older cache this log was started from (a Migrated record keyed by its name's hash)
struct MigratedRecord {
    std::string source;     ///< File name
    u64 source_size{};
    s64 source_mtime{};     ///< std::filesystem::file_time_type ticks
    u32 source_version{};
    u32 graphics{};
    u32 compute{};
    u32 used{};

    bool operator==(const MigratedRecord&) const = default;
};

[[nodiscard]] std::vector<u8> EncodeMigrated(const MigratedRecord& record);
[[nodiscard]] std::optional<MigratedRecord> DecodeMigrated(std::span<const u8> payload);

} // namespace VideoCommon::ShaderCacheV2
