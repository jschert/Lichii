// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader {
struct ProgramHeader;
}

namespace Shader::Ampere {
struct LoweredProgram;
enum class TessellationStage;
} // namespace Shader::Ampere

namespace VideoCommon {

/// A native (Switch 2) program after capture and lowering to Maxwell code. The shader cache keeps
/// it so the environments built for the program's pipelines don't capture and lower it again.
struct LoweredNativeCode {
    u64 native_program_key{};
    /// Shader header, lowered instructions with Maxwell scheduling words, then the native
    /// address map (what the environment's `code` used to be filled with).
    std::vector<u64> code;
    std::vector<Shader::Ampere::Instruction> trap_prefix;
    /// Bytes of the native program that were read, header included.
    size_t source_size{};
    /// Constant buffer holding the texture handles if the program requires one, otherwise ~0U
    /// (the engine's register value is used).
    u32 texture_bound = ~0U;
};

/// What the lowering of a native program takes besides its instructions
struct NativeLoweringInputs {
    bool geometry{};
    Shader::Ampere::TessellationStage tessellation_stage{};
    /// Where the code is in guest memory. Only programs loading their own address read it (see
    /// LoadsProgramCounter).
    u64 native_code_address{};
};

/// Byte offsets of the trap instructions, whose guards the caller verifies before lowering
[[nodiscard]] std::vector<u32> NativeTrapOffsets(
    std::span<const Shader::Ampere::Instruction> native);

/// Whether the lowering puts the address of the code into it: geometry programs loading the
/// program counter
[[nodiscard]] bool LoadsProgramCounter(std::span<const Shader::Ampere::Instruction> native,
                                       bool geometry);

/// Lowers a captured native program. It depends on its instructions, `inputs` and this build's
/// lowering (see ShaderCacheV2::LoweringStamp) alone, so it can run again without guest memory.
[[nodiscard]] Shader::Ampere::LoweredProgram LowerNativeProgram(
    std::span<const Shader::Ampere::Instruction> native, const NativeLoweringInputs& inputs);

/// Applies the fix-ups for single programs to a successful lowering and lays it out as an
/// environment's code: the shader header, the lowered instructions with a Maxwell scheduling word
/// before every fourth word, one zero word, then the native address map. `verified_traps` are
/// those the program was lowered with (NativeTrapOffsets). The fix-ups change `lowered`.
[[nodiscard]] std::shared_ptr<LoweredNativeCode> MakeLoweredNativeCode(
    std::span<const Shader::Ampere::Instruction> native, const Shader::ProgramHeader& sph,
    u32 header_size, std::span<const u32> verified_traps, Shader::Ampere::LoweredProgram& lowered);

/// The shader hash of lowered native code (frame dump "shaders", pipeline cache keys), with a
/// version tag that isolates this experimental ABI.
[[nodiscard]] u64 NativeShaderHash(std::span<const u64> code);

} // namespace VideoCommon
