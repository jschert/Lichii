// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "common/cityhash.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_override.h"
#include "shader_recompiler/program_header.h"
#include "video_core/mkw_wave_workaround.h"
#include "video_core/native_lowering.h"

namespace VideoCommon {

using Shader::Ampere::Instruction;

std::vector<u32> NativeTrapOffsets(std::span<const Instruction> native) {
    std::vector<u32> offsets;
    for (u32 i = 0; i < native.size(); ++i)
        if (native[i].Bits(0, 12) == 0x95c)
            offsets.push_back(i * sizeof(Instruction));
    return offsets;
}

bool LoadsProgramCounter(std::span<const Instruction> native, bool geometry) {
    return geometry && std::ranges::any_of(native, [](const Instruction& instruction) {
               return instruction.Bits(0, 12) == 0x34e;
           });
}

Shader::Ampere::LoweredProgram LowerNativeProgram(std::span<const Instruction> native,
                                                  const NativeLoweringInputs& inputs) {
    const std::vector<u32> verified_traps = NativeTrapOffsets(native);
    return Shader::Ampere::LowerToMaxwell(native, {.geometry = inputs.geometry,
                                                   .tensor = true,
                                                   .native_code_address = inputs.native_code_address,
                                                   .verified_traps = verified_traps,
                                                   .tessellation_stage = inputs.tessellation_stage});
}

std::shared_ptr<LoweredNativeCode> MakeLoweredNativeCode(std::span<const Instruction> native,
                                                         const Shader::ProgramHeader& sph,
                                                         u32 header_size,
                                                         std::span<const u32> verified_traps,
                                                         Shader::Ampere::LoweredProgram& lowered) {
    VideoCommon::ApplyMkwZeroDepthFallback(native, lowered);
    const size_t trap_count =
        verified_traps.empty() ? 0 : verified_traps.back() / sizeof(Instruction) + 1;
    auto result = std::make_shared<LoweredNativeCode>();
    result->native_program_key = Shader::Ampere::NativeProgramKey(native);
    result->texture_bound = lowered.required_texture_buffer;
    result->source_size = header_size + lowered.source_size;
    result->trap_prefix.assign(native.begin(), native.begin() + trap_count);
    auto& lowered_code = result->code;
    lowered_code.resize(sizeof(sph) / sizeof(u64));
    std::memcpy(lowered_code.data(), &sph, sizeof(sph));
    for (const u64 instruction : lowered.instructions) {
        if ((lowered_code.size() * sizeof(u64)) % 32 == 0) {
            lowered_code.push_back(0); // Maxwell scheduling word, absent from native instructions.
        }
        lowered_code.push_back(instruction);
    }
    // Keep the original padding ABI, then retain the native address map as
    // unreachable data. Existing disk-cache serialization carries it with code.
    lowered_code.push_back(0);
    Shader::Ampere::AppendNativeAddressMap(lowered_code, lowered.source_instruction_offsets);
    return result;
}

u64 NativeShaderHash(std::span<const u64> code) {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(u64)) ^
           0x4f554e4345000002ULL;
}

} // namespace VideoCommon
