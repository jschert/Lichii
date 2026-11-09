// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// nx2 shaderwarm (docs/shader-precompile.md): PrecompileEnvironment and the (de)serialization of a
// translation's environment answers (GenericEnvironment::SerializeAnswers / AdoptAnswers) for the
// per-stage translation cache.

#include <algorithm>
#include <array>
#include <cstring>

#include "common/alignment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "video_core/nx2_precompile_env.h"
#include "video_core/surface.h"

namespace VideoCommon {
namespace {
// The storage pointer classification keys GenericEnvironment uses (shader_environment.cpp
// MakeCbufKey / MakeIndirectCbufKey); the records must use the same.
u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

u64 MakeIndirectCbufKey(u32 index, u32 offset, u32 indirect) {
    return (u64{1} << 63) | (static_cast<u64>(indirect & 0x3fffff) << 40) |
           MakeCbufKey(index, offset);
}
} // Anonymous namespace

namespace {
template <typename T>
void PutBytes(std::vector<u8>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T GetBytes(std::span<const u8> data, size_t& offset) {
    if (offset + sizeof(T) > data.size()) {
        throw Shader::LogicError("Truncated environment answers");
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}
} // Anonymous namespace

void GenericEnvironment::SerializeAnswers(std::vector<u8>& out) const {
    PutBytes(out, static_cast<u32>(texture_types.size()));
    for (const auto& [key, value] : texture_types) {
        PutBytes(out, key);
        PutBytes(out, value);
    }
    PutBytes(out, static_cast<u32>(texture_pixel_formats.size()));
    for (const auto& [key, value] : texture_pixel_formats) {
        PutBytes(out, key);
        PutBytes(out, value);
    }
    PutBytes(out, static_cast<u32>(cbuf_values.size()));
    for (const auto& [key, value] : cbuf_values) {
        PutBytes(out, key);
        PutBytes(out, value);
    }
    PutBytes(out, static_cast<u32>(storage_pointer_types.size()));
    for (const auto& [key, value] : storage_pointer_types) {
        PutBytes(out, key);
        PutBytes(out, value);
    }
    PutBytes(out, viewport_transform_state);
}

void GenericEnvironment::AdoptAnswers(std::span<const u8> data) {
    size_t offset = 0;
    for (u32 n = GetBytes<u32>(data, offset); n > 0; --n) {
        const auto key = GetBytes<u32>(data, offset);
        texture_types.emplace(key, GetBytes<Shader::TextureType>(data, offset));
    }
    for (u32 n = GetBytes<u32>(data, offset); n > 0; --n) {
        const auto key = GetBytes<u32>(data, offset);
        texture_pixel_formats.emplace(key, GetBytes<Shader::TexturePixelFormat>(data, offset));
    }
    for (u32 n = GetBytes<u32>(data, offset); n > 0; --n) {
        const auto key = GetBytes<u64>(data, offset);
        cbuf_values.emplace(key, GetBytes<u32>(data, offset));
    }
    for (u32 n = GetBytes<u32>(data, offset); n > 0; --n) {
        const auto key = GetBytes<u64>(data, offset);
        storage_pointer_types.emplace(key, GetBytes<u8>(data, offset));
    }
    viewport_transform_state = GetBytes<u32>(data, offset);
}

PrecompileEnvironment::PrecompileEnvironment(std::span<const u8> blob_, u32 const_offset_,
                                             u32 const_size_, Shader::Stage stage_,
                                             u32 default_texture_bound, const ComputeLaunch& launch)
    : blob(blob_.begin(), blob_.end()), const_offset{const_offset_}, const_size{const_size_} {
    using Shader::Ampere::Instruction;
    const bool compute = stage_ == Shader::Stage::Compute;
    // Graphics: the 0x80-byte header, then code. Compute: code at +0 (ComputeEnvironment lowers
    // from the program address with header size 0 and a zero SPH).
    const size_t HeaderSize = compute ? 0 : 128;
    if (blob.size() < HeaderSize + sizeof(Instruction)) {
        throw Shader::RuntimeError("Native program blob too short ({} bytes)", blob.size());
    }
    if (!compute) {
        std::memcpy(&sph, blob.data(), sizeof(sph));
    }
    initial_offset = sizeof(sph);
    stage = stage_;
    if (compute) {
        workgroup_size = launch.workgroup;
        shared_memory_size = launch.shared_memory;
    }
    start_address = 0;
    texture_bound = default_texture_bound;
    is_proprietary_driver = true;
    std::vector<Instruction> native;
    Shader::Ampere::ProgramCaptureState capture;
    const size_t count = std::min<size_t>((blob.size() - HeaderSize) / sizeof(Instruction),
                                          Shader::Ampere::ProgramCaptureState::MaximumInstructions);
    for (u32 index = 0; index < count; ++index) {
        Instruction instruction;
        std::memcpy(&instruction, blob.data() + HeaderSize + index * sizeof(Instruction),
                    sizeof(instruction));
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            throw Shader::RuntimeError("Native shader branch outside capture bounds at +{:#x}",
                                       index * 16);
        }
        if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            break;
        }
    }
    // The constant section starts at the 256-byte boundary after the program, which ends with
    // padding the capture stops before: NOPs and the branch-to-self the compiler ends code with
    // (matches the NVN control section's offset for all 11,879 OoT VS/PS programs).
    size_t code_end = HeaderSize + native.size() * sizeof(Instruction);
    while (code_end + sizeof(Instruction) <= blob.size()) {
        std::array<u32, 4> words;
        std::memcpy(words.data(), blob.data() + code_end, sizeof(words));
        const u32 opcode = words[0] & 0xfff;
        const bool nop = opcode == 0x918;
        const bool branch_to_self =
            opcode == 0x947 && (words[1] & ~3U) == 0xfffffff0U && (words[2] & 0x3ffff) == 0x3ffff;
        if (!nop && !branch_to_self) {
            break;
        }
        code_end += sizeof(Instruction);
    }
    derived_const_offset = static_cast<u32>(Common::AlignUp(code_end, 256));
    if (const_offset == ~0U) {
        const_offset = derived_const_offset;
        const_size = blob.size() > const_offset ? static_cast<u32>(blob.size() - const_offset) : 0;
    }
    // Geometry programs lower with the geometry options, like GraphicsEnvironment's. Their LEPC
    // gives the program-relative address when it is dead (lower_to_maxwell.h), so the lowered
    // code matches the runtime's; otherwise the hash differs and the record is never used.
    const bool geometry = stage_ == Shader::Stage::Geometry;
    InitializeNativeCode(LowerCapturedNative(native, static_cast<u32>(HeaderSize), geometry),
                         static_cast<u32>(HeaderSize), geometry);
    if (compute) {
        // start_address stays 0 like the graphics programs': the CFG starts after the SPH words
        // (ComputeEnvironment has start_address = sizeof(sph) and starts the CFG there).
        local_memory_size = launch.local_memory;
        return;
    }
    const u64 local_size{sph.LocalMemorySize()};
    local_memory_size = static_cast<u32>(local_size) + sph.common3.shader_local_memory_crs_size;
    if (stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) {
        // Placeholder mesh state (triangles, 32 threads, one line passed, no per-primitive
        // outputs): only the mesh sizes and line map depend on it, and the runtime puts the live
        // values into the cached program (Shader::Maxwell::ApplyMeshState).
        mesh_registers = {4U | (64U << 4) | (126U << 13), 32U << 20, 32U | (1U << 12) | (1U << 22)};
        std::memset(&mesh_primitive_header, 0, sizeof(mesh_primitive_header));
    }
}

u32 PrecompileEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    u32 value{};
    // Bank 1 is the program's constant section; the bytes after it in the blob are what the GPU
    // reads past it too. Other banks hold texture handles: their values only select a descriptor.
    if (cbuf_index == 1 && static_cast<u64>(const_offset) + cbuf_offset + sizeof(u32) <= blob.size()) {
        std::memcpy(&value, blob.data() + const_offset + cbuf_offset, sizeof(value));
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

void PrecompileEnvironment::NoteTextureHandleSource(u32 index, u32 offset, u32 shift_left,
                                                    u32 secondary_index, u32 secondary_offset,
                                                    u32 secondary_shift_left) {
    last_handle_source = {index, offset, shift_left, secondary_index, secondary_offset,
                          secondary_shift_left};
    has_handle_source = true;
}

Shader::TextureType PrecompileEnvironment::ReadTextureType(u32 handle) {
    if (!has_handle_source) {
        throw Shader::RuntimeError("Texture type query without a handle source");
    }
    texture_queries.push_back({last_handle_source, 0, static_cast<u32>(Shader::TextureType::Color2D)});
    texture_types.emplace(handle, Shader::TextureType::Color2D);
    return Shader::TextureType::Color2D;
}

Shader::TexturePixelFormat PrecompileEnvironment::ReadTexturePixelFormat(u32 handle) {
    // An exact-format read (byte-addressed image stores, SNORM buffer fetches): the record is
    // only used where the live texture has this format (kind 2, as live translations log it).
    if (!has_handle_source) {
        throw Shader::RuntimeError("Texture format query without a handle source");
    }
    texture_queries.push_back(
        {last_handle_source, 2, static_cast<u32>(Shader::TexturePixelFormat::A8B8G8R8_UNORM)});
    texture_pixel_formats.emplace(handle, Shader::TexturePixelFormat::A8B8G8R8_UNORM);
    return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
}

bool PrecompileEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    if (!has_handle_source) {
        throw Shader::RuntimeError("Texture format query without a handle source");
    }
    // Only whether it is an integer format: checked as such (kind 1), not as the exact format.
    texture_queries.push_back({last_handle_source, 1, 0});
    texture_pixel_formats.emplace(handle, Shader::TexturePixelFormat::A8B8G8R8_UNORM);
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(Shader::TexturePixelFormat::A8B8G8R8_UNORM));
}

u32 PrecompileEnvironment::ReadViewportTransformState() {
    viewport_used = true;
    viewport_transform_state = 1;
    return viewport_transform_state;
}

std::optional<Shader::ReplaceConstant> PrecompileEnvironment::GetReplaceConstBuffer(u32, u32) {
    return std::nullopt;
}

bool PrecompileEnvironment::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    const u64 key = MakeCbufKey(index, offset);
    const bool mapped = std::ranges::find(unmapped_pointers, key) == unmapped_pointers.end();
    storage_pointer_types.emplace(key, u8{mapped});
    return mapped;
}

bool PrecompileEnvironment::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                               u32 indirect) {
    // As for direct pointers: a pointer table entry is taken as a mapped address (no GPU memory
    // to look into). Recorded, so the runtime re-asks it with the live table.
    const u64 key = MakeIndirectCbufKey(index, offset, indirect);
    const bool mapped = std::ranges::find(unmapped_pointers, key) == unmapped_pointers.end();
    storage_pointer_types.emplace(key, u8{mapped});
    return mapped;
}

u32 PrecompileEnvironment::TextureBoundBuffer() const {
    texture_bound_used = true;
    return texture_bound;
}

} // namespace VideoCommon
