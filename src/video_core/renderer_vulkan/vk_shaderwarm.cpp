// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// nx2 shaderwarm (docs/shader-precompile.md): the per-stage translation cache (vulkan_ir.bin), the
// precompile from game files (EDEN_NX2_PRECOMPILE), graphics pipeline library stages
// (EDEN_NX2_GPL) and their diagnostics. PipelineCache members; vk_pipeline_cache.cpp only calls
// them from short hooks marked "nx2 shaderwarm".

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_set>
#include <map>
#include <span>
#include <string_view>
#include <thread>
#include <vector>
#include <bit>
#include <numeric>
#include "common/cityhash.h"
#include "common/scope_exit.h"
#include "common/scm_rev.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/memory_detect.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/nx2_pipeline_stats.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shaderwarm.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2_ir_cache.h"
#include "video_core/nx2_precompile_env.h"
#include "video_core/renderer_vulkan/vk_gpl.h"
#include "video_core/renderer_vulkan/vk_shaderwarm.h"

namespace Vulkan {

using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::TranslateProgram;

namespace Nx2 {
// Files of the per-stage translation cache are only valid for the build (recompiler) and device
// (host translate info) that made them. Bump IrCacheFormat when the cached data changes meaning.
constexpr u32 IrCacheFormat = 2;
u64 IrCacheBuildTag(const Device& device) {
    const std::string text =
        fmt::format("{}|{}|{}|{}|{}|{}|{}|{}", IrCacheFormat, Common::g_scm_rev, Common::g_scm_desc,
                    sizeof(Shader::Info), sizeof(Shader::IR::Inst),
                    static_cast<u32>(device.GetDriverID()), device.GetDriverVersion(),
                    device.GetModelName());
    return Common::CityHash64(text.data(), text.size());
}

// EDEN_NX2_PIPELINE_LOG=1: one line per graphics pipeline built at run time: its key hash, each
// stage's slot, stage and SPH word 0, whether it is a task/mesh pipeline, whether its key was in
// the disk cache (loaded but not built at boot), and whether the build failed. Tells what a warm
// session still builds.
bool RuntimePipelineLog() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_LOG");
        return value && std::string_view{value} != "0";
    }();
    return enabled;
}

// EDEN_NX2_IR_CACHE_TASKMESH=1: task/mesh stages use and fill the IR cache (precompile from pack
// slots 7/8 and stage-buffer vertex candidates). Off by default: one precompile run with it stalled
// (cause unknown) and the cached task/mesh IR wasn't checked against live.
bool TaskMeshIrCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_IR_CACHE_TASKMESH");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

// EDEN_NX2_PRECOMPILE_POINTER_VARIANTS=0: no pointer variant records. On by default: a stage that
// asks whether constant-buffer storage pointers are mapped gets a record for every mapped/null
// combination (OoT's check route: most cached stages rejected at run time had some of them null;
// one all-null variant matched none of those).
bool PointerVariantsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_POINTER_VARIANTS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_PRECOMPILE_POINTER_MAX (default 6): up to this many pointers, all 2^n combinations.
size_t PointerVariantsMax() {
    static const size_t max = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_POINTER_MAX");
        return value ? std::min<size_t>(10, std::strtoull(value, nullptr, 0)) : size_t{6};
    }();
    return max;
}

// EDEN_NX2_IR_CACHE_COMPUTE=0: compute programs neither use nor fill the translation cache (on by
// default; records only apply to a launch they match, see ComputeLaunchMatches).
bool ComputeIrCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_IR_CACHE_COMPUTE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// Geometry records only with EDEN_NX2_RELATIVE_LEPC=1 (lower_to_maxwell.h).
bool GeometryIrCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RELATIVE_LEPC");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

// IR cache key of a task or mesh stage: the program's own hash (the pipeline key's stage hash
// with the mesh state XORed out), tagged by stage so a task, a mesh program and a vertex
// program translated from the same bytes (generic packs don't know the stage) don't collide.
u64 TaskMeshRecordHash(u64 program_hash, Shader::Stage stage) {
    return program_hash ^ (stage == Shader::Stage::Task ? 0x5441534b00000000ULL   // "TASK"
                                                        : 0x4d45534800000000ULL); // "MESH"
}

} // namespace Nx2

using namespace Nx2;

namespace Nx2 {
// A vertex program that loads the position attribute makes SPIR-V with a BuiltIn Position input,
// which Vulkan doesn't allow in a vertex shader (spirv-val VUID-Position-Position-04319); RADV
// crashes compiling it (radv_nir_lower_vs_inputs). Generic scans find such programs.
void LogComputeSpirv(const char* where, u64 unique_hash, const Shader::IR::Program& program,
                     std::span<const u32> spirv) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CS_LOG");
        return value != nullptr && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "NX2CS {} {:016x}: wg {}x{}x{} shared {} local {} (uses local {}) spirv {:016x} ({} words)",
             where, unique_hash, program.workgroup_size[0], program.workgroup_size[1],
             program.workgroup_size[2], program.shared_memory_size, program.local_memory_size,
             program.info.uses_local_memory,
             Common::CityHash64(reinterpret_cast<const char*>(spirv.data()), spirv.size_bytes()),
             spirv.size());
}

bool VertexReadsPosition(const Shader::Info& info) {
    return info.loads.AnyComponent(Shader::IR::Attribute::PositionX);
}

// Why a pipeline is not built from graphics pipeline libraries (nullptr: it is). Library pipelines
// are native VS+PS ones whose stages compile alone with the same result as linked: no legacy
// varyings, no fragment input the vertex stage leaves unwritten (the linked pipeline gives those
// (0,0,0,1), a library stage can't know), rescaling bits within the per-stage ranges.
const char* GplIneligible(const Device& device, const GraphicsPipelineCacheKey& key,
                          std::span<const Shader::IR::Program> programs, bool layer_emulation,
                          bool all_native) {
    const auto& hashes = key.unique_hashes;
    if (hashes[0] != 0 || hashes[1] == 0 || hashes[2] != 0 || hashes[3] != 0 || hashes[4] != 0 ||
        hashes[5] == 0) {
        return "stages other than VS+PS";
    }
    if (!all_native) {
        return "not native programs";
    }
    if (device.IsMoltenVK()) {
        return "MoltenVK";
    }
    const Shader::Info& vs = programs[1].info;
    const Shader::Info& ps = programs[5].info;
    if (vs.is_task || vs.is_mesh || layer_emulation) {
        return "mesh/task or layer emulation";
    }
    if (key.state.xfb_enabled != 0) {
        return "transform feedback";
    }
    if (key.state.extended_dynamic_state_2 == 0 && key.state.dynamic_state.rasterize_enable == 0) {
        return "static rasterizer discard";
    }
    if (vs.stores.Legacy() || ps.loads.Legacy()) {
        return "legacy varyings";
    }
    if (VertexReadsPosition(vs)) {
        return "vertex stage reads its position input";
    }
    for (size_t index = 0; index < Shader::IR::NUM_GENERICS; ++index) {
        if (ps.loads.Generic(index) && !vs.stores.Generic(index)) {
            return "fragment input the vertex stage does not write";
        }
    }
    constexpr u32 vs_textures = Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS * 32 -
                                GPL_VERTEX_TEXTURE_SCALING_BASE;
    constexpr u32 vs_images =
        Shader::Backend::SPIRV::NUM_IMAGE_SCALING_WORDS * 32 - GPL_VERTEX_IMAGE_SCALING_BASE;
    if (vs.texture_descriptors.size() > vs_textures || vs.image_descriptors.size() > vs_images ||
        ps.texture_descriptors.size() > GPL_VERTEX_TEXTURE_SCALING_BASE ||
        ps.image_descriptors.size() > GPL_VERTEX_IMAGE_SCALING_BASE) {
        return "too many textures/images for the per-stage rescaling ranges";
    }
    return nullptr;
}
} // namespace Nx2

namespace {
// EDEN_NX2_GPL_CHECK=1: for each library pipeline stage, which SPIR-V inputs differ from the
// precompile's defaults (GplDefaultKey); the stage IR is the same, so these are the reason a
// precompiled library misses.
std::string GplRuntimeDiff(const Shader::RuntimeInfo& live, const Shader::RuntimeInfo& def,
                           const Shader::Info& info, Shader::Stage stage) {
    std::string out;
    if (stage != Shader::Stage::Fragment) {
        for (size_t index = 0; index < Shader::IR::NUM_GENERICS; ++index) {
            if (info.loads.Generic(index) &&
                live.generic_input_types[index] != def.generic_input_types[index]) {
                out += fmt::format(" attr{}={}", index,
                                   static_cast<u32>(live.generic_input_types[index]));
            }
        }
        if (live.convert_depth_mode != def.convert_depth_mode) {
            out += " ndc";
        }
        if (live.fixed_state_point_size != def.fixed_state_point_size) {
            out += " point_size";
        }
    } else {
        if (live.alpha_test_func != def.alpha_test_func ||
            live.alpha_test_reference != def.alpha_test_reference) {
            out += " alpha_test";
        }
        if (live.dual_source_blend != def.dual_source_blend) {
            out += " dual_source";
        }
        if (live.force_early_z != def.force_early_z) {
            out += " early_z";
        }
        if (live.y_negate != def.y_negate) {
            out += " y_negate";
        }
    }
    return out.empty() ? std::string{" none"} : out;
}
} // Anonymous namespace

Shader::RuntimeInfo PipelineCache::MakeGplRuntimeInfo(const Shader::IR::Program& program,
                                                      const GraphicsPipelineCacheKey& key) const {
    // As the linked pipeline would, but with no previous stage (all fragment inputs declared).
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> no_programs{};
    Shader::RuntimeInfo runtime_info{MakeRuntimeInfo(no_programs, key, program, nullptr, device)};
    if (program.stage != Shader::Stage::Fragment && key.state.dynamic_vertex_input &&
        device.IsGplUntypedAttributes()) {
        // One SPIR-V for every attribute class: inputs declared float. The int/uint bits arrive
        // unchanged (VK_EXT_legacy_vertex_attributes, RADV), which is what the typed path's
        // bitcast gives; constant attributes get a null-buffer attribute (0,0,0,1) from
        // RasterizerVulkan::UpdateVertexInput, the typed path's constant.
        runtime_info.generic_input_types.fill(Shader::AttributeType::Float);
    }
    return runtime_info;
}

std::vector<u32> PipelineCache::EmitGplStage(Shader::IR::Program& program,
                                             const GraphicsPipelineCacheKey& key) const {
    Shader::RuntimeInfo runtime_info{MakeGplRuntimeInfo(program, key)};
    runtime_info.input_topology = Shader::InputTopology::Triangles;
    Shader::Backend::Bindings bindings{};
    if (program.stage == Shader::Stage::Fragment) {
        runtime_info.descriptor_set = GPL_FRAGMENT_SET;
    } else {
        runtime_info.descriptor_set = GPL_VERTEX_SET;
        bindings.texture_scaling_index = GPL_VERTEX_TEXTURE_SCALING_BASE;
        bindings.image_scaling_index = GPL_VERTEX_IMAGE_SCALING_BASE;
    }
    ConvertLegacyToGeneric(program, runtime_info);
    return EmitSPIRV(profile, runtime_info, program, bindings);
}

GraphicsPipelineCacheKey PipelineCache::GplDefaultKey(Shader::Stage stage) const {
    // Only what goes into a stage's SPIR-V or the driver's library key matters here: vertex
    // attributes float, no alpha test, no early-z, y not negated, depth 0..1, fill, last
    // provoking vertex, 1 sample, no alpha to coverage/one, triangles.
    GraphicsPipelineCacheKey key{};
    std::memset(&key, 0, sizeof(key));
    FixedPipelineState& state = key.state;
    state.extended_dynamic_state.Assign(dynamic_features.has_extended_dynamic_state ? 1 : 0);
    state.extended_dynamic_state_2.Assign(dynamic_features.has_extended_dynamic_state_2 ? 1 : 0);
    state.extended_dynamic_state_2_logic_op.Assign(
        dynamic_features.has_extended_dynamic_state_2_logic_op ? 1 : 0);
    state.extended_dynamic_state_3_blend.Assign(
        dynamic_features.has_extended_dynamic_state_3_blend ? 1 : 0);
    state.extended_dynamic_state_3_enables.Assign(
        dynamic_features.has_extended_dynamic_state_3_enables ? 1 : 0);
    state.color_write_enable_dynamic.Assign(dynamic_features.has_color_write_enable ? 1 : 0);
    state.dynamic_vertex_input.Assign(dynamic_features.has_dynamic_vertex_input ? 1 : 0);
    state.polygon_mode.Assign(FixedPipelineState::PackPolygonMode(Maxwell::PolygonMode::Fill));
    state.topology.Assign(Maxwell::PrimitiveTopology::Triangles);
    state.alpha_test_func.Assign(
        FixedPipelineState::PackComparisonOp(Maxwell::ComparisonOp::Always_GL));
    state.dynamic_state.rasterize_enable.Assign(1);
    // The last vertex provokes (the register's reset value, GL/NVN's default; every OoT and MKW
    // draw seen uses it). RADV puts it into a pre-rasterization library's key on GFX10+.
    const bool last_mode = dynamic_features.has_provoking_vertex &&
                           dynamic_features.has_provoking_vertex_last_mode;
    state.provoking_vertex_last.Assign(last_mode ? 1 : 0);
    if (dynamic_features.has_dynamic_vertex_input) {
        state.attribute_types = 0x5555'5555'5555'5555ULL; // every attribute float
    } else {
        for (auto& attribute : state.attributes) {
            attribute.enabled.Assign(1);
            attribute.type.Assign(static_cast<u32>(Maxwell::VertexAttribute::Type::Float));
        }
    }
    key.unique_hashes[stage == Shader::Stage::Fragment ? 5 : 1] = 1;
    return key;
}


// --- members called from hooks in vk_pipeline_cache.cpp ---------------------------------------

void PipelineCache::Nx2OpenStageCache(const std::filesystem::path& base_dir) {
        // Per-stage translation cache: used when the file exists (a precompile made it);
    // EDEN_NX2_IR_CACHE=0 ignores it. EDEN_NX2_PRECOMPILE=<pack> creates and fills it.
    const char* const pack = std::getenv("EDEN_NX2_PRECOMPILE");
    const char* const use = std::getenv("EDEN_NX2_IR_CACHE");
    const bool disabled = use != nullptr && use[0] == '0';
    if (!disabled && ir_cache.Open(base_dir / "vulkan_ir.bin", IrCacheBuildTag(device),
                                   pack != nullptr && pack[0] != '\0')) {
        LOG_INFO(Render_Vulkan, "NX2PC stage translation cache: {} records, {:.1f} MB",
                 ir_cache.NumRecords(), ir_cache.FileSize() / 1e6);
        if (pack != nullptr && pack[0] != '\0') {
            precompile_thread = std::jthread{
                [this, path = std::filesystem::path{pack}](std::stop_token stop) {
                    RunPrecompile(path, stop);
                }};
        }
    }
}

bool PipelineCache::Nx2CacheableStage(size_t index, Shader::Environment& env,
                                      bool uses_vertex_a) const {
    // Vertex and pixel stages, and task/mesh stages (vertex and tessellation slots) with
    // EDEN_NX2_IR_CACHE_TASKMESH=1.
    const bool mesh_stage{env.ShaderStage() == Shader::Stage::Task ||
                          env.ShaderStage() == Shader::Stage::Mesh};
    return !uses_vertex_a && (index == 1 || index == 5 || (index == 3 && mesh_stage)) &&
           ir_cache.IsOpen() && env.IsNativeProgram() && (!mesh_stage || TaskMeshIrCacheEnabled());
}

u64 PipelineCache::Nx2StageRecordHash(const GraphicsPipelineCacheKey& key, size_t index,
                                      Shader::Environment& env) const {
    // A task/mesh stage's hash in the key has the mesh state XORed in; its records are under the
    // program's own hash (TaskMeshRecordHash) and get the live mesh state put in (ApplyMeshState).
    const bool mesh_stage{env.ShaderStage() == Shader::Stage::Task ||
                          env.ShaderStage() == Shader::Stage::Mesh};
    return mesh_stage ? TaskMeshRecordHash(key.unique_hashes[index] ^
                                               VideoCommon::MeshStateHash(env.MeshRegisters(),
                                                                          env.MeshPrimitiveHeader()),
                                           env.ShaderStage())
                      : key.unique_hashes[index];
}

bool PipelineCache::Nx2TryCachedStage(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                      size_t index, Shader::Environment& env,
                                      Shader::IR::Program& out) {
    return TryCachedTranslation(pools, Nx2StageRecordHash(key, index, env), env, out);
}

void PipelineCache::Nx2CountMonolithic(const GraphicsPipelineCacheKey& key,
                                       const char* reason) const {
    if (!device.IsGplEnabled() || reason == nullptr || !Nx2Stats::Enabled() ||
        std::string_view{reason} == "stages other than VS+PS" ||
        std::string_view{reason} == "not native programs") {
        return;
    }
    Nx2Stats::Get().gpl_monolithic.fetch_add(1, std::memory_order_relaxed);
    if (Nx2Stats::Level() >= 2) {
        LOG_INFO(Render_Vulkan, "NX2STATS gpl {:016x}: monolithic, {}", key.Hash(), reason);
    }
}

void PipelineCache::Nx2StopPrecompile() {
    if (precompile_thread.joinable()) {
        precompile_thread.request_stop();
        precompile_thread.join();
    }
}

void PipelineCache::Nx2InitGpl() {
    if (device.IsGplEnabled()) {
        // EDEN_NX2_GPL_LTO=0: keep the fast-linked pipelines (no optimized link).
        const char* const lto = std::getenv("EDEN_NX2_GPL_LTO");
        if (lto == nullptr || lto[0] != '0') {
            const size_t threads = std::max<size_t>(1, std::thread::hardware_concurrency() / 4);
            gpl_lto_workers.reset(new Common::ThreadWorker(threads, "VkGplOptimize", {},
                                                           Common::ThreadPlacement::Background));
        }
    }
}

namespace {
// EDEN_NX2_GPL_DUMP=<dir> [EDEN_NX2_GPL_DUMP_ONLY=<hex hash>,...]: writes each library stage's
// SPIR-V and its library state (raw GplPreRasterState/GplFragmentState) as
// <dir>/<unique hash>-<rt|pc>.{spv,state}, to compare the runtime's with the precompile's.
void DumpGplStage(u64 unique_hash, bool runtime, std::span<const u32> spirv, const void* state,
                  size_t state_size) {
    static const char* const dir = std::getenv("EDEN_NX2_GPL_DUMP");
    if (dir == nullptr) {
        return;
    }
    static const std::string only = [] {
        const char* value = std::getenv("EDEN_NX2_GPL_DUMP_ONLY");
        return value ? std::string{value} : std::string{};
    }();
    const std::string name = fmt::format("{:016x}", unique_hash);
    if (!only.empty() && only.find(name) == std::string::npos) {
        return;
    }
    const std::string base = fmt::format("{}/{}-{}", dir, name, runtime ? "rt" : "pc");
    std::ofstream{base + ".spv", std::ios::binary}.write(
        reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size_bytes()));
    std::ofstream{base + ".state", std::ios::binary}.write(static_cast<const char*>(state),
                                                          static_cast<std::streamsize>(state_size));
}
} // Anonymous namespace

void PipelineCache::Nx2GplCheck(const GraphicsPipelineCacheKey& key,
                                const Shader::IR::Program& program, size_t index,
                                std::span<const u32> spirv) const {
    if (std::getenv("EDEN_NX2_GPL_DUMP") != nullptr) {
        if (index == 1) {
            const GplPreRasterState state{MakeGplPreRasterState(device, key.state, spirv)};
            DumpGplStage(key.unique_hashes[index], true, spirv, &state, sizeof(state));
        } else {
            const GplFragmentState state{MakeGplFragmentState(
                device, key.state, spirv, program.info.stores_frag_color[0])};
            DumpGplStage(key.unique_hashes[index], true, spirv, &state, sizeof(state));
        }
    }
    static const bool check = [] {
        const char* value = std::getenv("EDEN_NX2_GPL_CHECK");
        return value != nullptr && value[0] == '1';
    }();
    if (check) {
        const GraphicsPipelineCacheKey default_key{GplDefaultKey(program.stage)};
        LOG_INFO(Render_Vulkan, "NX2STATS gplcheck {:016x} {}:{}", key.unique_hashes[index],
                 index == 1 ? "vs" : "ps",
                 GplRuntimeDiff(MakeGplRuntimeInfo(program, key),
                                MakeGplRuntimeInfo(program, default_key), program.info,
                                program.stage));
    }
}

void PipelineCache::Nx2ReportGraphicsBuild(std::chrono::steady_clock::time_point begin,
                                           std::chrono::steady_clock::duration env_time) {
    if (Nx2Stats::Enabled()) {
        const auto& times = Nx2Stats::build_times;
        const auto total = std::chrono::steady_clock::now() - begin;
        Nx2Stats::Add(Nx2Stats::Phase::Env, env_time);
        Nx2Stats::Add(Nx2Stats::Phase::Translate, times.translate);
        Nx2Stats::Add(Nx2Stats::Phase::Spirv, times.spirv);
        Nx2Stats::Add(Nx2Stats::Phase::Module, times.module);
        Nx2Stats::Add(Nx2Stats::Phase::Layout, times.layout);
        Nx2Stats::Add(Nx2Stats::Phase::GpuTotal, total);
        if (times.cached_stages != 0 && times.translated_stages == 0) {
            Nx2Stats::Add(Nx2Stats::Phase::IrCached, total);
        }
        Nx2Stats::Get().ir_stage_hits.fetch_add(times.cached_stages, std::memory_order_relaxed);
        Nx2Stats::Get().ir_stage_misses.fetch_add(times.translated_stages,
                                                  std::memory_order_relaxed);
        if (Nx2Stats::Level() >= 2) {
            LOG_INFO(Render_Vulkan,
                     "NX2STATS pipeline {:016x}: lower {:.2f} env {:.2f} translate {:.2f} spirv {:.2f} "
                     "module {:.2f} layout {:.2f} total {:.2f} ms, stages cached {} translated {}",
                     graphics_key.Hash(), Nx2Stats::Ms(Nx2Stats::lowered_since_pipeline),
                     Nx2Stats::Ms(env_time), Nx2Stats::Ms(times.translate),
                     Nx2Stats::Ms(times.spirv), Nx2Stats::Ms(times.module),
                     Nx2Stats::Ms(times.layout), Nx2Stats::Ms(total), times.cached_stages,
                     times.translated_stages);
        }
        Nx2Stats::lowered_since_pipeline = {};
    }
}

void PipelineCache::Nx2LogRuntimePipeline(const GraphicsEnvironments& environments,
                                          const GraphicsPipeline* pipeline) const {
    if (RuntimePipelineLog()) {
        std::string stages;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (graphics_key.unique_hashes[index] == 0) {
                continue;
            }
            const auto& env{environments.envs[index]};
            u32 word0{};
            std::memcpy(&word0, &env.SPH(), sizeof(word0));
            stages += fmt::format(" s{}:{:016x}/st{}/{:08x}", index,
                                  graphics_key.unique_hashes[index],
                                  static_cast<u32>(env.ShaderStage()), word0);
            if (env.ShaderStage() == Shader::Stage::Task ||
                env.ShaderStage() == Shader::Stage::Mesh) {
                const auto& regs{env.MeshRegisters()};
                const auto hex = [](const Shader::ProgramHeader& header) {
                    std::array<u32, sizeof(header) / sizeof(u32)> words{};
                    std::memcpy(words.data(), &header, sizeof(header));
                    return fmt::format("{:08x}", fmt::join(words, ""));
                };
                stages += fmt::format("[mesh {:08x} {:08x} {:08x} sph {}", regs[0], regs[1],
                                      regs[2], hex(env.SPH()));
                if (env.ShaderStage() == Shader::Stage::Mesh) {
                    stages += fmt::format(" prim {}", hex(env.MeshPrimitiveHeader()));
                }
                // The program's own hash (the IR cache key before its stage tag) and its first
                // code bytes, to find it in the game files.
                stages += fmt::format(
                    " own {:016x} size {:#x} code {:02x}]",
                    graphics_key.unique_hashes[index] ^
                        VideoCommon::MeshStateHash(env.MeshRegisters(), env.MeshPrimitiveHeader()),
                    env.SourceSizeBytes(), fmt::join(env.ReadProgramBytes(0x80, 32), ""));
            }
        }
        LOG_INFO(Render_Vulkan, "NX2PIPE runtime {:016x} mesh={} disk={} ok={} ir={}/{}{}",
                 graphics_key.Hash(), pipeline && pipeline->IsMeshPipeline() ? 1 : 0,
                 disk_pipelines.contains(graphics_key.Hash()) ? 1 : 0, pipeline ? 1 : 0,
                 Nx2Stats::build_times.cached_stages,
                 Nx2Stats::build_times.cached_stages + Nx2Stats::build_times.translated_stages,
                 stages);
    }
}

namespace {
// Pointer-table entries (Lichii's IsCbufIndirectStorageBufferPointer); a base without them never
// records such keys.
template <typename Env>
bool LiveIndirectPointer(Env& env, u32 bank, u32 offset, u32 indirect) {
    if constexpr (requires { env.IsCbufIndirectStorageBufferPointer(bank, offset, indirect); }) {
        return env.IsCbufIndirectStorageBufferPointer(bank, offset, indirect);
    } else {
        return false;
    }
}

// Whether a storage buffer's writes have a bounded extent computed from the work group (Lichii's
// StorageWriteExtent); a base without write extents translates independently of the launch.
template <typename Descriptor>
bool HasBoundedWriteExtent(const Descriptor& desc) {
    if constexpr (requires { desc.write_extent.bounded; }) {
        return desc.write_extent.bounded;
    } else {
        return false;
    }
}

// Whether a cached stage record's environment inputs give the same answers in env (texture
// queries are asked again with the live handles, i.e. the live descriptors).
// nullptr when a cached stage record's environment inputs give the same answers in env (texture
// queries are asked again with the live handles, i.e. the live descriptors); else which one differs.
const char* RecordMismatch(const VideoCommon::Nx2IrCache::Record& record, Shader::Environment& env) {
    if (record.stage != env.ShaderStage()) {
        return "stage";
    }
    if (record.texture_bound_used && env.TextureBoundBuffer() != record.texture_bound) {
        return "texture bound buffer";
    }
    if (record.viewport_used &&
        env.ReadViewportTransformState() != record.viewport_transform_state) {
        return "viewport transform";
    }
    if (record.local_memory_used && env.LocalMemorySize() != record.local_memory_size) {
        return "local memory size";
    }
    for (const auto& [key, mapped] : record.storage_pointers) {
        // Keys as GenericEnvironment makes them: (bank << 32 | offset), and pointer-table entries
        // with bit 63 set and the table offset in bits 40-61.
        const u32 bank = static_cast<u32>(key >> 32) & 0xff;
        const u32 offset = static_cast<u32>(key);
        const bool live = (key >> 63) != 0
                              ? LiveIndirectPointer(env, bank, offset,
                                                    static_cast<u32>(key >> 40) & 0x3fffff)
                              : env.IsCbufStorageBufferPointer(static_cast<u32>(key >> 32), offset);
        if (live != (mapped != 0)) {
            return (key >> 63) != 0 ? (live ? "table pointer mapped live, unmapped cached"
                                            : "table pointer unmapped live, mapped cached")
                                    : (live ? "storage pointer mapped live, unmapped cached"
                                            : "storage pointer unmapped live, mapped cached");
        }
    }
    for (const auto& check : record.texture_checks) {
        const auto& source = check.source;
        const u32 handle = (env.ReadCbufValue(source[0], source[1]) << source[2]) |
                           (env.ReadCbufValue(source[3], source[4]) << source[5]);
        const u32 live = check.kind == 0
                             ? static_cast<u32>(env.ReadTextureType(handle))
                         : check.kind == 1
                             ? static_cast<u32>(env.IsTexturePixelFormatInteger(handle))
                             : static_cast<u32>(env.ReadTexturePixelFormat(handle));
        if (live != check.answer) {
            return check.kind == 0   ? "texture type"
                   : check.kind == 1 ? "texture integer format"
                                     : "texture pixel format";
        }
    }
    return nullptr;
}

// Lichii's storage write extents (global_memory_to_storage_buffer_pass, StoreExtentAnalysis) are
// computed during translation from the work group size: a compute record with a bounded write
// extent is only right for the launch it was translated with.
bool ComputeLaunchMatches(const Shader::IR::Program& program, const Shader::Environment& env) {
    if (program.stage != Shader::Stage::Compute || program.workgroup_size == env.WorkgroupSize()) {
        return true;
    }
    return std::ranges::none_of(program.info.storage_buffers_descriptors,
                                [](const auto& desc) { return HasBoundedWriteExtent(desc); });
}

// Otherwise a compute record serves every launch: the work group and shared memory size (and the
// local memory size when the program doesn't use local memory) are only copied into the program by
// the translation, so they are taken from the launch.
void ApplyComputeLaunch(Shader::IR::Program& program, const Shader::Environment& env) {
    if (program.stage != Shader::Stage::Compute) {
        return;
    }
    program.workgroup_size = env.WorkgroupSize();
    program.shared_memory_size = env.SharedMemorySize();
    if (!program.info.uses_local_memory) {
        program.local_memory_size = env.LocalMemorySize();
    }
}
} // Anonymous namespace

bool PipelineCache::TryCachedTranslation(ShaderPools& pools, u64 unique_hash,
                                         Shader::Environment& env, Shader::IR::Program& out) {
    if (env.HasHLEMacroState()) {
        return false;
    }
    std::vector<VideoCommon::Nx2IrCache::Record> records = ir_cache.Find(unique_hash);
    const char* mismatch = nullptr;
    int mismatch_stage = 0;
    SCOPE_EXIT {
        // Reached with mismatch set only when no record was used (a used one returns true first
        // and clears it).
        if (mismatch == nullptr) {
            return;
        }
        Nx2Stats::Get().ir_env_mismatches.fetch_add(1, std::memory_order_relaxed);
        // EDEN_NX2_IR_MISMATCH_LOG=1: which precompile default the live state contradicts (the
        // last record's, when there are variants).
        static const bool log_mismatch = [] {
            const char* value = std::getenv("EDEN_NX2_IR_MISMATCH_LOG");
            return value != nullptr && value[0] == '1';
        }();
        if (log_mismatch) {
            LOG_INFO(Render_Vulkan, "NX2PC mismatch {:016x} (stage {}): {} [{} records]",
                     unique_hash, mismatch_stage, mismatch, records.size());
        }
    };
    for (VideoCommon::Nx2IrCache::Record& record : records) {
        try {
            if (record.stage != env.ShaderStage()) {
                continue;
            }
            if (const char* const why = RecordMismatch(record, env)) {
                // Counted and logged once per lookup, below, when no record (variant) matches.
                mismatch = why;
                mismatch_stage = static_cast<int>(record.stage);
                continue;
            }
            const bool mesh_stage{record.stage == Shader::Stage::Task ||
                                  record.stage == Shader::Stage::Mesh};
            if (!ir_cache.LoadProgram(record)) {
                continue;
            }
            Shader::IR::Program program{VideoCommon::Nx2IrCache::DeserializeProgram(
                record.program, pools.inst, pools.block)};
            if (!ComputeLaunchMatches(program, env)) {
                Nx2Stats::Get().ir_env_mismatches.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (mesh_stage) {
                Shader::Maxwell::ApplyMeshState(program, env);
            }
            ApplyComputeLaunch(program, env);
            // EDEN_NX2_IR_CACHE_CHECK=1: also translate with the live answers and compare the
            // SPIR-V both give (default runtime info); on a difference use the live translation.
            static const bool check = [] {
                const char* value = std::getenv("EDEN_NX2_IR_CACHE_CHECK");
                return value != nullptr && value[0] == '1';
            }();
            if (check) {
                static ShaderPools check_pools;
                static std::mutex check_mutex;
                std::scoped_lock check_lock{check_mutex};
                check_pools.ReleaseContents();
                const bool compute = record.stage == Shader::Stage::Compute;
                // Compute environments start at the code (after the SPH words); graphics ones at
                // the SPH.
                const u32 cfg_offset{static_cast<u32>(
                    env.StartAddress() + (compute ? 0 : sizeof(Shader::ProgramHeader)))};
                Shader::Maxwell::Flow::CFG cfg(env, check_pools.flow_block, cfg_offset, false);
                Shader::IR::Program live{
                    TranslateProgram(check_pools.inst, check_pools.block, env, cfg, host_info)};
                Shader::IR::Program cached{VideoCommon::Nx2IrCache::DeserializeProgram(
                    record.program, check_pools.inst, check_pools.block)};
                if (mesh_stage) {
                    Shader::Maxwell::ApplyMeshState(cached, env);
                }
                ApplyComputeLaunch(cached, env);
                if (compute) {
                    const bool same = EmitSPIRV(profile, live) == EmitSPIRV(profile, cached);
                    LOG_INFO(Render_Vulkan, "NX2PC check {:016x} (cs): cached translation {}",
                             unique_hash, same ? "same as live" : "DIFFERS from live");
                    if (!same) {
                        Nx2Stats::Get().ir_env_mismatches.fetch_add(1, std::memory_order_relaxed);
                        return false;
                    }
                    env.AdoptAnswers(record.answers);
                    out = std::move(program);
                    mismatch = nullptr;
                    return true;
                }
                Shader::RuntimeInfo runtime_info{};
                runtime_info.previous_stage_stores.mask.set();
                runtime_info.input_topology = Shader::InputTopology::Triangles;
                if (record.stage == Shader::Stage::Fragment) {
                    runtime_info.alpha_test_func = Shader::CompareFunction::Always;
                }
                Shader::Backend::Bindings bindings_live, bindings_cached;
                ConvertLegacyToGeneric(live, runtime_info);
                ConvertLegacyToGeneric(cached, runtime_info);
                const bool same = EmitSPIRV(profile, runtime_info, live, bindings_live) ==
                                  EmitSPIRV(profile, runtime_info, cached, bindings_cached);
                LOG_INFO(Render_Vulkan, "NX2PC check {:016x} ({}): cached translation {}",
                         unique_hash,
                         record.stage == Shader::Stage::Fragment ? "ps"
                         : record.stage == Shader::Stage::Task   ? "task"
                         : record.stage == Shader::Stage::Mesh   ? "mesh"
                                                                 : "vs",
                         same ? "same as live" : "DIFFERS from live");
                if (!same) {
                    Nx2Stats::Get().ir_env_mismatches.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
            }
            env.AdoptAnswers(record.answers);
            out = std::move(program);
            mismatch = nullptr;
            return true;
        } catch (const std::exception& exception) {
            static std::atomic<u32> logged{};
            if (logged++ < 20) {
                LOG_ERROR(Render_Vulkan, "NX2PC cached translation of {:016x} unusable: {}",
                          unique_hash, exception.what());
            }
        }
    }
    return false;
}

namespace {
struct PrecompileStage {
    std::vector<u8> blob;
    u32 const_offset;
    u32 const_size;
    Shader::Stage stage;
    /// Compute: the launch from the NVN control section (work group 0: unknown, translate only).
    VideoCommon::PrecompileEnvironment::ComputeLaunch launch{{0, 0, 0}, 0, 0};
    std::string source; ///< where the pack says the program is from (file:offset)
};

// Pack (written by an external scan of the game files, not part of the tree): "S2PK", u32 version
// (1, 2, 3), u32 count; per entry u32 stage mask (bit 1 VertexB, bit 4 Geometry, bit 5 Pixel,
// bit 6 Compute, bit 7 task, bit 8 mesh: the BNSH stage record index and Shader::Stage), u32 source
// length + source text, per set bit: u32 size, blob, and in version >= 2 u32 const_offset, u32
// const_size (~0: unknown); version 3 compute stages then u32 work group x, y, z (0: unknown),
// u32 shared memory, u32 local memory (the QMD's values). Compute blobs have no header (code at
// +0). Read one stage at a time (a generic pack holds GBs of file windows); repeats are skipped.
class PackReader {
public:
    explicit PackReader(const std::filesystem::path& path) : file{path, std::ios::binary} {
        std::array<char, 4> magic{};
        file.read(magic.data(), magic.size());
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        valid = file && magic == std::array<char, 4>{'S', '2', 'P', 'K'} && version >= 1 &&
                version <= 3;
    }

    bool Valid() const {
        return valid;
    }

    /// Thread safe. False at the end of the pack.
    bool Next(PrecompileStage& out) {
        std::scoped_lock lock{mutex};
        while (valid) {
            if (pending_mask == 0) {
                if (entry == count) {
                    return false;
                }
                u32 source_length{};
                file.read(reinterpret_cast<char*>(&pending_mask), sizeof(pending_mask));
                file.read(reinterpret_cast<char*>(&source_length), sizeof(source_length));
                source.resize(std::min<u32>(source_length, 4096));
                file.read(source.data(), static_cast<std::streamsize>(source.size()));
                file.seekg(source_length - source.size(), std::ios::cur);
                if (!file) {
                    LOG_ERROR(Render_Vulkan, "NX2PC pack truncated at entry {}", entry);
                    valid = false;
                    return false;
                }
                ++entry;
                ++entries;
                continue;
            }
            const u32 slot = static_cast<u32>(std::countr_zero(pending_mask));
            pending_mask &= pending_mask - 1;
            u32 size{};
            file.read(reinterpret_cast<char*>(&size), sizeof(size));
            std::vector<u8> blob(size);
            file.read(reinterpret_cast<char*>(blob.data()), size);
            u32 const_offset = ~0U, const_size = 0;
            if (version >= 2) {
                file.read(reinterpret_cast<char*>(&const_offset), sizeof(const_offset));
                file.read(reinterpret_cast<char*>(&const_size), sizeof(const_size));
            }
            VideoCommon::PrecompileEnvironment::ComputeLaunch launch{{0, 0, 0}, 0, 0};
            if (version >= 3 && slot == 6) {
                file.read(reinterpret_cast<char*>(launch.workgroup.data()),
                          sizeof(launch.workgroup));
                file.read(reinterpret_cast<char*>(&launch.shared_memory),
                          sizeof(launch.shared_memory));
                file.read(reinterpret_cast<char*>(&launch.local_memory),
                          sizeof(launch.local_memory));
            }
            if (!file) {
                LOG_ERROR(Render_Vulkan, "NX2PC pack truncated at entry {}", entry);
                valid = false;
                return false;
            }
            if (slot != 1 && slot != 4 && slot != 5 && !(slot == 6 && version >= 3) && slot != 7 &&
                slot != 8) {
                ++unsupported;
                continue;
            }
            // Optional stage kinds: task/mesh (EDEN_NX2_IR_CACHE_TASKMESH=1), compute
            // (EDEN_NX2_IR_CACHE_COMPUTE, on), geometry (only useful with EDEN_NX2_RELATIVE_LEPC=1:
            // otherwise its lowering depends on its GPU address and a record never matches).
            if (((slot == 7 || slot == 8) && !TaskMeshIrCacheEnabled()) ||
                (slot == 6 && !ComputeIrCacheEnabled()) ||
                (slot == 4 && !GeometryIrCacheEnabled())) {
                ++unsupported;
                continue;
            }
            u64 hash = Common::CityHash64(reinterpret_cast<const char*>(blob.data()), blob.size()) ^
                       (u64{slot} << 56) ^ const_offset;
            if (slot == 6) {
                hash ^= Common::CityHash64(reinterpret_cast<const char*>(&launch), sizeof(launch));
            }
            if (!seen.insert(hash).second) {
                ++duplicates;
                continue;
            }
            ++stages;
            if (slot == 6) {
                ++compute;
            }
            out = {std::move(blob), const_offset, const_size,
                   slot == 1   ? Shader::Stage::VertexB
                   : slot == 4 ? Shader::Stage::Geometry
                   : slot == 5 ? Shader::Stage::Fragment
                   : slot == 6 ? Shader::Stage::Compute
                   : slot == 7 ? Shader::Stage::Task
                               : Shader::Stage::Mesh,
                   launch, source};
            return true;
        }
        return false;
    }

    u32 Count() const {
        return count;
    }

    size_t entries{}, stages{}, duplicates{}, unsupported{}, compute{};

private:
    std::mutex mutex;
    std::ifstream file;
    std::string source;
    u32 version{}, count{}, entry{}, pending_mask{};
    bool valid{};
    std::unordered_set<u64> seen;
};

// What a graphics stage can't do (a compute program, or bytes that only look like a program, from
// a generic scan): the SPIR-V backend aborts on shared memory outside compute.
std::optional<std::string_view> ComputeOnlyOperation(const Shader::IR::Program& program) {
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            const std::string_view name{Shader::IR::NameOf(inst.GetOpcode())};
            if (name.find("Shared") != std::string_view::npos || name == "WorkgroupId" ||
                name == "LocalInvocationId" || name == "Barrier" ||
                name == "WorkgroupMemoryBarrier") {
                return name;
            }
        }
    }
    return std::nullopt;
}

bool UsesSharedMemory(const Shader::IR::Program& program) {
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            if (Shader::IR::NameOf(inst.GetOpcode()).find("Shared") != std::string_view::npos) {
                return true;
            }
        }
    }
    return false;
}

std::string ShortReason(std::string_view what) {
    // "Native shader at 0x..., instruction 0x...: <reason> [op ...]": keep the reason.
    std::string text{what};
    if (const size_t colon = text.find(": "); text.starts_with("Native shader at") &&
                                              colon != std::string::npos) {
        text = text.substr(colon + 2);
    }
    if (const size_t bracket = text.find(" ["); bracket != std::string::npos) {
        text.resize(bracket);
    }
    if (text.size() > 80) {
        text.resize(80);
    }
    return text;
}
} // Anonymous namespace

void PipelineCache::RunPrecompile(std::filesystem::path pack_path, std::stop_token stop) {
    Common::SetCurrentThreadName("Nx2Precompile");
    const auto begin = std::chrono::steady_clock::now();
    PackReader reader{pack_path};
    if (!reader.Valid()) {
        LOG_ERROR(Render_Vulkan, "NX2PC {} is not a precompile pack", pack_path.string());
        return;
    }
    LOG_INFO(Render_Vulkan, "NX2PC precompile {}: {} entries", pack_path.string(), reader.Count());
    const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_VERIFY");
        return value == nullptr || value[0] != '0';
    }();
    const u32 default_texture_bound = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_TEXTURE_BOUND");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : 0U;
    }();

    std::atomic<size_t> gs_built{};
    std::atomic<size_t> done{}, built{}, cached{}, lowering_failed{}, translation_failed{},
        broken{}, verify_failed{}, const_rule_checked{}, const_rule_matched{}, gpl_built{},
        gpl_skipped{}, gpl_failed{}, retried_as_task_mesh{}, task_mesh_built{},
        pointer_variants_built{};
    std::atomic<u64> gpl_ns{};
    const bool gpl = device.IsGplEnabled();
    std::atomic<u64> translate_ns{}, raw_bytes{};
    std::mutex reasons_mutex;
    std::map<std::string, size_t> reasons;
    const auto fail = [&](std::atomic<size_t>& counter, std::string reason) {
        ++counter;
        std::scoped_lock lock{reasons_mutex};
        ++reasons[std::move(reason)];
    };
    // Library path: the stage's library from its cached IR with the default state, created once
    // so the driver's cache has it; a later session's first draw of the stage then finds it.
    const auto make_library = [&](ShaderPools& gpl_pools, const VideoCommon::Nx2IrCache::Record& record) {
        if (!gpl || (record.stage != Shader::Stage::VertexB &&
                     record.stage != Shader::Stage::Fragment)) {
            return;
        }
        try {
            gpl_pools.ReleaseContents();
            VideoCommon::Nx2IrCache::Record loaded{record};
            if (!ir_cache.LoadProgram(loaded)) {
                return;
            }
            Shader::IR::Program program{VideoCommon::Nx2IrCache::DeserializeProgram(
                loaded.program, gpl_pools.inst, gpl_pools.block)};
            const bool fragment = record.stage == Shader::Stage::Fragment;
            if ((fragment && (program.info.loads.Legacy() ||
                              program.info.texture_descriptors.size() >
                                  GPL_VERTEX_TEXTURE_SCALING_BASE ||
                              program.info.image_descriptors.size() >
                                  GPL_VERTEX_IMAGE_SCALING_BASE)) ||
                (!fragment && (program.info.stores.Legacy() || program.info.is_task ||
                               program.info.is_mesh || VertexReadsPosition(program.info) ||
                               program.info.texture_descriptors.size() > 32 ||
                               program.info.image_descriptors.size() > 16))) {
                ++gpl_skipped;
                return;
            }
            const GraphicsPipelineCacheKey key{GplDefaultKey(record.stage)};
            const std::vector<u32> spirv{EmitGplStage(program, key)};
            GplLibraryCache::Result result;
            if (fragment) {
                const GplFragmentState state{MakeGplFragmentState(
                    device, key.state, spirv, program.info.stores_frag_color[0])};
                DumpGplStage(record.unique_hash, false, spirv, &state, sizeof(state));
                result = gpl_libraries.CreateDetached(nullptr, &state, program.info, spirv);
            } else {
                const GplPreRasterState state{MakeGplPreRasterState(device, key.state, spirv)};
                DumpGplStage(record.unique_hash, false, spirv, &state, sizeof(state));
                result = gpl_libraries.CreateDetached(&state, nullptr, program.info, spirv);
            }
            gpl_ns += static_cast<u64>(result.ms * 1e6);
            ++gpl_built;
        } catch (const std::exception& exception) {
            fail(gpl_failed, "library: " + ShortReason(exception.what()));
        }
    };
    // Compute stages; those without a known launch get a record that only applies to work group
    // 1x1x1 when the program has bounded write extents (ComputeLaunchMatches).
    std::atomic<size_t> cs_stages{}, cs_built{}, cs_no_launch{};
    // EDEN_NX2_PRECOMPILE_COMPUTE_BUILD=0: no compute pipelines. Otherwise each compute record with a
    // known launch is built as the runtime would (same SPIR-V, layout and flags: ComputePipeline)
    // and dropped, so the driver's disk cache has it. Measured on OoT: the AV1 intro's first compute
    // pipeline compiles for ~60 s on a cold RADV cache and the game waits for it.
    const bool cs_build = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_COMPUTE_BUILD");
        return value == nullptr || value[0] != '0';
    }();
    std::atomic<size_t> cs_pipelines{}, cs_pipeline_failed{};
    std::atomic<u64> cs_pipeline_ns{};
    const auto build_compute = [&](ShaderPools& cs_pools,
                                   const VideoCommon::Nx2IrCache::Record& record,
                                   const VideoCommon::PrecompileEnvironment::ComputeLaunch& launch) {
        if (!cs_build || launch.workgroup[0] == 0 || launch.workgroup[1] == 0 ||
            launch.workgroup[2] == 0) {
            return;
        }
        if (record.local_memory_used && record.local_memory_size != launch.local_memory) {
            return;
        }
        try {
            const auto build_begin = Nx2Stats::Clock::now();
            cs_pools.ReleaseContents();
            VideoCommon::Nx2IrCache::Record loaded{record};
            if (!ir_cache.LoadProgram(loaded)) {
                return;
            }
            Shader::IR::Program program{VideoCommon::Nx2IrCache::DeserializeProgram(
                loaded.program, cs_pools.inst, cs_pools.block)};
            program.workgroup_size = launch.workgroup;
            program.shared_memory_size = launch.shared_memory;
            if (!program.info.uses_local_memory) {
                program.local_memory_size = launch.local_memory;
            }
            const std::vector<u32> code{EmitSPIRV(profile, program)};
            LogComputeSpirv("precompile", record.unique_hash, program, code);
            vk::ShaderModule module{BuildShader(device, code)};
            // Built here, on this thread (no build queue, no driver blob store: the driver's
            // own cache keeps it).
            ComputePipeline pipeline{device,
                                     scheduler,
                                     vulkan_pipeline_cache,
                                     nullptr,
                                     0,
                                     descriptor_pool,
                                     guest_descriptor_queue,
                                     descriptor_buffer_ring,
                                     PipelineBuildQueue{},
                                     nullptr,
                                     nullptr,
                                     program.info,
                                     std::move(module),
                                     record.unique_hash};
            cs_pipeline_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   Nx2Stats::Clock::now() - build_begin)
                                                   .count());
            Nx2Stats::Add(Nx2Stats::Phase::CsPrebuild, Nx2Stats::Clock::now() - build_begin);
            ++cs_pipelines;
        } catch (const std::exception& exception) {
            fail(cs_pipeline_failed, "compute pipeline: " + ShortReason(exception.what()));
        }
    };
    // EDEN_NX2_PRECOMPILE_TRACE=<file>: "begin <source>" / "end" per stage, flushed, so an abort
    // inside the recompiler or the driver names the program.
    std::FILE* trace = nullptr;
    if (const char* path = std::getenv("EDEN_NX2_PRECOMPILE_TRACE")) {
        trace = std::fopen(path, "w");
    }
    std::mutex trace_mutex;
    const auto trace_line = [&](std::string_view what, const std::string& source) {
        if (trace) {
            std::scoped_lock lock{trace_mutex};
            std::fprintf(trace, "%.*s %s\n", static_cast<int>(what.size()), what.data(),
                         source.c_str());
            std::fflush(trace);
        }
    };
    const auto work = [&] {
        ShaderPools pools;
        ShaderPools verify_pools;
        ShaderPools gpl_pools;
        PrecompileStage stage;
        while (!stop.stop_requested() && reader.Next(stage)) {
            trace_line("begin", stage.source);
            SCOPE_EXIT {
                ++done;
                trace_line("end", stage.source);
            };
            const bool compute = stage.stage == Shader::Stage::Compute;
            if (compute) {
                ++cs_stages;
            }
            // Unknown launch: work group 1x1x1 and shared memory declared (the SPIR-V emitted to
            // verify the record needs it when the program uses shared memory; Sirit aborts
            // otherwise). On Lichii such a record serves the runtime only where the launch doesn't
            // change the translation (ComputeLaunchMatches).
            VideoCommon::PrecompileEnvironment::ComputeLaunch env_launch{stage.launch};
            if (compute && (env_launch.workgroup[0] == 0 || env_launch.workgroup[1] == 0 ||
                            env_launch.workgroup[2] == 0)) {
                ++cs_no_launch;
                env_launch.workgroup = {1, 1, 1};
                env_launch.shared_memory = 48 * 1024;
            }
            // A generic pack doesn't know task and mesh programs from vertex programs (same
            // header): a vertex program that reads or writes the stage buffer is tried as both.
            // Each attempt: the stage kind, and whether constant-buffer storage pointers are taken
            // as mapped (a record that asked gets a second, unmapped-pointer variant, below).
            struct Attempt {
                Shader::Stage stage;
                std::vector<u64> unmapped; ///< storage pointer keys answered "not mapped"
                bool variant;              ///< a pointer variant of a record made just before
            };
            std::vector<Attempt> attempts{{stage.stage, {}, false}};
            for (size_t attempt = 0; attempt < attempts.size(); ++attempt) {
                const Shader::Stage stage_kind = attempts[attempt].stage;
                const bool pointers_mapped = !attempts[attempt].variant;
                std::optional<VideoCommon::PrecompileEnvironment> env;
                try {
                    env.emplace(stage.blob, stage.const_offset, stage.const_size, stage_kind,
                                default_texture_bound, env_launch);
                    env->SetUnmappedPointers(attempts[attempt].unmapped);
                } catch (const std::exception& exception) {
                    fail(lowering_failed, "lowering: " + ShortReason(exception.what()));
                    continue;
                }
                if (stage.const_offset != ~0U) {
                    ++const_rule_checked;
                    if (env->DerivedConstOffset() == stage.const_offset) {
                        ++const_rule_matched;
                    }
                }
                const bool mesh_stage =
                    stage_kind == Shader::Stage::Task || stage_kind == Shader::Stage::Mesh;
                const u64 unique_hash = mesh_stage ? TaskMeshRecordHash(*env->Analyze(), stage_kind)
                                                   : *env->Analyze();
                if (pointers_mapped && ir_cache.Contains(unique_hash)) {
                    ++cached;
                    if (compute) {
                        for (const auto& record : ir_cache.Find(unique_hash)) {
                            if (record.stage == Shader::Stage::Compute) {
                                build_compute(gpl_pools, record, stage.launch);
                                break;
                            }
                        }
                    }
                    if (gpl && !compute) {
                        const std::vector<VideoCommon::Nx2IrCache::Record> records{
                            ir_cache.Find(unique_hash)};
                        if (!records.empty()) {
                            make_library(gpl_pools, records.front());
                        }
                    }
                    continue;
                }
                pools.ReleaseContents();
                const auto translate_begin = std::chrono::steady_clock::now();
                Shader::IR::Program program;
                try {
                    // Graphics environments start at the SPH, compute ones (start address 0, no
                    // header) also skip the SPH words, as ComputeEnvironment's start address does.
                    const u32 cfg_offset{
                        static_cast<u32>(env->StartAddress() + sizeof(Shader::ProgramHeader))};
                    Shader::Maxwell::Flow::CFG cfg(*env, pools.flow_block, cfg_offset, false);
                    program = TranslateProgram(pools.inst, pools.block, *env, cfg, host_info);
                } catch (const std::exception& exception) {
                    if (stage_kind == Shader::Stage::VertexB && TaskMeshIrCacheEnabled() &&
                        std::string_view{exception.what()}.find(
                            "Stage buffer access outside task and mesh") != std::string_view::npos) {
                        ++retried_as_task_mesh;
                        attempts.push_back({Shader::Stage::Task, {}, false});
                        attempts.push_back({Shader::Stage::Mesh, {}, false});
                        continue;
                    }
                    fail(translation_failed, "translation: " + ShortReason(exception.what()) +
                                                 (stage_kind != stage.stage ? " (as task/mesh)" : "") +
                                                 (pointers_mapped ? "" : " (unmapped pointers)"));
                    continue;
                }
                translate_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - translate_begin)
                                                     .count());
                // Task and mesh programs are workgroups: shared memory and barriers are theirs.
                if (const auto operation = mesh_stage || compute ? std::nullopt
                                                                 : ComputeOnlyOperation(program)) {
                    fail(translation_failed,
                         fmt::format("translation: compute-only operation {} in a graphics stage",
                                     *operation));
                    continue;
                }
                if (compute && program.shared_memory_size == 0 && UsesSharedMemory(program)) {
                    // a launch without shared memory for a program that uses it: a wrong control
                    // section match (and the SPIR-V backend aborts on it)
                    fail(translation_failed, "compute: shared memory used, launch has none");
                    continue;
                }
                if (compute && program.local_memory_size == 0 && program.info.uses_local_memory) {
                    // the same for local memory (no control section field gives it): on Lichii
                    // the SPIR-V backend aborts on such a program (seen in a game's files), and
                    // its record could never match a real launch (local memory size checked)
                    fail(translation_failed, "compute: local memory used, launch has none");
                    continue;
                }
                if (stage_kind == Shader::Stage::Geometry && program.is_geometry_passthrough) {
                    // its translation reads the passthrough mask (draw state), which records
                    // don't check
                    fail(translation_failed, "geometry passthrough (not cached)");
                    continue;
                }
                if (!compute && device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY &&
                    Shader::Optimization::HasBrokenPattern(program)) {
                    fail(broken, "known broken pattern (the runtime skips these pipelines)");
                    continue;
                }
                VideoCommon::Nx2IrCache::Record record;
                try {
                    record.program = VideoCommon::Nx2IrCache::SerializeProgram(program);
                    if (verify) {
                        // The cached form must give the same SPIR-V as the translation it came
                        // from.
                        verify_pools.ReleaseContents();
                        Shader::IR::Program copy{VideoCommon::Nx2IrCache::DeserializeProgram(
                            record.program, verify_pools.inst, verify_pools.block)};
                        bool same;
                        if (compute) {
                            same = EmitSPIRV(profile, program) == EmitSPIRV(profile, copy);
                        } else {
                            Shader::RuntimeInfo runtime_info{};
                            runtime_info.previous_stage_stores.mask.set();
                            runtime_info.input_topology = Shader::InputTopology::Triangles;
                            if (stage_kind == Shader::Stage::Fragment) {
                                runtime_info.alpha_test_func = Shader::CompareFunction::Always;
                            }
                            Shader::Backend::Bindings bindings_a, bindings_b;
                            ConvertLegacyToGeneric(program, runtime_info);
                            ConvertLegacyToGeneric(copy, runtime_info);
                            same = EmitSPIRV(profile, runtime_info, program, bindings_a) ==
                                   EmitSPIRV(profile, runtime_info, copy, bindings_b);
                        }
                        if (!same) {
                            fail(verify_failed, "verify: SPIR-V of the cached form differs");
                            continue;
                        }
                    }
                } catch (const std::exception& exception) {
                    fail(verify_failed, "cache form: " + ShortReason(exception.what()));
                    continue;
                }
                record.unique_hash = unique_hash;
                record.stage = stage_kind;
                record.texture_bound_used = env->TextureBoundUsed();
                record.viewport_used = env->ViewportUsed();
                record.texture_bound = env->TextureBoundBuffer();
                record.viewport_transform_state = 1;
                record.local_memory_used = program.info.uses_local_memory;
                record.local_memory_size = env->LocalMemorySize();
                for (const auto& [key, mapped] : env->StoragePointerAnswers()) {
                    record.storage_pointers.emplace_back(key, mapped);
                }
                for (const auto& query : env->TextureQueries()) {
                    record.texture_checks.push_back({query.source, query.kind, query.answer});
                }
                env->SerializeAnswers(record.answers);
                raw_bytes += record.program.size();
                ir_cache.Append(record);
                ++built;
                if (!pointers_mapped) {
                    ++pointer_variants_built;
                } else if (PointerVariantsEnabled() && !env->StoragePointerAnswers().empty()) {
                    // Every mapped/null combination of the pointers this stage asked about (live
                    // they are a mix: some still null when the game first uses the program), up to
                    // EDEN_NX2_PRECOMPILE_POINTER_MAX pointers; beyond that each one null alone,
                    // and all null.
                    std::vector<u64> keys;
                    for (const auto& [key, mapped] : env->StoragePointerAnswers()) {
                        keys.push_back(key);
                    }
                    std::ranges::sort(keys);
                    if (keys.size() <= PointerVariantsMax()) {
                        for (u32 mask = 1; mask < (1U << keys.size()); ++mask) {
                            std::vector<u64> unmapped;
                            for (size_t bit = 0; bit < keys.size(); ++bit) {
                                if ((mask >> bit) & 1) {
                                    unmapped.push_back(keys[bit]);
                                }
                            }
                            attempts.push_back({stage_kind, std::move(unmapped), true});
                        }
                    } else {
                        for (const u64 key : keys) {
                            attempts.push_back({stage_kind, {key}, true});
                        }
                        attempts.push_back({stage_kind, keys, true});
                    }
                }
                if (mesh_stage) {
                    ++task_mesh_built;
                }
                if (stage_kind == Shader::Stage::Geometry) {
                    ++gs_built;
                }
                if (compute) {
                    ++cs_built;
                    build_compute(gpl_pools, record, stage.launch);
                } else {
                    make_library(gpl_pools, record);
                }
            }
        }
    };
    const size_t num_threads = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILE_THREADS");
        return value ? std::max<size_t>(1, std::strtoull(value, nullptr, 0))
                     : std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
    }();
    std::atomic<size_t> finished{};
    {
        std::vector<std::jthread> threads;
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&work, &finished] {
                Common::SetCurrentThreadName("Nx2PrecompileWorker");
                Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
                work();
                ++finished;
            });
        }
        auto last_log = std::chrono::steady_clock::now();
        while (!stop.stop_requested() &&
               std::ranges::any_of(threads, [](const std::jthread& t) { return t.joinable(); })) {
            std::this_thread::sleep_for(std::chrono::milliseconds{200});
            if (finished.load() == threads.size()) {
                break;
            }
            if (std::chrono::steady_clock::now() - last_log > std::chrono::seconds{5}) {
                last_log = std::chrono::steady_clock::now();
                LOG_INFO(Render_Vulkan, "NX2PC precompile progress: {} stages, {} built, {} libraries",
                         done.load(), built.load(), gpl_built.load());
            }
        }
    }
    if (trace) {
        std::fclose(trace);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan,
             "NX2PC precompile {} in {:.1f} s ({} threads): {} entries, {} distinct VS/PS stages ({} "
             "repeats, {} other stages), {} built, {} already cached, "
             "failed: {} lowering, {} translation, {} broken pattern, {} cache form; "
             "translation {:.1f} ms per stage; const offset rule matched {} / {}; "
             "cache now {} records, {:.1f} MB ({:.1f} MB of IR before compression this run)",
             stop.stop_requested() ? "stopped" : "done", seconds, num_threads, reader.entries,
             reader.stages, reader.duplicates, reader.unsupported, built.load(), cached.load(), lowering_failed.load(), translation_failed.load(),
             broken.load(), verify_failed.load(),
             built.load() ? translate_ns.load() / 1e6 / static_cast<double>(built.load()) : 0.0,
             const_rule_matched.load(), const_rule_checked.load(), ir_cache.NumRecords(),
             ir_cache.FileSize() / 1e6, raw_bytes.load() / 1e6);
    if (TaskMeshIrCacheEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "NX2PC task/mesh: {} built ({} vertex-slot programs that use the stage buffer "
                 "tried as task and mesh)",
                 task_mesh_built.load(), retried_as_task_mesh.load());
    }
    if (pointer_variants_built.load() != 0) {
        LOG_INFO(Render_Vulkan,
                 "NX2PC pointer variants: {} records with unmapped storage pointers",
                 pointer_variants_built.load());
    }
    if (gs_built.load() != 0) {
        LOG_INFO(Render_Vulkan, "NX2PC geometry: {} stages translated", gs_built.load());
    }
    if (cs_stages.load() != 0) {
        LOG_INFO(Render_Vulkan,
                 "NX2PC compute: {} stages, {} records built, {} without a known launch; {} "
                 "pipelines built for the driver cache ({:.2f} ms each), {} failed",
                 cs_stages.load(), cs_built.load(), cs_no_launch.load(), cs_pipelines.load(),
                 cs_pipelines.load() ? cs_pipeline_ns.load() / 1e6 /
                                           static_cast<double>(cs_pipelines.load())
                                     : 0.0,
                 cs_pipeline_failed.load());
    }
    if (gpl) {
        LOG_INFO(Render_Vulkan,
                 "NX2PC libraries: {} created ({:.2f} ms each), {} not usable as libraries, {} failed",
                 gpl_built.load(),
                 gpl_built.load() ? gpl_ns.load() / 1e6 / static_cast<double>(gpl_built.load()) : 0.0,
                 gpl_skipped.load(), gpl_failed.load());
    }
    std::vector<std::pair<size_t, std::string>> sorted;
    for (const auto& [reason, count] : reasons) {
        sorted.emplace_back(count, reason);
    }
    std::ranges::sort(sorted, std::greater{});
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 15); ++i) {
        LOG_INFO(Render_Vulkan, "NX2PC   {} x {}", sorted[i].first, sorted[i].second);
    }
}

namespace {
// EDEN_NX2_IR_LEARN=1: live translations of native vertex, pixel, task/mesh and compute stages are stored as
// per-stage cache records (with the answers and checks the precompile's records carry), so a stage
// the precompile missed or guessed wrong is translated once, not in every session.
bool IrLearnEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_IR_LEARN");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
} // Anonymous namespace

bool PipelineCache::Nx2LearnArm(Shader::Environment& env) const {
    if (!IrLearnEnabled() || !ir_cache.IsOpen() || !env.IsNativeProgram() ||
        env.HasHLEMacroState()) {
        return false;
    }
    // Task/mesh: the mesh state the translation put in is replaced by the live one when the
    // record is used (ApplyMeshState), as for precompiled task/mesh records.
    const Shader::Stage stage{env.ShaderStage()};
    const bool mesh_stage{stage == Shader::Stage::Task || stage == Shader::Stage::Mesh};
    if (stage != Shader::Stage::VertexB && stage != Shader::Stage::Fragment &&
        stage != Shader::Stage::Compute && !(mesh_stage && TaskMeshIrCacheEnabled())) {
        return false;
    }
    if (!env.IsLiveEnvironment()) {
        return false;
    }
    static_cast<VideoCommon::GenericEnvironment&>(env).BeginQueryLog();
    return true;
}

void PipelineCache::Nx2LearnTranslation(u64 record_hash, Shader::Environment& env,
                                        const Shader::IR::Program& program) {
    if (!env.IsLiveEnvironment()) {
        return;
    }
    auto* const generic{static_cast<VideoCommon::GenericEnvironment*>(&env)};
    generic->EndQueryLog();
    if (!IrLearnEnabled() || !ir_cache.IsOpen() || !generic->QueryLogComplete() ||
        generic->HasConstantReplacements()) {
        return;
    }
    // The same exclusions as the precompile's records.
    const bool compute{program.stage == Shader::Stage::Compute};
    if (!compute && device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY &&
        Shader::Optimization::HasBrokenPattern(program)) {
        return;
    }
    if (compute && ((program.shared_memory_size == 0 && UsesSharedMemory(program)) ||
                    (program.local_memory_size == 0 && program.info.uses_local_memory))) {
        return;
    }
    try {
        VideoCommon::Nx2IrCache::Record record;
        record.unique_hash = record_hash;
        record.stage = env.ShaderStage();
        // The texture bound buffer is checked whether or not the translation read it (stricter
        // than needed: a record may miss where it would have fit, never the other way round).
        record.texture_bound_used = true;
        record.texture_bound = env.TextureBoundBuffer();
        record.viewport_used = generic->ViewportTransformRead();
        record.viewport_transform_state = generic->ViewportTransformAnswer();
        record.local_memory_used = program.info.uses_local_memory;
        record.local_memory_size = env.LocalMemorySize();
        for (const auto& [key, mapped] : generic->StoragePointerAnswers()) {
            record.storage_pointers.emplace_back(key, mapped);
        }
        std::ranges::sort(record.storage_pointers);
        for (const auto& query : generic->LoggedTextureQueries()) {
            record.texture_checks.push_back({query.source, query.kind, query.answer});
        }
        // One record per set of checks: several pipelines can translate the stage before the
        // stored record is found.
        std::vector<u8> identity;
        const auto put = [&identity](const void* data, size_t size) {
            const auto* bytes = static_cast<const u8*>(data);
            identity.insert(identity.end(), bytes, bytes + size);
        };
        put(&record.unique_hash, sizeof(record.unique_hash));
        put(&record.stage, sizeof(record.stage));
        put(&record.texture_bound, sizeof(record.texture_bound));
        const u32 viewport{record.viewport_used ? record.viewport_transform_state : ~0U};
        put(&viewport, sizeof(viewport));
        put(&record.local_memory_size, sizeof(record.local_memory_size));
        for (const auto& [key, mapped] : record.storage_pointers) {
            put(&key, sizeof(key));
            put(&mapped, sizeof(mapped));
        }
        for (const auto& check : record.texture_checks) {
            put(&check, sizeof(check));
        }
        const u64 id{Common::CityHash64(reinterpret_cast<const char*>(identity.data()),
                                        identity.size())};
        {
            std::scoped_lock lock{learned_mutex};
            if (!learned_records.insert(id).second) {
                return;
            }
        }
        generic->SerializeAnswers(record.answers);
        record.program = VideoCommon::Nx2IrCache::SerializeProgram(program);
        Nx2Stats::Get().ir_learned.fetch_add(1, std::memory_order_relaxed);
        static const bool log_learn = [] {
            const char* value = std::getenv("EDEN_NX2_IR_MISMATCH_LOG");
            return value != nullptr && value[0] == '1';
        }();
        if (log_learn) {
            LOG_INFO(Render_Vulkan,
                     "NX2PC learned {:016x} (stage {}): {} pointers, {} texture checks",
                     record_hash, static_cast<int>(record.stage), record.storage_pointers.size(),
                     record.texture_checks.size());
        }
        // Compressing and writing off the pipeline's thread.
        serialization_thread.QueueWork(
            [this, record = std::move(record)] { ir_cache.Append(record); });
    } catch (const std::exception& exception) {
        static std::atomic<u32> logged{};
        if (logged++ < 20) {
            LOG_ERROR(Render_Vulkan, "NX2PC translation of {:016x} not stored: {}", record_hash,
                      exception.what());
        }
    }
}

} // namespace Vulkan
