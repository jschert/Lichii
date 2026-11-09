// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <fmt/format.h>

#include <boost/container/static_vector.hpp>

#include "common/bit_field.h"
#include "common/cityhash.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_gpl.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
using boost::container::static_vector;
using VideoCore::Surface::PixelFormat;

constexpr VkPipelineCreateFlags LIBRARY_FLAGS =
    VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;

u64 HashSpirv(std::span<const u32> spirv) {
    return Common::CityHash64(reinterpret_cast<const char*>(spirv.data()), spirv.size_bytes());
}

FixedPipelineState::DynamicState Dynamic(const FixedPipelineState& state) {
    FixedPipelineState::DynamicState dynamic{};
    if (!state.extended_dynamic_state) {
        dynamic = state.dynamic_state;
    } else {
        dynamic.raw1 = state.dynamic_state.raw1;
    }
    return dynamic;
}

VkViewportSwizzleNV UnpackViewportSwizzle(u16 swizzle) {
    union Swizzle {
        u32 raw;
        BitField<0, 3, Maxwell::ViewportSwizzle> x;
        BitField<4, 3, Maxwell::ViewportSwizzle> y;
        BitField<8, 3, Maxwell::ViewportSwizzle> z;
        BitField<12, 3, Maxwell::ViewportSwizzle> w;
    };
    const Swizzle unpacked{swizzle};
    return VkViewportSwizzleNV{
        .x = MaxwellToVK::ViewportSwizzle(unpacked.x),
        .y = MaxwellToVK::ViewportSwizzle(unpacked.y),
        .z = MaxwellToVK::ViewportSwizzle(unpacked.z),
        .w = MaxwellToVK::ViewportSwizzle(unpacked.w),
    };
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    static constexpr std::array unsupported_topologies{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
    };
    return std::ranges::find(unsupported_topologies, topology) == unsupported_topologies.end();
}

GplMultisampleState MakeMultisample(const Device& device, const FixedPipelineState& state,
                                    bool fragment_writes_color0) {
    GplMultisampleState ms{};
    ms.samples = MaxwellToVK::MsaaMode(state.msaa_mode);
    ms.sample_shading = Settings::values.sample_shading.GetValue() > 0 ? 1 : 0;
    const f32 min_shading = f32(Settings::values.sample_shading.GetValue()) / 100.0f;
    std::memcpy(&ms.min_sample_shading_bits, &min_shading, sizeof(u32));
    ms.alpha_to_coverage =
        fragment_writes_color0 && state.alpha_to_coverage_enabled != 0 ? 1 : 0;
    ms.alpha_to_one = fragment_writes_color0 && device.SupportsAlphaToOne() &&
                              state.alpha_to_one_enabled != 0
                          ? 1
                          : 0;
    if (state.extended_dynamic_state_3_enables) {
        if (device.SupportsDynamicState3AlphaToCoverageEnable()) {
            ms.dynamic_flags |= 1;
            ms.alpha_to_coverage = 0;
        }
        if (device.SupportsDynamicState3AlphaToOneEnable()) {
            ms.dynamic_flags |= 2;
            ms.alpha_to_one = 0;
        }
    }
    return ms;
}

VkPipelineMultisampleStateCreateInfo MultisampleInfo(const GplMultisampleState& ms) {
    f32 min_shading;
    std::memcpy(&min_shading, &ms.min_sample_shading_bits, sizeof(f32));
    return VkPipelineMultisampleStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = static_cast<VkSampleCountFlagBits>(ms.samples),
        .sampleShadingEnable = ms.sample_shading,
        .minSampleShading = min_shading,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = ms.alpha_to_coverage,
        .alphaToOneEnable = ms.alpha_to_one,
    };
}

void AddMultisampleDynamic(const GplMultisampleState& ms,
                           static_vector<VkDynamicState, 34>& dynamic_states) {
    if (ms.dynamic_flags & 1) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
    }
    if (ms.dynamic_flags & 2) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT);
    }
}

VkPipelineDynamicStateCreateInfo DynamicInfo(const static_vector<VkDynamicState, 34>& states) {
    return VkPipelineDynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(states.size()),
        .pDynamicStates = states.data(),
    };
}

template <typename T>
bool BytesEqual(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
} // Anonymous namespace

GplVertexInputState MakeGplVertexInputState(const Device& device, const FixedPipelineState& state,
                                            const Shader::Info& vs_info) {
    GplVertexInputState vi{};
    vi.dynamic_vertex_input = state.dynamic_vertex_input;
    vi.dynamic_stride = state.extended_dynamic_state && !state.dynamic_vertex_input ? 1 : 0;
    vi.dynamic_topology = state.extended_dynamic_state;
    vi.dynamic_restart = state.extended_dynamic_state_2;
    auto topology = MaxwellToVK::PrimitiveTopology(device, state.topology);
    if (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        // No tessellation in library pipelines (as the monolithic path does).
        topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    }
    vi.topology = topology;
    const FixedPipelineState::DynamicState dynamic = Dynamic(state);
    const bool restart =
        device.IsMoltenVK() ||
        (dynamic.primitive_restart_enable != 0 &&
         (device.IsTopologyListPrimitiveRestartSupported() || SupportsPrimitiveRestart(topology)));
    vi.primitive_restart = vi.dynamic_restart ? 0 : (restart ? 1 : 0);
    if (!state.dynamic_vertex_input) {
        const size_t num_vertex_arrays = (std::min)(
            Maxwell::NumVertexArrays, static_cast<size_t>(device.GetMaxVertexInputBindings()));
        for (size_t index = 0; index < num_vertex_arrays; ++index) {
            const bool instanced = state.binding_divisors[index] != 0;
            vi.bindings[vi.num_bindings++] = {
                static_cast<u32>(index), vi.dynamic_stride ? 0u : u32{state.vertex_strides[index]},
                instanced ? u32{VK_VERTEX_INPUT_RATE_INSTANCE} : u32{VK_VERTEX_INPUT_RATE_VERTEX}};
            if (instanced) {
                vi.divisors[vi.num_divisors++] = {static_cast<u32>(index),
                                                  state.binding_divisors[index]};
            }
        }
        for (size_t index = 0; index < state.attributes.size(); ++index) {
            const auto& attribute = state.attributes[index];
            if (!attribute.enabled || !vs_info.loads.Generic(index)) {
                continue;
            }
            vi.attributes[vi.num_attributes++] = {
                static_cast<u32>(index), attribute.buffer,
                static_cast<u32>(
                    MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size())),
                attribute.offset};
        }
    }
    return vi;
}

GplPreRasterState MakeGplPreRasterState(const Device& device, const FixedPipelineState& state,
                                        std::span<const u32> spirv) {
    GplPreRasterState pr{};
    pr.spirv_hash = HashSpirv(spirv);
    pr.spirv_words = static_cast<u32>(spirv.size());
    pr.num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
    const FixedPipelineState::DynamicState dynamic = Dynamic(state);
    const bool eds1 = state.extended_dynamic_state != 0;
    const bool eds2 = state.extended_dynamic_state_2 != 0;
    const bool eds3 = state.extended_dynamic_state_3_enables != 0;
    const bool dyn_clamp = eds3 && device.SupportsDynamicState3DepthClampEnable();
    const bool dyn_line = eds3 && device.SupportsDynamicState3LineRasterizationMode();
    const bool dyn_conservative = eds3 && device.SupportsDynamicState3ConservativeRasterizationMode();
    const bool dyn_stipple = eds3 && device.SupportsDynamicState3LineStippleEnable();
    pr.dynamic_flags = (eds1 ? 1u : 0u) | (eds2 ? 2u : 0u) | (dyn_clamp ? 4u : 0u) |
                       (dyn_line ? 8u : 0u) | (dyn_conservative ? 16u : 0u) |
                       (dyn_stipple ? 32u : 0u);
    pr.depth_clamp = dyn_clamp ? 0 : (dynamic.depth_clamp_disabled == 0 ? 1 : 0);
    pr.rasterizer_discard = eds2 ? 0 : (dynamic.rasterize_enable == 0 ? 1 : 0);
    pr.polygon_mode = static_cast<u32>(
        MaxwellToVK::PolygonMode(FixedPipelineState::UnpackPolygonMode(state.polygon_mode)));
    if (!eds1) {
        pr.cull_mode = dynamic.cull_enable ? MaxwellToVK::CullFace(dynamic.CullFace())
                                           : VK_CULL_MODE_NONE;
        pr.front_face = MaxwellToVK::FrontFace(dynamic.FrontFace());
    }
    pr.depth_bias_enable = eds2 ? 0 : (dynamic.depth_bias_enable != 0 ? 1 : 0);
    if (device.IsExtLineRasterizationSupported()) {
        // Always chained (the monolithic path chains it for line topologies only; the library
        // does not know the topology, and the state only applies to lines).
        pr.chain_line = 1;
        const bool smooth = device.SupportsSmoothLines();
        const bool stippled = device.SupportsStippledRectangularLines();
        pr.line_mode = dyn_line ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT
                                : (state.smooth_lines != 0 && smooth
                                       ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                       : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT);
        pr.stippled_line =
            dyn_stipple ? 0 : ((dynamic.line_stipple_enable && stippled) ? 1 : 0);
        pr.stipple_factor = state.line_stipple_factor;
        pr.stipple_pattern = static_cast<u16>(state.line_stipple_pattern);
    }
    if (device.IsExtConservativeRasterizationSupported()) {
        pr.chain_conservative = 1;
        pr.conservative_mode = dyn_conservative
                                   ? VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT
                                   : (state.conservative_raster_enable != 0
                                          ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                          : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    }
    if (device.IsExtProvokingVertexSupported()) {
        pr.chain_provoking = 1;
        const bool last = state.provoking_vertex_last != 0;
        pr.provoking_mode =
            last ? (device.SupportsProvokingVertexLastMode() ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                                             : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT)
                 : (device.SupportsProvokingVertexFirstMode() ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                                              : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
    }
    if (device.IsExtDepthClipControlSupported()) {
        pr.chain_depth_clip = 1;
        pr.negative_one_to_one = state.ndc_minus_one_to_one != 0 ? 1 : 0;
    }
    if (device.IsNvViewportSwizzleSupported()) {
        pr.chain_swizzle = 1;
        for (size_t i = 0; i < pr.swizzles.size(); ++i) {
            pr.swizzles[i] = state.viewport_swizzles[i];
        }
    }
    return pr;
}

GplFragmentState MakeGplFragmentState(const Device& device, const FixedPipelineState& state,
                                      std::span<const u32> spirv, bool fragment_writes_color0) {
    GplFragmentState fs{};
    fs.spirv_hash = HashSpirv(spirv);
    fs.spirv_words = static_cast<u32>(spirv.size());
    fs.ms = MakeMultisample(device, state, fragment_writes_color0);
    if (state.extended_dynamic_state) {
        fs.dynamic_flags = 1;
        return fs;
    }
    const FixedPipelineState::DynamicState dynamic = state.dynamic_state;
    fs.depth_test = dynamic.depth_test_enable;
    fs.depth_write = dynamic.depth_write_enable;
    fs.depth_compare = dynamic.depth_test_enable ? MaxwellToVK::ComparisonOp(dynamic.DepthTestFunc())
                                                 : VK_COMPARE_OP_ALWAYS;
    fs.depth_bounds_test = dynamic.depth_bounds_enable && device.IsDepthBoundsSupported();
    fs.stencil_test = dynamic.stencil_enable;
    const auto face = [](const auto& f) {
        return std::array<u32, 4>{
            static_cast<u32>(MaxwellToVK::StencilOp(f.ActionStencilFail())),
            static_cast<u32>(MaxwellToVK::StencilOp(f.ActionDepthPass())),
            static_cast<u32>(MaxwellToVK::StencilOp(f.ActionDepthFail())),
            static_cast<u32>(MaxwellToVK::ComparisonOp(f.TestFunc()))};
    };
    fs.front = face(dynamic.front);
    fs.back = face(dynamic.back);
    return fs;
}

GplFragmentOutputState MakeGplFragmentOutputState(const Device& device,
                                                  const FixedPipelineState& state,
                                                  bool fragment_writes_color0,
                                                  u32 color_output_mask) {
    using VideoCore::Surface::PixelFormatFromDepthFormat;
    using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
    GplFragmentOutputState fo{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; ++index) {
        const auto format{static_cast<Tegra::RenderTargetFormat>(state.color_formats[index])};
        if (format == Tegra::RenderTargetFormat::NONE) {
            continue;
        }
        fo.color_formats[index] = static_cast<u32>(
            MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true,
                                       PixelFormatFromRenderTargetFormat(format))
                .format);
        fo.num_colors = static_cast<u32>(index + 1);
    }
    if (state.depth_enabled != 0) {
        const PixelFormat depth{PixelFormatFromDepthFormat(
            static_cast<Tegra::DepthFormat>(state.depth_format.Value()))};
        const u32 vk_format = static_cast<u32>(
            MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, depth).format);
        const auto type = VideoCore::Surface::GetFormatType(depth);
        if (type == VideoCore::Surface::SurfaceType::Depth ||
            type == VideoCore::Surface::SurfaceType::DepthStencil) {
            fo.depth_format = vk_format;
        }
        if (type == VideoCore::Surface::SurfaceType::Stencil ||
            type == VideoCore::Surface::SurfaceType::DepthStencil) {
            fo.stencil_format = vk_format;
        }
    }
    fo.ms = MakeMultisample(device, state, fragment_writes_color0);
    const FixedPipelineState::DynamicState dynamic = Dynamic(state);
    const bool dyn_logic_op = state.extended_dynamic_state_2_logic_op != 0;
    const bool dyn_blend = state.extended_dynamic_state_3_blend != 0;
    const bool dyn_write = !dyn_blend && state.color_write_enable_dynamic != 0;
    const bool dyn_logic_enable =
        state.extended_dynamic_state_3_enables != 0 && device.SupportsDynamicState3LogicOpEnable();
    fo.dynamic_flags = (dyn_logic_op ? 1u : 0u) | (dyn_blend ? 2u : 0u) | (dyn_write ? 4u : 0u) |
                       (dyn_logic_enable ? 8u : 0u);
    fo.logic_op_enable = dyn_logic_enable ? 0 : (dynamic.logic_op_enable != 0 ? 1 : 0);
    fo.logic_op = dyn_logic_op ? 0 : static_cast<u32>(dynamic.logic_op.Value());
    fo.num_attachments = fo.num_colors;
    for (size_t index = 0; index < fo.num_attachments; ++index) {
        if (dyn_blend) {
            continue;
        }
        static constexpr std::array mask_table{
            VK_COLOR_COMPONENT_R_BIT,
            VK_COLOR_COMPONENT_G_BIT,
            VK_COLOR_COMPONENT_B_BIT,
            VK_COLOR_COMPONENT_A_BIT,
        };
        const auto& blend{state.attachments[index]};
        const std::array mask{blend.Mask()};
        VkColorComponentFlags write_mask{};
        for (size_t i = 0; i < mask_table.size(); ++i) {
            write_mask |= mask[i] ? mask_table[i] : 0;
        }
        // Components the fragment stage doesn't write stay untouched in the guest; Vulkan would
        // write undefined values to them (AMD writes garbage).
        write_mask &= (color_output_mask >> (index * 4)) & 0xF;
        fo.attachments[index] = {
            blend.enable != 0 ? 1u : 0u,
            static_cast<u32>(MaxwellToVK::BlendFactor(blend.SourceRGBFactor())),
            static_cast<u32>(MaxwellToVK::BlendFactor(blend.DestRGBFactor())),
            static_cast<u32>(MaxwellToVK::BlendEquation(blend.EquationRGB())),
            static_cast<u32>(MaxwellToVK::BlendFactor(blend.SourceAlphaFactor())),
            static_cast<u32>(MaxwellToVK::BlendFactor(blend.DestAlphaFactor())),
            static_cast<u32>(MaxwellToVK::BlendEquation(blend.EquationAlpha())),
            write_mask,
        };
    }
    return fo;
}

vk::DescriptorSetLayout MakeGplStageSetLayout(const Device& device, const Shader::Info& info,
                                              VkShaderStageFlags stage) {
    DescriptorLayoutBuilder builder{device};
    builder.Add(info, stage);
    return builder.CreateDescriptorSetLayout(false, false);
}

vk::PipelineLayout MakeGplPipelineLayout(const Device& device, VkDescriptorSetLayout vertex_set,
                                         VkDescriptorSetLayout fragment_set) {
    using Shader::Backend::SPIRV::RenderAreaLayout;
    using Shader::Backend::SPIRV::RescalingLayout;
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = static_cast<u32>(sizeof(RescalingLayout)) +
                static_cast<u32>(sizeof(RenderAreaLayout)),
    };
    const std::array<VkDescriptorSetLayout, 2> sets{vertex_set, fragment_set};
    return device.GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT,
        .setLayoutCount = static_cast<u32>(sets.size()),
        .pSetLayouts = sets.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

template <typename State>
size_t GplLibraryCache::Map<State>::Hasher::operator()(const State& s) const noexcept {
    return Common::CityHash64(reinterpret_cast<const char*>(&s), sizeof(State));
}

template <typename State>
bool GplLibraryCache::Map<State>::Equal::operator()(const State& a, const State& b) const noexcept {
    return BytesEqual(a, b);
}

GplLibraryCache::GplLibraryCache(const Device& device_) : device{device_} {}

GplLibraryCache::~GplLibraryCache() = default;

size_t GplLibraryCache::NumLibraries() const {
    size_t n = 0;
    const auto count = [&n](auto& map) {
        std::scoped_lock lock{map.mutex};
        n += map.entries.size();
    };
    count(const_cast<Map<GplVertexInputState>&>(vertex_input));
    count(const_cast<Map<GplPreRasterState>&>(pre_raster));
    count(const_cast<Map<GplFragmentState>&>(fragment));
    count(const_cast<Map<GplFragmentOutputState>&>(fragment_output));
    return n;
}

template <typename State, typename CreateFunc>
GplLibraryCache::Result GplLibraryCache::Get(Map<State>& map, const State& state,
                                             CreateFunc&& create) {
    Entry* entry;
    {
        std::scoped_lock lock{map.mutex};
        auto& slot = map.entries[state];
        if (!slot) {
            slot = std::make_unique<Entry>();
        }
        entry = slot.get();
    }
    std::scoped_lock lock{entry->mutex};
    Result result;
    if (!entry->done) {
        const auto begin = std::chrono::steady_clock::now();
        create(*entry, result.cache_hit);
        result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              begin)
                        .count();
        result.created = true;
        entry->done = true;
    }
    result.pipeline = *entry->pipeline;
    return result;
}

GplLibraryCache::Result GplLibraryCache::VertexInput(const GplVertexInputState& state) {
    return Get(vertex_input, state, [&](Entry& entry, bool& hit) {
        entry.pipeline = CreateVertexInput(state, hit);
    });
}

GplLibraryCache::Result GplLibraryCache::FragmentOutput(const GplFragmentOutputState& state) {
    return Get(fragment_output, state, [&](Entry& entry, bool& hit) {
        entry.pipeline = CreateFragmentOutput(state, hit);
    });
}

GplLibraryCache::Result GplLibraryCache::PreRaster(const GplPreRasterState& state,
                                                   const Shader::Info& info, VkShaderModule module,
                                                   std::span<const u32> spirv) {
    return Get(pre_raster, state, [&](Entry& entry, bool& hit) {
        entry.set_layout = MakeGplStageSetLayout(device, info, VK_SHADER_STAGE_VERTEX_BIT);
        entry.layout = MakeGplPipelineLayout(device, *entry.set_layout, VK_NULL_HANDLE);
        vk::ShaderModule own;
        if (!module) {
            own = BuildShader(device, spirv);
            module = *own;
        }
        entry.pipeline = CreatePreRaster(state, *entry.layout, module, hit);
    });
}

GplLibraryCache::Result GplLibraryCache::Fragment(const GplFragmentState& state,
                                                  const Shader::Info& info, VkShaderModule module,
                                                  std::span<const u32> spirv) {
    return Get(fragment, state, [&](Entry& entry, bool& hit) {
        entry.set_layout = MakeGplStageSetLayout(device, info, VK_SHADER_STAGE_FRAGMENT_BIT);
        entry.layout = MakeGplPipelineLayout(device, VK_NULL_HANDLE, *entry.set_layout);
        vk::ShaderModule own;
        if (!module) {
            own = BuildShader(device, spirv);
            module = *own;
        }
        entry.pipeline = CreateFragment(state, *entry.layout, module, hit);
    });
}

GplLibraryCache::Result GplLibraryCache::CreateDetached(const GplPreRasterState* pre_raster_state,
                                                        const GplFragmentState* fragment_state,
                                                        const Shader::Info& info,
                                                        std::span<const u32> spirv) {
    Result result;
    // EDEN_NX2_GPL_TRACE=<dir>: each stage's SPIR-V is written there while its library is created
    // (and removed after), so a driver crash leaves the stages it was compiling behind.
    static const char* const trace_dir = std::getenv("EDEN_NX2_GPL_TRACE");
    std::string trace_path;
    if (trace_dir != nullptr) {
        trace_path = fmt::format("{}/{}-{:016x}.spv", trace_dir, pre_raster_state ? "vs" : "ps",
                                 pre_raster_state ? pre_raster_state->spirv_hash
                                                  : fragment_state->spirv_hash);
        std::ofstream{trace_path, std::ios::binary}.write(
            reinterpret_cast<const char*>(spirv.data()),
            static_cast<std::streamsize>(spirv.size_bytes()));
    }
    SCOPE_EXIT {
        if (!trace_path.empty()) {
            std::filesystem::remove(trace_path);
        }
    };
    const auto begin = std::chrono::steady_clock::now();
    const vk::ShaderModule module = BuildShader(device, spirv);
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    if (pre_raster_state) {
        set_layout = MakeGplStageSetLayout(device, info, VK_SHADER_STAGE_VERTEX_BIT);
        layout = MakeGplPipelineLayout(device, *set_layout, VK_NULL_HANDLE);
        pipeline = CreatePreRaster(*pre_raster_state, *layout, *module, result.cache_hit);
    } else {
        set_layout = MakeGplStageSetLayout(device, info, VK_SHADER_STAGE_FRAGMENT_BIT);
        layout = MakeGplPipelineLayout(device, VK_NULL_HANDLE, *set_layout);
        pipeline = CreateFragment(*fragment_state, *layout, *module, result.cache_hit);
    }
    result.ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    result.created = true;
    return result;
}

vk::Pipeline GplLibraryCache::Create(const VkGraphicsPipelineCreateInfo& ci, bool& hit) {
    VkPipelineCreationFeedback feedback{};
    const VkPipelineCreationFeedbackCreateInfo feedback_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
        .pNext = ci.pNext,
        .pPipelineCreationFeedback = &feedback,
        .pipelineStageCreationFeedbackCount = 0,
        .pPipelineStageCreationFeedbacks = nullptr,
    };
    VkGraphicsPipelineCreateInfo with_feedback = ci;
    with_feedback.pNext = &feedback_ci;
    vk::Pipeline pipeline = device.GetLogical().CreateGraphicsPipeline(with_feedback, pipeline_cache);
    hit = (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0 &&
          (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
    return pipeline;
}

vk::Pipeline GplLibraryCache::CreateVertexInput(const GplVertexInputState& state, bool& hit) {
    static_vector<VkVertexInputBindingDescription, 32> bindings;
    static_vector<VkVertexInputBindingDivisorDescriptionEXT, 32> divisors;
    static_vector<VkVertexInputAttributeDescription, 32> attributes;
    for (u32 i = 0; i < state.num_bindings; ++i) {
        bindings.push_back({state.bindings[i][0], state.bindings[i][1],
                            static_cast<VkVertexInputRate>(state.bindings[i][2])});
    }
    for (u32 i = 0; i < state.num_divisors; ++i) {
        divisors.push_back({state.divisors[i][0], state.divisors[i][1]});
    }
    for (u32 i = 0; i < state.num_attributes; ++i) {
        attributes.push_back({state.attributes[i][0], state.attributes[i][1],
                              static_cast<VkFormat>(state.attributes[i][2]),
                              state.attributes[i][3]});
    }
    const VkPipelineVertexInputDivisorStateCreateInfoEXT divisor_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(divisors.size()),
        .pVertexBindingDivisors = divisors.data(),
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = divisors.empty() ? nullptr : &divisor_ci,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(bindings.size()),
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = static_cast<VkPrimitiveTopology>(state.topology),
        .primitiveRestartEnable = state.primitive_restart,
    };
    static_vector<VkDynamicState, 34> dynamic_states;
    if (state.dynamic_vertex_input) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }
    if (state.dynamic_stride) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT);
    }
    if (state.dynamic_topology) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT);
    }
    if (state.dynamic_restart) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT);
    }
    const VkPipelineDynamicStateCreateInfo dynamic_ci{DynamicInfo(dynamic_states)};
    const VkGraphicsPipelineLibraryCreateInfoEXT library_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
    };
    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &library_ci;
    ci.flags = LIBRARY_FLAGS;
    ci.pVertexInputState = state.dynamic_vertex_input ? nullptr : &vertex_input_ci;
    ci.pInputAssemblyState = &input_assembly_ci;
    ci.pDynamicState = &dynamic_ci;
    return Create(ci, hit);
}

vk::Pipeline GplLibraryCache::CreatePreRaster(const GplPreRasterState& state,
                                              VkPipelineLayout layout, VkShaderModule module,
                                              bool& hit) {
    std::array<VkViewportSwizzleNV, Maxwell::NumViewports> swizzles{};
    for (size_t i = 0; i < swizzles.size(); ++i) {
        swizzles[i] = UnpackViewportSwizzle(static_cast<u16>(state.swizzles[i]));
    }
    VkPipelineViewportSwizzleStateCreateInfoNV swizzle_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = Maxwell::NumViewports,
        .pViewportSwizzles = swizzles.data(),
    };
    VkPipelineViewportDepthClipControlCreateInfoEXT ndc_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = state.negative_one_to_one,
    };
    VkPipelineViewportStateCreateInfo viewport_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = state.num_viewports,
        .pViewports = nullptr,
        .scissorCount = state.num_viewports,
        .pScissors = nullptr,
    };
    if (state.chain_swizzle) {
        swizzle_ci.pNext = std::exchange(viewport_ci.pNext, &swizzle_ci);
    }
    if (state.chain_depth_clip) {
        ndc_info.pNext = std::exchange(viewport_ci.pNext, &ndc_info);
    }
    VkPipelineRasterizationStateCreateInfo rasterization_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = state.depth_clamp,
        .rasterizerDiscardEnable = state.rasterizer_discard,
        .polygonMode = static_cast<VkPolygonMode>(state.polygon_mode),
        .cullMode = state.cull_mode,
        .frontFace = static_cast<VkFrontFace>(state.front_face),
        .depthBiasEnable = state.depth_bias_enable,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
    VkPipelineRasterizationLineStateCreateInfoEXT line_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .lineRasterizationMode = static_cast<VkLineRasterizationModeEXT>(state.line_mode),
        .stippledLineEnable = state.stippled_line,
        .lineStippleFactor = state.stipple_factor,
        .lineStipplePattern = static_cast<uint16_t>(state.stipple_pattern),
    };
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .conservativeRasterizationMode =
            static_cast<VkConservativeRasterizationModeEXT>(state.conservative_mode),
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = static_cast<VkProvokingVertexModeEXT>(state.provoking_mode),
    };
    if (state.chain_line) {
        line_state.pNext = std::exchange(rasterization_ci.pNext, &line_state);
    }
    if (state.chain_conservative) {
        conservative_raster.pNext = std::exchange(rasterization_ci.pNext, &conservative_raster);
    }
    if (state.chain_provoking) {
        provoking_vertex.pNext = std::exchange(rasterization_ci.pNext, &provoking_vertex);
    }
    static_vector<VkDynamicState, 34> dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    if (state.dynamic_flags & 1) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_CULL_MODE_EXT);
        dynamic_states.push_back(VK_DYNAMIC_STATE_FRONT_FACE_EXT);
    }
    if (state.dynamic_flags & 2) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT);
        dynamic_states.push_back(VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT);
    }
    if (state.dynamic_flags & 4) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
    }
    if (state.dynamic_flags & 8) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
    }
    if (state.dynamic_flags & 16) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT);
    }
    if (state.dynamic_flags & 32) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
    }
    const VkPipelineDynamicStateCreateInfo dynamic_ci{DynamicInfo(dynamic_states)};
    const VkPipelineRenderingCreateInfo rendering_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 0,
        .pColorAttachmentFormats = nullptr,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };
    const VkGraphicsPipelineLibraryCreateInfoEXT library_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = &rendering_ci,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
    };
    const VkPipelineShaderStageCreateInfo stage_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &library_ci;
    ci.flags = LIBRARY_FLAGS;
    ci.stageCount = 1;
    ci.pStages = &stage_ci;
    ci.pViewportState = &viewport_ci;
    ci.pRasterizationState = &rasterization_ci;
    ci.pDynamicState = &dynamic_ci;
    ci.layout = layout;
    return Create(ci, hit);
}

vk::Pipeline GplLibraryCache::CreateFragment(const GplFragmentState& state,
                                             VkPipelineLayout layout, VkShaderModule module,
                                             bool& hit) {
    const VkPipelineMultisampleStateCreateInfo multisample_ci{MultisampleInfo(state.ms)};
    const auto face = [](const std::array<u32, 4>& f) {
        return VkStencilOpState{
            .failOp = static_cast<VkStencilOp>(f[0]),
            .passOp = static_cast<VkStencilOp>(f[1]),
            .depthFailOp = static_cast<VkStencilOp>(f[2]),
            .compareOp = static_cast<VkCompareOp>(f[3]),
            .compareMask = 0,
            .writeMask = 0,
            .reference = 0,
        };
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = state.depth_test,
        .depthWriteEnable = state.depth_write,
        .depthCompareOp = static_cast<VkCompareOp>(state.depth_compare),
        .depthBoundsTestEnable = state.depth_bounds_test,
        .stencilTestEnable = state.stencil_test,
        .front = face(state.front),
        .back = face(state.back),
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 0.0f,
    };
    static_vector<VkDynamicState, 34> dynamic_states{
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };
    if (state.dynamic_flags & 1) {
        static constexpr std::array extended{
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,  VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,   VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT, VK_DYNAMIC_STATE_STENCIL_OP_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());
    }
    AddMultisampleDynamic(state.ms, dynamic_states);
    const VkPipelineDynamicStateCreateInfo dynamic_ci{DynamicInfo(dynamic_states)};
    const VkPipelineRenderingCreateInfo rendering_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 0,
        .pColorAttachmentFormats = nullptr,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };
    const VkGraphicsPipelineLibraryCreateInfoEXT library_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = &rendering_ci,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
    };
    const VkPipelineShaderStageCreateInfo stage_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &library_ci;
    ci.flags = LIBRARY_FLAGS;
    ci.stageCount = 1;
    ci.pStages = &stage_ci;
    ci.pMultisampleState = &multisample_ci;
    ci.pDepthStencilState = &depth_stencil_ci;
    ci.pDynamicState = &dynamic_ci;
    ci.layout = layout;
    return Create(ci, hit);
}

vk::Pipeline GplLibraryCache::CreateFragmentOutput(const GplFragmentOutputState& state,
                                                   bool& hit) {
    const VkPipelineMultisampleStateCreateInfo multisample_ci{MultisampleInfo(state.ms)};
    std::array<VkFormat, 8> formats{};
    for (u32 i = 0; i < state.num_colors; ++i) {
        formats[i] = static_cast<VkFormat>(state.color_formats[i]);
    }
    static_vector<VkPipelineColorBlendAttachmentState, 8> attachments;
    for (u32 i = 0; i < state.num_attachments; ++i) {
        const auto& a = state.attachments[i];
        attachments.push_back({
            .blendEnable = a[0],
            .srcColorBlendFactor = static_cast<VkBlendFactor>(a[1]),
            .dstColorBlendFactor = static_cast<VkBlendFactor>(a[2]),
            .colorBlendOp = static_cast<VkBlendOp>(a[3]),
            .srcAlphaBlendFactor = static_cast<VkBlendFactor>(a[4]),
            .dstAlphaBlendFactor = static_cast<VkBlendFactor>(a[5]),
            .alphaBlendOp = static_cast<VkBlendOp>(a[6]),
            .colorWriteMask = a[7],
        });
    }
    const VkPipelineColorBlendStateCreateInfo color_blend_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = state.logic_op_enable,
        .logicOp = static_cast<VkLogicOp>(state.logic_op),
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .blendConstants = {},
    };
    static_vector<VkDynamicState, 34> dynamic_states{VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    if (state.dynamic_flags & 1) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
    }
    if (state.dynamic_flags & 2) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    }
    if (state.dynamic_flags & 4) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
    }
    if (state.dynamic_flags & 8) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT);
    }
    AddMultisampleDynamic(state.ms, dynamic_states);
    const VkPipelineDynamicStateCreateInfo dynamic_ci{DynamicInfo(dynamic_states)};
    const VkPipelineRenderingCreateInfo rendering_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = state.num_colors,
        .pColorAttachmentFormats = formats.data(),
        .depthAttachmentFormat = static_cast<VkFormat>(state.depth_format),
        .stencilAttachmentFormat = static_cast<VkFormat>(state.stencil_format),
    };
    const VkGraphicsPipelineLibraryCreateInfoEXT library_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = &rendering_ci,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
    };
    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &library_ci;
    ci.flags = LIBRARY_FLAGS;
    ci.pMultisampleState = &multisample_ci;
    ci.pColorBlendState = &color_blend_ci;
    ci.pDynamicState = &dynamic_ci;
    return Create(ci, hit);
}

std::optional<u32> GplNullAttributeBinding(const Device& device,
                                           const Tegra::Engines::Maxwell3D::Regs& regs,
                                           u32 max_attributes, u32 max_bindings) {
    if (!device.IsGplUntypedAttributes()) {
        return std::nullopt;
    }
    u32 used = 0;
    for (u32 index = 0; index < max_attributes; ++index) {
        const auto attribute{regs.vertex_attrib_format[index]};
        if (!attribute.constant && attribute.buffer < max_bindings) {
            used |= 1U << attribute.buffer;
        }
    }
    for (u32 binding = max_bindings; binding-- > 0;) {
        if ((used & (1U << binding)) == 0 && regs.vertex_streams[binding].enable == 0) {
            return binding;
        }
    }
    return std::nullopt;
}

} // namespace Vulkan
