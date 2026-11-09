// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

// Per-stage pipelines from VK_EXT_graphics_pipeline_library (EDEN_NX2_GPL=1, native VS+PS
// pipelines). A pipeline is linked from four libraries:
//   vertex input (VI) + pre-rasterization (the vertex shader, PR) + fragment shader (FS) +
//   fragment output (FO).
// PR and FS hold the driver compile of one stage each. Their SPIR-V depends on that stage only
// (own descriptor set, own binding numbers, fragment inputs not pruned by the vertex stage), and
// they are made without a render pass (the 3D pipelines are drawn with dynamic rendering), so the
// driver's cache key of a stage library does not depend on render targets, blending or vertex
// formats: a library created by the precompile from the game files alone is a cache hit when a
// pipeline with that stage is first drawn. VI and FO hold no shader code and are cheap.
// At draw time the four are fast-linked (no compile); an optimized (link-time optimization) link
// runs on a background worker and replaces it when done.
namespace Vulkan {

class Device;

/// Descriptor sets of a library pipeline: the vertex stage's resources are set 0, the fragment
/// stage's set 1 (VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT).
inline constexpr u32 GPL_VERTEX_SET = 0;
inline constexpr u32 GPL_FRAGMENT_SET = 1;
/// Rescaling bits: the fragment stage's textures/images start at bit 0, the vertex stage's at these
/// (so each stage's SPIR-V is independent of the other's resource counts).
inline constexpr u32 GPL_VERTEX_TEXTURE_SCALING_BASE = 96;
inline constexpr u32 GPL_VERTEX_IMAGE_SCALING_BASE = 48;

/// The create-info content of each library, canonical (state that is dynamic is zero) and
/// compared as bytes.
struct GplVertexInputState {
    u32 dynamic_vertex_input;
    u32 dynamic_stride;
    u32 dynamic_topology;
    u32 dynamic_restart;
    u32 topology;
    u32 primitive_restart;
    u32 num_bindings;
    u32 num_attributes;
    u32 num_divisors;
    std::array<std::array<u32, 3>, 32> bindings;   ///< binding, stride, input rate
    std::array<std::array<u32, 2>, 32> divisors;   ///< binding, divisor
    std::array<std::array<u32, 4>, 32> attributes; ///< location, binding, format, offset
};

struct GplPreRasterState {
    u64 spirv_hash;
    u32 spirv_words;
    u32 num_viewports;
    u32 depth_clamp;
    u32 rasterizer_discard;
    u32 polygon_mode;
    u32 cull_mode;
    u32 front_face;
    u32 depth_bias_enable;
    u32 chain_line;
    u32 line_mode;
    u32 stippled_line;
    u32 stipple_factor;
    u32 stipple_pattern;
    u32 chain_conservative;
    u32 conservative_mode;
    u32 chain_provoking;
    u32 provoking_mode;
    u32 chain_depth_clip;
    u32 negative_one_to_one;
    u32 chain_swizzle;
    std::array<u32, 16> swizzles;
    u32 dynamic_flags; ///< bit 0 EDS1, 1 EDS2, 2-5 EDS3 depth clamp / line mode / conservative / stipple
    u32 pad;
};

struct GplMultisampleState {
    u32 samples;
    u32 sample_shading;
    u32 min_sample_shading_bits;
    u32 alpha_to_coverage;
    u32 alpha_to_one;
    u32 dynamic_flags; ///< bit 0 alpha to coverage, 1 alpha to one (EDS3)
};

struct GplFragmentState {
    u64 spirv_hash;
    u32 spirv_words;
    GplMultisampleState ms;
    u32 depth_test;
    u32 depth_write;
    u32 depth_compare;
    u32 depth_bounds_test;
    u32 stencil_test;
    std::array<u32, 4> front; ///< fail, pass, depth fail, compare
    std::array<u32, 4> back;
    u32 depth_bounds_min;
    u32 depth_bounds_max;
    u32 dynamic_flags; ///< bit 0 EDS1
    u32 pad;
};

struct GplFragmentOutputState {
    u32 num_colors;
    std::array<u32, 8> color_formats;
    u32 depth_format;
    u32 stencil_format;
    GplMultisampleState ms;
    u32 logic_op_enable;
    u32 logic_op;
    u32 num_attachments;
    std::array<std::array<u32, 8>, 8> attachments; ///< VkPipelineColorBlendAttachmentState
    u32 dynamic_flags; ///< bit 0 EDS2 logic op, 1 EDS3 blend, 2 color write enable, 3 logic op enable
};

static_assert(std::has_unique_object_representations_v<GplVertexInputState>);
static_assert(std::has_unique_object_representations_v<GplPreRasterState>);
static_assert(std::has_unique_object_representations_v<GplFragmentState>);
static_assert(std::has_unique_object_representations_v<GplFragmentOutputState>);

/// The state of each library of the pipeline the key describes. vs_info decides which vertex
/// attributes are fetched; fragment_writes_color0 gates alpha to coverage/one (as for monolithic
/// pipelines).
GplVertexInputState MakeGplVertexInputState(const Device& device, const FixedPipelineState& state,
                                            const Shader::Info& vs_info);
GplPreRasterState MakeGplPreRasterState(const Device& device, const FixedPipelineState& state,
                                        std::span<const u32> spirv);
GplFragmentState MakeGplFragmentState(const Device& device, const FixedPipelineState& state,
                                      std::span<const u32> spirv, bool fragment_writes_color0);
/// `color_output_mask`: the fragment stage's written components, 4 bits per render target (the
/// static write masks leave the others alone, as the monolithic path does).
GplFragmentOutputState MakeGplFragmentOutputState(const Device& device,
                                                  const FixedPipelineState& state,
                                                  bool fragment_writes_color0,
                                                  u32 color_output_mask = ~0U);

/// Descriptor set layout of one stage's resources (null when it has none), binding numbers from 0
/// in DescriptorLayoutBuilder order, as the stage's library SPIR-V declares them.
vk::DescriptorSetLayout MakeGplStageSetLayout(const Device& device, const Shader::Info& info,
                                              VkShaderStageFlags stage);
/// Pipeline layout with independent sets: {vertex_set, fragment_set} (either may be null).
vk::PipelineLayout MakeGplPipelineLayout(const Device& device, VkDescriptorSetLayout vertex_set,
                                         VkDescriptorSetLayout fragment_set);

/// Libraries shared by every pipeline of the session, created on first use (the driver's cache
/// makes a precompiled one cheap). Thread safe.
class GplLibraryCache {
public:
    struct Result {
        VkPipeline pipeline{};
        bool created{};  ///< made by this call (else it existed in this session)
        double ms{};     ///< creation time when created
        bool cache_hit{}; ///< creation feedback: found in the application pipeline cache
    };

    explicit GplLibraryCache(const Device& device);
    ~GplLibraryCache();

    /// The pipeline cache libraries are created with (null: the driver's own cache).
    void SetPipelineCache(VkPipelineCache cache) {
        pipeline_cache = cache;
    }
    [[nodiscard]] VkPipelineCache PipelineCache() const noexcept {
        return pipeline_cache;
    }

    Result VertexInput(const GplVertexInputState& state);
    /// module may be null: then one is made from spirv (when the library is new).
    Result PreRaster(const GplPreRasterState& state, const Shader::Info& info,
                     VkShaderModule module, std::span<const u32> spirv);
    Result Fragment(const GplFragmentState& state, const Shader::Info& info, VkShaderModule module,
                    std::span<const u32> spirv);
    Result FragmentOutput(const GplFragmentOutputState& state);

    /// Creates a stage library without keeping it (precompile: fills the driver's cache only).
    Result CreateDetached(const GplPreRasterState* pre_raster, const GplFragmentState* fragment,
                          const Shader::Info& info, std::span<const u32> spirv);

    [[nodiscard]] size_t NumLibraries() const;

private:
    struct Entry {
        std::mutex mutex;
        bool done{};
        vk::DescriptorSetLayout set_layout;
        vk::PipelineLayout layout;
        vk::Pipeline pipeline;
    };
    template <typename State>
    struct Map {
        struct Hasher {
            size_t operator()(const State& s) const noexcept;
        };
        struct Equal {
            bool operator()(const State& a, const State& b) const noexcept;
        };
        std::mutex mutex;
        std::unordered_map<State, std::unique_ptr<Entry>, Hasher, Equal> entries;
    };

    template <typename State, typename Create>
    Result Get(Map<State>& map, const State& state, Create&& create);

    vk::Pipeline CreateVertexInput(const GplVertexInputState& state, bool& hit);
    vk::Pipeline CreatePreRaster(const GplPreRasterState& state, VkPipelineLayout layout,
                                 VkShaderModule module, bool& hit);
    vk::Pipeline CreateFragment(const GplFragmentState& state, VkPipelineLayout layout,
                                VkShaderModule module, bool& hit);
    vk::Pipeline CreateFragmentOutput(const GplFragmentOutputState& state, bool& hit);
    vk::Pipeline Create(const VkGraphicsPipelineCreateInfo& ci, bool& hit);

    const Device& device;
    VkPipelineCache pipeline_cache{};
    Map<GplVertexInputState> vertex_input;
    Map<GplPreRasterState> pre_raster;
    Map<GplFragmentState> fragment;
    Map<GplFragmentOutputState> fragment_output;
};

/// Library vertex stages declare every attribute (Device::IsGplUntypedAttributes), constant ones
/// too: those read a null vertex buffer (robustness2: zeros, (0,0,1) filled per format) as
/// R32_SFLOAT = (0,0,0,1.0), the constant the typed path compiles in. The binding is the highest
/// one no attribute uses and whose stream is disabled (the buffer cache binds only enabled
/// streams). nullopt: not needed (feature off) or no free binding.
[[nodiscard]] std::optional<u32> GplNullAttributeBinding(const Device& device,
                                                         const Tegra::Engines::Maxwell3D::Regs& regs,
                                                         u32 max_attributes, u32 max_bindings);

} // namespace Vulkan
