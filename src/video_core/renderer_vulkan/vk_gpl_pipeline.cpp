// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// nx2 shaderwarm (docs/shader-precompile.md): the graphics-pipeline-library side of
// GraphicsPipeline (EDEN_NX2_GPL=1): per-stage descriptor sets, getting the four libraries, the
// fast link and the background optimized link. vk_graphics_pipeline.cpp calls these from hooks
// marked "nx2 shaderwarm".

#include <array>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nx2_pipeline_stats.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_gpl.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

void GraphicsPipeline::Nx2InitLibraryPipeline(const GplParams& gpl, DescriptorPool& descriptor_pool) {
    is_gpl = true;
    gpl_libraries = gpl.libraries;
    gpl_lto_worker = gpl.lto_worker;
    gpl_vi = MakeGplVertexInputState(device, key.state, stage_infos[0]);
    gpl_pr = MakeGplPreRasterState(device, key.state, gpl.vertex_spirv);
    gpl_fs = MakeGplFragmentState(device, key.state, gpl.fragment_spirv,
                                  fragment_has_color0_output);
    gpl_fo = MakeGplFragmentOutputState(device, key.state, fragment_has_color0_output,
                                        color_output_mask);
    lto_control = std::make_shared<LtoControl>();
    SetupGplDescriptors(descriptor_pool);
}

void GraphicsPipeline::Nx2BindLibraryDescriptors(vk::CommandBuffer cmdbuf,
                                                 const void* descriptor_data) {
    const vk::Device& dev{device.GetLogical()};
    for (u32 set = 0; set < 2; ++set) {
        if (!gpl_set_layouts[set]) {
            continue;
        }
        const VkDescriptorSet descriptor_set{gpl_allocators[set].Commit()};
        dev.UpdateDescriptorSet(descriptor_set, *gpl_templates[set], descriptor_data);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, set,
                                  descriptor_set, nullptr);
    }
}

GraphicsPipeline::~GraphicsPipeline() {
    if (lto_control) {
        // Waits for a running optimized link of this pipeline; a queued one does nothing.
        std::scoped_lock lock{lto_control->mutex};
        lto_control->alive = false;
    }
}

void GraphicsPipeline::SetupGplDescriptors(DescriptorPool& descriptor_pool) {
    DescriptorLayoutBuilder vertex_builder{device};
    vertex_builder.Add(stage_infos[0], VK_SHADER_STAGE_VERTEX_BIT);
    DescriptorLayoutBuilder fragment_builder{device};
    // One payload for both stages (the vertex stage's entries first), a set per stage.
    fragment_builder.SetFirstEntry(NumDescriptorEntries(stage_infos[0]));
    fragment_builder.Add(stage_infos[4], VK_SHADER_STAGE_FRAGMENT_BIT);
    gpl_set_layouts[0] = vertex_builder.CreateDescriptorSetLayout(false, false);
    gpl_set_layouts[1] = fragment_builder.CreateDescriptorSetLayout(false, false);
    pipeline_layout = MakeGplPipelineLayout(device, *gpl_set_layouts[0], *gpl_set_layouts[1]);
    const std::array<const DescriptorLayoutBuilder*, 2> builders{&vertex_builder,
                                                                 &fragment_builder};
    for (size_t set = 0; set < 2; ++set) {
        if (!gpl_set_layouts[set]) {
            continue;
        }
        gpl_templates[set] =
            builders[set]->CreateTemplate(*gpl_set_layouts[set], *pipeline_layout, false);
        gpl_allocators[set] = descriptor_pool.Allocator(device, scheduler, *gpl_set_layouts[set],
                                                        stage_infos[set == 0 ? 0 : 4]);
    }
}

void GraphicsPipeline::MakeGplPipeline() {
    const auto begin = Nx2Stats::Clock::now();
    const GplLibraryCache::Result vi = gpl_libraries->VertexInput(gpl_vi);
    const GplLibraryCache::Result pr =
        gpl_libraries->PreRaster(gpl_pr, stage_infos[0], *spv_modules[0], {});
    const GplLibraryCache::Result fs =
        gpl_libraries->Fragment(gpl_fs, stage_infos[4], *spv_modules[4], {});
    const GplLibraryCache::Result fo = gpl_libraries->FragmentOutput(gpl_fo);
    const auto libraries_end = Nx2Stats::Clock::now();
    const std::array<VkPipeline, 4> libraries{vi.pipeline, pr.pipeline, fs.pipeline, fo.pipeline};
    const VkPipelineLibraryCreateInfoKHR library_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .libraryCount = static_cast<u32>(libraries.size()),
        .pLibraries = libraries.data(),
    };
    VkGraphicsPipelineCreateInfo link_ci{};
    link_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    link_ci.pNext = &library_ci;
    link_ci.layout = *pipeline_layout;
    pipeline = device.GetLogical().CreateGraphicsPipeline(link_ci, gpl_libraries->PipelineCache());
    active_pipeline.store(*pipeline, std::memory_order_release);
    if (Nx2Stats::Enabled()) {
        const auto end = Nx2Stats::Clock::now();
        Nx2Stats::Add(Nx2Stats::Phase::GplLibraries, libraries_end - begin);
        Nx2Stats::Add(Nx2Stats::Phase::GplLink, end - libraries_end);
        Nx2Stats::CountGplLibrary(pr);
        Nx2Stats::CountGplLibrary(fs);
        Nx2Stats::Get().gpl_pipelines.fetch_add(1, std::memory_order_relaxed);
        if (Nx2Stats::Level() >= 2) {
            // The state that goes into the stage SPIR-V / driver key, to tell why a stage was not
            // a driver cache hit: attribute classes of the attributes the vertex stage loads
            // (f float, i sint, u uint, - disabled), y negate, early z, alpha test, dual source,
            // samples, alpha to coverage, provoking vertex last.
            std::string attributes;
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                if (!stage_infos[0].loads.Generic(index)) {
                    continue;
                }
                const u32 type = key.state.dynamic_vertex_input
                                     ? key.state.DynamicAttributeType(index)
                                     : (key.state.attributes[index].enabled ? 1u : 0u);
                attributes += fmt::format("{}{}", index, "-fiu"[type & 3]);
            }
            LOG_INFO(Render_Vulkan,
                     "NX2STATS gpl {:016x}: vs {:016x} {} {:.2f} ms{} ps {:016x} {} {:.2f} ms{} vi {} "
                     "fo {} link {:.3f} ms | attrs {} yneg {} earlyz {} alpha {} dual {} msaa {} "
                     "a2c {} pvlast {}",
                     key.Hash(), key.unique_hashes[1], pr.created ? "new" : "reused", pr.ms,
                     pr.cache_hit ? " (hit)" : "", key.unique_hashes[5],
                     fs.created ? "new" : "reused", fs.ms, fs.cache_hit ? " (hit)" : "",
                     vi.created ? "new" : "reused", fo.created ? "new" : "reused",
                     Nx2Stats::Ms(end - libraries_end), attributes, key.state.y_negate.Value(),
                     key.state.early_z.Value(), key.state.alpha_test_func.Value(),
                     key.state.attachment0_dual_source_blend.Value(), key.state.msaa_mode.Value(),
                     key.state.alpha_to_coverage_enabled.Value(),
                     key.state.provoking_vertex_last.Value());
        }
    }
    if (!gpl_lto_worker) {
        return;
    }
    gpl_lto_worker->QueueWork([this, control = lto_control, libraries] {
        std::scoped_lock lock{control->mutex};
        if (!control->alive) {
            return;
        }
        const auto lto_begin = Nx2Stats::Clock::now();
        const VkPipelineLibraryCreateInfoKHR lto_library_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
            .pNext = nullptr,
            .libraryCount = static_cast<u32>(libraries.size()),
            .pLibraries = libraries.data(),
        };
        VkGraphicsPipelineCreateInfo lto_ci{};
        lto_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        lto_ci.pNext = &lto_library_ci;
        lto_ci.flags = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
        lto_ci.layout = *pipeline_layout;
        try {
            lto_pipeline =
                device.GetLogical().CreateGraphicsPipeline(lto_ci, gpl_libraries->PipelineCache());
        } catch (const vk::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "Optimized link of library pipeline failed: {}",
                      exception.what());
            return;
        }
        active_pipeline.store(*lto_pipeline, std::memory_order_release);
        lto_pipeline_done.store(true, std::memory_order_relaxed);
        if (Nx2Stats::Enabled()) {
            Nx2Stats::Add(Nx2Stats::Phase::GplLto, Nx2Stats::Clock::now() - lto_begin);
        }
    });
}

} // namespace Vulkan
