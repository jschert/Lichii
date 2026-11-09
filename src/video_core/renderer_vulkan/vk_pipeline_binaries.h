// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <memory>
#include <string_view>

#include "common/common_types.h"
#include "video_core/shader_cache_v2/driver_blobs.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/// Shader cache v2's driver layer on VK_KHR_pipeline_binary, which only v2 enables, and only on
/// drivers that don't prefer their own cache (Device). A pipeline is built from the binaries stored
/// for its v2 key when the driver's key of its create info is still theirs; otherwise, or when the
/// driver refuses them, it is compiled and the driver's binaries of it replace them. Elsewhere the
/// VkPipelineCache, or the driver's own disk cache, stays as it was.
class PipelineBinaries {
public:
    /// The store beside the title's v2 log in `directory`: null unless the device enabled
    /// VK_KHR_pipeline_binary, or when the store can't start.
    [[nodiscard]] static std::unique_ptr<PipelineBinaries> Open(
        const Device& device, const std::filesystem::path& directory);

    explicit PipelineBinaries(const Device& device, std::filesystem::path file,
                              const VideoCommon::ShaderCacheV2::DeviceIdentity& identity);

    /// As vk::Device::CreateGraphicsPipeline, for the graphics pipeline with this v2 key
    [[nodiscard]] vk::Pipeline CreateGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci,
                                                      VkPipelineCache cache, u64 key);

    /// As vk::Device::CreateComputePipeline, for the compute pipeline with this v2 key
    [[nodiscard]] vk::Pipeline CreateComputePipeline(const VkComputePipelineCreateInfo& ci,
                                                     VkPipelineCache cache, u64 key);

    [[nodiscard]] VideoCommon::ShaderCacheV2::DriverBlobCounts Counts() const {
        return store.Counts();
    }

    /// One log line with the counts so far; `when` names the point of the session
    void LogCounts(std::string_view when) const;

private:
    template <typename CreateInfo>
    vk::Pipeline Create(const CreateInfo& ci, VkPipelineCache cache,
                        VideoCommon::ShaderCacheV2::RecordKind kind, u64 key);

    const Device& device;
    VideoCommon::ShaderCacheV2::DriverBlobStore store;
};

} // namespace Vulkan
