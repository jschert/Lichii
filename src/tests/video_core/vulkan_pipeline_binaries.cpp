// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/renderer_vulkan/vk_pipeline_binaries.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace {

namespace V2 = VideoCommon::ShaderCacheV2;
namespace StutterTrace = VideoCommon::StutterTrace;

/// #version 450, layout(local_size_x = 1) in; void main() {}
constexpr std::array<u32, 70> EmptyCompute{
    0x07230203, 0x00010000, 0x0008000b, 0x0000000a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00040047, 0x00000009, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017,
    0x00000007, 0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x00000008, 0x00000001, 0x0006002c,
    0x00000007, 0x00000009, 0x00000008, 0x00000008, 0x00000008, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038,
};
/// The word of its LocalSize execution mode's x
constexpr size_t LocalSizeX = 24;

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-pipeline-binaries-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

u64 BinaryBuilds() {
    return StutterTrace::g_totals
        .driver_builds[static_cast<size_t>(StutterTrace::DriverBuild::Binary)]
        .load();
}

} // Anonymous namespace

// Needs a Vulkan driver with VK_KHR_pipeline_binary that doesn't prefer its own cache (RADV, for
// one: Mesa's amdgpu drm-shim runs it without a GPU):
//   ./tests "[vulkan_pipeline_binaries]"
TEST_CASE("Vulkan pipeline binaries: a compute pipeline is built again from its binaries",
          "[.][vulkan_pipeline_binaries]") {
    const auto library = Vulkan::OpenLibrary();
    REQUIRE(library);
    Vulkan::vk::InstanceDispatch dld;
    const Vulkan::vk::Instance instance = Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1);
    std::optional<Vulkan::Device> device;
    for (const VkPhysicalDevice physical : instance.EnumeratePhysicalDevices()) {
        try {
            device.emplace(*instance, Vulkan::vk::PhysicalDevice(physical, dld), VK_NULL_HANDLE,
                           dld);
        } catch (const Vulkan::vk::Exception&) {
            continue;
        }
        if (device->IsKhrPipelineBinarySupported()) {
            break;
        }
        device.reset();
    }
    if (!device) {
        SKIP("No device enables VK_KHR_pipeline_binary");
    }
    INFO(device->GetDriverName() << " " << device->GetModelName());
    const Vulkan::vk::Device& logical = device->GetLogical();
    const Vulkan::vk::PipelineLayout layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    });
    const auto make_module = [&](u32 local_size_x) {
        std::array<u32, EmptyCompute.size()> code = EmptyCompute;
        code[LocalSizeX] = local_size_x;
        return logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size() * sizeof(u32),
            .pCode = code.data(),
        });
    };
    const auto create_info = [&](VkShaderModule module) {
        return VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
            .layout = *layout,
        };
    };
    const Vulkan::vk::ShaderModule first = make_module(1);
    const Vulkan::vk::ShaderModule second = make_module(2);
    constexpr u64 Key = 0x1234;
    TempDir temp;

    // A first session compiles it and stores its binaries.
    {
        const auto binaries = Vulkan::PipelineBinaries::Open(*device, temp.path);
        REQUIRE(binaries);
        REQUIRE(binaries->Counts().loaded == 0);
        const Vulkan::vk::Pipeline pipeline =
            binaries->CreateComputePipeline(create_info(*first), VK_NULL_HANDLE, Key);
        REQUIRE(*pipeline != VK_NULL_HANDLE);
        REQUIRE(binaries->Counts().written == 1);
        REQUIRE(binaries->Counts().used == 0);
    }
    // The next one builds it from them.
    {
        const auto binaries = Vulkan::PipelineBinaries::Open(*device, temp.path);
        REQUIRE(binaries);
        REQUIRE(binaries->Counts().loaded == 1);
        const u64 binary_builds = BinaryBuilds();
        const Vulkan::vk::Pipeline pipeline =
            binaries->CreateComputePipeline(create_info(*first), VK_NULL_HANDLE, Key);
        REQUIRE(*pipeline != VK_NULL_HANDLE);
        REQUIRE(binaries->Counts().used == 1);
        REQUIRE(binaries->Counts().rejected == 0);
        REQUIRE(binaries->Counts().written == 0);
        REQUIRE(BinaryBuilds() == binary_builds + 1);
    }
    // New SPIR-V under the same key: the stale binaries are replaced.
    {
        const auto binaries = Vulkan::PipelineBinaries::Open(*device, temp.path);
        REQUIRE(binaries);
        const Vulkan::vk::Pipeline pipeline =
            binaries->CreateComputePipeline(create_info(*second), VK_NULL_HANDLE, Key);
        REQUIRE(*pipeline != VK_NULL_HANDLE);
        REQUIRE(binaries->Counts().used == 0);
        REQUIRE(binaries->Counts().rejected == 1);
        REQUIRE(binaries->Counts().written == 1);
    }
    {
        const auto binaries = Vulkan::PipelineBinaries::Open(*device, temp.path);
        REQUIRE(binaries);
        const Vulkan::vk::Pipeline pipeline =
            binaries->CreateComputePipeline(create_info(*second), VK_NULL_HANDLE, Key);
        REQUIRE(binaries->Counts().used == 1);
        REQUIRE(binaries->Counts().rejected == 0);
    }
}
