// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <exception>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/object_pool.h"

namespace Shader {
class Environment;
struct HostTranslateInfo;
} // namespace Shader

namespace VideoCommon {

class ShaderJobScheduler;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

/// One graphics stage of a pipeline translated on a CPU worker of the shader jobs, so the stages
/// of a pipeline translate at once (shader cache v2). Each stage has its own environment and IR
/// pools: an environment memoizes what it reads, and pools are not thread-safe.
struct StageTranslation {
    Shader::Environment* env{};
    ShaderPools* pools{};
    u32 cfg_offset{};
    bool exits_to_dispatcher{};

    std::optional<Shader::IR::Program> program;
    std::exception_ptr exception;
    /// Time the translation took on its thread
    std::chrono::nanoseconds time{};

    /// The translated program; rethrows what the translation threw
    Shader::IR::Program Take();
};

/// Translates every stage as a NeedNow job of `jobs`, the calling thread running those no worker
/// took, and returns once all are done. The environments must not be used elsewhere meanwhile.
void TranslateStages(ShaderJobScheduler& jobs, std::span<StageTranslation> stages,
                     const Shader::HostTranslateInfo& host_info);

} // namespace VideoCommon
