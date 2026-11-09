// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "video_core/shader_jobs.h"
#include "video_core/shader_stage_translation.h"

namespace VideoCommon {

Shader::IR::Program StageTranslation::Take() {
    if (exception) {
        std::rethrow_exception(exception);
    }
    if (!program) {
        throw Shader::RuntimeError("Stage translation cancelled by a shutdown");
    }
    return std::move(*program);
}

void TranslateStages(ShaderJobScheduler& jobs, std::span<StageTranslation> stages,
                     const Shader::HostTranslateInfo& host_info) {
    ParallelFor(jobs, stages.size(), [&](size_t index) {
        StageTranslation& stage = stages[index];
        const auto begin = std::chrono::steady_clock::now();
        try {
            Shader::Maxwell::Flow::CFG cfg{*stage.env, stage.pools->flow_block, stage.cfg_offset,
                                           stage.exits_to_dispatcher};
            stage.program = Shader::Maxwell::TranslateProgram(stage.pools->inst, stage.pools->block,
                                                              *stage.env, cfg, host_info);
        } catch (...) {
            stage.exception = std::current_exception();
        }
        stage.time = std::chrono::steady_clock::now() - begin;
    });
}

} // namespace VideoCommon
