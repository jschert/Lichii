// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "common/thread_worker.h"
#include "video_core/shader_jobs.h"

namespace Vulkan {

/// EDEN_NX2_FAST_FIRST=1: a pipeline built while the game runs is first created with
/// VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT and drawn with as soon as the driver returns it;
/// the optimized build follows on a worker and takes its place at a later draw. Trades the seconds
/// the driver spends on the heaviest pipelines (a demanding Switch 2 test title, first visit:
/// draws held up to 7 s of every 10 s) for a slower shader during that time.
inline bool FastFirstBuildEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FAST_FIRST");
        return value && *value && *value != '0';
    }();
    return enabled;
}

/// EDEN_NX2_HEAVY_SKIP: a native pipeline of at least this many converted instructions (all
/// stages) has its draws skipped while the driver builds it, instead of holding every draw
/// behind it. "1" picks 4000; 0 or unset turns it off. In a demanding Switch 2 test title's first
/// visits every driver build over 500 ms was above 4000, and they made ~90% of the compile time.
inline size_t HeavySkipWords() {
    static const size_t words = [] {
        const char* value = std::getenv("EDEN_NX2_HEAVY_SKIP");
        if (!value || !*value) {
            return size_t{0};
        }
        const size_t parsed = static_cast<size_t>(std::strtoull(value, nullptr, 10));
        return parsed == 1 ? size_t{4000} : parsed;
    }();
    return words;
}

/// Bytes of translated IR that pipelines hold until their job emits the SPIR-V. The scheduler
/// counts a job's bytes only while one of its stages runs, so a pipeline whose IR doesn't fit
/// beside the queued IR emits its SPIR-V where it was translated instead.
class QueuedIrBudget {
public:
    class Reservation {
    public:
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        ~Reservation() {
            queued->fetch_sub(bytes, std::memory_order_relaxed);
        }

    private:
        friend class QueuedIrBudget;
        Reservation(std::shared_ptr<std::atomic<u64>> queued_, u64 bytes_)
            : queued{std::move(queued_)}, bytes{bytes_} {}

        std::shared_ptr<std::atomic<u64>> queued;
        u64 bytes;
    };

    explicit QueuedIrBudget(u64 budget_) : budget{budget_} {}

    /// Null when `bytes` don't fit beside the IR queued now
    std::unique_ptr<Reservation> TryReserve(u64 bytes) {
        u64 current = queued->load(std::memory_order_relaxed);
        do {
            if (bytes > budget - std::min(current, budget)) {
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "{} MiB of IR waits for SPIR-V emission: pipelines past {} MiB emit "
                             "theirs where they were translated",
                             current >> 20, budget >> 20);
                }
                return nullptr;
            }
        } while (
            !queued->compare_exchange_weak(current, current + bytes, std::memory_order_relaxed));
        return std::unique_ptr<Reservation>{new Reservation{queued, bytes}};
    }

private:
    const u64 budget;
    std::shared_ptr<std::atomic<u64>> queued = std::make_shared<std::atomic<u64>>(0);
    std::atomic_bool logged{};
};

/// Where a pipeline is built. Without jobs it is built in its constructor.
struct PipelineBuildQueue {
    /// A Soon job of this scheduler, whose CPU stage is the pipeline's deferred SPIR-V emission if
    /// it has one, and whose driver stage builds it. The estimates are the job's.
    VideoCommon::ShaderJobScheduler* jobs{};
    u64 cpu_bytes{};
    u64 driver_bytes{};
    /// Left for Build(), the driver stage of the job that translated it
    bool owner_builds{};
    /// Priority of the job. Optional builds (the optimized build of a fast first build) are
    /// Background: a job still queued when the pipeline goes is dropped.
    VideoCommon::ShaderJobPriority priority{VideoCommon::ShaderJobPriority::Soon};
};

class PipelineBuild {
    struct State {
        explicit State(std::function<void()> task_) : task{std::move(task_)} {}

        void Run() noexcept {
            if (claimed.exchange(true)) {
                return;
            }
            try {
                task();
            } catch (const std::exception& exception) {
                failed = true;
                LOG_ERROR(Render_Vulkan, "Graphics pipeline worker failed: {}", exception.what());
            } catch (...) {
                failed = true;
                LOG_ERROR(Render_Vulkan,
                          "Graphics pipeline worker failed with an unknown exception");
            }
            task = {};
            Complete();
        }

        /// Ends a build that never ran, as failed
        void Fail() noexcept {
            if (claimed.exchange(true)) {
                return;
            }
            failed = true;
            task = {};
            Complete();
        }

        void Complete() {
            std::scoped_lock lock{mutex};
            complete.store(true, std::memory_order_release);
            condition.notify_all();
        }

        std::function<void()> task;
        bool failed{};
        std::atomic_bool claimed{};
        std::atomic_bool complete{};
        std::mutex mutex;
        std::condition_variable condition;
    };

public:
    /// With jobs, `emit` is the CPU stage that runs before `task`
    PipelineBuild(const PipelineBuildQueue& queue, std::function<void()> task,
                  std::function<void()> emit = {})
        : state{std::make_shared<State>(std::move(task))}, jobs{queue.jobs},
          owner_builds{queue.owner_builds} {
        if (jobs) {
            VideoCommon::ShaderJob job{
                .priority = queue.priority,
                .cpu_bytes = queue.cpu_bytes,
                .driver_bytes = queue.driver_bytes,
            };
            if (emit) {
                job.cpu = [state = state, emit = std::move(emit)](
                              VideoCommon::ShaderJobContext&) mutable -> VideoCommon::ShaderDriverStage {
                    try {
                        emit();
                    } catch (const std::exception& exception) {
                        LOG_ERROR(Render_Vulkan, "Graphics pipeline emission failed: {}",
                                  exception.what());
                        state->Fail();
                        return {};
                    } catch (...) {
                        LOG_ERROR(Render_Vulkan, "Graphics pipeline emission failed");
                        state->Fail();
                        return {};
                    }
                    return [state] { state->Run(); };
                };
            } else {
                job.driver = [state = state] { state->Run(); };
            }
            handle = jobs->Submit(std::move(job));
        } else if (!owner_builds) {
            state->Run();
        }
    }

    /// An optional build on a pool of its own (EDEN_NX2_GPL_OPTIMIZE_THREADS), outside the shader
    /// job scheduler. Without `workers` it is built here.
    /// @param drop_unstarted  Destroying the build before a worker started it drops the task
    ///                        instead of running it on the destroying thread.
    PipelineBuild(Common::ThreadWorker* workers, std::function<void()> task,
                  bool drop_unstarted_)
        : state{std::make_shared<State>(std::move(task))}, drop_unstarted{drop_unstarted_} {
        if (workers) {
            workers->QueueWork([state = state] { state->Run(); });
        } else {
            state->Run();
        }
    }

    ~PipelineBuild() {
        if (drop_unstarted && !state->claimed.exchange(true)) {
            state->task = {};
            state->Complete();
            return;
        }
        if (jobs) {
            // Nothing needs a pipeline that goes: a build still queued is dropped.
            jobs->Cancel(handle);
            handle.Wait();
        } else if (!owner_builds) {
            Wait(true);
        }
    }

    bool IsBuilt() const {
        return state->complete.load(std::memory_order_acquire);
    }

    bool Failed() const {
        return IsBuilt() && state->failed;
    }

    /// Builds it on this thread (owner_builds)
    void Build() const {
        state->Run();
    }

    void Wait(bool may_build = false) const {
        const auto shared = state;
        if (jobs) {
            if (IsBuilt()) {
                return;
            }
            // A draw waits for it. Its deferred emission is never run here.
            if (may_build) {
                jobs->RunOrWait(handle);
            } else {
                jobs->Promote(handle, VideoCommon::ShaderJobPriority::NeedNow);
                handle.Wait();
            }
            // Cancelled when the scheduler shut down
            shared->Fail();
            return;
        }
        if (may_build) {
            shared->Run();
        }
        std::unique_lock lock{shared->mutex};
        shared->condition.wait(lock,
                               [&] { return shared->complete.load(std::memory_order_acquire); });
    }

    /// Waits up to `timeout` for the build, without running it on this thread. The worker sets
    /// `complete` under the mutex before notifying, so the wakeup cannot be lost.
    /// @return True when the pipeline is built.
    bool WaitFor(std::chrono::microseconds timeout) const {
        const auto shared = state;
        if (jobs && !IsBuilt()) {
            jobs->Promote(handle, VideoCommon::ShaderJobPriority::NeedNow);
        }
        std::unique_lock lock{shared->mutex};
        return shared->condition.wait_for(lock, timeout, [&] {
            return shared->complete.load(std::memory_order_acquire);
        });
    }

    PipelineBuild(const PipelineBuild&) = delete;
    PipelineBuild& operator=(const PipelineBuild&) = delete;

private:
    std::shared_ptr<State> state;
    VideoCommon::ShaderJobScheduler* jobs{};
    VideoCommon::ShaderJobHandle handle;
    bool owner_builds{};
    bool drop_unstarted{};
};

} // namespace Vulkan
