# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

foreach(harness fuzz_partition_romfs fuzz_nca fuzz_package)
    add_executable(${harness} ${harness}.cpp fuzz_common.h stubs.cpp)
    target_link_libraries(${harness} PRIVATE common core video_core)
    target_link_options(${harness} PRIVATE -fsanitize=fuzzer)
endforeach()
