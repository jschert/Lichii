// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// One input as a raw PFS0/HFS0 partition and as a raw RomFS image.

#include "core/file_sys/partition_filesystem.h"
#include "fuzz/fuzz_common.h"

extern "C" int LLVMFuzzerTestOneInput(const u8* data, std::size_t size) {
    auto file = std::make_shared<FileSys::VectorVfsFile>(std::vector<u8>(data, data + size),
                                                         "fuzz.bin");
    Fuzz::Budget budget;
    FileSys::PartitionFilesystem pfs{file};
    if (pfs.GetStatus() == Loader::ResultStatus::Success) {
        for (const auto& entry : pfs.GetFiles()) {
            Fuzz::ReadAll(entry, budget);
        }
    }
    Fuzz::ReadRomFS(file, budget);
    return 0;
}
