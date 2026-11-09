// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/path_util.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/loader/loader.h"

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv);
extern "C" int LLVMFuzzerTestOneInput(const u8* data, std::size_t size);

namespace Fuzz {

// Bounds the work one input can cause, so a valid but huge tree does not read as a hang.
struct Budget {
    std::size_t files = 256;
    std::size_t dirs = 256;
    std::size_t bytes = 32u << 20;
    int depth = 16;
};

// Reads in chunks, so a bogus declared size does not make the harness itself allocate.
inline void ReadAll(const FileSys::VirtualFile& file, Budget& budget) {
    if (!file || budget.files == 0) {
        return;
    }
    --budget.files;
    std::vector<u8> chunk(64 * 1024);
    const std::size_t size = file->GetSize();
    std::size_t offset = 0;
    while (offset < size && budget.bytes > 0) {
        const std::size_t want = std::min({chunk.size(), size - offset, budget.bytes});
        const std::size_t got = file->Read(chunk.data(), want, offset);
        if (got == 0) {
            break;
        }
        offset += got;
        budget.bytes -= std::min(got, budget.bytes);
    }
}

inline void Walk(const FileSys::VirtualDir& dir, Budget& budget, int depth = 0) {
    if (!dir || depth > budget.depth || budget.dirs == 0) {
        return;
    }
    --budget.dirs;
    for (const auto& file : dir->GetFiles()) {
        ReadAll(file, budget);
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        Walk(sub, budget, depth + 1);
    }
}

inline void ReadRomFS(const FileSys::VirtualFile& romfs, Budget& budget) {
    if (!romfs) {
        return;
    }
    Walk(FileSys::ExtractRomFS(romfs), budget);
}

// Points the key manager at a scratch keys directory holding made-up key area keys for every
// generation, as a user with dumped keys has. Plaintext NCAs then get past the key check without
// any real keys, and the harness never reads or writes the user's own keys. Call it before
// anything touches Core::Crypto::KeyManager.
inline void UseSyntheticKeys() {
    const auto dir = std::filesystem::temp_directory_path() / fmt::format("lichii-fuzz-keys-{}", getpid());
    std::filesystem::create_directories(dir);
    std::ofstream keys{dir / "prod.keys"};
    for (const char* kind : {"application", "ocean", "system"}) {
        for (int generation = 0; generation < 0x20; ++generation) {
            keys << fmt::format("key_area_key_{}_{:02x} = {:032x}\n", kind, generation, 0x11);
        }
    }
    Common::FS::SetEdenPath(Common::FS::EdenPath::KeysDir, dir);
}

// A PFS0 (NSP layout) holding one file.
inline FileSys::VirtualFile WrapInPfs0(const u8* data, std::size_t size, const char* name) {
    const std::size_t name_size = std::strlen(name) + 1;
    const std::size_t strtab_size = (name_size + 0x1F) & ~std::size_t{0x1F};
    std::vector<u8> out(0x10 + 0x18 + strtab_size + size);
    const u32 magic = 0x30534650;
    const u32 num_entries = 1;
    const u32 strtab = static_cast<u32>(strtab_size);
    const u64 entry_size = size;
    std::memcpy(out.data(), &magic, 4);
    std::memcpy(out.data() + 4, &num_entries, 4);
    std::memcpy(out.data() + 8, &strtab, 4);
    std::memcpy(out.data() + 0x10 + 8, &entry_size, 8);
    std::memcpy(out.data() + 0x28, name, name_size);
    std::memcpy(out.data() + 0x28 + strtab_size, data, size);
    return std::make_shared<FileSys::VectorVfsFile>(std::move(out), "fuzz.nsp");
}

} // namespace Fuzz
