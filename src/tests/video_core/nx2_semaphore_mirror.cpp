// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2.h"

TEST_CASE("Nx2SemaphoreMirror maps second-page releases to the first page", "[video_core][nx2]") {
    constexpr u64 region = 0x2106027000;
    constexpr u64 other = 0x2106100000;
    Tegra::Nx2ClearSemaphoreRegions();
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1010) == 0);

    Tegra::Nx2AddSemaphoreRegion(region);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1000) == region);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1010) == region + 0x10);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1ffc) == region + 0xffc);
    CHECK(Tegra::Nx2SemaphoreMirror(region) == 0);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0xffc) == 0);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x2000) == 0);
    CHECK(Tegra::Nx2SemaphoreMirror(other + 0x1010) == 0);

    Tegra::Nx2AddSemaphoreRegion(other);
    Tegra::Nx2AddSemaphoreRegion(region);
    CHECK(Tegra::Nx2SemaphoreMirror(other + 0x1010) == other + 0x10);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1010) == region + 0x10);

    Tegra::Nx2RemoveSemaphoreRegion(region);
    CHECK(Tegra::Nx2SemaphoreMirror(region + 0x1010) == 0);
    CHECK(Tegra::Nx2SemaphoreMirror(other + 0x1010) == other + 0x10);

    Tegra::Nx2ClearSemaphoreRegions();
    CHECK(Tegra::Nx2SemaphoreMirror(other + 0x1010) == 0);
}


TEST_CASE("Nx2MirrorShadow keeps mirrored values out of words with direct releases", "[video_core][nx2]") {
    constexpr u64 region = 0x2106027000;
    constexpr u64 device = 0x80001000;
    auto& shadow = Tegra::Nx2MirrorShadow::Get();
    Tegra::Nx2ClearSemaphoreRegions();
    Tegra::Nx2AddSemaphoreRegion(region);

    CHECK_FALSE(shadow.Diverts(device));
    shadow.NoteDirectRelease(device + 0x10, region + 0x1010);
    CHECK_FALSE(shadow.Diverts(device + 0x10));

    shadow.NoteDirectRelease(device, region);
    CHECK(shadow.Diverts(device));
    CHECK(shadow.Diverts(device + 2));
    CHECK_FALSE(shadow.Diverts(device + 4));

    CHECK_FALSE(shadow.Read(device, 7).has_value());
    shadow.Set(device, 42, 7);
    CHECK(shadow.Read(device, 7) == 42u);
    CHECK(shadow.Read(device, 8) == std::nullopt);
    CHECK(shadow.Read(device, 7) == std::nullopt);

    shadow.Set(device, 43, 7);
    shadow.NoteDirectRelease(device, region);
    CHECK(shadow.Read(device, 7) == std::nullopt);

    const auto max_with = [](u32 payload) { return [payload](u32 current) { return std::max(current, payload); }; };
    shadow.Set(device, 10, 0);
    CHECK(shadow.NoteCombinedRelease(device, region, 0, 3, max_with(3)) == 10u);
    CHECK(shadow.Read(device, 3) == 10u);
    CHECK(shadow.NoteCombinedRelease(device, region, 3, 12, max_with(12)) == 12u);
    CHECK(shadow.Read(device, 12) == 12u);
    CHECK(shadow.NoteCombinedRelease(device + 4, region + 4, 5, 6, max_with(6)) == 6u);
    CHECK(shadow.Read(device + 4, 6) == std::nullopt);

    shadow.Set(device, 44, 7);
    Tegra::Nx2ClearSemaphoreRegions();
    CHECK_FALSE(shadow.Diverts(device));
    CHECK(shadow.Read(device, 7) == std::nullopt);
}

TEST_CASE("Nx2SemaphoreMirror finds every region, past the first 32", "[video_core][nx2]") {
    constexpr u64 base = 0x2106000000;
    constexpr u64 count = 140;
    const auto start = [](u64 index) { return base + index * 0x3000; };
    Tegra::Nx2ClearSemaphoreRegions();
    for (u64 index = 0; index < count; ++index) {
        Tegra::Nx2AddSemaphoreRegion(start(index));
    }
    Tegra::Nx2AddSemaphoreRegion(start(0));
    CHECK(Tegra::g_nx2_semaphore_regions.size() == count);
    for (u64 index = 0; index < count; ++index) {
        CAPTURE(index);
        CHECK(Tegra::Nx2SemaphoreMirror(start(index) + 0x1010) == start(index) + 0x10);
        CHECK(Tegra::Nx2InSemaphoreFirstPage(start(index) + 0x10));
        CHECK(Tegra::Nx2SemaphoreMirror(start(index) + 0x2000) == 0);
        CHECK_FALSE(Tegra::Nx2InSemaphoreFirstPage(start(index) + 0x2000));
    }

    Tegra::Nx2RemoveSemaphoreRegion(start(count - 1));
    CHECK(Tegra::Nx2SemaphoreMirror(start(count - 1) + 0x1010) == 0);
    CHECK(Tegra::Nx2SemaphoreMirror(start(count - 2) + 0x1010) == start(count - 2) + 0x10);
    Tegra::Nx2ClearSemaphoreRegions();
    CHECK(Tegra::Nx2SemaphoreMirror(start(40) + 0x1010) == 0);
}
