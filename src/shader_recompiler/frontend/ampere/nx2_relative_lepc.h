// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// nx2 shaderwarm (docs/shader-precompile.md): address-independent lowering of the compiler's
// geometry prologue LEPC, so a geometry program's lowered code (and its stage hash) doesn't depend
// on where it sits in GPU memory and can be precompiled. Off unless EDEN_NX2_RELATIVE_LEPC=1.

#include <cstdlib>
#include <span>

#include "common/common_types.h"

namespace Shader::Ampere {

[[nodiscard]] inline bool Nx2RelativeLepcEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RELATIVE_LEPC");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// The prologue LEPC Rd; MOV Rt, imm; IADD3 Rd = Rd + Rt (carry); IADD3.X Rd+1; exit stub setup
/// (0x357), the lowering drops the stub. The address is dead when Rd and Rd+1 are written again
/// before anything else names them.
template <typename Insn>
[[nodiscard]] bool Nx2LepcAddressIsDead(std::span<const Insn> code, u64 index, u64 dest) {
    if (index + 4 >= code.size() || code[index + 4].Bits(0, 12) != 0x357) return false;
    const u32 temp = code[index + 1].Bits(16, 8);
    if (code[index + 1].Bits(0, 12) != 0x802 || code[index + 2].Bits(0, 12) != 0x210 ||
        code[index + 3].Bits(0, 12) != 0x210 || code[index + 2].Bits(16, 8) != dest ||
        code[index + 3].Bits(16, 8) != dest + 1 || temp == dest || temp == dest + 1)
        return false;
    // After the stub setup: each of Rd, Rd+1 must be an ALU destination (bits 16-23)
    // before it appears in any other register field (24-31, 32-39, 64-71) or a branch.
    bool written[2]{};
    for (u64 next = index + 5; next < code.size() && next < index + 64; ++next) {
        const Insn& n = code[next];
        const u32 op = n.Bits(0, 12);
        if (op == 0x947 || op == 0x944 || op == 0x945 || op == 0x949 || op == 0x950 ||
            op == 0x94d || op == 0x958)
            return false;
        for (u32 r = 0; r < 2; ++r) {
            const u32 reg = static_cast<u32>(dest) + r;
            if (written[r]) continue;
            // S2R, S2UR, CS2R, MOV immediate: no register source (zero fields)
            const bool no_source =
                op == 0x919 || op == 0x9c3 || op == 0x805 || op == 0x802;
            if (!no_source && (n.Bits(24, 8) == reg || n.Bits(32, 8) == reg ||
                               n.Bits(64, 8) == reg))
                return false;
            const u32 base = op & 0x1ff;
            const bool no_destination = base == 0x00b || base == 0x00c ||
                                        (base >= 0x185 && base <= 0x188) ||
                                        base == 0x18e || op == 0x918 || op == 0x357 ||
                                        op == 0x957;
            // Only an unconditional write (guard PT, bits 12-15 == 7) replaces the address on
            // every lane; a predicated one may leave it.
            if (n.Bits(16, 8) == reg && !no_destination && n.Bits(12, 4) == 7) {
                written[r] = true;
            }
        }
        if (written[0] && written[1]) return true;
    }
    return false;
}

} // namespace Shader::Ampere
