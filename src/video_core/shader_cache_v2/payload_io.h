// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "common/common_types.h"

/// Writes and reads record payloads: values in the host's byte order, back to back.
namespace VideoCommon::ShaderCacheV2 {

class PayloadWriter {
public:
    template <typename T>
    void Put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        PutBytes(&value, sizeof(T));
    }

    template <typename T>
    void PutSpan(std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        PutBytes(values.data(), values.size_bytes());
    }

    std::vector<u8> Take() {
        return std::move(bytes);
    }

private:
    void PutBytes(const void* data, size_t size) {
        const size_t at = bytes.size();
        bytes.resize(at + size);
        if (size != 0) {
            std::memcpy(bytes.data() + at, data, size);
        }
    }

    std::vector<u8> bytes;
};

class PayloadReader {
public:
    explicit PayloadReader(std::span<const u8> bytes_) : bytes{bytes_} {}

    template <typename T>
    bool Get(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return GetBytes(&value, sizeof(T));
    }

    template <typename T>
    bool GetVector(std::vector<T>& values, u64 count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (count > (bytes.size() - offset) / sizeof(T)) {
            return false;
        }
        values.resize(static_cast<size_t>(count));
        return GetBytes(values.data(), values.size() * sizeof(T));
    }

    bool AtEnd() const {
        return offset == bytes.size();
    }

private:
    bool GetBytes(void* data, size_t size) {
        if (size > bytes.size() - offset) {
            return false;
        }
        if (size != 0) {
            std::memcpy(data, bytes.data() + offset, size);
        }
        offset += size;
        return true;
    }

    std::span<const u8> bytes;
    size_t offset{};
};

} // namespace VideoCommon::ShaderCacheV2
