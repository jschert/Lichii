// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "common/zstd_compression.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "video_core/nx2_ir_cache.h"

namespace VideoCommon::Nx2IrCache {
namespace {
using Shader::IR::Block;
using Shader::IR::Inst;
using Shader::IR::Opcode;
using Shader::IR::Value;

constexpr u32 ProgramMagic = 0x32505249; // "IRP2"
constexpr u32 RecordMagic = 0x52493253;  // "S2IR"
constexpr std::array<char, 8> FileMagic{'S', '2', 'I', 'R', 'C', 'A', 'C', 'H'};
constexpr u32 FileVersion = 3; // 3: compute stages (local memory check)
constexpr u32 NoBlock = ~0U;

class Writer {
public:
    template <typename T>
    void Put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes = reinterpret_cast<const u8*>(&value);
        out.insert(out.end(), bytes, bytes + sizeof(T));
    }
    void PutBytes(const void* data, size_t size) {
        const auto* bytes = static_cast<const u8*>(data);
        out.insert(out.end(), bytes, bytes + size);
    }
    void PutVar(u64 value) {
        while (value >= 0x80) {
            out.push_back(static_cast<u8>(value | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<u8>(value));
    }
    template <typename Container>
    void PutVector(const Container& container) {
        Put(static_cast<u32>(container.size()));
        for (const auto& item : container) {
            Put(item);
        }
    }
    std::vector<u8> out;
};

class Reader {
public:
    explicit Reader(std::span<const u8> data_) : data{data_} {}
    template <typename T>
    T Get() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value;
        GetBytes(&value, sizeof(T));
        return value;
    }
    void GetBytes(void* dest, size_t size) {
        if (offset + size > data.size()) {
            throw Shader::LogicError("Truncated cached translation");
        }
        std::memcpy(dest, data.data() + offset, size);
        offset += size;
    }
    template <typename Container>
    void GetVector(Container& container) {
        const u32 count = Get<u32>();
        if (count > data.size()) {
            throw Shader::LogicError("Invalid cached container size");
        }
        container.clear();
        for (u32 i = 0; i < count; ++i) {
            container.push_back(Get<typename Container::value_type>());
        }
    }
    u64 GetVar() {
        u64 value = 0;
        for (u32 shift = 0; shift < 64; shift += 7) {
            if (offset >= data.size()) {
                throw Shader::LogicError("Truncated cached translation");
            }
            const u8 byte = data[offset++];
            value |= u64{byte & 0x7fU} << shift;
            if ((byte & 0x80) == 0) {
                return value;
            }
        }
        throw Shader::LogicError("Invalid cached varint");
    }
    bool AtEnd() const {
        return offset == data.size();
    }
    std::span<const u8> data;
    size_t offset{};
};

// Shader::Info: two runs of trivially copyable members, copied as bytes, and its containers.
// The build tag (sizeof(Info)) invalidates files when the struct changes.
struct InfoLayout {
    size_t region1_end;
    size_t region2_begin;
    size_t region2_end;
};

InfoLayout GetInfoLayout() {
    static const Shader::Info probe{};
    const auto* base = reinterpret_cast<const char*>(&probe);
    const InfoLayout layout{
        .region1_end = static_cast<size_t>(
            reinterpret_cast<const char*>(&probe.legacy_stores_mapping) - base),
        .region2_begin = static_cast<size_t>(
            reinterpret_cast<const char*>(&probe.loads_indexed_attributes) - base),
        .region2_end = static_cast<size_t>(
            reinterpret_cast<const char*>(&probe.constant_buffer_descriptors) - base),
    };
    if (!(layout.region1_end < layout.region2_begin && layout.region2_begin < layout.region2_end)) {
        throw Shader::LogicError("Unexpected Shader::Info layout");
    }
    return layout;
}

void SerializeInfo(Writer& w, const Shader::Info& info) {
    static const InfoLayout layout = GetInfoLayout();
    const auto* base = reinterpret_cast<const u8*>(&info);
    w.PutBytes(base, layout.region1_end);
    w.Put(static_cast<u32>(info.legacy_stores_mapping.size()));
    for (const auto& [from, to] : info.legacy_stores_mapping) {
        w.Put(from);
        w.Put(to);
    }
    w.PutBytes(base + layout.region2_begin, layout.region2_end - layout.region2_begin);
    w.PutVector(info.constant_buffer_descriptors);
    w.PutVector(info.storage_buffers_descriptors);
    w.PutVector(info.texture_buffer_descriptors);
    w.PutVector(info.image_buffer_descriptors);
    w.PutVector(info.texture_descriptors);
    w.Put(static_cast<u32>(info.handle_programs.size()));
    for (const auto& program : info.handle_programs) {
        w.PutVector(program);
    }
    w.PutVector(info.image_descriptors);
}

void DeserializeInfo(Reader& r, Shader::Info& info) {
    static const InfoLayout layout = GetInfoLayout();
    auto* base = reinterpret_cast<u8*>(&info);
    r.GetBytes(base, layout.region1_end);
    info.legacy_stores_mapping.clear();
    for (u32 n = r.Get<u32>(); n > 0; --n) {
        const auto from = r.Get<Shader::IR::Attribute>();
        info.legacy_stores_mapping.emplace(from, r.Get<Shader::IR::Attribute>());
    }
    r.GetBytes(base + layout.region2_begin, layout.region2_end - layout.region2_begin);
    r.GetVector(info.constant_buffer_descriptors);
    r.GetVector(info.storage_buffers_descriptors);
    r.GetVector(info.texture_buffer_descriptors);
    r.GetVector(info.image_buffer_descriptors);
    r.GetVector(info.texture_descriptors);
    const u32 num_programs = r.Get<u32>();
    if (num_programs > r.data.size()) {
        throw Shader::LogicError("Invalid cached handle program count");
    }
    info.handle_programs.assign(num_programs, {});
    for (auto& program : info.handle_programs) {
        r.GetVector(program);
    }
    r.GetVector(info.image_descriptors);
}


// The stored kind of a value (Value::Type() reports an instruction's result type instead): the
// IR::Type that is Value's first member.
bool RefersToInst(const Value& value) {
    Shader::IR::Type raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw == Shader::IR::Type::Opaque;
}

// A value: varint (instruction index + 1) << 1 | 0 for an instruction, or (raw IR::Type) << 1 | 1
// followed by the 8 payload bytes as a varint (Value = {IR::Type, 8-byte union}).
static_assert(sizeof(Value) == 16);
constexpr size_t ValuePayloadOffset = 8;

void PutRawValue(Writer& w, const Value& value) {
    u32 type;
    u64 payload;
    std::memcpy(&type, &value, sizeof(type));
    std::memcpy(&payload, reinterpret_cast<const u8*>(&value) + ValuePayloadOffset, sizeof(payload));
    w.PutVar((u64{type} << 1) | 1);
    w.PutVar(payload);
}

Value GetRawValue(u64 tag, Reader& r) {
    const u32 type = static_cast<u32>(tag >> 1);
    if (type == static_cast<u32>(Shader::IR::Type::Opaque)) {
        throw Shader::LogicError("Raw cached value with an instruction type");
    }
    const u64 payload = r.GetVar();
    Value value{};
    std::memcpy(&value, &type, sizeof(type));
    std::memcpy(reinterpret_cast<u8*>(&value) + ValuePayloadOffset, &payload, sizeof(payload));
    return value;
}
} // Anonymous namespace

std::vector<u8> SerializeProgram(const Shader::IR::Program& program) {
    Writer w;
    w.Put(ProgramMagic);
    std::unordered_map<const Block*, u32> block_index;
    std::unordered_map<const Inst*, u32> inst_index;
    for (const Block* block : program.blocks) {
        block_index.emplace(block, static_cast<u32>(block_index.size()));
        for (const Inst& inst : block->Instructions()) {
            inst_index.emplace(&inst, static_cast<u32>(inst_index.size()));
        }
    }
    const auto block_id = [&](const Block* block) -> u32 {
        if (block == nullptr) {
            return NoBlock;
        }
        const auto it = block_index.find(block);
        if (it == block_index.end()) {
            throw Shader::LogicError("Block outside the program's block list");
        }
        return it->second;
    };
    const auto put_value = [&](const Value& value) {
        if (RefersToInst(value)) {
            const auto it = inst_index.find(value.Inst());
            if (it == inst_index.end()) {
                throw Shader::LogicError("Value refers to an instruction outside the program");
            }
            w.PutVar(u64{it->second + 1} << 1);
        } else {
            PutRawValue(w, value);
        }
    };
    w.Put(static_cast<u32>(program.blocks.size()));
    w.Put(static_cast<u32>(inst_index.size()));
    for (const Block* block : program.blocks) {
        w.Put(block->GetOrder());
        const auto successors = block->ImmSuccessors();
        w.Put(static_cast<u32>(successors.size()));
        for (const Block* successor : successors) {
            w.Put(block_id(successor));
        }
        w.PutVar(block->size());
        for (const Inst& inst : block->Instructions()) {
            w.PutVar(static_cast<u32>(inst.GetOpcode()));
            w.PutVar(inst.Flags<u32>());
            w.PutVar(static_cast<u32>(std::max(inst.UseCount(), 0)));
            const size_t num_args = inst.NumArgs();
            if (inst.GetOpcode() == Opcode::Phi) {
                w.PutVar(num_args);
            }
            for (size_t i = 0; i < num_args; ++i) {
                if (inst.GetOpcode() == Opcode::Phi) {
                    w.PutVar(block_id(inst.PhiBlock(i)));
                }
                put_value(inst.Arg(i));
            }
        }
    }
    w.Put(static_cast<u32>(program.post_order_blocks.size()));
    for (const Block* block : program.post_order_blocks) {
        w.Put(block_id(block));
    }
    w.Put(static_cast<u32>(program.syntax_list.size()));
    for (const auto& node : program.syntax_list) {
        using Type = Shader::IR::AbstractSyntaxNode::Type;
        w.Put(static_cast<u32>(node.type));
        const auto& d = node.data;
        switch (node.type) {
        case Type::Block:
            w.Put(block_id(d.block));
            break;
        case Type::If:
            put_value(d.if_node.cond);
            w.Put(block_id(d.if_node.body));
            w.Put(block_id(d.if_node.merge));
            break;
        case Type::EndIf:
            w.Put(block_id(d.end_if.merge));
            break;
        case Type::Loop:
            w.Put(block_id(d.loop.body));
            w.Put(block_id(d.loop.continue_block));
            w.Put(block_id(d.loop.merge));
            break;
        case Type::Repeat:
            put_value(d.repeat.cond);
            w.Put(block_id(d.repeat.loop_header));
            w.Put(block_id(d.repeat.merge));
            break;
        case Type::Break:
            put_value(d.break_node.cond);
            w.Put(block_id(d.break_node.merge));
            w.Put(block_id(d.break_node.skip));
            break;
        case Type::Return:
        case Type::Unreachable:
            break;
        }
    }
    SerializeInfo(w, program.info);
    w.Put(program.stage);
    w.Put(program.workgroup_size);
    w.Put(program.output_topology);
    w.Put(program.output_vertices);
    w.Put(program.invocations);
    w.Put(program.local_memory_size);
    w.Put(program.shared_memory_size);
    w.Put(program.is_geometry_passthrough);
    w.Put(program.mesh.payload_words);
    w.Put(program.mesh.max_vertices);
    w.Put(program.mesh.max_primitives);
    w.Put(program.mesh.topology);
    w.PutVector(program.mesh.vertex_lines);
    w.PutVector(program.mesh.primitive_lines);
    return std::move(w.out);
}

Shader::IR::Program DeserializeProgram(std::span<const u8> data,
                                       Shader::ObjectPool<Shader::IR::Inst>& inst_pool,
                                       Shader::ObjectPool<Shader::IR::Block>& block_pool) {
    Reader r{data};
    if (r.Get<u32>() != ProgramMagic) {
        throw Shader::LogicError("Not a cached translation");
    }
    const u32 num_blocks = r.Get<u32>();
    const u32 num_insts = r.Get<u32>();
    if (num_blocks > data.size() || num_insts > data.size()) {
        throw Shader::LogicError("Invalid cached translation counts");
    }
    Shader::IR::Program program;
    program.blocks.reserve(num_blocks);
    for (u32 i = 0; i < num_blocks; ++i) {
        program.blocks.push_back(block_pool.Create(inst_pool));
    }
    const auto block_at = [&](u32 index) -> Block* {
        if (index == NoBlock) {
            return nullptr;
        }
        if (index >= num_blocks) {
            throw Shader::LogicError("Invalid cached block index");
        }
        return program.blocks[index];
    };
    // Pass 1 creates the instructions and remembers where each one's arguments are; pass 2 sets
    // them (an argument may refer to a later instruction).
    struct PendingInst {
        Inst* inst;
        u32 use_count;
        u32 args_offset;
        u32 num_args;
    };
    std::vector<PendingInst> insts;
    insts.reserve(num_insts);
    const auto skip_value = [&r] {
        const u64 tag = r.GetVar();
        if ((tag & 1) != 0) {
            r.GetVar();
        }
    };
    for (Block* block : program.blocks) {
        block->SetOrder(r.Get<u32>());
        for (u32 n = r.Get<u32>(); n > 0; --n) {
            Block* const successor = block_at(r.Get<u32>());
            if (successor == nullptr) {
                throw Shader::LogicError("Null cached successor");
            }
            block->AddBranch(successor);
        }
        for (u64 n = r.GetVar(); n > 0; --n) {
            const auto opcode = static_cast<Opcode>(r.GetVar());
            const u32 flags = static_cast<u32>(r.GetVar());
            const u32 use_count = static_cast<u32>(r.GetVar());
            if (static_cast<u32>(opcode) > static_cast<u32>(Opcode::DPdyCoarse)) {
                throw Shader::LogicError("Invalid cached opcode");
            }
            Inst* const inst = inst_pool.Create(opcode, flags);
            block->Instructions().push_back(*inst);
            const u32 num_args = opcode == Opcode::Phi ? static_cast<u32>(r.GetVar())
                                                       : static_cast<u32>(inst->NumArgs());
            if (num_args > data.size()) {
                throw Shader::LogicError("Invalid cached argument count");
            }
            insts.push_back({inst, use_count, static_cast<u32>(r.offset), num_args});
            for (u32 a = 0; a < num_args; ++a) {
                if (opcode == Opcode::Phi) {
                    r.GetVar();
                }
                skip_value();
            }
        }
    }
    if (insts.size() != num_insts) {
        throw Shader::LogicError("Cached instruction count mismatch");
    }
    const auto get_value_at = [&](Reader& reader) -> Value {
        const u64 tag = reader.GetVar();
        if ((tag & 1) != 0) {
            return GetRawValue(tag, reader);
        }
        const u64 index = (tag >> 1) - 1;
        if ((tag >> 1) == 0 || index >= insts.size()) {
            throw Shader::LogicError("Invalid cached instruction index");
        }
        return Value{insts[index].inst};
    };
    const auto set_args = [&](PendingInst& pending) {
        Reader args{data};
        args.offset = pending.args_offset;
        for (u32 a = 0; a < pending.num_args; ++a) {
            if (pending.inst->GetOpcode() == Opcode::Phi) {
                Block* const predecessor = block_at(static_cast<u32>(args.GetVar()));
                pending.inst->AddPhiOperand(predecessor, get_value_at(args));
            } else {
                pending.inst->SetArg(a, get_value_at(args));
            }
        }
    };
    // Value::IsImmediate looks through Identity instructions, so their arguments go first (an
    // Identity of an Identity after the one it refers to).
    std::vector<u32> identities;
    for (u32 i = 0; i < insts.size(); ++i) {
        if (insts[i].inst->GetOpcode() == Opcode::Identity) {
            identities.push_back(i);
        }
    }
    std::vector<bool> done(insts.size());
    for (size_t remaining = identities.size(); remaining > 0;) {
        const size_t before = remaining;
        for (const u32 i : identities) {
            if (done[i]) {
                continue;
            }
            Reader args{data};
            args.offset = insts[i].args_offset;
            const u64 tag = args.GetVar();
            if ((tag & 1) == 0 && (tag >> 1) != 0 && (tag >> 1) - 1 < insts.size()) {
                const u64 target = (tag >> 1) - 1;
                if (insts[target].inst->GetOpcode() == Opcode::Identity && !done[target]) {
                    continue;
                }
            }
            set_args(insts[i]);
            done[i] = true;
            --remaining;
        }
        if (remaining == before) {
            throw Shader::LogicError("Cyclic cached Identity instructions");
        }
    }
    for (u32 i = 0; i < insts.size(); ++i) {
        if (!done[i]) {
            set_args(insts[i]);
        }
    }
    // Uses through an Identity of an immediate aren't counted when the argument is set (the value
    // looks immediate), but were when the instruction became that Identity: take the stored count.
    for (const PendingInst& pending : insts) {
        const int missing = static_cast<int>(pending.use_count) - pending.inst->UseCount();
        if (missing < 0 || (missing > 0 && pending.inst->GetOpcode() != Opcode::Identity)) {
            throw Shader::LogicError("Cached translation use count mismatch ({} {} vs {})",
                                     pending.inst->GetOpcode(), pending.inst->UseCount(),
                                     pending.use_count);
        }
        pending.inst->DestructiveAddUsage(missing);
    }
    const u32 num_post_order = r.Get<u32>();
    if (num_post_order > num_blocks) {
        throw Shader::LogicError("Invalid cached post order");
    }
    for (u32 i = 0; i < num_post_order; ++i) {
        program.post_order_blocks.push_back(block_at(r.Get<u32>()));
    }
    const u32 num_nodes = r.Get<u32>();
    if (num_nodes > data.size()) {
        throw Shader::LogicError("Invalid cached syntax list");
    }
    const auto get_value = [&]() { return Shader::IR::U1{get_value_at(r)}; };
    program.syntax_list.resize(num_nodes);
    for (auto& node : program.syntax_list) {
        using Type = Shader::IR::AbstractSyntaxNode::Type;
        node.type = static_cast<Type>(r.Get<u32>());
        auto& d = node.data;
        switch (node.type) {
        case Type::Block:
            d.block = block_at(r.Get<u32>());
            break;
        case Type::If:
            d.if_node.cond = get_value();
            d.if_node.body = block_at(r.Get<u32>());
            d.if_node.merge = block_at(r.Get<u32>());
            break;
        case Type::EndIf:
            d.end_if.merge = block_at(r.Get<u32>());
            break;
        case Type::Loop:
            d.loop.body = block_at(r.Get<u32>());
            d.loop.continue_block = block_at(r.Get<u32>());
            d.loop.merge = block_at(r.Get<u32>());
            break;
        case Type::Repeat:
            d.repeat.cond = get_value();
            d.repeat.loop_header = block_at(r.Get<u32>());
            d.repeat.merge = block_at(r.Get<u32>());
            break;
        case Type::Break:
            d.break_node.cond = get_value();
            d.break_node.merge = block_at(r.Get<u32>());
            d.break_node.skip = block_at(r.Get<u32>());
            break;
        case Type::Return:
        case Type::Unreachable:
            break;
        default:
            throw Shader::LogicError("Invalid cached syntax node");
        }
    }
    DeserializeInfo(r, program.info);
    program.stage = r.Get<Shader::Stage>();
    program.workgroup_size = r.Get<std::array<u32, 3>>();
    program.output_topology = r.Get<Shader::OutputTopology>();
    program.output_vertices = r.Get<u32>();
    program.invocations = r.Get<u32>();
    program.local_memory_size = r.Get<u32>();
    program.shared_memory_size = r.Get<u32>();
    program.is_geometry_passthrough = r.Get<bool>();
    program.mesh.payload_words = r.Get<u32>();
    program.mesh.max_vertices = r.Get<u32>();
    program.mesh.max_primitives = r.Get<u32>();
    program.mesh.topology = r.Get<u32>();
    {
        const u32 n = r.Get<u32>();
        if (n > program.mesh.vertex_lines.capacity()) {
            throw Shader::LogicError("Invalid cached mesh lines");
        }
        program.mesh.vertex_lines.clear();
        for (u32 i = 0; i < n; ++i) {
            program.mesh.vertex_lines.push_back(r.Get<u32>());
        }
    }
    {
        const u32 n = r.Get<u32>();
        if (n > program.mesh.primitive_lines.capacity()) {
            throw Shader::LogicError("Invalid cached mesh lines");
        }
        program.mesh.primitive_lines.clear();
        for (u32 i = 0; i < n; ++i) {
            program.mesh.primitive_lines.push_back(r.Get<u32>());
        }
    }
    if (!r.AtEnd()) {
        throw Shader::LogicError("Trailing bytes after the cached translation");
    }
    return program;
}

bool StageCache::Open(const std::filesystem::path& path_, u64 build_tag, bool create) {
    std::scoped_lock lock{mutex};
    path = path_;
    is_open = false;
    index.clear();
    parsed.clear();
    program_lru.clear();
    program_lookup.clear();
    program_cache_bytes = 0;
    num_records = 0;
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    bool valid = false;
    if (exists) {
        std::ifstream in{path, std::ios::binary};
        std::array<char, 8> magic{};
        u32 version{};
        u64 tag{};
        in.read(magic.data(), magic.size());
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        valid = in && magic == FileMagic && version == FileVersion && tag == build_tag;
    }
    if (!valid) {
        if (!create) {
            return false;
        }
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        out.write(FileMagic.data(), FileMagic.size());
        out.write(reinterpret_cast<const char*>(&FileVersion), sizeof(FileVersion));
        out.write(reinterpret_cast<const char*>(&build_tag), sizeof(build_tag));
        if (!out) {
            return false;
        }
    }
    // Open again for another title (LoadDiskResources runs once per game): an open stream would
    // fail to open and leave the cache off.
    if (file.is_open()) {
        file.close();
    }
    file.clear();
    file.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        return false;
    }
    // Index: magic, size, hash at the start of each record. A record cut short (the emulator
    // stopped while writing) ends the index; later appends overwrite it.
    u64 offset = FileMagic.size() + sizeof(u32) + sizeof(u64);
    file.seekg(0, std::ios::end);
    const u64 size = static_cast<u64>(file.tellg());
    while (offset + 16 <= size) {
        file.seekg(static_cast<std::streamoff>(offset));
        u32 magic{}, record_size{};
        u64 hash{};
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&record_size), sizeof(record_size));
        file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
        if (!file || magic != RecordMagic || record_size < 16 || offset + record_size > size) {
            break;
        }
        index[hash].push_back(offset);
        ++num_records;
        offset += record_size;
    }
    file.clear();
    end_offset = offset;
    is_open = true;
    return true;
}

bool StageCache::Contains(u64 unique_hash) const {
    std::scoped_lock lock{mutex};
    return index.contains(unique_hash);
}

std::vector<Record> StageCache::Find(u64 unique_hash) {
    // The checks and answers lead each record and are small; the compressed program (most of the
    // record, ~10 KB median for OoT) is read only for the record a lookup uses (LoadProgram). A stage
    // with many pointer variants used to read every variant's program here, on the GPU thread.
    constexpr u64 HeaderReadSize = 2048; // OoT: header median 140 bytes, p99 1.1 KB
    std::vector<Record> result;
    std::scoped_lock lock{mutex};
    const auto it = index.find(unique_hash);
    if (!is_open || it == index.end()) {
        return result;
    }
    if (const auto cached = parsed.find(unique_hash); cached != parsed.end()) {
        return cached->second;
    }
    std::vector<u8> bytes;
    for (const u64 offset : it->second) {
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset + sizeof(u32)));
        u32 record_size{};
        file.read(reinterpret_cast<char*>(&record_size), sizeof(record_size));
        if (!file || record_size < 16) {
            file.clear();
            continue;
        }
        const u64 body_size = record_size - 8; // after the magic and the size
        bytes.resize(static_cast<size_t>((std::min)(body_size, HeaderReadSize)));
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            file.clear();
            continue;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                Reader r{bytes};
                Record record;
                record.unique_hash = r.Get<u64>();
                record.stage = r.Get<Shader::Stage>();
                const u32 flags = r.Get<u32>();
                record.texture_bound_used = (flags & 1) != 0;
                record.viewport_used = (flags & 2) != 0;
                record.local_memory_used = (flags & 4) != 0;
                record.texture_bound = r.Get<u32>();
                record.viewport_transform_state = r.Get<u32>();
                record.local_memory_size = r.Get<u32>();
                for (u32 n = r.Get<u32>(); n > 0; --n) {
                    const u64 key = r.Get<u64>();
                    record.storage_pointers.emplace_back(key, r.Get<u8>());
                }
                for (u32 n = r.Get<u32>(); n > 0; --n) {
                    record.texture_checks.push_back(r.Get<Record::TextureCheck>());
                }
                // The answers resize allocates: bound the count against the record's remaining
                // bytes first and reject an over-count. A count that fits the record but runs
                // past the HeaderReadSize prefix throws into the full-record retry instead.
                const u32 answer_count = r.Get<u32>();
                if (static_cast<size_t>(answer_count) > body_size - r.offset) {
                    break; // not a record this build wrote
                }
                if (static_cast<size_t>(answer_count) > r.data.size() - r.offset) {
                    throw Shader::LogicError("Cached answers run past the header read");
                }
                record.answers.resize(answer_count);
                r.GetBytes(record.answers.data(), record.answers.size());
                record.raw_program_size = r.Get<u32>();
                record.compressed_size = r.Get<u32>();
                if (r.offset + record.compressed_size != body_size) {
                    break; // not a record this build wrote
                }
                record.program_offset = offset + 8 + r.offset;
                result.push_back(std::move(record));
                break;
            } catch (const Shader::Exception&) {
                if (bytes.size() == body_size) {
                    break;
                }
                // Checks and answers longer than the first read: read the whole record.
                bytes.resize(static_cast<size_t>(body_size));
                file.clear();
                file.seekg(static_cast<std::streamoff>(offset + 8));
                file.read(reinterpret_cast<char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
                if (!file) {
                    file.clear();
                    break;
                }
            }
        }
    }
    parsed.emplace(unique_hash, result);
    return result;
}

bool StageCache::LoadProgram(Record& record) {
    // A few hundred MiB of decompressed programs (OoT: ~50 KB each) keep the stages of the
    // area being played; the file holds GBs.
    constexpr size_t ProgramCacheBytes = 256ULL << 20;
    if (!record.program.empty()) {
        return true;
    }
    if (record.compressed_size == 0) {
        return false;
    }
    std::vector<u8> compressed(record.compressed_size);
    {
        std::scoped_lock lock{mutex};
        if (!is_open) {
            return false;
        }
        if (const auto hit = program_lookup.find(record.program_offset);
            hit != program_lookup.end()) {
            program_lru.splice(program_lru.begin(), program_lru, hit->second);
            record.program = *hit->second->second;
            return true;
        }
        file.clear();
        file.seekg(static_cast<std::streamoff>(record.program_offset));
        file.read(reinterpret_cast<char*>(compressed.data()),
                  static_cast<std::streamsize>(compressed.size()));
        if (!file) {
            file.clear();
            return false;
        }
    }
    record.program = Common::Compression::DecompressDataZSTD(compressed);
    if (record.program.size() != record.raw_program_size) {
        record.program.clear();
        return false;
    }
    std::scoped_lock lock{mutex};
    if (!program_lookup.contains(record.program_offset)) {
        program_lru.emplace_front(record.program_offset,
                                  std::make_shared<const std::vector<u8>>(record.program));
        program_lookup.emplace(record.program_offset, program_lru.begin());
        program_cache_bytes += record.program.size();
        while (program_cache_bytes > ProgramCacheBytes && program_lru.size() > 1) {
            program_cache_bytes -= program_lru.back().second->size();
            program_lookup.erase(program_lru.back().first);
            program_lru.pop_back();
        }
    }
    return true;
}

void StageCache::Append(const Record& record) {
    Writer w;
    w.Put(RecordMagic);
    w.Put(u32{}); // size, below
    w.Put(record.unique_hash);
    w.Put(record.stage);
    w.Put(static_cast<u32>((record.texture_bound_used ? 1 : 0) | (record.viewport_used ? 2 : 0) |
                           (record.local_memory_used ? 4 : 0)));
    w.Put(record.texture_bound);
    w.Put(record.viewport_transform_state);
    w.Put(record.local_memory_size);
    w.Put(static_cast<u32>(record.storage_pointers.size()));
    for (const auto& [key, value] : record.storage_pointers) {
        w.Put(key);
        w.Put(value);
    }
    w.Put(static_cast<u32>(record.texture_checks.size()));
    for (const auto& check : record.texture_checks) {
        w.Put(check);
    }
    w.Put(static_cast<u32>(record.answers.size()));
    w.PutBytes(record.answers.data(), record.answers.size());
    const std::vector<u8> compressed =
        Common::Compression::CompressDataZSTDDefault(record.program.data(), record.program.size());
    w.Put(static_cast<u32>(record.program.size()));
    w.Put(static_cast<u32>(compressed.size()));
    w.PutBytes(compressed.data(), compressed.size());
    const u32 size = static_cast<u32>(w.out.size());
    std::memcpy(w.out.data() + sizeof(u32), &size, sizeof(size));

    std::scoped_lock lock{mutex};
    if (!is_open) {
        return;
    }
    file.clear();
    file.seekp(static_cast<std::streamoff>(end_offset));
    file.write(reinterpret_cast<const char*>(w.out.data()), static_cast<std::streamsize>(w.out.size()));
    file.flush();
    if (!file) {
        file.clear();
        return;
    }
    index[record.unique_hash].push_back(end_offset);
    parsed.erase(record.unique_hash); // the next Find reads the new record too
    end_offset += size;
    ++num_records;
}

size_t StageCache::NumRecords() const {
    std::scoped_lock lock{mutex};
    return num_records;
}

u64 StageCache::FileSize() const {
    std::scoped_lock lock{mutex};
    return end_offset;
}

} // namespace VideoCommon::Nx2IrCache
