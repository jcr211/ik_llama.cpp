#include "stateos-model.h"
#include "stateos-header.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

std::vector<std::pair<uint64_t, uint64_t>> stateos_fingerprint_windows(uint64_t data_offset, uint64_t file_size) {
    std::vector<std::pair<uint64_t, uint64_t>> w;
    if (data_offset >= file_size) {
        return w;
    }
    const uint64_t data = file_size - data_offset;
    if (data <= (uint64_t) STATEOS_FP_SAMPLES * STATEOS_FP_WINDOW) {
        w.emplace_back(data_offset, data);
        return w;
    }
    for (int i = 0; i < STATEOS_FP_SAMPLES; ++i) {
        const uint64_t off = data_offset + (data - STATEOS_FP_WINDOW) * (uint64_t) i / (uint64_t) (STATEOS_FP_SAMPLES - 1);
        w.emplace_back(off, STATEOS_FP_WINDOW);
    }
    return w;
}

static void put_u64(stateos_sha256 & h, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = (uint8_t) (v >> (8 * i));
    }
    h.update(b, sizeof(b));
}

// data offset and split count of one GGUF (split count 1 when the key is absent)
static bool gguf_probe(const std::string & path, uint64_t & data_offset, int & n_split, std::string * err) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        *err = "cannot parse the GGUF header at '" + path + "'";
        return false;
    }
    data_offset = gguf_get_data_offset(g);
    n_split = 1;
    const int kid = gguf_find_key(g, "split.count");
    if (kid >= 0) {
        switch (gguf_get_kv_type(g, kid)) {
            case GGUF_TYPE_UINT16: n_split = (int) gguf_get_val_u16(g, kid); break;
            case GGUF_TYPE_INT32:  n_split = (int) gguf_get_val_i32(g, kid); break;
            case GGUF_TYPE_UINT32: n_split = (int) gguf_get_val_u32(g, kid); break;
            default:               n_split = -1; break;
        }
    }
    gguf_free(g);
    if (n_split == 0) {
        n_split = 1; // llama-gguf-split --merge writes 0; the loader treats anything <= 1 as one file
    }
    if (n_split < 1) {
        *err = "unreadable split.count in '" + path + "'";
        return false;
    }
    return true;
}

static bool hash_shard(stateos_sha256 & h, const std::string & path, std::string * err) {
    uint64_t data_offset = 0;
    int n_split = 1;
    if (!gguf_probe(path, data_offset, n_split, err)) {
        return false;
    }
    std::error_code ec;
    const std::filesystem::path p = stateos_path(path);
    const uint64_t file_size = (uint64_t) std::filesystem::file_size(p, ec);
    if (ec) {
        *err = "cannot size '" + path + "'";
        return false;
    }
    // a GGUF without tensor data (vocab-only) ends before its aligned data offset
    data_offset = std::min(data_offset, file_size);

    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot open '" + path + "'";
        return false;
    }
    std::vector<char> chunk(1u << 20);
    auto hash_range = [&](uint64_t off, uint64_t len) {
        f.clear();
        f.seekg((std::streamoff) off, std::ios::beg);
        while (len > 0) {
            const size_t n = (size_t) std::min<uint64_t>(chunk.size(), len);
            f.read(chunk.data(), (std::streamsize) n);
            if ((size_t) f.gcount() != n) {
                return false;
            }
            h.update(chunk.data(), n);
            len -= n;
        }
        return true;
    };
    if (!hash_range(0, data_offset)) {
        *err = "short read in the GGUF header of '" + path + "'";
        return false;
    }
    put_u64(h, file_size);
    for (const auto & w : stateos_fingerprint_windows(data_offset, file_size)) {
        put_u64(h, w.first);
        if (!hash_range(w.first, w.second)) {
            *err = "short read while sampling tensor data of '" + path + "'";
            return false;
        }
    }
    return true;
}

std::string stateos_model_fingerprint(const std::string & path, std::string * err) {
    uint64_t data_offset = 0;
    int n_split = 1;
    if (!gguf_probe(path, data_offset, n_split, err)) {
        return std::string();
    }

    std::vector<std::string> shards;
    if (n_split == 1) {
        shards.push_back(path);
    } else {
        // the model is loaded from its first shard: "<prefix>-00001-of-0000N.gguf"
        char suffix[64];
        std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", 1, n_split);
        const std::string sfx(suffix);
        if (path.size() <= sfx.size() || path.compare(path.size() - sfx.size(), sfx.size(), sfx) != 0) {
            *err = "'" + path + "' declares split.count=" + std::to_string(n_split) + " but is not named <prefix>" + sfx;
            return std::string();
        }
        const std::string prefix = path.substr(0, path.size() - sfx.size());
        for (int i = 0; i < n_split; ++i) {
            std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i + 1, n_split);
            shards.push_back(prefix + suffix);
        }
    }

    stateos_sha256 h;
    const std::string tag = "stateos-model-fp/2";
    h.update(tag.data(), tag.size());
    put_u64(h, shards.size());
    for (const auto & shard : shards) {
        if (!hash_shard(h, shard, err)) {
            return std::string();
        }
    }
    return h.final_hex();
}
