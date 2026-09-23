#include "stateos-model.h"
#include "stateos-header.h"

#include "ggml.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

std::string stateos_model_fingerprint(const std::string & path, std::string * err) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        *err = "cannot parse the model GGUF header at '" + path + "'";
        return std::string();
    }
    const uint64_t data_offset = gguf_get_data_offset(g);
    gguf_free(g);

    std::error_code ec;
    const std::filesystem::path p = std::filesystem::u8path(path);
    const uint64_t file_size = (uint64_t) std::filesystem::file_size(p, ec);
    if (ec || data_offset > file_size) {
        *err = "cannot size the model file '" + path + "'";
        return std::string();
    }
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot open the model file '" + path + "'";
        return std::string();
    }

    stateos_sha256 h;
    const std::string tag = "stateos-model-fp/1";
    h.update(tag.data(), tag.size());
    std::vector<char> chunk(1u << 20);
    uint64_t done = 0;
    while (done < data_offset) {
        const size_t n = (size_t) std::min<uint64_t>(chunk.size(), data_offset - done);
        f.read(chunk.data(), (std::streamsize) n);
        if ((size_t) f.gcount() != n) {
            *err = "short read in the model GGUF header";
            return std::string();
        }
        h.update(chunk.data(), n);
        done += n;
    }
    uint8_t size_le[8];
    for (int i = 0; i < 8; ++i) {
        size_le[i] = (uint8_t) (file_size >> (8 * i));
    }
    h.update(size_le, sizeof(size_le));

    const uint64_t sample = 64u << 10;
    const uint64_t data_size = file_size - data_offset;
    const uint64_t probes[3] = { data_offset, data_offset + data_size / 2, file_size > sample ? file_size - sample : 0 };
    for (const uint64_t off : probes) {
        const size_t n = (size_t) std::min<uint64_t>(sample, file_size - std::min(off, file_size));
        f.clear();
        f.seekg((std::streamoff) off, std::ios::beg);
        f.read(chunk.data(), (std::streamsize) n);
        if ((size_t) f.gcount() != n) {
            *err = "short read while sampling model tensor data";
            return std::string();
        }
        h.update(chunk.data(), n);
    }
    return h.final_hex();
}
