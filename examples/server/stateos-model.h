#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// State-OS "model_fingerprint_v2" hard field. For EVERY shard of the model (split GGUFs: "<prefix>-0000i-of-0000N.gguf",
// shard count from "split.count"): sha256 over the GGUF header bytes [0, data offset) (all metadata and the full tensor
// table: names, types, shapes, offsets), the file size and STATEOS_FP_SAMPLES evenly spaced 64 KiB windows of tensor
// data (the whole data region when it is smaller than that), plus the shard count.
//
// Not a whole-file sha256 (minutes on a 180 GB model at every start). Residual, by design: an edit of tensor bytes that
// keeps every header byte and the file size, and falls outside all sampled windows, is not detected.
// Returns an empty string (and sets *err) when a shard is missing or is not a readable GGUF.
std::string stateos_model_fingerprint(const std::string & path, std::string * err);

constexpr uint64_t STATEOS_FP_WINDOW  = 64u << 10;
constexpr int      STATEOS_FP_SAMPLES = 16;

// [offset, length) windows sampled from a shard's tensor-data region [data_offset, file_size)
std::vector<std::pair<uint64_t, uint64_t>> stateos_fingerprint_windows(uint64_t data_offset, uint64_t file_size);
