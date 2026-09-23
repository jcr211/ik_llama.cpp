#pragma once

#include <string>

// State-OS "model_fingerprint" hard field: sha256 over the GGUF header bytes [0, data offset) (all metadata and the
// full tensor table: names, types, shapes, offsets), the file size and three 64 KiB samples of tensor data. Not a
// whole-file sha256 (minutes on a 180 GB model); it changes with any metadata, quantization, layout or size change.
// Returns an empty string (and sets *err) when the file is not a readable GGUF.
std::string stateos_model_fingerprint(const std::string & path, std::string * err);
