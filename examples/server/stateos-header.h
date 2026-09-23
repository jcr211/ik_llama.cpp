#pragma once

// State-OS v1 keyed slot-state container (Longspear fork).
//
// Pure helpers only: no llama/ggml dependency, so tests/test-stateos-header.cpp exercises them without a model.
// The server (server-context.cpp) composes these with the llama_state_seq_* file APIs.
//
// Container layout (all integers little-endian):
//   u32 magic 'LSOS' | u32 container version | u32 header_len | header text (header_len bytes)
//   then sections, each: u32 tag | u32 reserved (0) | u64 payload size | payload
//   and a final 'END\0' section with size 0 that must end exactly at EOF.
//
// Header text: one field per line, "<class> <key>=<value>\n", class H (hard: refuse on mismatch),
// S (soft: warn and proceed) or I (info: never compared). The CURRENT server decides a field's class;
// a saved file cannot downgrade a hard field.

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// UTF-8 std::string <-> filesystem::path, identical under C++17 and C++20 (u8path is deprecated in C++20 and
// path::u8string() changes its return type there)
std::filesystem::path stateos_path(const std::string & utf8);
std::string stateos_path_utf8(const std::filesystem::path & p);

constexpr uint32_t STATEOS_MAGIC             = 0x534F534Cu; // bytes "LSOS"
constexpr uint32_t STATEOS_CONTAINER_VERSION = 1;
constexpr uint32_t STATEOS_LEGACY_SEQ_MAGIC  = 0x67677371u; // 'ggsq' = LLAMA_STATE_SEQ_MAGIC (headerless llama file)
constexpr uint32_t STATEOS_MAX_HEADER_BYTES  = 1u << 20;

// Bump when the bytes this fork writes into a MAIN or COMP section change meaning without the llama
// state-seq version changing (e.g. a new per-layer cache the reader expects). Part of the hard header.
constexpr uint32_t STATEOS_KV_LAYOUT_VERSION   = 1;
constexpr uint32_t STATEOS_COMP_LAYOUT_VERSION = 1;
constexpr uint32_t STATEOS_CKPT_VERSION        = 1;

constexpr uint32_t stateos_tag(char a, char b, char c, char d) {
    return (uint32_t) (uint8_t) a | ((uint32_t) (uint8_t) b << 8) | ((uint32_t) (uint8_t) c << 16) | ((uint32_t) (uint8_t) d << 24);
}

constexpr uint32_t STATEOS_TAG_TOKS = stateos_tag('T', 'O', 'K', 'S'); // int32 token ids
constexpr uint32_t STATEOS_TAG_MAIN = stateos_tag('M', 'A', 'I', 'N'); // target llama seq state (flags 0)
constexpr uint32_t STATEOS_TAG_CKPT = stateos_tag('C', 'K', 'P', 'T'); // server context checkpoints
constexpr uint32_t STATEOS_TAG_COMP = stateos_tag('C', 'O', 'M', 'P'); // MTP companion: sub-header + llama seq state
constexpr uint32_t STATEOS_TAG_END  = stateos_tag('E', 'N', 'D', '\0');

std::string stateos_tag_name(uint32_t tag);

// ---- SHA-256 -------------------------------------------------------------------------------------

struct stateos_sha256 {
    stateos_sha256();
    void update(const void * data, size_t len);
    std::array<uint8_t, 32> final_bytes();
    std::string final_hex();

private:
    void block(const uint8_t * p);
    uint32_t h[8];
    uint8_t  buf[64];
    size_t   buf_len = 0;
    uint64_t total   = 0;
    bool     done    = false;
};

std::string stateos_sha256_hex(const void * data, size_t len);
std::string stateos_hex(const uint8_t * data, size_t len);

// sha256 over the little-endian int32 token ids (the "token_sha256" hard field)
std::string stateos_token_sha256(const int32_t * ids, size_t n);

// ---- header fields --------------------------------------------------------------------------------

enum stateos_class : char {
    STATEOS_HARD = 'H',
    STATEOS_SOFT = 'S',
    STATEOS_INFO = 'I',
};

struct stateos_field {
    char        cls;   // stateos_class
    std::string key;
    std::string value;
};

using stateos_fields = std::vector<stateos_field>;

// false (and *err set) on an invalid key/value/class or a duplicate key
bool stateos_encode_header(const stateos_fields & fields, std::string & out, std::string * err);
bool stateos_decode_header(const std::string & text, stateos_fields & out, std::string * err);

const stateos_field * stateos_find(const stateos_fields & fields, const std::string & key);

struct stateos_mismatch {
    std::string key;
    std::string saved;
    std::string current;
};

struct stateos_verdict {
    bool ok = true;                          // no hard refusal
    std::vector<stateos_mismatch> refused;   // hard fields that differ / are missing / are unknown
    std::vector<stateos_mismatch> warnings;  // soft fields that differ / are missing / are unknown
};

// Compare a saved header against the fields of the current server (plus any integrity fields the caller
// recomputed from the file itself). The current side's class wins.
stateos_verdict stateos_verify(const stateos_fields & saved, const stateos_fields & current);

// ---- container ------------------------------------------------------------------------------------

struct stateos_section {
    uint32_t tag    = 0;
    uint64_t offset = 0; // payload offset in the file
    uint64_t size   = 0; // payload size
};

enum stateos_scan_status {
    STATEOS_SCAN_OK,
    STATEOS_SCAN_NOT_FOUND,
    STATEOS_SCAN_LEGACY,       // a headerless llama state-seq file: refused, fail closed
    STATEOS_SCAN_UNRECOGNIZED, // neither a State-OS container nor a llama state file
    STATEOS_SCAN_UNSUPPORTED,  // a State-OS container of another container version
    STATEOS_SCAN_CORRUPT,      // truncated / inconsistent structure
};

struct stateos_scan_result {
    stateos_scan_status status = STATEOS_SCAN_CORRUPT;
    std::string         error;
    uint32_t            version = 0;
    uint64_t            file_size = 0;
    std::string         header_text;
    std::vector<stateos_section> sections;

    const stateos_section * find(uint32_t tag) const;
};

// Reads only the preamble, the header and the section table (seeking over payloads); never loads state.
stateos_scan_result stateos_scan_file(const std::string & path);

bool stateos_read_range(const std::string & path, uint64_t offset, uint64_t size, std::vector<uint8_t> & out, std::string * err);

// Writers over a stdio FILE* opened in binary mode. All return false on a short write.
bool stateos_write_preamble(std::FILE * f, const std::string & header_text);
bool stateos_write_section_header(std::FILE * f, uint32_t tag, uint64_t size);
bool stateos_write_bytes(std::FILE * f, const void * data, size_t size);

// ---- checkpoints (CKPT payload) -----------------------------------------------------------------

struct stateos_checkpoint_rec {
    int32_t pos_min        = 0;
    int32_t pos_max        = 0;
    int32_t pos_min_prompt = 0;
    int32_t pos_max_prompt = 0;
    int64_t n_tokens       = 0;
    std::vector<uint8_t> data;
};

void stateos_encode_checkpoints(const std::vector<stateos_checkpoint_rec> & in, std::vector<uint8_t> & out);
bool stateos_decode_checkpoints(const uint8_t * data, size_t size, std::vector<stateos_checkpoint_rec> & out, std::string * err);

// ---- companion (COMP payload prefix) ----------------------------------------------------------------

// COMP payload = u32 sub-header length | sub-header text (same line format) | companion llama seq state.
void stateos_encode_companion_prefix(const std::string & subheader, std::vector<uint8_t> & out);

// Reads the COMP sub-header from the file; state_offset/state_size locate the companion llama seq state.
bool stateos_read_companion(const std::string & path, const stateos_section & comp, std::string & subheader,
                            uint64_t & state_offset, uint64_t & state_size, std::string * err);
