#include "stateos-header.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// ---- small LE helpers -------------------------------------------------------------------------------

static void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back((uint8_t) (v >> (8 * i)));
    }
}

static void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t) (v >> (8 * i)));
    }
}

static uint32_t get_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static uint64_t get_u64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

std::filesystem::path stateos_path(const std::string & utf8) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
#else
    return std::filesystem::u8path(utf8);
#endif
}

std::string stateos_path_utf8(const std::filesystem::path & p) {
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::string stateos_tag_name(uint32_t tag) {
    std::string s;
    for (int i = 0; i < 4; ++i) {
        const char c = (char) (tag >> (8 * i));
        if (c == '\0') {
            break;
        }
        s.push_back((c >= 32 && c < 127) ? c : '?');
    }
    return s;
}

// ---- SHA-256 (FIPS 180-4) -----------------------------------------------------------------------------

static const uint32_t k_sha256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

stateos_sha256::stateos_sha256() {
    h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
    h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
}

void stateos_sha256::block(const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t) p[4 * i] << 24) | ((uint32_t) p[4 * i + 1] << 16) | ((uint32_t) p[4 * i + 2] << 8) | (uint32_t) p[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch  = (e & f) ^ (~e & g);
        const uint32_t t1  = hh + S1 + ch + k_sha256[i] + w[i];
        const uint32_t S0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2  = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void stateos_sha256::update(const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    total += len;
    if (buf_len > 0) {
        const size_t take = std::min(len, (size_t) 64 - buf_len);
        std::memcpy(buf + buf_len, p, take);
        buf_len += take;
        p += take;
        len -= take;
        if (buf_len == 64) {
            block(buf);
            buf_len = 0;
        }
    }
    while (len >= 64) {
        block(p);
        p += 64;
        len -= 64;
    }
    if (len > 0) {
        std::memcpy(buf, p, len);
        buf_len = len;
    }
}

std::array<uint8_t, 32> stateos_sha256::final_bytes() {
    if (!done) {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80;
        const uint8_t zero = 0x00;
        update(&one, 1);
        while (buf_len != 56) {
            update(&zero, 1);
        }
        uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) {
            len_be[i] = (uint8_t) (bits >> (56 - 8 * i));
        }
        update(len_be, 8);
        done = true;
    }
    std::array<uint8_t, 32> out;
    for (int i = 0; i < 8; ++i) {
        out[4 * i]     = (uint8_t) (h[i] >> 24);
        out[4 * i + 1] = (uint8_t) (h[i] >> 16);
        out[4 * i + 2] = (uint8_t) (h[i] >> 8);
        out[4 * i + 3] = (uint8_t) (h[i]);
    }
    return out;
}

std::string stateos_sha256::final_hex() {
    const auto b = final_bytes();
    return stateos_hex(b.data(), b.size());
}

std::string stateos_hex(const uint8_t * data, size_t len) {
    static const char * digits = "0123456789abcdef";
    std::string s;
    s.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(digits[data[i] >> 4]);
        s.push_back(digits[data[i] & 15]);
    }
    return s;
}

std::string stateos_sha256_hex(const void * data, size_t len) {
    stateos_sha256 h;
    h.update(data, len);
    return h.final_hex();
}

std::string stateos_token_sha256(const int32_t * ids, size_t n) {
    stateos_sha256 h;
    uint8_t chunk[4096];
    size_t fill = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = (uint32_t) ids[i];
        chunk[fill++] = (uint8_t) v;
        chunk[fill++] = (uint8_t) (v >> 8);
        chunk[fill++] = (uint8_t) (v >> 16);
        chunk[fill++] = (uint8_t) (v >> 24);
        if (fill == sizeof(chunk)) {
            h.update(chunk, fill);
            fill = 0;
        }
    }
    h.update(chunk, fill);
    return h.final_hex();
}

// ---- header fields --------------------------------------------------------------------------------

static bool valid_key(const std::string & key) {
    if (key.empty() || key.size() > 128) {
        return false;
    }
    for (const char c : key) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

static bool valid_class(char c) {
    return c == STATEOS_HARD || c == STATEOS_SOFT || c == STATEOS_INFO;
}

static void set_err(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
}

bool stateos_encode_header(const stateos_fields & fields, std::string & out, std::string * err) {
    out.clear();
    std::set<std::string> seen;
    for (const auto & f : fields) {
        if (!valid_class(f.cls)) {
            set_err(err, "invalid class for field '" + f.key + "'");
            return false;
        }
        if (!valid_key(f.key)) {
            set_err(err, "invalid field key '" + f.key + "'");
            return false;
        }
        if (f.value.find_first_of("\r\n") != std::string::npos || f.value.find('\0') != std::string::npos) {
            set_err(err, "field '" + f.key + "' value contains a line break or NUL");
            return false;
        }
        if (!seen.insert(f.key).second) {
            set_err(err, "duplicate field '" + f.key + "'");
            return false;
        }
        out.push_back(f.cls);
        out.push_back(' ');
        out += f.key;
        out.push_back('=');
        out += f.value;
        out.push_back('\n');
    }
    if (out.size() > STATEOS_MAX_HEADER_BYTES) {
        set_err(err, "header exceeds the size limit");
        return false;
    }
    return true;
}

bool stateos_decode_header(const std::string & text, stateos_fields & out, std::string * err) {
    out.clear();
    if (text.size() > STATEOS_MAX_HEADER_BYTES) {
        set_err(err, "header exceeds the size limit");
        return false;
    }
    if (!text.empty() && text.back() != '\n') {
        set_err(err, "header is not newline-terminated");
        return false;
    }
    std::set<std::string> seen;
    size_t pos = 0;
    int line_no = 0;
    while (pos < text.size()) {
        const size_t nl = text.find('\n', pos);
        const std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        if (line.size() < 4 || !valid_class(line[0]) || line[1] != ' ') {
            set_err(err, "malformed header line " + std::to_string(line_no));
            return false;
        }
        const size_t eq = line.find('=', 2);
        if (eq == std::string::npos) {
            set_err(err, "header line " + std::to_string(line_no) + " has no '='");
            return false;
        }
        stateos_field f;
        f.cls   = line[0];
        f.key   = line.substr(2, eq - 2);
        f.value = line.substr(eq + 1);
        if (!valid_key(f.key)) {
            set_err(err, "invalid field key on header line " + std::to_string(line_no));
            return false;
        }
        if (f.value.find('\r') != std::string::npos || f.value.find('\0') != std::string::npos) {
            set_err(err, "invalid value on header line " + std::to_string(line_no));
            return false;
        }
        if (!seen.insert(f.key).second) {
            set_err(err, "duplicate field '" + f.key + "'");
            return false;
        }
        out.push_back(std::move(f));
    }
    return true;
}

const stateos_field * stateos_find(const stateos_fields & fields, const std::string & key) {
    for (const auto & f : fields) {
        if (f.key == key) {
            return &f;
        }
    }
    return nullptr;
}

stateos_verdict stateos_verify(const stateos_fields & saved, const stateos_fields & current) {
    static const std::string missing = "<missing>";
    static const std::string unknown = "<unknown to this build>";

    stateos_verdict v;
    for (const auto & cur : current) {
        if (cur.cls == STATEOS_INFO) {
            continue;
        }
        const stateos_field * s = stateos_find(saved, cur.key);
        if (s == nullptr || s->value != cur.value) {
            stateos_mismatch m { cur.key, s ? s->value : missing, cur.value };
            if (cur.cls == STATEOS_HARD) {
                v.refused.push_back(std::move(m));
            } else {
                v.warnings.push_back(std::move(m));
            }
        }
    }
    for (const auto & s : saved) {
        if (s.cls == STATEOS_INFO || stateos_find(current, s.key) != nullptr) {
            continue;
        }
        stateos_mismatch m { s.key, s.value, unknown };
        if (s.cls == STATEOS_HARD) {
            v.refused.push_back(std::move(m));
        } else {
            v.warnings.push_back(std::move(m));
        }
    }
    v.ok = v.refused.empty();
    return v;
}

// ---- container ------------------------------------------------------------------------------------

const stateos_section * stateos_scan_result::find(uint32_t tag) const {
    for (const auto & s : sections) {
        if (s.tag == tag) {
            return &s;
        }
    }
    return nullptr;
}

static bool read_exact(std::ifstream & f, void * dst, size_t n) {
    f.read((char *) dst, (std::streamsize) n);
    return (size_t) f.gcount() == n;
}

stateos_scan_result stateos_scan_file(const std::string & path) {
    stateos_scan_result r;
    std::error_code ec;
    const std::filesystem::path p = stateos_path(path);
    if (!std::filesystem::is_regular_file(p, ec) || ec) {
        r.status = STATEOS_SCAN_NOT_FOUND;
        r.error  = "state file not found";
        return r;
    }
    r.file_size = (uint64_t) std::filesystem::file_size(p, ec);
    if (ec) {
        r.status = STATEOS_SCAN_NOT_FOUND;
        r.error  = "cannot stat state file";
        return r;
    }
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) {
        r.status = STATEOS_SCAN_NOT_FOUND;
        r.error  = "cannot open state file";
        return r;
    }
    uint8_t pre[12];
    if (r.file_size < 4 || !read_exact(f, pre, 4)) {
        r.status = STATEOS_SCAN_UNRECOGNIZED;
        r.error  = "file is too small to be a state file";
        return r;
    }
    const uint32_t magic = get_u32(pre);
    if (magic == STATEOS_LEGACY_SEQ_MAGIC) {
        r.status = STATEOS_SCAN_LEGACY;
        r.error  = "legacy/unkeyed state file (headerless llama state-seq format, no State-OS header): refused, fail closed; re-save it with this build";
        return r;
    }
    if (magic != STATEOS_MAGIC) {
        r.status = STATEOS_SCAN_UNRECOGNIZED;
        r.error  = "unrecognized state file (neither a State-OS container nor a llama state file)";
        return r;
    }
    if (r.file_size < 12 || !read_exact(f, pre + 4, 8)) {
        r.status = STATEOS_SCAN_CORRUPT;
        r.error  = "truncated State-OS preamble";
        return r;
    }
    r.version = get_u32(pre + 4);
    if (r.version != STATEOS_CONTAINER_VERSION) {
        r.status = STATEOS_SCAN_UNSUPPORTED;
        r.error  = "unsupported State-OS container version " + std::to_string(r.version) + " (this build reads " +
                   std::to_string(STATEOS_CONTAINER_VERSION) + ")";
        return r;
    }
    const uint32_t header_len = get_u32(pre + 8);
    if (header_len > STATEOS_MAX_HEADER_BYTES || 12 + (uint64_t) header_len > r.file_size) {
        r.status = STATEOS_SCAN_CORRUPT;
        r.error  = "State-OS header length is out of range (truncated or corrupt file)";
        return r;
    }
    r.header_text.resize(header_len);
    if (header_len > 0 && !read_exact(f, r.header_text.data(), header_len)) {
        r.status = STATEOS_SCAN_CORRUPT;
        r.error  = "truncated State-OS header";
        return r;
    }

    uint64_t pos = 12 + (uint64_t) header_len;
    std::set<uint32_t> seen;
    while (true) {
        if (pos + 16 > r.file_size) {
            r.status = STATEOS_SCAN_CORRUPT;
            r.error  = "truncated section table (no END section before EOF)";
            return r;
        }
        f.seekg((std::streamoff) pos, std::ios::beg);
        uint8_t sh[16];
        if (!f.good() || !read_exact(f, sh, sizeof(sh))) {
            r.status = STATEOS_SCAN_CORRUPT;
            r.error  = "cannot read section header";
            return r;
        }
        stateos_section s;
        s.tag    = get_u32(sh);
        s.offset = pos + 16;
        s.size   = get_u64(sh + 8);
        if (s.size > r.file_size - s.offset) {
            r.status = STATEOS_SCAN_CORRUPT;
            r.error  = "section '" + stateos_tag_name(s.tag) + "' is truncated: needs " + std::to_string(s.size) +
                       " bytes, file has " + std::to_string(r.file_size - s.offset);
            return r;
        }
        if (s.tag == STATEOS_TAG_END) {
            if (s.size != 0 || s.offset != r.file_size) {
                r.status = STATEOS_SCAN_CORRUPT;
                r.error  = "END section is not the last 16 bytes of the file";
                return r;
            }
            break;
        }
        if (!seen.insert(s.tag).second) {
            r.status = STATEOS_SCAN_CORRUPT;
            r.error  = "duplicate section '" + stateos_tag_name(s.tag) + "'";
            return r;
        }
        if (r.sections.size() >= STATEOS_MAX_SECTIONS) {
            r.status = STATEOS_SCAN_CORRUPT;
            r.error  = "more than " + std::to_string(STATEOS_MAX_SECTIONS) + " sections";
            return r;
        }
        r.sections.push_back(s);
        pos = s.offset + s.size;
    }
    r.status = STATEOS_SCAN_OK;
    return r;
}

bool stateos_read_range(const std::string & path, uint64_t offset, uint64_t size, std::vector<uint8_t> & out, std::string * err) {
    std::ifstream f(stateos_path(path), std::ios::binary);
    if (!f.is_open()) {
        set_err(err, "cannot open state file");
        return false;
    }
    f.seekg((std::streamoff) offset, std::ios::beg);
    out.resize((size_t) size);
    if (!f.good() || (size > 0 && !read_exact(f, out.data(), (size_t) size))) {
        set_err(err, "short read at offset " + std::to_string(offset));
        return false;
    }
    return true;
}

stateos_section_check stateos_check_sections(const stateos_scan_result & scan, size_t n_ctx_slot) {
    stateos_section_check c;
    const stateos_section * toks = scan.find(STATEOS_TAG_TOKS);
    const stateos_section * target = scan.find(STATEOS_TAG_MAIN);
    if (toks == nullptr || target == nullptr) {
        c.field = toks == nullptr ? "section:TOKS" : "section:MAIN";
        c.error = "a required section is missing";
        return c;
    }
    if (toks->size % 4 != 0) {
        c.field = "section:TOKS";
        c.error = "token section size is not a multiple of 4";
        return c;
    }
    c.n_tokens = (size_t) (toks->size / 4);
    if (c.n_tokens > n_ctx_slot) {
        c.field = "n_tokens";
        c.error = std::to_string(c.n_tokens) + " tokens exceed the slot context of " + std::to_string(n_ctx_slot);
        return c;
    }
    if (target->size == 0) {
        // the writer never produces this (a target state holds at least its cell count); 0 is also the loader's
        // failure value, so it must never reach the load
        c.field = "section:MAIN";
        c.error = "empty target state";
        return c;
    }
    c.empty = c.n_tokens == 0;
    c.ok    = true;
    return c;
}

bool stateos_replace_file(const std::string & src, const std::string & dst, std::string * err) {
#if defined(_WIN32)
    const std::wstring wsrc = stateos_path(src).wstring();
    const std::wstring wdst = stateos_path(dst).wstring();
    DWORD last = 0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (MoveFileExW(wsrc.c_str(), wdst.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return true;
        }
        last = GetLastError();
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
    }
    set_err(err, "MoveFileExW failed (Win32 error " + std::to_string((unsigned long) last) + ")");
    return false;
#else
    std::error_code ec;
    std::filesystem::rename(stateos_path(src), stateos_path(dst), ec);
    if (ec) {
        set_err(err, ec.message());
        return false;
    }
    return true;
#endif
}

bool stateos_write_bytes(std::FILE * f, const void * data, size_t size) {
    return size == 0 || std::fwrite(data, 1, size, f) == size;
}

bool stateos_write_preamble(std::FILE * f, const std::string & header_text) {
    std::vector<uint8_t> b;
    put_u32(b, STATEOS_MAGIC);
    put_u32(b, STATEOS_CONTAINER_VERSION);
    put_u32(b, (uint32_t) header_text.size());
    return stateos_write_bytes(f, b.data(), b.size()) && stateos_write_bytes(f, header_text.data(), header_text.size());
}

bool stateos_write_section_header(std::FILE * f, uint32_t tag, uint64_t size) {
    std::vector<uint8_t> b;
    put_u32(b, tag);
    put_u32(b, 0);
    put_u64(b, size);
    return stateos_write_bytes(f, b.data(), b.size());
}

// ---- checkpoints --------------------------------------------------------------------------------

void stateos_encode_checkpoints(const std::vector<stateos_checkpoint_rec> & in, std::vector<uint8_t> & out) {
    out.clear();
    put_u32(out, STATEOS_CKPT_VERSION);
    put_u64(out, in.size());
    for (const auto & c : in) {
        put_u32(out, (uint32_t) c.pos_min);
        put_u32(out, (uint32_t) c.pos_max);
        put_u32(out, (uint32_t) c.pos_min_prompt);
        put_u32(out, (uint32_t) c.pos_max_prompt);
        put_u64(out, (uint64_t) c.n_tokens);
        put_u64(out, c.data.size());
        out.insert(out.end(), c.data.begin(), c.data.end());
    }
}

bool stateos_decode_checkpoints(const uint8_t * data, size_t size, std::vector<stateos_checkpoint_rec> & out, std::string * err) {
    out.clear();
    size_t pos = 0;
    auto need = [&](size_t n) { return n <= size - pos; };
    if (size < 12) {
        set_err(err, "checkpoint section is truncated");
        return false;
    }
    const uint32_t version = get_u32(data);
    const uint64_t count   = get_u64(data + 4);
    pos = 12;
    if (version != STATEOS_CKPT_VERSION) {
        set_err(err, "unsupported checkpoint section version " + std::to_string(version));
        return false;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (!need(32)) {
            set_err(err, "checkpoint " + std::to_string(i) + " header is truncated");
            return false;
        }
        stateos_checkpoint_rec c;
        c.pos_min        = (int32_t) get_u32(data + pos);
        c.pos_max        = (int32_t) get_u32(data + pos + 4);
        c.pos_min_prompt = (int32_t) get_u32(data + pos + 8);
        c.pos_max_prompt = (int32_t) get_u32(data + pos + 12);
        c.n_tokens       = (int64_t) get_u64(data + pos + 16);
        const uint64_t len = get_u64(data + pos + 24);
        pos += 32;
        if (len > size - pos) {
            set_err(err, "checkpoint " + std::to_string(i) + " data is truncated");
            return false;
        }
        c.data.assign(data + pos, data + pos + len);
        pos += (size_t) len;
        out.push_back(std::move(c));
    }
    if (pos != size) {
        set_err(err, "checkpoint section has trailing bytes");
        return false;
    }
    return true;
}

bool stateos_checkpoints_sane(const std::vector<stateos_checkpoint_rec> & recs, std::string * err) {
    for (size_t i = 0; i < recs.size(); ++i) {
        const auto & c = recs[i];
        const bool ok = c.pos_min >= 0 && c.pos_min <= c.pos_max &&
                        c.pos_min_prompt <= c.pos_max_prompt && c.pos_max_prompt < INT32_MAX &&
                        c.n_tokens >= 0;
        if (!ok) {
            set_err(err, "checkpoint " + std::to_string(i) + " has inconsistent positions");
            return false;
        }
    }
    return true;
}

// ---- companion --------------------------------------------------------------------------------------

void stateos_encode_companion_prefix(const std::string & subheader, std::vector<uint8_t> & out) {
    out.clear();
    put_u32(out, (uint32_t) subheader.size());
    out.insert(out.end(), subheader.begin(), subheader.end());
}

bool stateos_read_companion(const std::string & path, const stateos_section & comp, std::string & subheader,
                            uint64_t & state_offset, uint64_t & state_size, std::string * err) {
    std::vector<uint8_t> len_bytes;
    if (comp.size < 4 || !stateos_read_range(path, comp.offset, 4, len_bytes, err)) {
        set_err(err, "companion section is truncated");
        return false;
    }
    const uint32_t len = get_u32(len_bytes.data());
    if (len > STATEOS_MAX_HEADER_BYTES || 4 + (uint64_t) len > comp.size) {
        set_err(err, "companion sub-header length is out of range");
        return false;
    }
    std::vector<uint8_t> text;
    if (!stateos_read_range(path, comp.offset + 4, len, text, err)) {
        return false;
    }
    subheader.assign(text.begin(), text.end());
    state_offset = comp.offset + 4 + len;
    state_size   = comp.size - 4 - len;
    return true;
}
