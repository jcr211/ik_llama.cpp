#pragma once

// LONGSPEAR State-OS v2 (SV2-E1): env flags, the divergence-telemetry vocabulary, the divergence
// classifier, the tail-snapshot eligibility predicate, the checkpoint-list order/search rules and a
// small SHA-256 (for the tail snapshot's token-prefix hash).
//
// Every behaviour added by SV2-E1 sits behind one of three env flags, read once per process:
//   LONGSPEAR_STATEOS_DIV_LOG=1        [stateos-div] telemetry lines on stderr
//   LONGSPEAR_STATEOS_TAIL_SNAPSHOT=1  the tail snapshot at release (C1)
//   LONGSPEAR_STATEOS_TAIL_XCHECK=1    diagnostic cross-check after a tail restore
// With all three unset the server behaves exactly like the base build.
//
// This header is pure (no llama or server dependencies) so tests/ can include it directly.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

inline bool stateos_env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

inline bool stateos_div_log() {
    static const bool v = stateos_env_flag("LONGSPEAR_STATEOS_DIV_LOG");
    return v;
}

inline bool stateos_tail_snapshot() {
    static const bool v = stateos_env_flag("LONGSPEAR_STATEOS_TAIL_SNAPSHOT");
    return v;
}

inline bool stateos_tail_xcheck() {
    static const bool v = stateos_env_flag("LONGSPEAR_STATEOS_TAIL_XCHECK");
    return v;
}

// Where a context checkpoint came from. Kept in server_prompt_checkpoint (not serialized).
enum stateos_ckpt_origin : uint8_t {
    STATEOS_ORIGIN_UNKNOWN = 0,     // loaded from a slot file
    STATEOS_ORIGIN_TOLERANCE,       // prompt end (n_prompt - ctx_checkpoints_tolerance, or first sample)
    STATEOS_ORIGIN_PROMPT_INTERVAL, // every ctx_checkpoints_interval tokens while processing the prompt
    STATEOS_ORIGIN_GEN_INTERVAL,    // every ctx_checkpoints_interval tokens on the non-speculative sample path
    STATEOS_ORIGIN_RELEASE,         // end of generation, includes the last cached token
    STATEOS_ORIGIN_TAIL,            // C1: built from the speculative shadow, before the final round
};

inline const char * stateos_origin_name(int origin) {
    switch (origin) {
        case STATEOS_ORIGIN_TOLERANCE:       return "tolerance";
        case STATEOS_ORIGIN_PROMPT_INTERVAL: return "prompt-interval";
        case STATEOS_ORIGIN_GEN_INTERVAL:    return "gen-interval";
        case STATEOS_ORIGIN_RELEASE:         return "release";
        case STATEOS_ORIGIN_TAIL:            return "tail";
        default:                             return "unknown";
    }
}

// Kind of the last decode round of a generation.
enum stateos_round_kind : uint8_t {
    STATEOS_ROUND_NONE = 0,  // no generation recorded
    STATEOS_ROUND_DRAFTED,   // verify batch with >= 1 drafted token (n accepted recorded separately)
    STATEOS_ROUND_ROOT_ONLY, // speculation active, but the round decoded the root alone
    STATEOS_ROUND_NON_SPEC,  // speculation off for this slot
};

inline const char * stateos_round_name(int kind) {
    switch (kind) {
        case STATEOS_ROUND_DRAFTED:   return "drafted";
        case STATEOS_ROUND_ROOT_ONLY: return "root-only";
        case STATEOS_ROUND_NON_SPEC:  return "non-spec";
        default:                      return "none";
    }
}

enum stateos_stop_cause : uint8_t {
    STATEOS_STOP_UNKNOWN = 0, // no generation recorded (or already reported)
    STATEOS_STOP_EOG,
    STATEOS_STOP_STRING,
    STATEOS_STOP_N_PREDICT,
    STATEOS_STOP_OTHER,
};

inline const char * stateos_stop_name(int cause) {
    switch (cause) {
        case STATEOS_STOP_EOG:       return "eog";
        case STATEOS_STOP_STRING:    return "stop-string";
        case STATEOS_STOP_N_PREDICT: return "n_predict";
        case STATEOS_STOP_OTHER:     return "other";
        default:                     return "unknown";
    }
}

inline int stateos_stop_cause_of(bool stopped_eos, bool stopped_word, bool stopped_limit) {
    if (stopped_eos)   return STATEOS_STOP_EOG;
    if (stopped_word)  return STATEOS_STOP_STRING;
    if (stopped_limit) return STATEOS_STOP_N_PREDICT;
    return STATEOS_STOP_OTHER;
}

// ---- divergence classifier -------------------------------------------------------------------

// Buckets of tail distance = cached tokens - first divergent cache index (1 = only the last cached
// token differs). Same buckets as the read-only census of the production log.
inline const char * stateos_tail_bucket(int32_t tail_dist) {
    if (tail_dist <= 1)   return "1";
    if (tail_dist <= 5)   return "2-5";
    if (tail_dist <= 64)  return "6-64";
    if (tail_dist <= 512) return "65-512";
    return ">512";
}

struct stateos_div_input {
    int32_t cache_n = 0;               // tokens in the slot cache
    int32_t n_past = 0;                // first divergent cache index D
    bool    cache_tok_is_eog = false;  // the cached token at D is an end-of-generation token
    int     prev_stop = STATEOS_STOP_UNKNOWN;
};

// last-token:eog | last-token:stop-string | last-token:other | interior
inline const char * stateos_classify_divergence(const stateos_div_input & in) {
    if (in.cache_n - in.n_past != 1) {
        return "interior";
    }
    if (in.cache_tok_is_eog) {
        return "last-token:eog";
    }
    if (in.prev_stop == STATEOS_STOP_STRING) {
        return "last-token:stop-string";
    }
    return "last-token:other";
}

// ---- log helpers -----------------------------------------------------------------------------

// Escape a token piece for a single-line, double-quoted log field.
inline std::string stateos_escape_piece(const std::string & s) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    out += "\\x";
                    out += hex[c >> 4];
                    out += hex[c & 0xf];
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

// ---- tail snapshot (C1): eligibility, list order, restore search ----------------------------

struct stateos_tail_input {
    bool    flag = false;              // LONGSPEAR_STATEOS_TAIL_SNAPSHOT
    bool    defrag_on = false;         // defrag_thold >= 0: cell index != position (V1)
    bool    per_step = false;          // PER_STEP spec checkpoints keep no full shadow (V2)
    bool    media = false;             // media chunks before the shadow position: indices != positions
    int32_t shadow_pos = -1;           // position the spec shadow holds, -1 = none
    int32_t last_ckpt_pos_max = -1;    // pos_max of the newest checkpoint in the list, -1 = empty list
    int32_t cache_pos_max = -1;        // last cached position of the sequence
};

struct stateos_tail_verdict {
    bool         eligible;
    const char * cause;
};

// Eligible iff the flag is set, shadow_pos >= 64, shadow_pos is newer than the newest checkpoint,
// and shadow_pos <= cache_pos_max - 2 (the final round accepted >= 1 draft, so the snapshot can
// serve a divergence at the last cached token).
inline stateos_tail_verdict stateos_tail_eligibility(const stateos_tail_input & in) {
    if (!in.flag)                                return { false, "flag-off" };
    if (in.defrag_on)                            return { false, "defrag" };
    if (in.per_step)                             return { false, "per-step" };
    if (in.shadow_pos < 0)                       return { false, "no-shadow" };
    if (in.media)                                return { false, "media" };
    if (in.shadow_pos < 64)                      return { false, "short" };
    if (in.shadow_pos <= in.last_ckpt_pos_max)   return { false, "not-newer" };
    if (in.shadow_pos > in.cache_pos_max - 2)    return { false, "no-accepted-draft" };
    return { true, "eligible" };
}

// The checkpoint list must stay ascending in pos_max: apply_checkpoint's reverse search takes the
// first usable entry from the back (M4).
template <typename List>
bool stateos_is_ascending(const List & list) {
    bool first = true;
    int64_t prev = 0;
    for (const auto & cur : list) {
        if (!first && (int64_t) cur.pos_max < prev) {
            return false;
        }
        prev = cur.pos_max;
        first = false;
    }
    return true;
}

// apply_checkpoint's search: the newest entry with pos_max < thold (thold = D - 1 for a divergence
// at position D; DSV4/openPangu pass pos_next) that `usable` accepts. The server's `usable` accepts
// every non-tail entry, so with no tail snapshots in the list this is the base search.
template <typename List, typename Usable>
auto stateos_find_restore(List & list, int64_t thold, Usable && usable) -> decltype(list.rbegin()) {
    auto it = list.rbegin();
    for (; it != list.rend(); ++it) {
        if ((int64_t) it->pos_max < thold && usable(*it)) {
            break;
        }
    }
    return it;
}

// ---- tail crosscheck (LONGSPEAR_STATEOS_TAIL_XCHECK=1): read a PARTIAL_ONLY payload --------------

struct stateos_row_view {
    int32_t  type     = -1;
    uint64_t row_size = 0;
    uint32_t n_rows   = 0;
    size_t   offset   = 0; // byte offset of the rows in the payload
};

struct stateos_partial_view {
    uint32_t cell_count = 0;
    int32_t  pos_max    = -1;
    std::vector<stateos_row_view> layers;
};

// Walks a single-sequence PARTIAL_ONLY payload as llama_data_write::write_kv_cache emits it when no
// layer is SWA-compacted and the arch keeps no position-indexed side state (qwen4exp): u32 cell_count,
// 8 B/cell metadata, u32 v_state, u32 n_layer, empty K (and V) headers, u32 qnext flag, per-layer row
// blocks, u32 DSA marker. Returns false on anything else (then the caller reports, never guesses).
inline bool stateos_parse_partial(const uint8_t * data, size_t size, stateos_partial_view & out) {
    size_t off = 0;
    auto take = [&](void * dst, size_t n) {
        if (size - off < n) {
            return false;
        }
        std::memcpy(dst, data + off, n);
        off += n;
        return true;
    };
    out = stateos_partial_view();
    if (!take(&out.cell_count, 4) || (size_t) out.cell_count > (size - off) / 8) {
        return false;
    }
    for (uint32_t i = 0; i < out.cell_count; ++i) {
        int32_t  pos = 0;
        uint32_t n_seq = 0;
        if (!take(&pos, 4) || !take(&n_seq, 4) || n_seq != 0) {
            return false;
        }
        out.pos_max = pos > out.pos_max ? pos : out.pos_max;
    }
    uint32_t v_state = 0, n_layer = 0;
    if (!take(&v_state, 4) || !take(&n_layer, 4) || v_state > 2 || n_layer > 4096) {
        return false;
    }
    for (uint32_t il = 0; il < n_layer; ++il) {
        int32_t  type = 0;
        uint64_t row = 0;
        if (!take(&type, 4) || !take(&row, 8) || type != -1 || row != 0) {
            return false; // a K row block: compacted layer or full state
        }
    }
    for (uint32_t il = 0; v_state != 2 && il < n_layer; ++il) {
        int32_t  type = 0;
        uint64_t row = 0;
        uint32_t el = 0, embd = 0;
        const bool ok = v_state == 0 ? take(&type, 4) && take(&row, 8) && row == 0
                                     : take(&type, 4) && take(&el, 4) && take(&embd, 4) && el == 0 && embd == 0;
        if (!ok || type != -1) {
            return false;
        }
    }
    uint32_t qnext = 0;
    if (!take(&qnext, 4) || qnext > 1) {
        return false;
    }
    for (uint32_t il = 0; qnext == 1 && il < n_layer; ++il) {
        stateos_row_view v;
        if (!take(&v.type, 4) || !take(&v.row_size, 8) || !take(&v.n_rows, 4)) {
            return false;
        }
        if (v.n_rows > 0 && (v.row_size == 0 || v.row_size > (size - off) / v.n_rows)) {
            return false;
        }
        v.offset = off;
        off += (size_t) v.n_rows * (size_t) v.row_size;
        out.layers.push_back(v);
    }
    uint32_t dsa = 0;
    return take(&dsa, 4) && off == size;
}

struct stateos_row_diff {
    size_t n          = 0;    // compared elements (f32) or bytes (other types)
    size_t n_bitequal = 0;
    double rel_l2     = -1.0; // ||a - b|| / ||b|| for f32 rows, -1 otherwise
};

// Compare one layer's rows; b is the reference (the flag-off path). type 0 = GGML_TYPE_F32.
inline stateos_row_diff stateos_compare_rows(const uint8_t * a, const uint8_t * b, size_t bytes, int32_t type) {
    stateos_row_diff d;
    if (type != 0 || bytes % 4 != 0) {
        d.n = bytes;
        for (size_t i = 0; i < bytes; ++i) {
            d.n_bitequal += a[i] == b[i];
        }
        return d;
    }
    d.n = bytes / 4;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < d.n; ++i) {
        float x = 0.0f, y = 0.0f;
        std::memcpy(&x, a + 4*i, 4);
        std::memcpy(&y, b + 4*i, 4);
        d.n_bitequal += std::memcmp(a + 4*i, b + 4*i, 4) == 0;
        num += ((double) x - (double) y) * ((double) x - (double) y);
        den += (double) y * (double) y;
    }
    d.rel_l2 = den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
    return d;
}

// ---- SHA-256 (FIPS 180-4) --------------------------------------------------------------------

struct stateos_sha256 {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    uint8_t  buf[64] = {};
    size_t   n_buf = 0;
    uint64_t n_bits = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i + 1] << 16 | (uint32_t) p[4*i + 2] << 8 | (uint32_t) p[4*i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        n_bits += (uint64_t) size * 8;
        while (size > 0) {
            const size_t take = (64 - n_buf) < size ? (64 - n_buf) : size;
            std::memcpy(buf + n_buf, p, take);
            n_buf += take;
            p     += take;
            size  -= take;
            if (n_buf == 64) {
                block(buf);
                n_buf = 0;
            }
        }
    }

    std::array<uint8_t, 32> digest() {
        const uint64_t bits = n_bits;
        const uint8_t pad = 0x80;
        const uint8_t zero = 0x00;
        update(&pad, 1);
        while (n_buf != 56) {
            update(&zero, 1);
        }
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) {
            len[i] = (uint8_t) (bits >> (56 - 8*i));
        }
        update(len, 8);
        std::array<uint8_t, 32> out;
        for (int i = 0; i < 8; ++i) {
            out[4*i]     = (uint8_t) (h[i] >> 24);
            out[4*i + 1] = (uint8_t) (h[i] >> 16);
            out[4*i + 2] = (uint8_t) (h[i] >> 8);
            out[4*i + 3] = (uint8_t) (h[i]);
        }
        return out;
    }
};

inline std::string stateos_hex(const std::array<uint8_t, 32> & d) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t b : d) {
        out += hex[b >> 4];
        out += hex[b & 0xf];
    }
    return out;
}
