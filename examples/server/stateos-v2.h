#pragma once

// LONGSPEAR State-OS v2 (SV2-E1): env flags, the divergence-telemetry vocabulary and the
// divergence classifier.
//
// Every behaviour added by SV2-E1 sits behind one of three env flags, read once per process:
//   LONGSPEAR_STATEOS_DIV_LOG=1        [stateos-div] telemetry lines on stderr
//   LONGSPEAR_STATEOS_TAIL_SNAPSHOT=1  the tail snapshot at release (C1)
//   LONGSPEAR_STATEOS_TAIL_XCHECK=1    diagnostic cross-check after a tail restore
// With all three unset the server behaves exactly like the base build.
//
// This header is pure (no llama or server dependencies) so tests/ can include it directly.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

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
