// State-OS v1 keyed slot-state container: model-free unit tests for examples/server/stateos-header.*
// (encode/decode round trip, one refusal per hard field, soft warnings, container scan, checkpoint codec).

#include "stateos-header.h"
#include "stateos-model.h"
#include "stateos-props.h"

#include "ggml.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// the hard fields the server writes (server_context::stateos_identity_fields + the two integrity fields)
static stateos_fields server_like_fields() {
    return {
        { STATEOS_HARD, "model_fingerprint_v2", "3f1c0d5e9a" },
        { STATEOS_HARD, "effective_model",      "none" },
        { STATEOS_HARD, "n_ctx",                "196608" },
        { STATEOS_HARD, "cache_type_k",         "q8_0" },
        { STATEOS_HARD, "cache_type_v",         "q8_0" },
        { STATEOS_HARD, "rope",                 "type=40 base=1e+07 scale=1 orig_yarn=262144 ext=0 attn=1 beta_fast=32 beta_slow=1" },
        { STATEOS_HARD, "kv_layout_version",    "stateos-kv/1 llama-seq/4" },
        { STATEOS_HARD, "system_prompt_sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { STATEOS_HARD, "kv_geometry",          "9b2e" },
        { STATEOS_SOFT, "build",                "4321-abc1234" },
        { STATEOS_HARD, "n_tokens",             "4096" },
        { STATEOS_HARD, "token_sha256",         "aa" },
    };
}

static void set_value(stateos_fields & f, const std::string & key, const std::string & value) {
    for (auto & x : f) {
        if (x.key == key) {
            x.value = value;
        }
    }
}

static void test_sha256() {
    CHECK(stateos_sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(stateos_sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(stateos_sha256_hex(two_blocks.data(), two_blocks.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // incremental updates across block boundaries equal the one-shot digest
    std::string million(1000000, 'a');
    stateos_sha256 h;
    for (size_t i = 0; i < million.size(); i += 777) {
        h.update(million.data() + i, std::min<size_t>(777, million.size() - i));
    }
    CHECK(h.final_hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    // token hash is over little-endian int32 ids and is order sensitive
    const int32_t ids[3] = { 1, 2, 3 };
    const int32_t rev[3] = { 3, 2, 1 };
    const uint8_t le[12] = { 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0 };
    CHECK(stateos_token_sha256(ids, 3) == stateos_sha256_hex(le, sizeof(le)));
    CHECK(stateos_token_sha256(ids, 3) != stateos_token_sha256(rev, 3));
    CHECK(stateos_token_sha256(nullptr, 0) == stateos_sha256_hex("", 0));
}

static void test_header_codec() {
    stateos_fields in = server_like_fields();
    in.push_back({ STATEOS_INFO, "saved_unix", "1790000000" });
    in.push_back({ STATEOS_SOFT, "empty_value", "" });
    in.push_back({ STATEOS_INFO, "has.equals-and spaces", "a=b c=d" }); // invalid key: space

    std::string text;
    std::string err;
    CHECK(!stateos_encode_header(in, text, &err));
    CHECK(err.find("invalid field key") != std::string::npos);

    in.pop_back();
    in.push_back({ STATEOS_INFO, "equals.in-value", "a=b c=d" });
    CHECK(stateos_encode_header(in, text, &err));

    stateos_fields out;
    CHECK(stateos_decode_header(text, out, &err));
    CHECK(out.size() == in.size());
    bool same = out.size() == in.size();
    for (size_t i = 0; same && i < in.size(); ++i) {
        same = out[i].cls == in[i].cls && out[i].key == in[i].key && out[i].value == in[i].value;
    }
    CHECK(same);

    // encode rejects what decode could not read back
    CHECK(!stateos_encode_header({ { STATEOS_HARD, "k", "line\nbreak" } }, text, &err));
    CHECK(!stateos_encode_header({ { 'X', "k", "v" } }, text, &err));
    CHECK(!stateos_encode_header({ { STATEOS_HARD, "k", "1" }, { STATEOS_SOFT, "k", "2" } }, text, &err));
    CHECK(!stateos_encode_header({ { STATEOS_HARD, "", "v" } }, text, &err));

    // decode rejects malformed text
    CHECK(!stateos_decode_header("H k=v", out, &err));          // not newline-terminated
    CHECK(!stateos_decode_header("Q k=v\n", out, &err));        // bad class
    CHECK(!stateos_decode_header("H kv\n", out, &err));         // no '='
    CHECK(!stateos_decode_header("H k=1\nH k=2\n", out, &err)); // duplicate
    CHECK(!stateos_decode_header("\n", out, &err));             // empty line
    CHECK(!stateos_decode_header("Hk=v\n", out, &err));         // missing space
    CHECK(stateos_decode_header("", out, &err) && out.empty()); // an empty header is well-formed
}

static void test_verify() {
    const stateos_fields base = server_like_fields();

    // identical: accepted, no warnings
    {
        const stateos_verdict v = stateos_verify(base, base);
        CHECK(v.ok);
        CHECK(v.refused.empty());
        CHECK(v.warnings.empty());
    }

    // every hard field refuses on its own, and the refusal names it
    for (const auto & f : base) {
        if (f.cls != STATEOS_HARD) {
            continue;
        }
        stateos_fields saved = base;
        set_value(saved, f.key, f.value + "-other");
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(!v.ok);
        CHECK(v.refused.size() == 1);
        CHECK(!v.refused.empty() && v.refused.front().key == f.key);
        CHECK(!v.refused.empty() && v.refused.front().saved == f.value + "-other");
        CHECK(!v.refused.empty() && v.refused.front().current == f.value);
        if (v.refused.empty() || v.refused.front().key != f.key) {
            std::fprintf(stderr, "  (hard field under test: %s)\n", f.key.c_str());
        }
    }

    // soft mismatch: accepted with a warning naming the field
    {
        stateos_fields saved = base;
        set_value(saved, "build", "1-deadbee");
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(v.ok);
        CHECK(v.warnings.size() == 1);
        CHECK(!v.warnings.empty() && v.warnings.front().key == "build" && v.warnings.front().saved == "1-deadbee");
    }

    // a hard field the saved header lacks is refused as missing
    {
        stateos_fields saved;
        for (const auto & f : base) {
            if (f.key != "cache_type_v") {
                saved.push_back(f);
            }
        }
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(!v.ok);
        CHECK(!v.refused.empty() && v.refused.front().key == "cache_type_v" && v.refused.front().saved == "<missing>");
    }

    // a hard field this build does not know is refused (fail closed); an unknown soft one only warns
    {
        stateos_fields saved = base;
        saved.push_back({ STATEOS_HARD, "future_field", "x" });
        saved.push_back({ STATEOS_SOFT, "future_soft", "y" });
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(!v.ok);
        CHECK(v.refused.size() == 1 && v.refused.front().key == "future_field");
        CHECK(v.warnings.size() == 1 && v.warnings.front().key == "future_soft");
    }

    // the current server's class wins: a file cannot demote a hard field to soft
    {
        stateos_fields saved = base;
        for (auto & f : saved) {
            if (f.key == "n_ctx") {
                f.cls   = STATEOS_SOFT;
                f.value = "65536";
            }
        }
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(!v.ok);
        CHECK(!v.refused.empty() && v.refused.front().key == "n_ctx");
    }

    // info fields are never compared
    {
        stateos_fields saved = base;
        saved.push_back({ STATEOS_INFO, "saved_unix", "1" });
        stateos_fields current = base;
        current.push_back({ STATEOS_INFO, "saved_unix", "2" });
        const stateos_verdict v = stateos_verify(saved, current);
        CHECK(v.ok && v.warnings.empty());
    }

    // several differences: all reported, the first in the current server's field order
    {
        stateos_fields saved = base;
        set_value(saved, "kv_geometry", "zz");
        set_value(saved, "cache_type_k", "f16");
        const stateos_verdict v = stateos_verify(saved, base);
        CHECK(v.refused.size() == 2);
        CHECK(!v.refused.empty() && v.refused.front().key == "cache_type_k");
    }
}

// ---- container ------------------------------------------------------------------------------------

static std::filesystem::path g_tmp;

static std::string tmp_file(const std::string & name) {
    return stateos_path_utf8(g_tmp / name);
}

static bool write_container(const std::string & path, const std::string & header,
                            const std::vector<std::pair<uint32_t, std::vector<uint8_t>>> & sections, bool with_end) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    bool ok = stateos_write_preamble(f, header);
    for (const auto & s : sections) {
        ok = ok && stateos_write_section_header(f, s.first, s.second.size()) && stateos_write_bytes(f, s.second.data(), s.second.size());
    }
    if (with_end) {
        ok = ok && stateos_write_section_header(f, STATEOS_TAG_END, 0);
    }
    return (std::fclose(f) == 0) && ok;
}

static void write_raw(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(stateos_path(path), std::ios::binary);
    f.write((const char *) bytes.data(), (std::streamsize) bytes.size());
}

static std::vector<uint8_t> read_all(const std::string & path) {
    std::ifstream f(stateos_path(path), std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void test_container() {
    std::string header;
    std::string err;
    CHECK(stateos_encode_header(server_like_fields(), header, &err));

    const std::vector<uint8_t> toks = { 1, 0, 0, 0, 2, 0, 0, 0 };
    const std::vector<uint8_t> main_payload(1000, 0xAB);
    std::vector<uint8_t> ckpt;
    stateos_encode_checkpoints({}, ckpt);

    const std::string good = tmp_file("good.state");
    CHECK(write_container(good, header, { { STATEOS_TAG_TOKS, toks }, { STATEOS_TAG_MAIN, main_payload }, { STATEOS_TAG_CKPT, ckpt } }, true));

    const stateos_scan_result r = stateos_scan_file(good);
    CHECK(r.status == STATEOS_SCAN_OK);
    CHECK(r.version == STATEOS_CONTAINER_VERSION);
    CHECK(r.header_text == header);
    CHECK(r.sections.size() == 3);
    const stateos_section * m = r.find(STATEOS_TAG_MAIN);
    CHECK(m != nullptr && m->size == main_payload.size());
    CHECK(r.find(STATEOS_TAG_COMP) == nullptr);
    if (m != nullptr) {
        std::vector<uint8_t> back;
        CHECK(stateos_read_range(good, m->offset, m->size, back, &err));
        CHECK(back == main_payload);
    }
    const stateos_section * t = r.find(STATEOS_TAG_TOKS);
    CHECK(t != nullptr && t->offset == 12 + header.size() + 16 && t->size == toks.size());

    // not found vs present-but-unreadable (a directory; a file another process holds with no read sharing)
    CHECK(stateos_scan_file(tmp_file("missing.state")).status == STATEOS_SCAN_NOT_FOUND);
    std::filesystem::create_directories(stateos_path(tmp_file("a-directory.state")));
    CHECK(stateos_scan_file(tmp_file("a-directory.state")).status == STATEOS_SCAN_UNREADABLE);
#if defined(_WIN32)
    {
        int fd = -1;
        const std::wstring wpath = stateos_path(good).wstring();
        if (_wsopen_s(&fd, wpath.c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRW, 0) == 0) {
            const stateos_scan_result locked = stateos_scan_file(good);
            CHECK(locked.status == STATEOS_SCAN_UNREADABLE);
            CHECK(locked.error.find("cannot open") != std::string::npos);
            _close(fd);
        } else {
            CHECK(false); // could not take the exclusive handle the test needs
        }
        CHECK(stateos_scan_file(good).status == STATEOS_SCAN_OK); // readable again once released
    }
#endif

    // a headerless llama state-seq file (magic 'ggsq', version 4, token count ...) is legacy/unkeyed
    const std::string legacy = tmp_file("legacy.state");
    write_raw(legacy, { 0x71, 0x73, 0x67, 0x67, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4 });
    const stateos_scan_result rl = stateos_scan_file(legacy);
    CHECK(rl.status == STATEOS_SCAN_LEGACY);
    CHECK(rl.error.find("legacy/unkeyed") != std::string::npos);

    // unrecognized magic
    const std::string junk = tmp_file("junk.state");
    write_raw(junk, { 'J', 'U', 'N', 'K', 0, 0, 0, 0 });
    CHECK(stateos_scan_file(junk).status == STATEOS_SCAN_UNRECOGNIZED);

    // another container version
    std::vector<uint8_t> bytes = read_all(good);
    bytes[4] = 2;
    const std::string v2 = tmp_file("v2.state");
    write_raw(v2, bytes);
    CHECK(stateos_scan_file(v2).status == STATEOS_SCAN_UNSUPPORTED);

    // truncation anywhere is detected before any payload is used
    const std::vector<uint8_t> full = read_all(good);
    const size_t cuts[] = { 3, 11, 12 + header.size() / 2, 12 + header.size() + 10, full.size() - 500, full.size() - 1 };
    for (const size_t cut : cuts) {
        const std::string p = tmp_file("cut.state");
        write_raw(p, std::vector<uint8_t>(full.begin(), full.begin() + cut));
        const stateos_scan_status st = stateos_scan_file(p).status;
        CHECK(st == STATEOS_SCAN_CORRUPT || (cut < 4 && st == STATEOS_SCAN_UNRECOGNIZED));
    }

    // missing END, trailing garbage, oversized header length and duplicate sections are corrupt
    const std::string no_end = tmp_file("noend.state");
    CHECK(write_container(no_end, header, { { STATEOS_TAG_TOKS, toks } }, false));
    CHECK(stateos_scan_file(no_end).status == STATEOS_SCAN_CORRUPT);

    std::vector<uint8_t> trailing = full;
    trailing.push_back(0);
    const std::string tr = tmp_file("trailing.state");
    write_raw(tr, trailing);
    CHECK(stateos_scan_file(tr).status == STATEOS_SCAN_CORRUPT);

    std::vector<uint8_t> big_header = full;
    big_header[8] = 0xFF; big_header[9] = 0xFF; big_header[10] = 0xFF; big_header[11] = 0x7F;
    const std::string bh = tmp_file("bigheader.state");
    write_raw(bh, big_header);
    CHECK(stateos_scan_file(bh).status == STATEOS_SCAN_CORRUPT);

    const std::string dup = tmp_file("dup.state");
    CHECK(write_container(dup, header, { { STATEOS_TAG_TOKS, toks }, { STATEOS_TAG_TOKS, toks } }, true));
    CHECK(stateos_scan_file(dup).status == STATEOS_SCAN_CORRUPT);

    // a section whose declared size runs past EOF is named in the error
    if (m == nullptr) {
        return;
    }
    std::vector<uint8_t> overlong = full;
    const size_t main_size_at = (size_t) m->offset - 8;
    overlong[main_size_at + 4] = 0x10; // bump the MAIN size far past EOF
    const std::string ol = tmp_file("overlong.state");
    write_raw(ol, overlong);
    const stateos_scan_result rol = stateos_scan_file(ol);
    CHECK(rol.status == STATEOS_SCAN_CORRUPT);
    CHECK(rol.error.find("MAIN") != std::string::npos);
}

static void test_checkpoints() {
    std::vector<stateos_checkpoint_rec> in(3);
    for (int i = 0; i < 3; ++i) {
        in[i].pos_min        = 100 * i;
        in[i].pos_max        = 100 * i + 99;
        in[i].pos_min_prompt = 100 * i + 1;
        in[i].pos_max_prompt = 100 * i + 98;
        in[i].n_tokens       = 1000000000000LL + i;
        in[i].data.assign((size_t) (i * 37), (uint8_t) (i + 1));
    }
    std::vector<uint8_t> bytes;
    stateos_encode_checkpoints(in, bytes);

    std::vector<stateos_checkpoint_rec> out;
    std::string err;
    CHECK(stateos_decode_checkpoints(bytes.data(), bytes.size(), out, &err));
    CHECK(out.size() == in.size());
    bool same = out.size() == in.size();
    for (size_t i = 0; same && i < in.size(); ++i) {
        same = out[i].pos_min == in[i].pos_min && out[i].pos_max == in[i].pos_max &&
               out[i].pos_min_prompt == in[i].pos_min_prompt && out[i].pos_max_prompt == in[i].pos_max_prompt &&
               out[i].n_tokens == in[i].n_tokens && out[i].data == in[i].data;
    }
    CHECK(same);

    // every truncation point fails cleanly
    bool all_fail = true;
    for (size_t cut = 0; cut < bytes.size(); ++cut) {
        std::vector<stateos_checkpoint_rec> tmp;
        if (stateos_decode_checkpoints(bytes.data(), cut, tmp, &err)) {
            all_fail = false;
        }
    }
    CHECK(all_fail);

    // trailing bytes and an unknown version are refused
    std::vector<uint8_t> extra = bytes;
    extra.push_back(0);
    CHECK(!stateos_decode_checkpoints(extra.data(), extra.size(), out, &err));
    std::vector<uint8_t> v9 = bytes;
    v9[0] = 9;
    CHECK(!stateos_decode_checkpoints(v9.data(), v9.size(), out, &err));

    // empty list round-trips
    stateos_encode_checkpoints({}, bytes);
    CHECK(stateos_decode_checkpoints(bytes.data(), bytes.size(), out, &err) && out.empty());
}

static void test_companion() {
    std::string err;
    std::string sub;
    CHECK(stateos_encode_header({ { STATEOS_HARD, "companion_layout_version", "stateos-comp/1 llama-seq/4" },
                                  { STATEOS_INFO, "mtp_warmed_heads", "1" } }, sub, &err));
    std::vector<uint8_t> payload;
    stateos_encode_companion_prefix(sub, payload);
    const std::vector<uint8_t> state(333, 0x5A);
    payload.insert(payload.end(), state.begin(), state.end());

    std::string header;
    CHECK(stateos_encode_header(server_like_fields(), header, &err));
    const std::string path = tmp_file("comp.state");
    CHECK(write_container(path, header, { { STATEOS_TAG_TOKS, {} }, { STATEOS_TAG_MAIN, { 1, 2, 3 } }, { STATEOS_TAG_COMP, payload } }, true));

    const stateos_scan_result r = stateos_scan_file(path);
    CHECK(r.status == STATEOS_SCAN_OK);
    const stateos_section * c = r.find(STATEOS_TAG_COMP);
    CHECK(c != nullptr);
    if (c != nullptr) {
        std::string sub_back;
        uint64_t off = 0;
        uint64_t size = 0;
        CHECK(stateos_read_companion(path, *c, sub_back, off, size, &err));
        CHECK(sub_back == sub);
        CHECK(size == state.size());
        std::vector<uint8_t> back;
        CHECK(stateos_read_range(path, off, size, back, &err) && back == state);

        // a sub-header length past the section is refused, not trusted
        stateos_section bad = *c;
        bad.size = 3;
        CHECK(!stateos_read_companion(path, bad, sub_back, off, size, &err));
    }
}

// GET /props "stateos": the appliance enables model-state rewind only when this object is present with version >= 1
static void test_props_capability() {
    for (const bool companion : { true, false }) {
        const nlohmann::ordered_json caps = stateos_props_entry(true, true, companion);
        nlohmann::ordered_json props = { { "n_ctx", 196608 } };
        if (!caps.is_null()) {
            props["stateos"] = caps; // as handle_props embeds it
        }
        const nlohmann::ordered_json back = nlohmann::ordered_json::parse(props.dump());
        CHECK(back.contains("stateos") && back["stateos"].is_object());
        CHECK(back["stateos"]["version"].is_number_integer() && back["stateos"]["version"].get<int>() >= 1);
        CHECK(back["stateos"]["version"].get<int>() == (int) STATEOS_CONTAINER_VERSION);
        CHECK(back["stateos"]["keyed_header"].is_boolean() && back["stateos"]["keyed_header"].get<bool>());
        CHECK(back["stateos"]["companion"].is_boolean() && back["stateos"]["companion"].get<bool>() == companion);
        CHECK(back["stateos"].size() == 3);
    }
    // not advertised without the /slots routes (no --slot-save-path) or without a model identity
    CHECK(stateos_props_entry(false, true, true).is_null());
    CHECK(stateos_props_entry(true, false, true).is_null());
    CHECK(stateos_props_entry(false, false, false).is_null());
}

// runtime adapters/overrides are part of the identity: a state from one adapter set is refused on another
static void test_effective_model() {
    CHECK(stateos_effective_model_value({}) == "none");
    const std::string a = stateos_effective_model_value({ "lora path=a.gguf scale=1" });
    CHECK(a.size() == 64 && a != "none");
    CHECK(stateos_effective_model_value({ "lora path=a.gguf scale=1" }) == a);             // stable
    CHECK(stateos_effective_model_value({ "lora path=a.gguf scale=0.5" }) != a);           // scale matters
    CHECK(stateos_effective_model_value({ "lora path=b.gguf scale=1" }) != a);             // adapter matters
    const std::string ab = stateos_effective_model_value({ "lora path=a.gguf scale=1", "cvec path=c.gguf scale=1 layers=0..9" });
    const std::string ba = stateos_effective_model_value({ "cvec path=c.gguf scale=1 layers=0..9", "lora path=a.gguf scale=1" });
    CHECK(ab != a && ab != ba);                                                            // order is part of it
    // no line-joining ambiguity: two items are not one item containing a separator
    CHECK(stateos_effective_model_value({ "x", "y" }) != stateos_effective_model_value({ "x\ny" }));
    CHECK(stateos_effective_model_value({ "xy" }) != stateos_effective_model_value({ "x", "y" }));

    // the verify path refuses a different adapter set by name
    stateos_fields saved   = server_like_fields();
    stateos_fields current = server_like_fields();
    set_value(current, "effective_model", a);
    const stateos_verdict v = stateos_verify(saved, current);
    CHECK(!v.ok && !v.refused.empty() && v.refused.front().key == "effective_model");
}

// section-level restore checks: MAIN must not be empty (0 is the loader's failure value), TOKS is bounded first
static void test_section_checks() {
    stateos_scan_result r;
    r.status   = STATEOS_SCAN_OK;
    r.sections = { { STATEOS_TAG_TOKS, 100, 8 }, { STATEOS_TAG_MAIN, 200, 50 } };
    stateos_section_check c = stateos_check_sections(r, 196608);
    CHECK(c.ok && c.n_tokens == 2 && !c.empty);

    r.sections[1].size = 0;
    c = stateos_check_sections(r, 196608);
    CHECK(!c.ok && c.field == "section:MAIN" && c.type == "state_corrupt");

    r.sections[1].size = 50;
    r.sections[0].size = 0; // a state saved from an empty slot
    c = stateos_check_sections(r, 196608);
    CHECK(c.ok && c.empty && c.n_tokens == 0);

    r.sections[0].size = 6;
    c = stateos_check_sections(r, 196608);
    CHECK(!c.ok && c.field == "section:TOKS");

    r.sections[0].size = 8;
    c = stateos_check_sections(r, 1);
    CHECK(!c.ok && c.field == "n_tokens" && c.type == "state_refused"); // valid file, larger context: refused

    r.sections = { { STATEOS_TAG_TOKS, 100, 8 } };
    c = stateos_check_sections(r, 196608);
    CHECK(!c.ok && c.field == "section:MAIN");
    r.sections = { { STATEOS_TAG_MAIN, 100, 8 } };
    c = stateos_check_sections(r, 196608);
    CHECK(!c.ok && c.field == "section:TOKS");

    // vocabulary range (restore refuses, /list withholds the prompt)
    {
        const int32_t ok_ids[3]  = { 0, 5, 9 };
        const int32_t bad_hi[3]  = { 0, 10, 1 };
        const int32_t bad_neg[2] = { 3, -1 };
        size_t bad = 99;
        CHECK(stateos_tokens_in_vocab(ok_ids, 3, 10, &bad) && bad == 99);
        CHECK(!stateos_tokens_in_vocab(bad_hi, 3, 10, &bad) && bad == 1);
        CHECK(!stateos_tokens_in_vocab(bad_neg, 2, 10, &bad) && bad == 1);
        CHECK(stateos_tokens_in_vocab(nullptr, 0, 10, nullptr));
    }

    // KV <-> tokens: tokens need at least one KV cell (save refuses, restore fails and clears the slot)
    CHECK(stateos_kv_consistent(0, -1));      // empty slot
    CHECK(stateos_kv_consistent(4096, 4095)); // the normal case
    CHECK(stateos_kv_consistent(4096, 4094)); // off-by-one stays report-only (kv_pos_max in the responses)
    CHECK(!stateos_kv_consistent(4096, -1));  // tokens over an empty KV
    CHECK(!stateos_kv_consistent(1, -1));

    // checkpoint positions must be ordered and non-negative
    std::vector<stateos_checkpoint_rec> recs(2);
    recs[0].pos_min = 0;  recs[0].pos_max = 99;  recs[0].pos_min_prompt = 0;  recs[0].pos_max_prompt = 99;
    recs[1].pos_min = 100; recs[1].pos_max = 199; recs[1].pos_min_prompt = 100; recs[1].pos_max_prompt = 199;
    std::string err;
    CHECK(stateos_checkpoints_sane(recs, &err));
    recs[1].pos_min = 300;
    CHECK(!stateos_checkpoints_sane(recs, &err));
    recs[1].pos_min = -1;
    CHECK(!stateos_checkpoints_sane(recs, &err));
    recs[1].pos_min = 100;
    recs[1].pos_max_prompt = INT32_MAX; // pos_max_prompt + 1 would overflow
    CHECK(!stateos_checkpoints_sane(recs, &err));

    // CKPT budget before reading: 12 framing + records x (32 + state bytes)
    CHECK(stateos_ckpt_within_budget(12, 0, 100));                 // an empty list
    CHECK(stateos_ckpt_within_budget(12 + 2 * (32 + 100), 2, 100));
    CHECK(!stateos_ckpt_within_budget(12 + 2 * (32 + 100) + 1, 2, 100));
    CHECK(!stateos_ckpt_within_budget(1ull << 40, 32, 112u << 20)); // a crafted terabyte section is never read
    CHECK(stateos_ckpt_within_budget(UINT64_MAX, UINT64_MAX, UINT64_MAX)); // no overflow in the bound itself
    // the per-record bound does not depend on the target slot's current length (review P1): a checkpoint taken in a
    // 4K conversation must fit when measured against a slot that now holds 10 cells, or none
    {
        const uint64_t fixed = 112u << 20;                       // recurrent rows
        const uint64_t ckpt_4k = fixed + STATEOS_CELL_META_BYTES * 4096;
        const uint64_t partial_short = fixed + STATEOS_CELL_META_BYTES * 10;
        const uint64_t partial_empty = fixed;
        CHECK(stateos_ckpt_record_bound(partial_short, 196608, false, 0) >= ckpt_4k);
        CHECK(stateos_ckpt_record_bound(partial_empty, 196608, false, 0) >= ckpt_4k);
        CHECK(stateos_ckpt_record_bound(partial_empty, 196608, false, 0) >= fixed + STATEOS_CELL_META_BYTES * 196608);
        CHECK(stateos_ckpt_record_bound(partial_short, 196608, true, 5000) == 5000); // compacted: the MAIN size
        CHECK(stateos_ckpt_record_bound(UINT64_MAX - 1, UINT64_MAX, false, 0) == UINT64_MAX); // saturates
        std::vector<stateos_checkpoint_rec> long_ckpt(32);
        for (auto & c : long_ckpt) {
            c.data.resize(1024 + STATEOS_CELL_META_BYTES * 4096);
        }
        const uint64_t bound = stateos_ckpt_record_bound(1024 + STATEOS_CELL_META_BYTES * 10, 196608, false, 0);
        CHECK(stateos_checkpoints_fit(long_ckpt, bound, &err));
        uint64_t section = 12;
        for (const auto & c : long_ckpt) {
            section += 32 + c.data.size();
        }
        CHECK(stateos_ckpt_within_budget(section, 32, bound)); // a full list of long checkpoints is not skipped
    }

    // after decode: every record fits a partial state of this context
    std::vector<stateos_checkpoint_rec> fit(2);
    fit[0].data.assign(100, 1);
    fit[1].data.assign(100, 2);
    CHECK(stateos_checkpoints_fit(fit, 100, &err));
    fit[1].data.push_back(3);
    CHECK(!stateos_checkpoints_fit(fit, 100, &err) && err.find("checkpoint 1") != std::string::npos);
}

// the save's commit point: replace the previous state without deleting it first
static void test_replace_file() {
    const std::string a = tmp_file("replace-a.state");
    const std::string b = tmp_file("replace-b.state");
    write_raw(a, { 'o', 'l', 'd' });
    write_raw(b, { 'n', 'e', 'w', '!' });
    std::string err;
    CHECK(stateos_replace_file(b, a, &err));
    CHECK(read_all(a) == std::vector<uint8_t>({ 'n', 'e', 'w', '!' }));
    CHECK(!std::filesystem::exists(stateos_path(b)));
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!stateos_replace_file(tmp_file("replace-missing.state"), a, &err));
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(waited < 300); // a missing source is not retried (was ~1.5 s of sleeps, one after the last attempt)
    CHECK(read_all(a) == std::vector<uint8_t>({ 'n', 'e', 'w', '!' })); // a failed replace keeps the old file

    // durability before the commit rename
    CHECK(stateos_flush_file(a, &err));
    CHECK(!stateos_flush_file(tmp_file("flush-missing.state"), &err));
}

// startup cleanup: only our temp suffix, only regular files, only older than the age limit
static void test_stale_tmp_cleanup() {
    const std::filesystem::path dir = g_tmp / "cleanup";
    std::filesystem::create_directories(dir);
    const auto old_time = std::filesystem::file_time_type::clock::now() - std::chrono::hours(2);
    auto make = [&](const std::string & name, bool old) {
        const std::string p = stateos_path_utf8(dir / name);
        write_raw(p, { 'x' });
        if (old) {
            std::filesystem::last_write_time(stateos_path(p), old_time);
        }
    };
    make("crashed.state.stateos.tmp", true);  // removed
    make("in-flight.state.stateos.tmp", false); // too young: a save may be writing it
    make("user-file.tmp", true);               // not our suffix
    make("saved.state", true);                 // a real state
    make(".stateos.tmp", true);                // suffix only, no name: left alone
    std::filesystem::create_directories(dir / "dir.stateos.tmp"); // not a regular file

    std::vector<std::string> removed;
    const size_t n = stateos_cleanup_stale_tmp(stateos_path_utf8(dir), 3600, &removed);
    CHECK(n == 1 && removed.size() == 1 && removed[0] == "crashed.state.stateos.tmp");
    CHECK(!std::filesystem::exists(dir / "crashed.state.stateos.tmp"));
    CHECK(std::filesystem::exists(dir / "in-flight.state.stateos.tmp"));
    CHECK(std::filesystem::exists(dir / "user-file.tmp"));
    CHECK(std::filesystem::exists(dir / "saved.state"));
    CHECK(std::filesystem::exists(dir / ".stateos.tmp"));
    CHECK(std::filesystem::exists(dir / "dir.stateos.tmp"));
    CHECK(stateos_cleanup_stale_tmp(stateos_path_utf8(dir / "does-not-exist"), 3600, nullptr) == 0);
}

// a GGUF with real tensor data (the vocab fixtures have none, so their sampling path is empty)
static bool write_tensor_gguf(const std::string & path, size_t n_floats, int split_count, float seed) {
    ggml_init_params ip = { n_floats * sizeof(float) + (1u << 20), nullptr, false };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) {
        return false;
    }
    ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, (int64_t) n_floats);
    ggml_set_name(t, "w");
    float * d = (float *) t->data;
    for (size_t i = 0; i < n_floats; ++i) {
        d[i] = seed + (float) (i % 1000) * 0.5f;
    }
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.name", "stateos-fp-test");
    if (split_count != 1) {
        gguf_set_val_u16(g, "split.count", (uint16_t) split_count); // 0 = what llama-gguf-split --merge writes
    }
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(gctx);
    return std::filesystem::exists(stateos_path(path));
}

static uint64_t gguf_data_offset_of(const std::string & path) {
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        return 0;
    }
    const uint64_t off = gguf_get_data_offset(g);
    gguf_free(g);
    return off;
}

static void flip_byte(const std::string & path, uint64_t off) {
    std::fstream f(stateos_path(path), std::ios::in | std::ios::out | std::ios::binary);
    f.seekg((std::streamoff) off);
    char c = 0;
    f.read(&c, 1);
    c = (char) (c ^ 0x5A);
    f.seekp((std::streamoff) off);
    f.write(&c, 1);
}

static bool in_windows(const std::vector<std::pair<uint64_t, uint64_t>> & w, uint64_t off) {
    for (const auto & x : w) {
        if (off >= x.first && off < x.first + x.second) {
            return true;
        }
    }
    return false;
}

static void test_fingerprint_v2() {
    // window placement
    auto w = stateos_fingerprint_windows(1000, 1000 + 10 * STATEOS_FP_WINDOW);
    CHECK(w.size() == 1 && w[0].first == 1000 && w[0].second == 10 * STATEOS_FP_WINDOW); // small: all of it
    CHECK(stateos_fingerprint_windows(5000, 5000).empty());
    w = stateos_fingerprint_windows(4096, 4096 + (64u << 20));
    CHECK(w.size() == (size_t) STATEOS_FP_SAMPLES);
    CHECK(w.front().first == 4096);
    CHECK(w.back().first + w.back().second == 4096 + (64u << 20));
    bool ordered = true;
    for (size_t i = 1; i < w.size(); ++i) {
        ordered = ordered && w[i].first >= w[i - 1].first + w[i - 1].second;
    }
    CHECK(ordered);

    // one file with 4 MiB of tensor data: sampled bytes are covered, unsampled ones are the documented residual
    std::string err;
    const std::string one = tmp_file("fp-one.gguf");
    CHECK(write_tensor_gguf(one, 1u << 20, 1, 0.0f));
    const std::string fp0 = stateos_model_fingerprint(one, &err);
    CHECK(fp0.size() == 64);
    const uint64_t d0 = gguf_data_offset_of(one);
    const auto wins = stateos_fingerprint_windows(d0, (uint64_t) std::filesystem::file_size(stateos_path(one)));
    CHECK(wins.size() == (size_t) STATEOS_FP_SAMPLES);
    flip_byte(one, d0 + 100);
    CHECK(stateos_model_fingerprint(one, &err) != fp0);
    flip_byte(one, d0 + 100);
    CHECK(stateos_model_fingerprint(one, &err) == fp0);
    const uint64_t gap = d0 + STATEOS_FP_WINDOW + 4096;
    CHECK(!in_windows(wins, gap));
    flip_byte(one, gap);
    CHECK(stateos_model_fingerprint(one, &err) == fp0); // residual: same header, same size, unsampled offset
    flip_byte(one, gap);

    // split model: every shard is covered, and the shard set must be complete
    const std::string s1 = tmp_file("fp-split-00001-of-00002.gguf");
    const std::string s2 = tmp_file("fp-split-00002-of-00002.gguf");
    CHECK(write_tensor_gguf(s1, 1u << 18, 2, 1.0f));
    CHECK(write_tensor_gguf(s2, 1u << 18, 2, 2.0f));
    const std::string fps = stateos_model_fingerprint(s1, &err);
    CHECK(fps.size() == 64);
    if (fps.size() != 64) {
        std::fprintf(stderr, "  split fingerprint error: %s\n", err.c_str());
    }
    flip_byte(s2, gguf_data_offset_of(s2) + 100);
    CHECK(stateos_model_fingerprint(s1, &err) != fps); // a change in shard 2 changes the identity
    std::filesystem::remove(stateos_path(s2));
    err.clear();
    CHECK(stateos_model_fingerprint(s1, &err).empty() && !err.empty()); // a missing shard: no identity

    const std::string misnamed = tmp_file("fp-misnamed.gguf");
    CHECK(write_tensor_gguf(misnamed, 1024, 2, 3.0f));
    err.clear();
    CHECK(stateos_model_fingerprint(misnamed, &err).empty() && !err.empty());

    // a merged GGUF (llama-gguf-split --merge writes split.count = 0) is one file, like the loader treats it
    const std::string merged = tmp_file("fp-merged.gguf");
    CHECK(write_tensor_gguf(merged, 1024, 0, 4.0f));
    err.clear();
    const std::string fpm = stateos_model_fingerprint(merged, &err);
    CHECK(fpm.size() == 64);
    if (fpm.size() != 64) {
        std::fprintf(stderr, "  merged fingerprint error: %s\n", err.c_str());
    }
}

// the current server's model identity (argv[1] = a small GGUF, e.g. models/ggml-vocab-qwen2.gguf)
static void test_model_fingerprint(const std::string & gguf, const std::string & other_gguf) {
    std::string err;
    const std::string a = stateos_model_fingerprint(gguf, &err);
    CHECK(a.size() == 64);
    if (a.size() != 64) {
        std::fprintf(stderr, "  fingerprint error for %s: %s\n", gguf.c_str(), err.c_str());
    }
    CHECK(stateos_model_fingerprint(gguf, &err) == a); // stable

    // same bytes plus a tail: the size is part of the identity
    const std::string grown = tmp_file("grown.gguf");
    std::filesystem::copy_file(stateos_path(gguf), stateos_path(grown));
    {
        std::ofstream f(stateos_path(grown), std::ios::binary | std::ios::app);
        const std::string tail(64, 'z');
        f.write(tail.data(), (std::streamsize) tail.size());
    }
    const std::string b = stateos_model_fingerprint(grown, &err);
    CHECK(b.size() == 64 && b != a);

    if (!other_gguf.empty()) {
        const std::string c = stateos_model_fingerprint(other_gguf, &err);
        CHECK(c.size() == 64 && c != a);
    }

    // not a GGUF: refused with a reason, never an identity
    const std::string junk = tmp_file("junk.gguf");
    write_raw(junk, { 'J', 'U', 'N', 'K', 0, 0, 0, 0 });
    err.clear();
    CHECK(stateos_model_fingerprint(junk, &err).empty());
    CHECK(!err.empty());
    CHECK(stateos_model_fingerprint(tmp_file("missing.gguf"), &err).empty());
}

int main(int argc, char ** argv) {
    g_tmp = std::filesystem::temp_directory_path() / ("stateos-header-test-" + std::to_string((long long) std::time(nullptr)));
    std::filesystem::create_directories(g_tmp);

    test_sha256();
    test_header_codec();
    test_verify();
    test_container();
    test_checkpoints();
    test_companion();
    test_props_capability();
    test_effective_model();
    test_section_checks();
    test_replace_file();
    test_stale_tmp_cleanup();
    test_fingerprint_v2();
    if (argc > 1) {
        test_model_fingerprint(argv[1], argc > 2 ? argv[2] : "");
    } else {
        std::printf("test-stateos-header: SKIP model fingerprint (pass a GGUF path as argv[1])\n");
    }

    std::error_code ec;
    std::filesystem::remove_all(g_tmp, ec);

    std::printf("test-stateos-header: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
