// State-OS v1 keyed slot-state container: model-free unit tests for examples/server/stateos-header.*
// (encode/decode round trip, one refusal per hard field, soft warnings, container scan, checkpoint codec).

#include "stateos-header.h"
#include "stateos-model.h"
#include "stateos-props.h"

#include <algorithm>
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
        { STATEOS_HARD, "model_fingerprint",    "3f1c0d5e9a" },
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

    // not found
    CHECK(stateos_scan_file(tmp_file("missing.state")).status == STATEOS_SCAN_NOT_FOUND);

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
        const nlohmann::ordered_json caps = stateos_props_capability(companion);
        const nlohmann::ordered_json props = { { "n_ctx", 196608 }, { "stateos", caps } }; // as handle_props embeds it
        const nlohmann::ordered_json back = nlohmann::ordered_json::parse(props.dump());
        CHECK(back.contains("stateos") && back["stateos"].is_object());
        CHECK(back["stateos"]["version"].is_number_integer() && back["stateos"]["version"].get<int>() >= 1);
        CHECK(back["stateos"]["version"].get<int>() == (int) STATEOS_CONTAINER_VERSION);
        CHECK(back["stateos"]["keyed_header"].is_boolean() && back["stateos"]["keyed_header"].get<bool>());
        CHECK(back["stateos"]["companion"].is_boolean() && back["stateos"]["companion"].get<bool>() == companion);
        CHECK(back["stateos"].size() == 3);
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
