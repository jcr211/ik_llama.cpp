// LONGSPEAR State-OS v2 (SV2-E1): CPU unit tests for examples/server/stateos-v2.h.
//   (a) tail-snapshot eligibility over the speculative path matrix
//   (b) ordered insert: apply_checkpoint's reverse search returns the tail, not the tolerance checkpoint
//   (c) divergence classifier
// plus SHA-256 vectors, the PARTIAL_ONLY payload walker used by the crosscheck, and the row comparator.
// No llama/ggml link: header-only, never loads a backend.

#include "stateos-v2.h"
#include "llama-partial-state.h"

#include <algorithm>
#include <cstdio>
#include <list>
#include <set>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

// ---- (a) eligibility over the path matrix ------------------------------------------------------
//
// Engine rules being modelled (llama.cpp, SV2-E1 commit 3): a successful full gpu-fallback save before
// the verify batch of the round rooted at `root` records shadow_pos = root - 1; the cache then holds
// the root and the accepted drafts, so the last cached position is root + n_acc. A round that decodes
// the root alone does not save (the shadow keeps the older position). A failed save, a failed commit,
// a non-speculative generation and PER_STEP mode leave shadow_pos = -1.

struct round_sim {
    int32_t shadow_pos = -1;
    int32_t cache_pos_max = -1;

    void drafted(int32_t root, int32_t n_acc) {
        shadow_pos = root - 1;
        cache_pos_max = root + n_acc;
    }
    void root_only(int32_t root) {
        cache_pos_max = root;
    }
    void failed() {
        shadow_pos = -1;
    }
};

static stateos_tail_input input_of(const round_sim & r, int32_t last_ckpt = 100) {
    stateos_tail_input in;
    in.flag = true;
    in.shadow_pos = r.shadow_pos;
    in.cache_pos_max = r.cache_pos_max;
    in.last_ckpt_pos_max = last_ckpt;
    return in;
}

static bool cause_is(const stateos_tail_verdict & v, const char * cause) {
    return std::string(v.cause) == cause;
}

static void test_eligibility_path_matrix() {
    {   // full accept: 4 of 4 drafts
        round_sim r; r.drafted(200, 4);
        const auto v = stateos_tail_eligibility(input_of(r));
        CHECK(v.eligible && cause_is(v, "eligible"));
        CHECK(r.shadow_pos == 199 && r.cache_pos_max == 204);
    }
    {   // partial accept: 2 of 4 (restore + re-decode; the shadow is only read)
        round_sim r; r.drafted(200, 2);
        CHECK(stateos_tail_eligibility(input_of(r)).eligible);
    }
    {   // drafted round with 0 accepted: the snapshot would sit at the last cached position - 1
        round_sim r; r.drafted(200, 0);
        const auto v = stateos_tail_eligibility(input_of(r));
        CHECK(!v.eligible && cause_is(v, "no-accepted-draft"));
    }
    {   // exactly one accepted draft is the boundary
        round_sim r; r.drafted(200, 1);
        CHECK(stateos_tail_eligibility(input_of(r)).eligible);
    }
    {   // root-only rounds after a drafted round: older valid shadow_pos, eligible with a larger gap
        round_sim r; r.drafted(200, 3); r.root_only(204); r.root_only(205);
        const auto v = stateos_tail_eligibility(input_of(r));
        CHECK(v.eligible);
        CHECK(r.cache_pos_max - r.shadow_pos == 6);
    }
    {   // root-only rounds only (never drafted): no shadow
        round_sim r; r.root_only(150); r.root_only(151);
        CHECK(cause_is(stateos_tail_eligibility(input_of(r)), "no-shadow"));
    }
    {   // stop-cut final round: the stop hit ids[1] of 4 accepted; the cache still holds all 4
        round_sim r; r.drafted(200, 4);
        CHECK(stateos_tail_eligibility(input_of(r)).eligible);
    }
    {   // commit failure (failed restore/replay): shadow_pos invalidated
        round_sim r; r.drafted(200, 3); r.failed();
        CHECK(cause_is(stateos_tail_eligibility(input_of(r)), "no-shadow"));
    }
    {   // alloc failure: _save_at sets -1 before the copy and the copy never succeeds
        round_sim r; r.drafted(180, 2); r.failed(); r.root_only(183);
        CHECK(cause_is(stateos_tail_eligibility(input_of(r)), "no-shadow"));
    }
    {   // non-speculative generation
        round_sim r; r.cache_pos_max = 500;
        CHECK(cause_is(stateos_tail_eligibility(input_of(r)), "no-shadow"));
    }
    {   // PER_STEP (SL-1): no full shadow, inert and reported as such
        round_sim r; r.drafted(200, 3);
        stateos_tail_input in = input_of(r);
        in.per_step = true;
        CHECK(cause_is(stateos_tail_eligibility(in), "per-step"));
    }
    {   // flag off, defrag on, media, short, not newer than the newest checkpoint
        round_sim r; r.drafted(200, 3);
        stateos_tail_input in = input_of(r);
        in.flag = false;
        CHECK(cause_is(stateos_tail_eligibility(in), "flag-off"));
        in = input_of(r); in.defrag_on = true;
        CHECK(cause_is(stateos_tail_eligibility(in), "defrag"));
        in = input_of(r); in.media = true;
        CHECK(cause_is(stateos_tail_eligibility(in), "media"));
        round_sim s; s.drafted(60, 3);
        CHECK(cause_is(stateos_tail_eligibility(input_of(s, -1)), "short"));
        round_sim s2; s2.drafted(65, 3);
        CHECK(stateos_tail_eligibility(input_of(s2, -1)).eligible);
        CHECK(cause_is(stateos_tail_eligibility(input_of(r, 199)), "not-newer"));
        CHECK(cause_is(stateos_tail_eligibility(input_of(r, 250)), "not-newer"));
        CHECK(stateos_tail_eligibility(input_of(r, 198)).eligible);
    }
}

// ---- (b) ordered insert and the reverse search ---------------------------------------------------

struct test_ckpt {
    int32_t pos_max;
    uint8_t origin;
};

static void test_ordered_insert_and_search() {
    std::list<test_ckpt> l = {
        { 1000, STATEOS_ORIGIN_TOLERANCE },
        { 1600, STATEOS_ORIGIN_GEN_INTERVAL },
    };
    // release: tail first (shadow_pos 1799), then the release checkpoint at the last cached position
    l.push_back({ 1799, STATEOS_ORIGIN_TAIL });
    l.push_back({ 1803, STATEOS_ORIGIN_RELEASE });
    CHECK(stateos_is_ascending(l));

    auto accept_all = [](const test_ckpt &) { return true; };
    auto reject_tail = [](const test_ckpt & c) { return c.origin != STATEOS_ORIGIN_TAIL; };

    // divergence at the last cached token D = 1803: threshold D - 1 = 1802
    auto it = stateos_find_restore(l, 1802, accept_all);
    CHECK(it != l.rend() && it->origin == STATEOS_ORIGIN_TAIL && it->pos_max == 1799);

    // without the tail (flag off) the same search falls back to the older checkpoint
    std::list<test_ckpt> base = { { 1000, STATEOS_ORIGIN_TOLERANCE }, { 1803, STATEOS_ORIGIN_RELEASE } };
    it = stateos_find_restore(base, 1802, accept_all);
    CHECK(it != base.rend() && it->origin == STATEOS_ORIGIN_TOLERANCE);

    // a tail whose token prefix no longer matches is skipped, never restored
    it = stateos_find_restore(l, 1802, reject_tail);
    CHECK(it != l.rend() && it->origin == STATEOS_ORIGIN_GEN_INTERVAL);

    // divergence right after the shadow position: the tail includes the divergent token's predecessor
    // only, pos_max 1799 is not < 1799, so the older checkpoint serves
    it = stateos_find_restore(l, 1799, accept_all);
    CHECK(it != l.rend() && it->pos_max == 1600);

    // nothing below the threshold: reset
    it = stateos_find_restore(l, 900, accept_all);
    CHECK(it == l.rend());

    // an out-of-order insert is detected
    std::list<test_ckpt> bad = { { 1000, 1 }, { 1800, 4 }, { 1799, 5 } };
    CHECK(!stateos_is_ascending(bad));
    std::list<test_ckpt> empty;
    CHECK(stateos_is_ascending(empty));
}

// ---- (c) divergence classifier --------------------------------------------------------------------

static void test_classifier() {
    stateos_div_input in;
    in.cache_n = 1000;
    in.n_past = 999;
    CHECK(std::string(stateos_classify_divergence(in)) == "last-token:other");
    in.cache_tok_is_eog = true;
    CHECK(std::string(stateos_classify_divergence(in)) == "last-token:eog");
    in.cache_tok_is_eog = false;
    in.prev_stop = STATEOS_STOP_STRING;
    CHECK(std::string(stateos_classify_divergence(in)) == "last-token:stop-string");
    in.n_past = 998;
    CHECK(std::string(stateos_classify_divergence(in)) == "interior");
    in.n_past = 1000;
    CHECK(std::string(stateos_classify_divergence(in)) == "interior");

    CHECK(std::string(stateos_tail_bucket(1)) == "1");
    CHECK(std::string(stateos_tail_bucket(2)) == "2-5");
    CHECK(std::string(stateos_tail_bucket(5)) == "2-5");
    CHECK(std::string(stateos_tail_bucket(6)) == "6-64");
    CHECK(std::string(stateos_tail_bucket(64)) == "6-64");
    CHECK(std::string(stateos_tail_bucket(65)) == "65-512");
    CHECK(std::string(stateos_tail_bucket(512)) == "65-512");
    CHECK(std::string(stateos_tail_bucket(513)) == ">512");

    CHECK(stateos_stop_cause_of(true, true, true) == STATEOS_STOP_EOG);
    CHECK(stateos_stop_cause_of(false, true, true) == STATEOS_STOP_STRING);
    CHECK(stateos_stop_cause_of(false, false, true) == STATEOS_STOP_N_PREDICT);
    CHECK(stateos_stop_cause_of(false, false, false) == STATEOS_STOP_OTHER);
    CHECK(std::string(stateos_origin_name(STATEOS_ORIGIN_TAIL)) == "tail");
    CHECK(std::string(stateos_origin_name(STATEOS_ORIGIN_UNKNOWN)) == "unknown");

    CHECK(stateos_escape_piece("a\"b\\c\nd\te\x01") == "a\\\"b\\\\c\\nd\\te\\x01");
}

// ---- SHA-256 --------------------------------------------------------------------------------------

static std::string sha_hex(const std::string & s) {
    stateos_sha256 h;
    h.update(s.data(), s.size());
    return stateos_hex(h.digest());
}

static void test_sha256() {
    CHECK(sha_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // chunked updates across block boundaries equal one update
    std::string m(1000, 'x');
    for (size_t i = 0; i < m.size(); ++i) m[i] = (char) ('a' + i % 26);
    stateos_sha256 h;
    for (size_t off = 0; off < m.size(); off += 37) {
        h.update(m.data() + off, std::min<size_t>(37, m.size() - off));
    }
    CHECK(stateos_hex(h.digest()) == sha_hex(m));
    // one million 'a' (FIPS 180-2 vector)
    stateos_sha256 big;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) big.update(chunk.data(), chunk.size());
    CHECK(stateos_hex(big.digest()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// ---- crosscheck payload walker and comparator ----------------------------------------------------

struct vec_sink {
    std::vector<uint8_t> bytes;
    void write(const void * src, size_t size) {
        const uint8_t * p = (const uint8_t *) src;
        bytes.insert(bytes.end(), p, p + size);
    }
};

struct cell {
    int32_t pos = -1;
    std::set<int32_t> seq_id;
    bool has_seq_id(int32_t id) const { return seq_id.count(id) > 0; }
    bool is_empty() const { return seq_id.empty(); }
};

// a PARTIAL_ONLY payload shaped like write_kv_cache's (v_state 0, three layers, layer 1 without state)
static std::vector<uint8_t> make_payload(int32_t cap, const std::vector<float> & row0, uint32_t dsa = 1) {
    std::vector<cell> cells(10);
    for (int i = 0; i < 10; ++i) { cells[(size_t) i].pos = i; cells[(size_t) i].seq_id = {0}; }
    std::vector<llama_partial_state::cell_range> ranges;
    const uint32_t n = llama_partial_state::select_cells(cells.data(), 10, 0, cap, ranges);
    vec_sink s;
    s.write(&n, 4);
    llama_partial_state::emit_cells_meta(s, cells.data(), ranges, 0);
    const uint32_t v_state = 0, n_layer = 3, qnext = 1;
    const int32_t none = -1;
    const uint64_t zero = 0;
    s.write(&v_state, 4);
    s.write(&n_layer, 4);
    for (int k = 0; k < 6; ++k) { s.write(&none, 4); s.write(&zero, 8); } // K then V headers
    s.write(&qnext, 4);
    std::vector<llama_partial_state::row_block> blocks(3);
    blocks[0] = { 0, (uint64_t) (row0.size() * 4), 1, 0 };
    blocks[1] = { -1, 0, 0, 0 };
    blocks[2] = { 1, 6, 1, 0 };
    const uint8_t f16row[6] = { 1, 2, 3, 4, 5, 6 };
    llama_partial_state::emit_rows(s, blocks, [&](uint32_t il, size_t, size_t size) {
        if (il == 0) s.write(row0.data(), size); else s.write(f16row, size);
    });
    s.write(&dsa, 4);
    return s.bytes;
}

static void test_partial_walker_and_compare() {
    const std::vector<float> ref = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> near = ref;
    near[3] = 4.5f;

    const auto a = make_payload(6, ref);
    const auto b = make_payload(6, near);
    stateos_partial_view va, vb;
    CHECK(stateos_parse_partial(a.data(), a.size(), va));
    CHECK(stateos_parse_partial(b.data(), b.size(), vb));
    CHECK(va.cell_count == 7 && va.pos_max == 6);
    CHECK(va.layers.size() == 3 && va.layers[1].n_rows == 0 && va.layers[0].row_size == 16);

    const auto same = stateos_compare_rows(a.data() + va.layers[0].offset, a.data() + va.layers[0].offset, 16, 0);
    CHECK(same.n == 4 && same.n_bitequal == 4 && same.rel_l2 == 0.0);

    const auto diff = stateos_compare_rows(b.data() + vb.layers[0].offset, a.data() + va.layers[0].offset, 16, 0);
    CHECK(diff.n == 4 && diff.n_bitequal == 3);
    CHECK(diff.rel_l2 > 0.091 && diff.rel_l2 < 0.092); // 0.5 / sqrt(30)

    const auto bytes = stateos_compare_rows(a.data() + va.layers[2].offset, b.data() + vb.layers[2].offset, 6, 1);
    CHECK(bytes.n == 6 && bytes.n_bitequal == 6 && bytes.rel_l2 < 0.0);

    // truncated, trailing byte, and a K row block (compacted layer / full state) are refused
    CHECK(!stateos_parse_partial(a.data(), a.size() - 1, va));
    std::vector<uint8_t> longer = a;
    longer.push_back(0);
    CHECK(!stateos_parse_partial(longer.data(), longer.size(), va));
    std::vector<uint8_t> k_rows = a;
    const size_t k0 = 4 + 7 * 8 + 8; // first K header row size
    k_rows[k0 + 4] = 1;
    CHECK(!stateos_parse_partial(k_rows.data(), k_rows.size(), va));
}

int main() {
    test_eligibility_path_matrix();
    test_ordered_insert_and_search();
    test_classifier();
    test_sha256();
    test_partial_walker_and_compare();

    std::printf("test-stateos-tail: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
