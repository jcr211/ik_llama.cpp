// LONGSPEAR State-OS v2 (SV2-E1): CPU unit tests for src/llama-partial-state.h, synthetic data only.
// No llama/ggml link: the helpers are pure, so this binary never loads a backend.

#include "llama-partial-state.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace lps = llama_partial_state;

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

struct test_cell {
    int32_t pos = -1;
    std::set<int32_t> seq_id;
    bool has_seq_id(int32_t id) const { return seq_id.count(id) > 0; }
    bool is_empty() const { return seq_id.empty(); }
};

struct vec_sink {
    std::vector<uint8_t> bytes;
    void write(const void * src, size_t size) {
        const uint8_t * p = (const uint8_t *) src;
        bytes.insert(bytes.end(), p, p + size);
    }
};

// what llama_data_write_dummy does: count, never read the source
struct count_sink {
    size_t n = 0;
    void write(const void *, size_t size) { n += size; }
};

static void le32(std::vector<uint8_t> & v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back((uint8_t) (x >> (8*i)));
}
static void le64(std::vector<uint8_t> & v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back((uint8_t) (x >> (8*i)));
}

// cells: 0..2 seq 0 at pos 0..2, 3 empty, 4 seq 1 at pos 3, 5 seqs {0,1} at pos 4
static std::vector<test_cell> make_cells() {
    std::vector<test_cell> c(6);
    c[0].pos = 0; c[0].seq_id = {0};
    c[1].pos = 1; c[1].seq_id = {0};
    c[2].pos = 2; c[2].seq_id = {0};
    c[4].pos = 3; c[4].seq_id = {1};
    c[5].pos = 4; c[5].seq_id = {0, 1};
    return c;
}

// three layers: layer 0 has 2 rows of 8 B, layer 1 has no state, layer 2 has 2 rows of 4 B
struct synth_state {
    std::vector<std::vector<uint8_t>> tensors;
    std::vector<lps::row_block> blocks; // for sequence row 1
    explicit synth_state(uint8_t salt) {
        tensors.resize(3);
        for (int i = 0; i < 16; ++i) tensors[0].push_back((uint8_t) (salt + i));
        for (int i = 0; i < 8; ++i)  tensors[2].push_back((uint8_t) (salt + 0x40 + i));
        blocks.resize(3);
        blocks[0] = { 0, 8, 1, 8 };
        blocks[1] = { -1, 0, 0, 0 };
        blocks[2] = { 0, 4, 1, 4 };
    }
};

template <typename Sink>
static void emit_partial(Sink & sink, const std::vector<test_cell> & cells, int32_t seq_id, int32_t cap,
                         const synth_state & rows) {
    std::vector<lps::cell_range> ranges;
    const uint32_t cell_count = lps::select_cells(cells.data(), (uint32_t) cells.size(), seq_id, cap, ranges);
    sink.write(&cell_count, sizeof(cell_count));
    lps::emit_cells_meta(sink, cells.data(), ranges, seq_id);
    lps::emit_rows(sink, rows.blocks, [&](uint32_t il, size_t offset, size_t size) {
        sink.write(rows.tensors[il].data() + offset, size);
    });
}

static void test_select_ranges() {
    const auto cells = make_cells();
    std::vector<lps::cell_range> r;

    CHECK(lps::select_cells(cells.data(), 6, 0, -1, r) == 4);
    CHECK(r.size() == 2 && r[0] == lps::cell_range(0, 3) && r[1] == lps::cell_range(5, 6));

    CHECK(lps::select_cells(cells.data(), 6, -1, -1, r) == 5);
    CHECK(r.size() == 2 && r[0] == lps::cell_range(0, 3) && r[1] == lps::cell_range(4, 6));

    CHECK(lps::select_cells(cells.data(), 6, 1, -1, r) == 2);
    CHECK(r.size() == 1 && r[0] == lps::cell_range(4, 6));

    CHECK(lps::select_cells(cells.data(), 6, 7, -1, r) == 0);
    CHECK(r.empty());
}

static void test_golden_single_seq() {
    const auto cells = make_cells();
    const synth_state rows(0x10);
    vec_sink sink;
    emit_partial(sink, cells, 0, -1, rows);

    std::vector<uint8_t> want;
    le32(want, 4);                                  // cell_count
    for (int32_t pos : { 0, 1, 2, 4 }) {            // meta: pos, n_seq_id = 0
        le32(want, (uint32_t) pos);
        le32(want, 0);
    }
    le32(want, 0); le64(want, 8); le32(want, 1);    // layer 0 header
    for (int i = 8; i < 16; ++i) want.push_back((uint8_t) (0x10 + i));
    le32(want, 0xffffffffu); le64(want, 0); le32(want, 0); // layer 1: no state
    le32(want, 0); le64(want, 4); le32(want, 1);    // layer 2 header
    for (int i = 4; i < 8; ++i) want.push_back((uint8_t) (0x50 + i));

    CHECK(sink.bytes == want);
    CHECK(sink.bytes.size() == 4 + 4*8 + (16 + 8) + 16 + (16 + 4));
}

static void test_golden_all_seqs_meta() {
    const auto cells = make_cells();
    std::vector<lps::cell_range> r;
    lps::select_cells(cells.data(), 6, -1, -1, r);
    vec_sink sink;
    lps::emit_cells_meta(sink, cells.data(), r, -1);

    std::vector<uint8_t> want;
    for (int i : { 0, 1, 2 }) { le32(want, (uint32_t) i); le32(want, 1); le32(want, 0); }
    le32(want, 3); le32(want, 1); le32(want, 1);            // cell 4: pos 3, {1}
    le32(want, 4); le32(want, 2); le32(want, 0); le32(want, 1); // cell 5: pos 4, {0,1}
    CHECK(sink.bytes == want);
}

static void test_size_equals_emitted() {
    const auto cells = make_cells();
    const synth_state rows(0x20);
    for (int32_t seq : { -1, 0, 1 }) {
        vec_sink v;
        count_sink c;
        emit_partial(v, cells, seq, -1, rows);
        // the size pass must not read row bytes: the dummy writer only counts
        std::vector<lps::cell_range> ranges;
        const uint32_t n = lps::select_cells(cells.data(), (uint32_t) cells.size(), seq, -1, ranges);
        c.write(&n, sizeof(n));
        lps::emit_cells_meta(c, cells.data(), ranges, seq);
        lps::emit_rows(c, rows.blocks, [&](uint32_t, size_t, size_t size) { c.n += size; });
        CHECK(c.n == v.bytes.size());
    }
}

// ---- capped / shadow-sourced writer (commit 3) -------------------------------------------------

// one sequence, cells 0..9 at pos 0..9 (cell index == position, as with defrag off)
static std::vector<test_cell> make_seq(int n) {
    std::vector<test_cell> c((size_t) n);
    for (int i = 0; i < n; ++i) {
        c[(size_t) i].pos = i;
        c[(size_t) i].seq_id = {0};
    }
    return c;
}

static void test_cap_filter() {
    const auto cells = make_seq(10);
    std::vector<lps::cell_range> r;
    CHECK(lps::select_cells(cells.data(), 10, 0, 6, r) == 7);
    CHECK(r.size() == 1 && r[0] == lps::cell_range(0, 7));
    CHECK(lps::select_cells(cells.data(), 10, 0, 0, r) == 1);
    CHECK(lps::select_cells(cells.data(), 10, 0, 9, r) == 10);

    // the cap is on position, not index: a gap in the cells keeps the ranges exact
    auto holes = make_seq(10);
    holes[3].seq_id.clear();
    holes[3].pos = -1;
    CHECK(lps::select_cells(holes.data(), 10, 0, 5, r) == 5);
    CHECK(r.size() == 2 && r[0] == lps::cell_range(0, 3) && r[1] == lps::cell_range(4, 6));

    // metadata of the capped write lists exactly positions 0..cap
    vec_sink sink;
    lps::select_cells(cells.data(), 10, 0, 6, r);
    lps::emit_cells_meta(sink, cells.data(), r, 0);
    CHECK(sink.bytes.size() == 7 * 8);
    int32_t last_pos = -1;
    std::memcpy(&last_pos, sink.bytes.data() + 6 * 8, 4);
    CHECK(last_pos == 6);
}

static void test_cap_above_last_cell_is_legacy() {
    const auto cells = make_seq(10);
    const synth_state rows(0x30);
    vec_sink legacy, capped, capped_far;
    emit_partial(legacy, cells, 0, -1, rows);
    emit_partial(capped, cells, 0, 9, rows);
    emit_partial(capped_far, cells, 0, 1 << 30, rows);
    CHECK(capped.bytes == legacy.bytes);
    CHECK(capped_far.bytes == legacy.bytes);
}

static void test_source_swap() {
    const auto cells = make_seq(10);
    const synth_state live(0x40);
    const synth_state shadow(0x90);
    vec_sink from_live, from_shadow;
    emit_partial(from_live, cells, 0, 6, live);
    emit_partial(from_shadow, cells, 0, 6, shadow);

    CHECK(from_live.bytes.size() == from_shadow.bytes.size());
    // cell_count + metadata + layer-0 header are identical; the row bytes are the shadow's
    const size_t head = 4 + 7 * 8 + 16;
    CHECK(std::memcmp(from_live.bytes.data(), from_shadow.bytes.data(), head) == 0);
    CHECK(std::memcmp(from_shadow.bytes.data() + head, shadow.tensors[0].data() + 8, 8) == 0);
    CHECK(std::memcmp(from_live.bytes.data() + head, live.tensors[0].data() + 8, 8) == 0);
    CHECK(from_live.bytes != from_shadow.bytes);
}

static void test_capped_size_equals_emitted() {
    const auto cells = make_seq(10);
    const synth_state rows(0x50);
    for (int32_t cap : { -1, 0, 3, 6, 9, 100 }) {
        vec_sink v;
        emit_partial(v, cells, 0, cap, rows);
        count_sink c;
        std::vector<lps::cell_range> ranges;
        const uint32_t n = lps::select_cells(cells.data(), (uint32_t) cells.size(), 0, cap, ranges);
        c.write(&n, sizeof(n));
        lps::emit_cells_meta(c, cells.data(), ranges, 0);
        lps::emit_rows(c, rows.blocks, [&](uint32_t, size_t, size_t size) { c.n += size; });
        CHECK(c.n == v.bytes.size());
        const uint32_t kept = cap < 0 ? 10u : (uint32_t) std::min(cap + 1, 10);
        CHECK(v.bytes.size() == 4 + 8 * (size_t) kept + (16 + 8) + 16 + (16 + 4));
    }
}

int main() {
    test_select_ranges();
    test_golden_single_seq();
    test_golden_all_seqs_meta();
    test_size_equals_emitted();
    test_cap_filter();
    test_cap_above_last_cell_is_legacy();
    test_source_swap();
    test_capped_size_equals_emitted();

    std::printf("test-partial-state: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
