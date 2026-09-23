#pragma once

// LONGSPEAR State-OS v2 (SV2-E1): pure emission helpers for the cell metadata and the recurrent-row
// block of a sequence state (the part a LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY checkpoint consists of).
//
// llama_data_write::write_kv_cache / write_kv_cache_data route through these, so the writer and the
// size function (llama_data_write_dummy) emit through one code path. The helpers take plain arrays
// and callbacks only, so tests/test-partial-state.cpp checks the bytes on the CPU with synthetic data.
//
// Layout emitted (little-endian, as the writer always wrote it):
//   emit_cells_meta: per selected cell  i32 pos, u32 n_seq_id (0 for a single-sequence write),
//                    then n_seq_id x i32 seq id                 -> 8 B per cell for one sequence
//   emit_rows:       per layer          i32 type (-1 = none), u64 row size, u32 row count,
//                    then row count x row size bytes from the row reader

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace llama_partial_state {

using cell_range = std::pair<uint32_t, uint32_t>; // [first, second)

// Selects the cells written for `seq_id` (-1: every non-empty cell) as contiguous index ranges and
// returns their count. With pos_max_cap >= 0 only cells with pos <= pos_max_cap are kept.
// Cell needs: `pos`, `bool has_seq_id(int32_t) const`, `bool is_empty() const`.
template <typename Cell>
uint32_t select_cells(const Cell * cells, uint32_t n_cells, int32_t seq_id, int32_t pos_max_cap,
                      std::vector<cell_range> & ranges) {
    ranges.clear();
    uint32_t cell_count = 0;
    uint32_t range_begin = n_cells;
    for (uint32_t i = 0; i < n_cells; ++i) {
        const Cell & cell = cells[i];
        const bool in_seq = (seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id);
        const bool keep = in_seq && (pos_max_cap < 0 || cell.pos <= pos_max_cap);
        if (keep) {
            ++cell_count;
            if (range_begin == n_cells) {
                range_begin = i;
            }
        } else if (range_begin != n_cells) {
            ranges.emplace_back(range_begin, i);
            range_begin = n_cells;
        }
    }
    if (range_begin != n_cells) {
        ranges.emplace_back(range_begin, n_cells);
    }
    return cell_count;
}

// Per-cell metadata. Sink needs `void write(const void *, size_t)`; Cell additionally needs an
// iterable `seq_id` container of int32 ids (only read for seq_id == -1).
template <typename Sink, typename Cell>
void emit_cells_meta(Sink & sink, const Cell * cells, const std::vector<cell_range> & ranges, int32_t seq_id) {
    for (const auto & range : ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const Cell & cell = cells[i];
            const int32_t  pos      = cell.pos;
            const uint32_t n_seq_id = seq_id == -1 ? (uint32_t) cell.seq_id.size() : 0;

            sink.write(&pos,      sizeof(pos));
            sink.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto id : cell.seq_id) {
                    const int32_t id32 = id;
                    sink.write(&id32, sizeof(id32));
                }
            }
        }
    }
}

// One layer's recurrent-state rows.
struct row_block {
    int32_t  type     = -1; // ggml type of the layer's state tensor, -1 = the layer has none
    uint64_t row_size = 0;  // bytes per row, 0 = the layer has none
    uint32_t n_rows   = 0;  // rows written (0/1 for one sequence, every slot for seq -1)
    size_t   offset   = 0;  // byte offset of the first written row in the source tensor
};

// Per-layer row blocks. RowReader is called as read_rows(il, offset, size) and must append exactly
// `size` bytes to the sink (the live tensor or the speculative shadow, chosen by the caller).
template <typename Sink, typename RowReader>
void emit_rows(Sink & sink, const std::vector<row_block> & layers, RowReader && read_rows) {
    for (uint32_t il = 0; il < (uint32_t) layers.size(); ++il) {
        const row_block & b = layers[il];
        sink.write(&b.type,     sizeof(b.type));
        sink.write(&b.row_size, sizeof(b.row_size));
        sink.write(&b.n_rows,   sizeof(b.n_rows));
        if (b.n_rows > 0) {
            read_rows(il, b.offset, (size_t) b.n_rows * (size_t) b.row_size);
        }
    }
}

} // namespace llama_partial_state
