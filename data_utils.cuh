#pragma once
#include "common.cuh"

void build_BALS_format(int rows, int cols, int xb, int yb,
                       const vector<int>& row_ptr, const vector<int>& col_idx, const vector<float>& values,
                       vector<int>& tile_ptr, vector<int>& tile_colidx,
                       vector<int>& seg_ptr, vector<int>& seg_colidx, vector<float>& seg_values,
                       vector<float>& tile_density,
                       vector<int>& nz_tile_list, vector<int>& nz_tile_ptr) {
    int num_tiles_x = (rows + xb - 1) / xb;
    int num_tiles_y = (cols + yb - 1) / yb;
    int tile_number = num_tiles_x * num_tiles_y;

    tile_ptr.assign(tile_number + 1, 0);
    seg_ptr.assign((long long)num_tiles_x * num_tiles_y * xb + 1, 0);
    tile_density.resize(tile_number, 0.0f);
    nz_tile_ptr.resize(num_tiles_x + 1, 0);

    int seg_count = 0;
    int tile_idx_count = 0;
    seg_ptr[0] = 0;
    tile_ptr[0] = 0;

    for (int tx = 0; tx < num_tiles_x; tx++) {
        int start_row = tx * xb;
        vector<vector<vector<pair<int, float>>>> tile_rows(num_tiles_y, vector<vector<pair<int, float>>>(xb));

        for (int r = 0; r < xb; r++) {
            int global_row = start_row + r;
            if (global_row < rows) {
                for (int i = row_ptr[global_row]; i < row_ptr[global_row + 1]; i++) {
                    int c = col_idx[i];
                    int ty = c / yb;
                    tile_rows[ty][r].push_back({c, values[i]});
                }
            }
        }

        nz_tile_ptr[tx] = (int)nz_tile_list.size();

        for (int ty = 0; ty < num_tiles_y; ty++) {
            int tile_id = tx * num_tiles_y + ty;

            vector<int> unique_cols;
            for (int r = 0; r < xb; r++)
                for (auto& p : tile_rows[ty][r])
                    unique_cols.push_back(p.first);

            sort(unique_cols.begin(), unique_cols.end());
            unique_cols.erase(unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());

            for (int c : unique_cols) tile_colidx.push_back(c);
            tile_idx_count += (int)unique_cols.size();
            tile_ptr[tile_id + 1] = tile_idx_count;

            int tile_nnz = 0;
            for (int r = 0; r < xb; r++) tile_nnz += (int)tile_rows[ty][r].size();
            tile_density[tile_id] = (float)tile_nnz / ((float)xb * yb);

            if (tile_nnz > 0) nz_tile_list.push_back(tile_id);

            long long seg_base = (long long)tile_id * xb;
            seg_ptr[seg_base] = seg_count;
            for (int r = 0; r < xb; r++) {
                for (auto& p : tile_rows[ty][r]) {
                    int c = p.first;
                    int local_c = (int)(lower_bound(unique_cols.begin(), unique_cols.end(), c) - unique_cols.begin());
                    seg_colidx.push_back(local_c);
                    seg_values.push_back(p.second);
                    seg_count++;
                }
                seg_ptr[seg_base + r + 1] = seg_count;
            }
        }
    }
    nz_tile_ptr[num_tiles_x] = (int)nz_tile_list.size();
}
