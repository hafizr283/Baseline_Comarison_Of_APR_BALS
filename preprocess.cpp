#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

using namespace std;

template<typename T>
void write_vec(FILE* f, const vector<T>& v) {
    size_t sz = v.size();
    fwrite(&sz, sizeof(size_t), 1, f);
    if(sz > 0) fwrite(v.data(), sizeof(T), sz, f);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_csv> <output_bin>" << endl;
        return 1;
    }

    const char* csv_path = argv[1];
    const char* bin_path = argv[2];

    cout << "Reading CSV: " << csv_path << endl;
    ifstream f(csv_path);
    if (!f) {
        cerr << "Error opening CSV: " << csv_path << endl;
        return 1;
    }

    vector<int> raw_users, raw_items;
    vector<float> raw_ratings;
    int num_users = 0, num_items = 0;

    string line;
    getline(f, line); // header
    while (getline(f, line)) {
        int u, item, ts; float r;
        if (sscanf(line.c_str(), "%d,%d,%f,%d", &u, &item, &r, &ts) == 4) {
            raw_users.push_back(u - 1);
            raw_items.push_back(item - 1);
            raw_ratings.push_back(r);
            if (u > num_users)    num_users = u;
            if (item > num_items) num_items = item;
        }
    }
    int nnz = (int)raw_users.size();
    cout << "Parsed " << nnz << " ratings." << endl;

    vector<int> user_freq(num_users, 0);
    vector<int> item_freq(num_items, 0);
    for (int i = 0; i < nnz; i++) {
        user_freq[raw_users[i]]++;
        item_freq[raw_items[i]]++;
    }

    vector<int> user_order_freq(num_users);
    for (int i = 0; i < num_users; i++) user_order_freq[i] = i;
    sort(user_order_freq.begin(), user_order_freq.end(), [&](int a, int b) {
        return user_freq[a] > user_freq[b];
    });

    vector<int> item_order_freq(num_items);
    for (int i = 0; i < num_items; i++) item_order_freq[i] = i;
    sort(item_order_freq.begin(), item_order_freq.end(), [&](int a, int b) {
        return item_freq[a] > item_freq[b];
    });

    vector<int> user_map(num_users);
    for (int i = 0; i < num_users; i++) user_map[user_order_freq[i]] = i;
    vector<int> item_map(num_items);
    for (int i = 0; i < num_items; i++) item_map[item_order_freq[i]] = i;

    for (int i = 0; i < nnz; i++) {
        raw_users[i] = user_map[raw_users[i]];
        raw_items[i] = item_map[raw_items[i]];
    }

    int actual_num_users = 0, actual_num_items = 0;
    for (int i = 0; i < num_users; i++) if (user_freq[i] > 0) actual_num_users++;
    for (int i = 0; i < num_items; i++) if (item_freq[i] > 0) actual_num_items++;
    num_users = actual_num_users;
    num_items = actual_num_items;

    cout << "Reordered users/items. Actual users: " << num_users << ", items: " << num_items << endl;

    // 80/20 train/test split stratified by user
    vector<int>   test_users_h, test_items_h;
    vector<float> test_ratings_h;
    {
        vector<vector<int>> per_user(num_users);
        for (int i = 0; i < nnz; i++) per_user[raw_users[i]].push_back(i);

        mt19937 rng_split(42);
        vector<int> tag(nnz, 1);
        for (int u = 0; u < num_users; u++) {
            auto& ui = per_user[u];
            if ((int)ui.size() <= 1) continue;
            shuffle(ui.begin(), ui.end(), rng_split);
            int n_test = max(1, (int)(ui.size() / 5));
            for (int j = (int)ui.size() - n_test; j < (int)ui.size(); j++)
                tag[ui[j]] = 0;
        }

        vector<int>   train_u, train_i;
        vector<float> train_r;
        for (int i = 0; i < nnz; i++) {
            if (tag[i]) {
                train_u.push_back(raw_users[i]);
                train_i.push_back(raw_items[i]);
                train_r.push_back(raw_ratings[i]);
            } else {
                test_users_h.push_back(raw_users[i]);
                test_items_h.push_back(raw_items[i]);
                test_ratings_h.push_back(raw_ratings[i]);
            }
        }
        raw_users   = move(train_u);
        raw_items   = move(train_i);
        raw_ratings = move(train_r);
    }
    int nnz_train = (int)raw_users.size();
    int nnz_test  = (int)test_users_h.size();
    cout << "Split: train=" << nnz_train << ", test=" << nnz_test << endl;

    // Build user CSR (sorted by user id)
    vector<int> order(nnz_train);
    for (int i = 0; i < nnz_train; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){ return raw_users[a] < raw_users[b]; });

    vector<int>   h_item_indices(nnz_train);
    vector<float> h_user_ratings(nnz_train);
    for (int i = 0; i < nnz_train; i++) {
        h_item_indices[i] = raw_items[order[i]];
        h_user_ratings[i] = raw_ratings[order[i]];
    }
    vector<int> h_user_offsets(num_users + 1, 0);
    for (int i = 0; i < nnz_train; i++) h_user_offsets[raw_users[order[i]] + 1]++;
    for (int i = 0; i < num_users; i++) h_user_offsets[i + 1] += h_user_offsets[i];

    // Build item CSR (sorted by item id)
    for (int i = 0; i < nnz_train; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){ return raw_items[a] < raw_items[b]; });

    vector<int>   h_user_indices(nnz_train);
    vector<float> h_item_ratings(nnz_train);
    for (int i = 0; i < nnz_train; i++) {
        h_user_indices[i] = raw_users[order[i]];
        h_item_ratings[i] = raw_ratings[order[i]];
    }
    vector<int> h_item_offsets(num_items + 1, 0);
    for (int i = 0; i < nnz_train; i++) h_item_offsets[raw_items[order[i]] + 1]++;
    for (int i = 0; i < num_items; i++) h_item_offsets[i + 1] += h_item_offsets[i];

    cout << "Writing to binary file: " << bin_path << endl;
    FILE* out = fopen(bin_path, "wb");
    if (!out) {
        cerr << "Error creating " << bin_path << endl;
        return 1;
    }

    int version = 1;
    fwrite(&version, sizeof(int), 1, out);
    fwrite(&num_users, sizeof(int), 1, out);
    fwrite(&num_items, sizeof(int), 1, out);
    fwrite(&nnz_train, sizeof(int), 1, out);
    fwrite(&nnz_test, sizeof(int), 1, out);

    write_vec(out, raw_users);
    write_vec(out, raw_items);
    write_vec(out, raw_ratings);

    write_vec(out, test_users_h);
    write_vec(out, test_items_h);
    write_vec(out, test_ratings_h);

    write_vec(out, h_user_offsets);
    write_vec(out, h_item_indices);
    write_vec(out, h_user_ratings);

    write_vec(out, h_item_offsets);
    write_vec(out, h_user_indices);
    write_vec(out, h_item_ratings);

    fclose(out);
    cout << "Done!" << endl;

    return 0;
}
