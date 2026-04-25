#pragma once
#include <vector>

// Tracks label equivalences across tile boundaries.
// Union() always points the larger root → smaller root, consistent with
// the atomicMin strategy used in the GPU version later.
class UnionFind {
public:
    std::vector<int> parent;

    UnionFind(int num_labels) : parent(num_labels) {
        for (int label = 0; label < num_labels; label++)
            parent[label] = label;
    }

    int Find(int label) {
        if (parent[label] != label)
            parent[label] = Find(parent[label]); // path compression: flatten tree on the way up
        return parent[label];
    }

    void Union(int label_a, int label_b) {
        int root_a = Find(label_a);
        int root_b = Find(label_b);
        if (root_a == root_b) return;
        if (root_a < root_b) parent[root_b] = root_a;
        else parent[root_a] = root_b;
    }
};
