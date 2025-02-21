#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdint>

// **Global Variables**
std::vector<int> bins;           // Flat 1D vector for all particle indices
std::vector<int> bin_starts;     // Start index of each bin in bins
int nbins;                       // Number of bins per dimension
double bin_size;                 // Size of each bin

// **Morton Index Helper Functions**
inline uint32_t interleave_bits(int x, int y) {
    uint32_t z = 0;
    for (int i = 0; i < 16; ++i) {  // Assuming nbins < 2^16
        z |= ((x >> i) & 1) << (2 * i);
        z |= ((y >> i) & 1) << (2 * i + 1);
    }
    return z;
}

inline void decode_morton(int morton, int& row, int& col) {
    row = 0;
    col = 0;
    for (int i = 0; i < 16; ++i) {
        row |= ((morton >> (2 * i)) & 1) << i;
        col |= ((morton >> (2 * i + 1)) & 1) << i;
    }
}

inline int morton_index(int row, int col) {
    return static_cast<int>(interleave_bits(row, col));
}

// **Inline Apply Force**
inline void apply_force(particle_t* __restrict particle, const particle_t* __restrict neighbor) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = std::fmax(r2, min_r * min_r);
    double r = std::sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

// **Inline Move Particle**
inline void move(particle_t* __restrict p, double size) {
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// **Inline Calculate Bin Index with Clamping**
inline int calc_blocked_index(double x, double y, int nbins) {
    int bin_col = static_cast<int>(x / bin_size);
    int bin_row = static_cast<int>(y / bin_size);
    bin_col = std::max(0, std::min(nbins - 1, bin_col));
    bin_row = std::max(0, std::min(nbins - 1, bin_row));
    return morton_index(bin_row, bin_col);
}

// **Initialize Simulation**
void init_simulation(particle_t* __restrict parts, int num_parts, double size, double bin_size_param, int block_size) {
    if (size <= 0 || bin_size_param <= 0 || num_parts <= 0 || !parts) {
        std::cerr << "Invalid parameters: size=" << size
                  << ", bin_size=" << bin_size_param
                  << ", num_parts=" << num_parts << std::endl;
        exit(1);
    }

    bin_size = bin_size_param;
    nbins = std::max(1, static_cast<int>(std::ceil(size / bin_size)));
    int total_bins = nbins * nbins;

    bins.resize(num_parts + total_bins);
    bin_starts.resize(total_bins + 1, 0);

    std::vector<std::vector<int>> temp_bins(total_bins);
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = calc_blocked_index(parts[i].x, parts[i].y, nbins);
        temp_bins[bin_index].push_back(i);
    }

    int offset = 0;
    for (int i = 0; i < total_bins; ++i) {
        bin_starts[i] = offset;
        for (int idx : temp_bins[i]) {
            if (offset >= bins.size()) {
                bins.resize(bins.size() * 2);
            }
            bins[offset++] = idx;
        }
    }
    bin_starts[total_bins] = offset;
}

// **Simulate One Step**
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size) {
    int total_bins = nbins * nbins;

    // Reset accelerations and compute forces
    for (int bin_index = 0; bin_index < total_bins; ++bin_index) {
        int start = bin_starts[bin_index];
        int end = bin_starts[bin_index + 1];
        if (start < 0 || end > bins.size() || start > end) {
            std::cerr << "Invalid bin range at " << bin_index << ": start=" << start << ", end=" << end << std::endl;
            exit(1);
        }

        int bin_row, bin_col;
        decode_morton(bin_index, bin_row, bin_col);

        for (int p_idx = start; p_idx < end; ++p_idx) {
            int particle_index = bins[p_idx];
            if (particle_index < 0 || particle_index >= num_parts) {
                std::cerr << "Invalid particle index: " << particle_index << std::endl;
                exit(1);
            }
            parts[particle_index].ax = 0.0;
            parts[particle_index].ay = 0.0;

            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    int nr = bin_row + dr;
                    int nc = bin_col + dc;
                    if (nr < 0 || nr >= nbins || nc < 0 || nc >= nbins) continue;
                    int neighbor_bin = morton_index(nr, nc);
                    int n_start = bin_starts[neighbor_bin];
                    int n_end = bin_starts[neighbor_bin + 1];
                    if (n_start < 0 || n_end > bins.size() || n_start > n_end) {
                        std::cerr << "Invalid neighbor bin range at " << neighbor_bin << std::endl;
                        exit(1);
                    }
                    for (int n_idx = n_start; n_idx < n_end; ++n_idx) {
                        int neighbor_index = bins[n_idx];
                        if (neighbor_index < 0 || neighbor_index >= num_parts) {
                            std::cerr << "Invalid neighbor index: " << neighbor_index << std::endl;
                            exit(1);
                        }
                        apply_force(&parts[particle_index], &parts[neighbor_index]);
                    }
                }
            }
        }
    }

    // Move all particles
    for (int i = 0; i < num_parts; ++i) {
        move(&parts[i], size);
    }

    // Rebuild bins
    bins.clear();
    bin_starts.resize(total_bins + 1, 0);
    std::vector<std::vector<int>> temp_bins(total_bins);
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = calc_blocked_index(parts[i].x, parts[i].y, nbins);
        temp_bins[bin_index].push_back(i);
    }

    int offset = 0;
    for (int i = 0; i < total_bins; ++i) {
        bin_starts[i] = offset;
        for (int idx : temp_bins[i]) {
            if (offset >= bins.size()) {
                bins.resize(offset + num_parts);
            }
            bins[offset++] = idx;
        }
    }
    bin_starts[total_bins] = offset;
}