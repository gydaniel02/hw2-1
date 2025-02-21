#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdint>  // Added to define uint32_t

// **Global Variables**
std::vector<int> bins;           // Flat 1D vector for all particle indices
std::vector<int> bin_starts;     // Start index of each bin in bins
int nbins;                       // Number of bins per dimension (assumed square grid)
int nblocks;                     // Number of blocks per dimension
int bins_per_block;              // Bins per block (for compatibility)
int block_size;                  // Size of each block (in bins)
double bin_size;                 // Size of each bin (in simulation units)

// **Morton Index Helper Functions**
inline uint32_t interleave_bits(int x, int y) {
    // Interleave bits of x and y to create a Morton code
    uint32_t z = 0;
    for (int i = 0; i < 16; ++i) {  // Assuming nbins < 2^16
        z |= ((x >> i) & 1) << (2 * i);     // x bits to even positions
        z |= ((y >> i) & 1) << (2 * i + 1); // y bits to odd positions
    }
    return z;
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
void init_simulation(particle_t* __restrict parts, int num_parts, double size, double bin_size_param, int block_size_param) {
    if (size <= 0 || bin_size_param <= 0 || block_size_param <= 0 || num_parts <= 0 || !parts) {
        std::cerr << "Invalid parameters: size=" << size
                  << ", bin_size=" << bin_size_param
                  << ", block_size=" << block_size_param
                  << ", num_parts=" << num_parts << std::endl;
        exit(1);
    }

    bin_size = bin_size_param;
    block_size = block_size_param;
    nbins = std::max(1, static_cast<int>(std::ceil(size / bin_size)));
    nblocks = (nbins + block_size - 1) / block_size;
    bins_per_block = (nbins / nblocks) * (nbins / nblocks);

    // Initialize bins and bin_starts
    bins.clear();
    bins.resize(num_parts + nbins * nbins);  // Overestimate size to avoid frequent resizing
    bin_starts.resize(nbins * nbins + 1, 0); // +1 for end marker

    // Temporary vector to collect particles per bin
    std::vector<std::vector<int>> temp_bins(nbins * nbins);
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = calc_blocked_index(parts[i].x, parts[i].y, nbins);
        temp_bins[bin_index].push_back(i);
    }

    // Populate flat bins array
    int offset = 0;
    for (int i = 0; i < nbins * nbins; ++i) {
        bin_starts[i] = offset;
        for (int idx : temp_bins[i]) {
            bins[offset++] = idx;
        }
    }
    bin_starts[nbins * nbins] = offset;  // Mark end of last bin
}

// **Simulate One Step**
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size) {
    // Reset accelerations and compute forces
    for (int block_row = 0; block_row < nblocks; ++block_row) {
        for (int block_col = 0; block_col < nblocks; ++block_col) {
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    int bin_row = block_row * block_size + i;
                    int bin_col = block_col * block_size + j;
                    if (bin_row >= nbins || bin_col >= nbins) continue;
                    int bin_index = morton_index(bin_row, bin_col);

                    int start = bin_starts[bin_index];
                    int end = bin_starts[bin_index + 1];
                    for (int p_idx = start; p_idx < end; ++p_idx) {
                        int particle_index = bins[p_idx];
                        parts[particle_index].ax = 0.0;
                        parts[particle_index].ay = 0.0;

                        // Calculate neighbor bins dynamically with Morton indexing
                        for (int dr = -1; dr <= 1; ++dr) {
                            for (int dc = -1; dc <= 1; ++dc) {
                                int nr = bin_row + dr;
                                int nc = bin_col + dc;
                                if (nr >= 0 && nr < nbins && nc >= 0 && nc < nbins) {
                                    int neighbor_bin = morton_index(nr, nc);
                                    int n_start = bin_starts[neighbor_bin];
                                    int n_end = bin_starts[neighbor_bin + 1];
                                    for (int n_idx = n_start; n_idx < n_end; ++n_idx) {
                                        int neighbor_index = bins[n_idx];
                                        apply_force(&parts[particle_index], &parts[neighbor_index]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Move particles and track movers
    std::vector<std::pair<int, int>> movers;
    movers.reserve(num_parts / 10);  // Reserve space for efficiency
    for (int bin_index = 0; bin_index < nbins * nbins; ++bin_index) {
        int start = bin_starts[bin_index];
        int end = bin_starts[bin_index + 1];
        for (int p_idx = start; p_idx < end; ) {
            int particle_index = bins[p_idx];
            move(&parts[particle_index], size);
            int bin_index_fin = calc_blocked_index(parts[particle_index].x, parts[particle_index].y, nbins);
            if (bin_index != bin_index_fin) {
                movers.push_back({particle_index, bin_index_fin});
                bins[p_idx] = bins[end - 1];  // Replace with last particle
                end--;                        // Reduce bin size
            } else {
                p_idx++;
            }
        }
        bin_starts[bin_index + 1] = end;  // Update bin end
    }

    // Update bins with movers
    for (const auto& mover : movers) {
        int particle_index = mover.first;
        int bin_index_fin = mover.second;
        int end = bin_starts[bin_index_fin + 1];
        bins[end] = particle_index;
        bin_starts[bin_index_fin + 1]++;
    }
}