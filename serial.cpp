#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>

// **Global Variables**
std::vector<std::vector<int>> bins;  // Stores particle indices in each bin
int nbins;                           // Number of bins per dimension
int nblocks;                         // Number of blocks per dimension
int bins_per_block;                  // Bins per block (kept for compatibility)
int block_size;                      // Size of each block (in bins)
double bin_size;                     // Size of each bin (in simulation units)

// **Inline Apply Force**
inline void apply_force(particle_t* __restrict particle, const particle_t* __restrict neighbor) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;  // Skip if distance exceeds cutoff
    r2 = std::fmax(r2, min_r * min_r); // Prevent division by zero
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
    // Handle boundary conditions
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// **Inline Calculate Blocked Index with Clamping**
inline int calc_blocked_index(double x, double y, int nbins, int nblocks, int bins_per_block) {
    int bin_col = static_cast<int>(x / bin_size);
    int bin_row = static_cast<int>(y / bin_size);
    // Clamp to valid bin range
    bin_col = std::max(0, std::min(nbins - 1, bin_col));
    bin_row = std::max(0, std::min(nbins - 1, bin_row));
    return bin_row * nbins + bin_col;
}

// **Initialize Simulation**
void init_simulation(particle_t* __restrict parts, int num_parts, double size, double bin_size_param, int block_size_param) {
    // **Input Validation**
    if (size <= 0 || bin_size_param <= 0 || block_size_param <= 0 || num_parts <= 0 || !parts) {
        std::cerr << "Invalid parameters: size=" << size
                  << ", bin_size=" << bin_size_param
                  << ", block_size=" << block_size_param
                  << ", num_parts=" << num_parts
                  << ", parts=" << (parts ? "non-null" : "null") << std::endl;
        exit(1);
    }

    bin_size = bin_size_param;
    block_size = block_size_param;  // Already an int

    // **Calculate Grid Parameters**
    nbins = std::max(1, static_cast<int>(std::ceil(size / bin_size)));  // Ensure at least 1 bin
    nblocks = (nbins + block_size - 1) / block_size;                    // Number of blocks to cover all bins
    bins_per_block = (nbins / nblocks) * (nbins / nblocks);             // Kept for compatibility

    bins.resize(nbins * nbins);  // Resize bins array

    // **Debug Output** (optional, removable in production)
    std::cout << "nbins: " << nbins << ", nblocks: " << nblocks
              << ", block_size: " << block_size << std::endl;

    // **Bin Particles**
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = calc_blocked_index(parts[i].x, parts[i].y, nbins, nblocks, bins_per_block);
        bins[bin_index].push_back(i);
    }
}

// **Simulate One Step**
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size) {
    // **Reset Accelerations and Compute Forces**
    for (int block_row = 0; block_row < nblocks; ++block_row) {
        for (int block_col = 0; block_col < nblocks; ++block_col) {
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    int bin_row = block_row * block_size + i;
                    int bin_col = block_col * block_size + j;
                    if (bin_row >= nbins || bin_col >= nbins) continue;  // Skip out-of-bounds bins
                    int bin_index = bin_row * nbins + bin_col;

                    for (int particle_index : bins[bin_index]) {
                        parts[particle_index].ax = 0.0;
                        parts[particle_index].ay = 0.0;

                        // **Calculate Neighboring Bins Dynamically**
                        for (int dr = -1; dr <= 1; ++dr) {
                            for (int dc = -1; dc <= 1; ++dc) {
                                int nr = bin_row + dr;
                                int nc = bin_col + dc;
                                if (nr >= 0 && nr < nbins && nc >= 0 && nc < nbins) {
                                    int neighbor_bin_index = nr * nbins + nc;
                                    const auto& neighbor_particles = bins[neighbor_bin_index];
                                    for (int neighbor_index : neighbor_particles) {
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

    // **Move Particles and Track Movers**
    std::vector<std::pair<int, int>> movers;
    for (int bin_index = 0; bin_index < nbins * nbins; ++bin_index) {
        for (int particle_index : bins[bin_index]) {
            move(&parts[particle_index], size);
            int bin_index_fin = calc_blocked_index(parts[particle_index].x, parts[particle_index].y, nbins, nblocks, bins_per_block);
            if (bin_index != bin_index_fin) {
                movers.push_back({particle_index, bin_index_fin});
            }
        }
    }

    // **Update Bins**
    for (const auto& mover : movers) {
        int particle_index = mover.first;
        int bin_index_fin = mover.second;
        for (int bin_index = 0; bin_index < nbins * nbins; ++bin_index) {
            auto& bin = bins[bin_index];
            for (auto it = bin.begin(); it != bin.end(); ) {
                if (*it == particle_index) {
                    it = bin.erase(it);
                    break;
                } else {
                    ++it;
                }
            }
        }
        bins[bin_index_fin].push_back(particle_index);
    }
}