#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>

// Global variables
std::vector<std::vector<int>> bins;
std::vector<std::vector<int>> neighbors;
int nbins;
int nblocks;
int bins_per_block;
int block_size;    // Changed to int for simplicity and correctness
double bin_size;   // Global variable for bin size

// Inline Apply Force
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

// Inline Move Particle
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

// Optimized Neighbor Calculation
std::vector<int> binNeighbors(int bin_index, int nbins) {
    std::vector<int> neighbors;
    neighbors.reserve(9);
    int row = bin_index / nbins;
    int col = bin_index % nbins;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int nr = row + dr;
            int nc = col + dc;
            if (nr >= 0 && nr < nbins && nc >= 0 && nc < nbins) {
                neighbors.push_back(nr * nbins + nc);
            }
        }
    }
    return neighbors;
}

// Inline Calculate Blocked Index with Clamping
inline int calc_blocked_index(double x, double y, int nbins, int nblocks, int bins_per_block) {
    int bin_col = static_cast<int>(x / bin_size);
    int bin_row = static_cast<int>(y / bin_size);
    // Clamp to valid range
    bin_col = std::max(0, std::min(nbins - 1, bin_col));
    bin_row = std::max(0, std::min(nbins - 1, bin_row));
    return bin_row * nbins + bin_col;
}

// Initialize Simulation
void init_simulation(particle_t* __restrict parts, int num_parts, double size, double bin_size_param, int block_size_param) {
    // Input validation
    if (size <= 0 || bin_size_param <= 0 || block_size_param <= 0 || num_parts <= 0 || !parts) {
        std::cerr << "Invalid parameters: size=" << size
                  << ", bin_size=" << bin_size_param
                  << ", block_size=" << block_size_param
                  << ", num_parts=" << num_parts
                  << ", parts=" << (parts ? "non-null" : "null") << std::endl;
        exit(1);
    }

    bin_size = bin_size_param;
    block_size = static_cast<int>(block_size_param); // Convert to int
    if (block_size <= 0) {
        std::cerr << "Invalid block_size after conversion: " << block_size << std::endl;
        exit(1);
    }

    // Ensure nbins >= 1
    nbins = std::max(1, static_cast<int>(std::ceil(size / bin_size)));
    
    // Calculate nblocks to cover all bins
    nblocks = (nbins + block_size - 1) / block_size;
    
    // bins_per_block is not used in calc_blocked_index, but kept for compatibility
    bins_per_block = (nbins / nblocks) * (nbins / nblocks);
    
    bins.resize(nbins * nbins);
    neighbors.resize(nbins * nbins);

    // Debug output (optional, can be removed in production)
    std::cout << "nbins: " << nbins << ", nblocks: " << nblocks
              << ", block_size: " << block_size << std::endl;

    // Precompute neighbors
    for (int i = 0; i < nbins * nbins; ++i) {
        neighbors[i] = binNeighbors(i, nbins);
    }

    // Bin particles
    for (int i = 0; i < num_parts; ++i) {
        int bin_index = calc_blocked_index(parts[i].x, parts[i].y, nbins, nblocks, bins_per_block);
        bins[bin_index].push_back(i);
    }
}

// Simulate One Step
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size) {
    // Reset accelerations and compute forces
    for (int block_row = 0; block_row < nblocks; ++block_row) {
        for (int block_col = 0; block_col < nblocks; ++block_col) {
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    int bin_row = block_row * block_size + i;
                    int bin_col = block_col * block_size + j;
                    if (bin_row >= nbins || bin_col >= nbins) continue;
                    int bin_index = bin_row * nbins + bin_col;

                    // Debug check (optional)
                    if (bin_index < 0 || bin_index >= nbins * nbins) {
                        std::cerr << "Invalid bin_index: " << bin_index << std::endl;
                        continue;
                    }

                    for (int particle_index : bins[bin_index]) {
                        // Debug check (optional)
                        if (particle_index < 0 || particle_index >= num_parts) {
                            std::cerr << "Invalid particle_index: " << particle_index << std::endl;
                            continue;
                        }

                        parts[particle_index].ax = 0.0;
                        parts[particle_index].ay = 0.0;
                        const auto& neighbor_bins = neighbors[bin_index];
                        for (int neighbor_bin_index : neighbor_bins) {
                            const auto& neighbor_particles = bins[neighbor_bin_index];
                            for (int neighbor_index : neighbor_particles) {
                                // Debug check (optional)
                                if (neighbor_index < 0 || neighbor_index >= num_parts) {
                                    std::cerr << "Invalid neighbor_index: " << neighbor_index << std::endl;
                                    continue;
                                }
                                apply_force(&parts[particle_index], &parts[neighbor_index]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Move particles and track movers
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

    // Update bins
    for (const auto& mover : movers) {
        int particle_index = mover.first;
        int bin_index_fin = mover.second;

        // Debug check (optional)
        if (bin_index_fin < 0 || bin_index_fin >= nbins * nbins) {
            std::cerr << "Invalid bin_index_fin: " << bin_index_fin << std::endl;
            continue;
        }

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