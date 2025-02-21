#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>

// Global constants
#define CUTOFF 0.01
#define MIN_R 0.0001
#define MASS 1.0
#define DT 0.0005
#define BIN_SIZE 0.1
#define BLOCK_SIZE 8

// Global variables
std::vector<std::vector<int>> bins;
std::vector<std::vector<int>> neighbors;
int nbins;
int nblocks;
int bins_per_block;

// Inline Apply Force
inline void apply_force(particle_t* __restrict particle, const particle_t* __restrict neighbor) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > CUTOFF * CUTOFF) return;
    r2 = std::fmax(r2, MIN_R * MIN_R);
    double r = std::sqrt(r2);
    double coef = (1 - CUTOFF / r) / r2 / MASS;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

// Inline Move Particle
inline void move(particle_t* __restrict p, double size) {
    p->vx += p->ax * DT;
    p->vy += p->ay * DT;
    p->x += p->vx * DT;
    p->y += p->vy * DT;
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

// Inline Calculate Blocked Index
inline int calc_blocked_index(double x, double y, int nbins, int nblocks, int bins_per_block) {
    int bin_col = static_cast<int>(x / BIN_SIZE);
    int bin_row = static_cast<int>(y / BIN_SIZE);
    int bin_index = bin_row * nbins + bin_col;
    int block_col = bin_col / (nbins / nblocks);
    int block_row = bin_row / (nbins / nblocks);
    int block_index = block_row * nblocks + block_col;
    return bin_index + block_index * bins_per_block;
}

// Initialize Simulation
void init_simulation(particle_t* __restrict parts, int num_parts, double size) {
    nbins = static_cast<int>(size / BIN_SIZE);
    nblocks = nbins / BLOCK_SIZE;
    bins_per_block = (nbins / nblocks) * (nbins / nblocks);
    bins.resize(nbins * nbins);
    neighbors.resize(nbins * nbins);

    // Precompute neighbors
    for (int i = 0; i < nbins * nbins; ++i) {
        neighbors[i] = binNeighbors(i, nbins);
    }

    // Bin particles
    for (int i = 0; i < num_parts; ++i) {
        int blocked_index = calc_blocked_index(parts[i].x, parts[i].y, nbins, nblocks, bins_per_block);
        bins[blocked_index].push_back(i);
    }
}

// Simulate One Step
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size) {
    int nblockssqrd = nblocks * nblocks;

    // Reset accelerations and compute forces
    for (int block = 0; block < nblockssqrd; ++block) {
        for (int j = 0; j < bins_per_block; ++j) {
            int bin_index = block * bins_per_block + j;
            for (int particle_index : bins[bin_index]) {
                parts[particle_index].ax = 0.0;
                parts[particle_index].ay = 0.0;
                const auto& neighbor_bins = neighbors[bin_index];
                for (int neighbor_bin_index : neighbor_bins) {
                    const auto& neighbor_particles = bins[neighbor_bin_index];
                    for (int neighbor_index : neighbor_particles) {
                        apply_force(&parts[particle_index], &parts[neighbor_index]);
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