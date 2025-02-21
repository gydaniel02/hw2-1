//#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>

// Particle structure with cache alignment
struct alignas(64) particle_t {
    double x, y, vx, vy, ax, ay;
};

// Global constants
constexpr double CUTOFF = 0.01;
constexpr double min_r = 0.0001;
constexpr double mass = 1.0;
constexpr double dt = 0.0005;

// Global variables
std::vector<std::vector<int>> bins_morton;
std::vector<std::vector<unsigned int>> neighbors_morton;
std::vector<unsigned int> particle_bins;

// Inline Morton Encoding
inline unsigned int mortonEncode(unsigned int x, unsigned int y) {
    unsigned int z = 0;
    for (unsigned int i = 0; i < sizeof(unsigned int) * 8 / 2; ++i) {
        z |= (x & (1u << i)) << i | (y & (1u << i)) << (i + 1);
    }
    return z;
}

// Inline Remove from Bin
inline void remove_from_bin(std::vector<int>& bin, int particle_idx) {
    for (size_t k = 0; k < bin.size(); ++k) {
        if (bin[k] == particle_idx) {
            bin[k] = bin.back();
            bin.pop_back();
            return;
        }
    }
}

// Inline Apply Force Locally
inline void apply_force_local(const particle_t* __restrict particle, 
                             const particle_t* __restrict neighbor, 
                             double& ax_local, double& ay_local) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > CUTOFF * CUTOFF) return;
    r2 = std::fmax(r2, min_r * min_r);
    double r = std::sqrt(r2);
    double coef = (1 - CUTOFF / r) / r2 / mass;
    ax_local += coef * dx;
    ay_local += coef * dy;
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

// Initialize Simulation
void init_simulation(particle_t* __restrict parts, int num_parts, double size) {
    int nbins = static_cast<int>(size / CUTOFF);
    double bin_size = size / nbins;
    bins_morton.resize(nbins * nbins);
    neighbors_morton.resize(nbins * nbins);
    particle_bins.resize(num_parts);

    // Precompute neighbors
    for (int row = 0; row < nbins; ++row) {
        for (int col = 0; col < nbins; ++col) {
            unsigned int morton_idx = mortonEncode(row, col);
            std::vector<unsigned int> neighbor_mortons;
            neighbor_mortons.reserve(9); // Max 9 neighbors
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    int nr = row + dr;
                    int nc = col + dc;
                    if (nr >= 0 && nr < nbins && nc >= 0 && nc < nbins) {
                        neighbor_mortons.push_back(mortonEncode(nr, nc));
                    }
                }
            }
            neighbors_morton[morton_idx] = std::move(neighbor_mortons);
        }
    }

    // Bin particles
    for (int i = 0; i < num_parts; ++i) {
        int binrow = static_cast<int>(parts[i].y / bin_size);
        int bincol = static_cast<int>(parts[i].x / bin_size);
        unsigned int morton_idx = mortonEncode(binrow, bincol);
        bins_morton[morton_idx].push_back(i);
        particle_bins[i] = morton_idx;
    }
}

// Simulate One Step
void simulate_one_step(particle_t* __restrict parts, int num_parts, double size, double bin_size, double block_size) {
    int nbins = static_cast<int>(size / CUTOFF);
    double bin_size = size / nbins;

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> ax_local(num_threads, std::vector<double>(num_parts, 0.0));
    std::vector<std::vector<double>> ay_local(num_threads, std::vector<double>(num_parts, 0.0));
    std::vector<std::vector<std::pair<int, unsigned int>>> movers_per_thread(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (unsigned int morton_idx = 0; morton_idx < nbins * nbins; ++morton_idx) {
            for (int i : bins_morton[morton_idx]) {
                const auto& neighbors = neighbors_morton[morton_idx];
                size_t num_neighbors = neighbors.size();
                for (size_t k = 0; k < num_neighbors; k += 2) {
                    unsigned int neighbor1 = neighbors[k];
                    const auto& bin1 = bins_morton[neighbor1];
                    #pragma omp simd
                    for (int j : bin1) {
                        apply_force_local(&parts[i], &parts[j], ax_local[thread_id][i], ay_local[thread_id][i]);
                    }
                    if (k + 1 < num_neighbors) {
                        unsigned int neighbor2 = neighbors[k + 1];
                        const auto& bin2 = bins_morton[neighbor2];
                        #pragma omp simd
                        for (int j : bin2) {
                            apply_force_local(&parts[i], &parts[j], ax_local[thread_id][i], ay_local[thread_id][i]);
                        }
                    }
                }
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < num_parts; ++i) {
            unsigned int morton_idx_ini = particle_bins[i];
            move(&parts[i], size);
            int binrow_fin = static_cast<int>(parts[i].y / bin_size);
            int bincol_fin = static_cast<int>(parts[i].x / bin_size);
            unsigned int morton_idx_fin = mortonEncode(binrow_fin, bincol_fin);
            if (morton_idx_ini != morton_idx_fin) {
                movers_per_thread[thread_id].push_back({i, morton_idx_ini});
            }
        }
    }

    // Combine accelerations
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_parts; ++i) {
        double ax_sum = 0.0, ay_sum = 0.0;
        for (int t = 0; t < num_threads; ++t) {
            ax_sum += ax_local[t][i];
            ay_sum += ay_local[t][i];
        }
        parts[i].ax = ax_sum;
        parts[i].ay = ay_sum;
    }

    // Update bins
    for (const auto& thread_movers : movers_per_thread) {
        for (const auto& mover : thread_movers) {
            int i = mover.first;
            unsigned int morton_idx_ini = mover.second;
            remove_from_bin(bins_morton[morton_idx_ini], i);
            int binrow_fin = static_cast<int>(parts[i].y / bin_size);
            int bincol_fin = static_cast<int>(parts[i].x / bin_size);
            unsigned int morton_idx_fin = mortonEncode(binrow_fin, bincol_fin);
            bins_morton[morton_idx_fin].push_back(i);
            particle_bins[i] = morton_idx_fin;
        }
    }
}
