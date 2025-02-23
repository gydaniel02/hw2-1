#include "common.h"
#include <cmath>
#include <vector>

// Global bins storing particle indices
static std::vector<std::vector<int>> bins;

// Apply force symmetrically to both particles
inline void apply_force_both(particle_t* __restrict p1, particle_t* __restrict p2) {
    double dx = p2->x - p1->x;
    double dy = p2->y - p1->y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    double fx = coef * dx;
    double fy = coef * dy;

    // Update accelerations symmetrically
    p1->ax += fx;
    p1->ay += fy;
    p2->ax -= fx;
    p2->ay -= fy;
}

// Move a particle and handle boundary conditions
inline void move(particle_t* __restrict p, double size) {
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Periodic boundary conditions
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -p->x : 2 * size - p->x;
        p->vx = -p->vx;
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -p->y : 2 * size - p->y;
        p->vy = -p->vy;
    }
}

// Initialize the simulation by assigning particles to bins
void init_simulation(particle_t* parts, int num_parts, double size) {
    int nbins = static_cast<int>(size / cutoff);
    double bin_size = size / nbins;
    bins.resize(nbins * nbins);

    // Assign particles to bins
    for (int i = 0; i < num_parts; i++) {
        int binrow = static_cast<int>(parts[i].y / bin_size);
        int bincol = static_cast<int>(parts[i].x / bin_size);
        int bin_index = bincol + nbins * binrow;
        bins[bin_index].push_back(i);
    }
}

// Simulate one time step
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int nbins = static_cast<int>(size / cutoff);
    double bin_size = size / nbins;

    // Reset all accelerations
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = parts[i].ay = 0;
    }

    // Compute forces using dynamic neighbor calculation
    for (int ii = 0; ii < nbins * nbins; ii++) {
        int row = ii / nbins;
        int col = ii % nbins;

        // Intra-bin interactions (within the same bin)
        const auto& bin_ii = bins[ii];
        for (size_t k = 0; k < bin_ii.size(); k++) {
            int i = bin_ii[k];
            for (size_t m = k + 1; m < bin_ii.size(); m++) {
                int j = bin_ii[m];
                apply_force_both(&parts[i], &parts[j]);
            }
        }

        // Inter-bin interactions (between neighboring bins, jj > ii)
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue; // Skip intra-bin case
                int nrow = row + dr;
                int ncol = col + dc;
                if (nrow >= 0 && nrow < nbins && ncol >= 0 && ncol < nbins) {
                    int jj = nrow * nbins + ncol;
                    if (jj > ii) { // Ensure each pair is processed only once
                        const auto& bin_jj = bins[jj];
                        for (int i : bin_ii) {
                            for (int j : bin_jj) {
                                apply_force_both(&parts[i], &parts[j]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Update particle positions
    for (int i = 0; i < num_parts; i++) {
        move(&parts[i], size);
    }

    // Rebuild bins from scratch
    for (auto& bin : bins) {
        bin.clear();
    }
    for (int i = 0; i < num_parts; i++) {
        int binrow = static_cast<int>(parts[i].y / bin_size);
        int bincol = static_cast<int>(parts[i].x / bin_size);
        int bin_index = bincol + nbins * binrow;
        bins[bin_index].push_back(i);
    }
}