#include "common.h"
#include <vector>
#include <iostream>

std::vector<int> bins;
std::vector<int> bin_starts;
int nbins;
double bin_size;

inline void apply_force(particle_t* particle, const particle_t* neighbor) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = std::fmax(r2, min_r * min_r);
    double coef = (1 - cutoff / std::sqrt(r2)) / r2 / mass;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

inline void move(particle_t* p, double size) {
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    while (p->x < 0 || p->x > size) p->x = p->x < 0 ? -p->x : 2 * size - p->x, p->vx = -p->vx;
    while (p->y < 0 || p->y > size) p->y = p->y < 0 ? -p->y : 2 * size - p->y, p->vy = -p->vy;
}

inline int calc_bin_index(double x, double y) {
    int col = std::min(nbins - 1, std::max(0, static_cast<int>(x / bin_size)));
    int row = std::min(nbins - 1, std::max(0, static_cast<int>(y / bin_size)));
    return row * nbins + col;
}

void init_simulation(particle_t* parts, int num_parts, double size, double bin_size_param, int block_size) {
    bin_size = bin_size_param;
    nbins = std::max(1, static_cast<int>(std::ceil(size / bin_size)));
    int total_bins = nbins * nbins;

    bins.resize(num_parts);
    bin_starts.assign(total_bins + 1, 0);

    // Count particles per bin
    for (int i = 0; i < num_parts; ++i) {
        int bin = calc_bin_index(parts[i].x, parts[i].y);
        bin_starts[bin + 1]++;
    }
    // Compute cumulative starts
    for (int i = 1; i <= total_bins; ++i) {
        bin_starts[i] += bin_starts[i - 1];
    }
    // Fill bins
    std::vector<int> counts(total_bins, 0);
    for (int i = 0; i < num_parts; ++i) {
        int bin = calc_bin_index(parts[i].x, parts[i].y);
        bins[bin_starts[bin] + counts[bin]++] = i;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int total_bins = nbins * nbins;

    // Move and reset forces
    for (int i = 0; i < num_parts; ++i) {
        move(&parts[i], size);
    }

    // Rebuild bins efficiently
    bin_starts.assign(total_bins + 1, 0);
    for (int i = 0; i < num_parts; ++i) {
        int bin = calc_bin_index(parts[i].x, parts[i].y);
        bin_starts[bin + 1]++;
    }
    for (int i = 1; i <= total_bins; ++i) {
        bin_starts[i] += bin_starts[i - 1];
    }
    std::vector<int> counts(total_bins, 0);
    for (int i = 0; i < num_parts; ++i) {
        int bin = calc_bin_index(parts[i].x, parts[i].y);
        bins[bin_starts[bin] + counts[bin]++] = i;
    }

    // Compute forces
    for (int bin = 0; bin < total_bins; ++bin) {
        int start = bin_starts[bin];
        int end = bin_starts[bin + 1];
        int row = bin / nbins;
        int col = bin % nbins;

        for (int i = start; i < end; ++i) {
            particle_t& p = parts[bins[i]];
            p.ax = p.ay = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    int nr = row + dr;
                    int nc = col + dc;
                    if (nr < 0 || nr >= nbins || nc < 0 || nc >= nbins) continue;
                    int nbin = nr * nbins + nc;
                    for (int j = bin_starts[nbin]; j < bin_starts[nbin + 1]; ++j) {
                        apply_force(&p, &parts[bins[j]]);
                    }
                }
            }
        }
    }
}