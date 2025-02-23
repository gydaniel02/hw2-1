#include "common.h"
#include <cmath>
#include <vector>

// Global data structures
static std::vector<std::vector<int>> bins;      // Bins storing particle indices
static std::vector<std::vector<int>> neighbors; // All neighbors for each bin (including self)

// Original apply_force function
void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE (unchanged)
void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Compute all neighboring bins for a given bin (unchanged)
std::vector<int> binNeighbors(int bin_index, double size) {
    std::vector<int> neighbors;
    int nbins = size / cutoff;
    int row = bin_index / nbins;
    int col = bin_index % nbins;

    neighbors.push_back(bin_index);

    bool leftedge = (col == 0);
    bool rightedge = (col == nbins - 1);
    bool topedge = (row == nbins - 1);
    bool botedge = (row == 0);

    if (botedge) {
        neighbors.push_back(col + (row + 1) * nbins); // top neighbor
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right neighbor
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left neighbor
        } else {
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right neighbor
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left neighbor
        }
    } else if (topedge) {
        neighbors.push_back(col + (row - 1) * nbins); // bottom neighbor
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right neighbor
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left neighbor
        } else {
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right neighbor
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left neighbor
        }
    } else {
        neighbors.push_back(col + (row - 1) * nbins); // bottom neighbor
        neighbors.push_back(col + (row + 1) * nbins); // top neighbor
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right neighbor
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right neighbor
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left neighbor
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left neighbor
        } else {
            neighbors.push_back(col - 1 + row * nbins); // left neighbor
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left neighbor
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left neighbor
            neighbors.push_back(col + 1 + row * nbins); // right neighbor
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right neighbor
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right neighbor
        }
    }
    return neighbors;
}

// Initialize the simulation
void init_simulation(particle_t* parts, int num_parts, double size) {
    int nbins = size / cutoff;
    double bin_size = size / nbins;

    // Initialize bins
    bins.resize(nbins * nbins);
    for (int i = 0; i < num_parts; i++) {
        int binrow = parts[i].y / bin_size;
        int bincol = parts[i].x / bin_size;
        int bin_index = bincol + nbins * binrow;
        bins[bin_index].push_back(i);
    }

    // Compute neighbors
    neighbors.resize(nbins * nbins);
    for (int i = 0; i < nbins * nbins; i++) {
        neighbors[i] = binNeighbors(i, size);
    }
}

// Simulate one time step
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int nbins = size / cutoff;
    double bin_size = size / nbins;

    // Compute forces
    for (int ii = 0; ii < nbins * nbins; ii++) {
        for (int i : bins[ii]) {
            parts[i].ax = parts[i].ay = 0;
            for (int jj : neighbors[ii]) {
                for (int j : bins[jj]) {
                    apply_force(parts[i], parts[j]);
                }
            }
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }

    // Rebuild bins
    for (auto& bin : bins) {
        bin.clear();
    }
    for (int i = 0; i < num_parts; i++) {
        int binrow = parts[i].y / bin_size;
        int bincol = parts[i].x / bin_size;
        int bin_index = bincol + nbins * binrow;
        bins[bin_index].push_back(i);
    }
}