#include "common.h" // Assumes definitions for particle_t, cutoff, min_r, mass, dt
#include <cmath>
#include <vector>

// Global data structures
static std::vector<std::vector<int>> bins;      // Bins storing particle indices
static std::vector<std::vector<int>> neighbors; // Neighbors for each bin

// Optimized apply_force function
inline void apply_force(particle_t* restrict particle, particle_t* restrict neighbor) {
    double dx = neighbor->x - particle->x;
    double dy = neighbor->y - particle->y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

// Optimized move function
inline void move(particle_t* restrict p, double size) {
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

// Compute neighboring bins for a given bin
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
        neighbors.push_back(col + (row + 1) * nbins); // top
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left
        } else {
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left
        }
    } else if (topedge) {
        neighbors.push_back(col + (row - 1) * nbins); // bottom
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left
        } else {
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left
        }
    } else {
        neighbors.push_back(col + (row - 1) * nbins); // bottom
        neighbors.push_back(col + (row + 1) * nbins); // top
        if (leftedge) {
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right
        } else if (rightedge) {
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left
        } else {
            neighbors.push_back(col - 1 + row * nbins); // left
            neighbors.push_back(col - 1 + (row - 1) * nbins); // bottom left
            neighbors.push_back(col - 1 + (row + 1) * nbins); // top left
            neighbors.push_back(col + 1 + row * nbins); // right
            neighbors.push_back(col + 1 + (row - 1) * nbins); // bottom right
            neighbors.push_back(col + 1 + (row + 1) * nbins); // top right
        }
    }
    return neighbors;
}

// Initialize the simulation
void init_simulation(particle_t* parts, int num_parts, double size) {
    int nbins = size / cutoff;
    double bin_size = size / nbins;

    bins.resize(nbins * nbins);
    for (int i = 0; i < num_parts; i++) {
        int binrow = parts[i].y / bin_size;
        int bincol = parts[i].x / bin_size;
        int bin_index = bincol + nbins * binrow;
        bins[bin_index].push_back(i);
    }

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
                    apply_force(&parts[i], &parts[j]);
                }
            }
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(&parts[i], size);
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