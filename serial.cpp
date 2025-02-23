#include "common.h"
#include <cmath>
#include <math.h>
#include <vector>
#include <set>
#include <iostream>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


std::vector<int> binNeighbors(int bin_index, double size) {
    std::vector<int> neighbors;
    int nbins = size/cutoff;
    int row = bin_index / nbins;
    int col = bin_index % nbins;

    neighbors.push_back(bin_index);

    bool leftedge = (col == 0);
    bool rightedge = (col == nbins-1);
    bool topedge = (row == nbins-1);
    bool botedge = (row == 0);

    if (botedge) {
    neighbors.push_back(col+(row+1)*nbins); // top neighbor
        if (leftedge) {
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row+1)*nbins); //top right neighbor
        }
        else if (rightedge) {
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row+1)*nbins); //top left neighbor
        }
        else {
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row+1)*nbins); //top right neighbor
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row+1)*nbins); //top left neighbor
        }
    }
    else if (topedge) {
        neighbors.push_back(col+(row-1)*nbins); // bot neighbor
        if (leftedge) {
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row-1)*nbins); //bot right neighbor
        }
        else if (rightedge) {
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row-1)*nbins); //bot left neighbor
        }
        else {
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row-1)*nbins); //bot right neighbor
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row-1)*nbins); //bot left neighbor
        }
    }
    else {
        neighbors.push_back(col+(row-1)*nbins); // bot neighbor
        neighbors.push_back(col+(row+1)*nbins); // top neighbor
        if (leftedge) {
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row-1)*nbins); //bot right neighbor
            neighbors.push_back(col+1 + (row+1)*nbins); //top right neighbor
        }
        else if (rightedge) {
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row-1)*nbins); //bot left neighbor
            neighbors.push_back(col-1 + (row+1)*nbins); //top left neighbor
        }
        else {
            neighbors.push_back(col-1 + row*nbins); //left neighbor
            neighbors.push_back(col-1 + (row-1)*nbins); //bot left neighbor
            neighbors.push_back(col-1 + (row+1)*nbins); //top left neighbor
            neighbors.push_back(col+1 + row*nbins); //right neighbor
            neighbors.push_back(col+1 + (row-1)*nbins); //bot right neighbor
            neighbors.push_back(col+1 + (row+1)*nbins); //top right neighbor
        }
    }
    return neighbors;
}
static std::vector<std::set<int>> bins;
std::vector<std::vector<int>> neighbors;

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    int nbins = size/cutoff;
    double bin_size = size/nbins;

    // Initialize bins
    bins.resize(nbins*nbins);

    // Bin particles
    for (int i = 0; i < num_parts; ++i) {
        int binrow = parts[i].y / bin_size;
        int bincol = parts[i].x / bin_size;
        bins[bincol + nbins*binrow].insert(i);
    }
    for (int i = 0; i < nbins*nbins; ++i) {
        neighbors.push_back(binNeighbors(i,size));
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int nbins = size/cutoff;
    double bin_size = size/nbins;
    //std::vector<int> neighbors(9);
    // Compute forces
    // For every bin
    for (int ii = 0; ii < nbins*nbins; ii++) {
        // Calculate which bins are neighbors
        //neighbors = binNeighbors(ii,size);
        // For each particle in bin
        for (int i : bins[ii]) {
            // Set particle acceleration to 0
            parts[i].ax = parts[i].ay = 0;
            // For each neighboring bin
            for (int jj : neighbors[ii]) {
                // For each particle in neighboring bin
                for (int j : bins[jj]) {
                    // Apply force on original particle
                    apply_force(parts[i],parts[j]);
                }
            }
        }
    }
    //
    std::set<std::pair<int,int>> movers;
    // Move Particles
    for (int i = 0; i < num_parts; i++) {
        int binrow_ini = parts[i].y / bin_size;
        int bincol_ini = parts[i].x / bin_size;
        move(parts[i], size);
        int binrow_fin = parts[i].y / bin_size;
        int bincol_fin = parts[i].x / bin_size;
        if (binrow_ini != binrow_fin || bincol_ini != bincol_fin) {
            std::pair<int,int> temp(i,bincol_ini + binrow_ini*nbins);
            movers.insert(temp);
        }
    }
    // Update bins
    //std::cout << "New step" << "\n";
    for (std::pair<int,int> i : movers) {
        //std::cout << i.first << "\n";
        bins[i.second].erase(i.first);
        int binrow = parts[i.first].y / bin_size;
        int bincol = parts[i.first].x / bin_size;
        bins[bincol + nbins*binrow].insert(i.first);
    }
}