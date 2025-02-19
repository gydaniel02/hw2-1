#include "common.h"
#include <omp.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <set>

// Put any static global variables here that you will use throughout the simulation.
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
    //#pragma omp atomic
    particle.ax += coef * dx;
    //#pragma omp atomic
    particle.ay += coef * dy;
}
double get_ax(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return 0;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    return coef * dx;
}
double get_ay(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return 0;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    return coef * dy;
}

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

static std::vector<std::set<int>> bins;

static std::vector<std::set<int>> bins_00;
static std::vector<std::set<int>> bins_01;
static std::vector<std::set<int>> bins_10;
static std::vector<std::set<int>> bins_11;

static std::vector<std::set<int>> bins_00_neighbors;
static std::vector<std::set<int>> bins_01_neighbors;
static std::vector<std::set<int>> bins_10_neighbors;
static std::vector<std::set<int>> bins_11_neighbors;


std::vector<int> binNeighbors(int bin_index, double size) {
    std::vector<int> neighbors;
    neighbors.reserve(9);

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
// Initialize globals
std::vector<std::pair<int,int>> movers;
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
        neighbors.push_back(binNeighbors(nbins*nbins-1-i,size));
    }
}


inline void simulate_one_step_shift_grid(particle_t* parts, const int& num_parts, const double& size, const int& h_shift, const int& v_shift) {
    double bin_size = cutoff;
    int nbins = size/bin_size;

    // todo: index recalculated in every iteration
    // todo: go through grid in blocks for better cache performance 
    #pragma omp parallel for schedule(dynamic) collapse(2) 
    for (size_t i = h_shift; i < nbins; i += 2) {
        for (size_t j = v_shift; j < nbins; j += 2) {
            size_t bin_index = i + j * nbins;
            // For each particle in the bin
            for (auto particle_idx : bins[bin_index]) {

                parts[particle_idx].ax = 0;
                parts[particle_idx].ay = 0;

                // Iterate over all neighboring bins
                for (int h = -1; h <= 1; ++h) {
                    for (int v = -1; v <= 1; ++v) {
                        int n_i = i + h;
                        int n_j = j + v;
                        if (n_i < 0 || n_i >= nbins || n_j < 0 || n_j >= nbins)
                            continue;
                        int neighbor_bin = n_i + n_j * nbins;
                        // For each particle in the neighbor bin
                        for (auto neighbor_particle_idx : bins[neighbor_bin]) {
                            apply_force(parts[particle_idx], parts[neighbor_particle_idx]);
                        }
                    }
                }
            }
        }
    }


}



void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int nbins = size/cutoff;
    double bin_size = size/nbins;
    //std::vector<int> neighbors(9);
    // Compute forces
    // For every bin
    

    /*
    #pragma omp schedule(dynamic) for collapse(4)
    for (int ii = 0; ii < nbins*nbins; ii++) {
        // For each particle in bin
        for (int i : bins[ii]) {
            // Set particle acceleration to 0
            parts[i].ax = parts[i].ay = 0;
            double ax,ay;
            // For each neighboring bin
            for (int jj : neighbors[ii]) {
                // For each particle in neighboring bin
                for (int j : bins[jj]) {
                    // Apply force on original particle
                    apply_force(parts[i],parts[j]);
                    // ax = get_ax(parts[i],parts[j]);
                    // ay = get_ay(parts[i],parts[j]);
                }
            }
            // #pragma omp single
            // {
            //     parts[i].ax = ax;
            //     parts[i].ay = ay;
            // }
        }
    }
    */

   #pragma omp single {
        simulate_one_step_shift_grid(parts, num_parts, size, 0, 0);
        simulate_one_step_shift_grid(parts, num_parts, size, 1, 0);
        simulate_one_step_shift_grid(parts, num_parts, size, 0, 1);
        simulate_one_step_shift_grid(parts, num_parts, size, 1, 1);
   }
    
    // Move Particles
    #pragma omp schedule(static) for collapse(2) shared (movers)
    for (int ii = 0; ii < nbins*nbins; ii++) {
        for (const int& i : bins[ii]) {
            int binrow_ini = parts[i].y / bin_size;
            int bincol_ini = parts[i].x / bin_size;
            #pragma omp critical
            {
                move(parts[i], size);
            }
            int binrow_fin = parts[i].y / bin_size;
            int bincol_fin = parts[i].x / bin_size;
            #pragma omp critical 
            {
                if (binrow_ini != binrow_fin || bincol_ini != bincol_fin) {
                    std::pair<int,int> temp(i,bincol_ini + binrow_ini*nbins);
                    movers.push_back(temp);
                }
            }
        }
    }
    // Update bins
    #pragma omp single
    {
        for (std::pair<int,int> i : movers) {
            bins[i.second].erase(i.first);
            int binrow = parts[i.first].y / bin_size;
            int bincol = parts[i.first].x / bin_size;
            bins[bincol + nbins*binrow].insert(i.first);
        }
    }

    movers.clear();
}
