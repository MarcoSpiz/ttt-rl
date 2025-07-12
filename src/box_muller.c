//
// Created by Marco  on 11/07/25.
//

#include "box_muller.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_uniform_0_1() {
    return (double)(rand() + 1) / (RAND_MAX + 1.0);
}

double box_muller_transform(double mu, double sigma) {
    static int have_spare = 0;
    static double spare_value;

    if (have_spare) {
        have_spare = 0;
        return spare_value * sigma + mu;
    }

    double u1 = rand_uniform_0_1();
    double u2 = rand_uniform_0_1();

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    have_spare = 1;
    spare_value = z1;
    return z0 * sigma + mu;
}