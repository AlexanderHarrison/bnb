#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "node.h"
#include "sparse_mat.h"

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x01 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0')


float get_cost(uint32_t x, uint32_t width, const float *c) {
    float cost = 0.0f;
    for (uint32_t i = 0; i < width; ++i) {
        if ((x >> i) & 1) {
            cost += c[i];
        }
    }

    return cost;
}

bool check_valid(SparseBitMat *A, uint32_t x, int32_t *ties, float *scratch) {
    uint32_t width = A->width;
    uint32_t height = A->height;

    for (uint32_t i = 0; i < width; ++i) {
        uint32_t x_i = (x >> i) & 1;
        int32_t t_i = ties[i];

        if ((t_i != -1) & (x_i != (uint32_t) t_i)) {
            return false;
        }
    }

    mat_bitvec_mul(A, x, scratch);
    for (uint32_t i = 0; i < height; ++i) {
        if (scratch[i] > 1.0f) {
            return false;
        }
    }

    return true;
}

void solve_node(SparseBitMat *A, const float *c, Node *node) {
    uint32_t width = A->width;
    uint32_t height = A->height;
    uint32_t itercount = 1 << width;

    float *scratch = malloc(sizeof *scratch * height);

    uint32_t best_sol = 0;
    float best_cost = 0.0f;
    for (uint32_t x = 0; x < itercount; ++x) {
        if (check_valid(A, x, node->ties, scratch)) {

            float cost = get_cost(x, width, c);
            if (cost < best_cost) {
                best_cost = cost;
                best_sol = x;
            }
        }
    }

    float *x = node->x;
    for (uint32_t i = 0; i < width; ++i) {
        x[i] = (best_sol >> i) & 1;
    }
    node->cost = best_cost;

    free(scratch);
}

