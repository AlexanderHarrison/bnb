#pragma once

#include <stdlib.h>
#include <stdint.h>

typedef struct {
    uint16_t col;
    uint16_t row;
} MatBit;

typedef struct {
    MatBit *bits;
    uint32_t bitcount;
    uint32_t width;
    uint32_t height;
} SparseBitMat;

SparseBitMat init_sparse_bit_mat(uint32_t width, uint32_t height, const float* rawdata) {
    uint32_t len = width * height;
    uint32_t bitcount = 0;
    for (uint32_t i = 0; i < len; ++i) {
        if (rawdata[i] != 0.0f) {
            bitcount += 1;
        }
    }

    MatBit *bits = malloc(sizeof *bits * bitcount);

    uint32_t bitnum = 0;
    for (uint32_t i = 0; i < len; ++i) {
        if (rawdata[i] != 0.0f) {
            MatBit *bit = &bits[bitnum];
            bit->col = i % width;
            bit->row = i / width;
            ++bitnum;
        }
    }

    SparseBitMat mat;
    mat.bitcount = bitcount;
    mat.bits = bits;
    mat.width = width;
    mat.height = height;
    return mat;
}

void mat_vec_mul(const SparseBitMat *mat, const float *vec, float *out) {
    uint32_t height = mat->height;
    uint32_t bitcount = mat->bitcount;
    MatBit *bits = mat->bits;

    for (uint32_t i = 0; i < height; ++i) {
        out[i] = 0.0f;
    }

    for (uint32_t i = 0; i < bitcount; ++i) {
        uint16_t col = bits[i].col;
        uint16_t row = bits[i].row;
        out[row] += vec[col];
    }
}

void mat_bitvec_mul(const SparseBitMat *mat, uint32_t bitvec, float *out) {
    uint32_t height = mat->height;
    uint32_t bitcount = mat->bitcount;
    MatBit *bits = mat->bits;

    for (uint32_t i = 0; i < height; ++i) {
        out[i] = 0.0f;
    }

    for (uint32_t i = 0; i < bitcount; ++i) {
        uint16_t col = bits[i].col;
        uint16_t row = bits[i].row;
        out[row] += (bitvec >> col) & 1;
    }
}
