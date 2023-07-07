#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "node.h"

typedef struct {
    uint32_t len;
    uint32_t capacity;
    Node *data;
} NodeHeap;

NodeHeap init_heap() {
    NodeHeap heap;
    heap.len = 0;
    heap.capacity = 16;
    heap.data = (Node*)malloc(sizeof(Node) * 16);

    return heap;
}

void dealloc_heap(NodeHeap *heap) {
    uint32_t len = heap->len;
    Node *data = heap->data;
    for (uint32_t i = 0; i < len; ++i) {
        dealloc_node(&data[i]);
    }
    free(data);
}

void add_to_heap(NodeHeap *heap, Node node) {
    uint32_t len = heap->len;
    uint32_t capacity = heap->capacity;

    if (len == capacity) {
        uint32_t new_capacity = capacity * 2;
        heap->data = (Node*)realloc(heap->data, new_capacity);
        heap->capacity = new_capacity;
    }

    heap->data[len] = node;
    ++(heap->len);
}

Node pop_heap(NodeHeap *heap) {
    uint32_t min_node_idx = 0;
    uint32_t len = heap->len;
    Node *data = heap->data;
    Node *min_node_ptr = &data[0];

    for (uint32_t i = 1; i < len; ++i) {
        Node *node = &data[i];
        if (node->cost < min_node_ptr->cost) {
            min_node_idx = i;
            min_node_ptr = node;
        }
    }

    Node min_node = data[min_node_idx];
    memmove(&data[min_node_idx], &data[min_node_idx+1], sizeof(Node) * (len - min_node_idx - 1));
    --(heap->len);

    return min_node;
}
