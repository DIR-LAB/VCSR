// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <cstring>

#include "pvector.h"
#include "util.h"
using namespace std;

#define debug 0

/*
GAP Benchmark Suite
Class:  CSRGraph
Author: Scott Beamer

Simple container for graph in CSR format
 - Intended to be constructed by a Builder
 - To make weighted, set DestID_ template type to NodeWeight
 - MakeInverse parameter controls whether graph stores its inverse
*/


// each node has an associated sentinel (max_int, offset) that gets back to its
// offset into the node array
// UINT32_MAX
//
// if value == UINT32_MAX, read it as null.
template <typename NodeID_, typename WeightT_, typename TimestampT_>
struct NodeWeight {
  NodeID_ v;        // destination of this edge in the graph, MAX_INT if this is a sentinel
  WeightT_ w;       // edge value of zero means it a null since we don't store 0 edges
  TimestampT_ t;    // timestamp when this edge inserted
  NodeWeight() {}
  NodeWeight(NodeID_ v) : v(v), w(1), t(1) {}
  NodeWeight(NodeID_ v, WeightT_ w) : v(v), w(w), t(1) {}
  NodeWeight(NodeID_ v, WeightT_ w, TimestampT_ t) : v(v), w(w), t(t) {}

  bool operator< (const NodeWeight& rhs) const {
    return v == rhs.v ? w < rhs.w : v < rhs.v;
  }

  // doesn't check WeightT_s, needed to remove duplicate edges
  bool operator== (const NodeWeight& rhs) const {
    return v == rhs.v;
  }

  // doesn't check WeightT_s, needed to remove self edges
  bool operator== (const NodeID_& rhs) const {
    return v == rhs;
  }

  operator NodeID_() {
    return v;
  }
};

// Syntatic sugar for an edge
template <typename SrcT, typename DstT = SrcT>
struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int32_t SGID;
typedef int64_t TimestampT;
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;

/*****************************************************************************
*                                                                           *
*   PCSR                                                                    *
*                                                                           *
*****************************************************************************/

typedef struct _node {
  // beginning and end of the associated region in the edge list
  uint32_t beginning;     // deleted = max int
  uint32_t end;           // end pointer is exclusive
  uint32_t num_neighbors; // number of edges with this node as source
} node_t;

// find index of first 1-bit (least significant bit)
static inline int bsf_word(int word) {
  int result;
  __asm__ volatile("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

// find index of first 1-bit (most significant bit)
static inline int bsr_word(int word) {
  int result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

typedef struct _pair_int {
  int x; // length in array
  int y; // depth
} pair_int;

typedef struct _pair_double {
  double x;
  double y;
} pair_double;

template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class CSRGraph {
  // Used for *non-negative* offsets within a neighborhood
  typedef std::make_unsigned<std::ptrdiff_t>::type OffsetT;
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    DestID_* g_index_;
    OffsetT start_index_;
    OffsetT end_index_;
    OffsetT start_offset_;
   public:
    Neighborhood(DestID_* g_index, OffsetT start_index, OffsetT end_index, OffsetT start_skip, uint32_t degree) :
        g_index_(g_index), start_index_(start_index), end_index_(end_index), start_offset_(0) {
      if(start_skip >= degree) {
        start_index_ = end_index_;
      }
      else {
        uint32_t i = 0;
        while (i < start_skip && start_index_ < end_index_) {
          if (g_index_[start_index_].w != 0) i += 1;
          start_index_ += 1;
        }
      }
    }
    typedef DestID_ *iterator;
    iterator begin() { return g_index_ + start_index_; }
    iterator end()   { return g_index_ + end_index_; }
  };

  void ReleaseResources() {
    if (items != nullptr)
      delete[] items;
  }

 public:
  CSRGraph() : directed_(false), N(-1), H(-1), logN(-1),
               items(nullptr), num_edges_(-1), num_vertices(-1) {}

  CSRGraph(CSRGraph&& other) : directed_(other.directed_), nodes(other.nodes), N(other.N), H(other.H), logN(other.logN),
                               items(other.items), num_edges_(other.num_edges_), num_vertices(other.num_vertices) {
    other.nodes.clear();
    other.N = -1;
    other.H = -1;
    other.logN = -1;
    other.items = nullptr;
    other.num_edges_ = -1;
    other.num_vertices = -1;
  }

  ~CSRGraph() {
    ReleaseResources();
  }

  CSRGraph& operator=(CSRGraph&& other) {
    if (this != &other) {
      ReleaseResources();
      directed_ = other.directed_;
      nodes = other.nodes;
      N = other.N;
      H = other.H;
      logN = other.logN;
      items = other.items;
      num_edges_ = other.num_edges_;
      num_vertices = other.num_vertices;

      other.nodes.clear();
      other.N = -1;
      other.H = -1;
      other.logN = -1;
      other.items = nullptr;
      other.num_edges_ = -1;
      other.num_vertices = -1;
    }
    return *this;
  }

  CSRGraph(EdgeList &edge_list, bool is_directed, uint64_t init_e, uint64_t init_n) {
    directed_ = is_directed;
    N = 2 << bsr_word(init_n + init_e);
    // cout << "N: " << N << endl;
    logN = (1 << bsr_word(bsr_word(N) + 1));
    H = bsr_word(N / logN);

    items = (DestID_ *) malloc(N * sizeof(*(items)));
    for (int i = 0; i < N; i++) {
      items[i].w = 0;
      items[i].v = 0;
      items[i].t = 0;
    }

    num_edges_ = 0;
    num_vertices = init_n;
    delta_up = (up_0 - up_h) / H;
    delta_low = (low_h - low_0) / H;

    // adding the nodes and sentinel edges
    for (int i = 0; i < init_n; i++) {
      add_node();
    }

    // adding the edges from the base graph
    for (int i = 0; i < init_e; i++) {
      add_edge(edge_list[i].u, edge_list[i].v.v, edge_list[i].v.w);
    }

    redistribute(0, N);
  }

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_vertices;
  }

  int64_t num_edges() const {
    return num_edges_;
  }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2*num_edges_;
  }

  int64_t out_degree(NodeID_ v) const {
    return nodes[v].num_neighbors;
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return nodes[v].num_neighbors;
  }

  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    return Neighborhood(items, nodes[n].beginning + 1, nodes[n].end, start_offset, nodes[n].num_neighbors);
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return Neighborhood(items, nodes[n].beginning + 1, nodes[n].end, start_offset, nodes[n].num_neighbors);
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_vertices << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_vertices << std::endl;
  }

  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

  int isPowerOfTwo(int x) {
    return ((x != 0) && !(x & (x - 1)));
  }

  // same as find_leaf, but does it for any level in the tree
  // index: index in array
  // len: length of sub-level.
  int find_node(int index, int len) {
    return (index / len) * len;
  }

  // null overrides sentinel
  // e.g. in rebalance, we check if an edge is null
  // first before copying it into temp, then fix the sentinels.
  bool is_sentinel(DestID_ e) {
    return e.v == UINT32_MAX || e.w == UINT32_MAX;
  }

  bool is_null(DestID_ e) {
    return e.w == 0;
  }

  // get density of a node
  // todo: we can done this density calculation in more optimized way
  //  (e.g., keeping information in a complete binary-tree)
  double get_density(int index, int len) {
    int full = 0;
    for (int i = index; i < index + len; i++) {
      full += (!is_null(items[i]));
    }
    double full_d = (double)full;
    return full_d / len;
  }

  // height of this node in the tree
  int get_depth(int len) {
    return bsr_word(N / len);
  }

  // get parent of this node in the tree
  pair_int get_parent(int index, int len) {
    int parent_len = len * 2;
    int depth = get_depth(len);
    pair_int pair;
    pair.x = parent_len;
    pair.y = depth;
    return pair;
  }

  // given index, return the starting index of the leaf it is in
  int find_leaf(int index) {
    return (index / logN) * logN;
  }

  // true if e1, e2 are equals
  bool edge_equals(DestID_ e1, DestID_ e2) {
    return e1.v == e2.v && e1.w == e2.w;
  }

  // return index of the edge elem
  // takes in edge list and place to start looking
  uint32_t find_elem_pointer(uint32_t index, DestID_ elem) {
    DestID_ item = items[index];
    while (!edge_equals(item, elem)) {
      item = items[++index];
    }
    return index;
  }

  // return index of the edge elem
  // takes in edge list and place to start looking
  // looks in reverse
  uint32_t find_elem_pointer_reverse(uint32_t index, DestID_ elem) {
    DestID_ item = items[index];
    while (!edge_equals(item, elem)) {
      item = items[--index];
    }
    return index;
  }

  // important: make sure start, end don't include sentinels
  // returns the index of the smallest element bigger than you in the range
  // [start, end) if no such element is found, returns end (because insert shifts
  // everything to the right)
  uint32_t binary_search(DestID_ *elem, uint32_t start, uint32_t end) {
    while (start + 1 < end) {
      uint32_t mid = (start + end) / 2;
      DestID_ item = items[mid];

      uint32_t change = 1;
      uint32_t check = mid;

      bool flag = true;
      while (is_null(item) && flag) {
        flag = false;
        check = mid + change;
        if (check < end) {
          flag = true;
          if (check <= end) {
            item = items[check];

            if (!is_null(item)) {
              break;
            } else if (check == end) {
              break;
            }
          }
        }
        check = mid - change;
        if (check >= start) {
          flag = true;
          item = items[check];
        }
        change++;
      }

      if (is_null(item) || start == check || end == check) {
        if (!is_null(item) && start == check && elem->t <= item.t) {
          return check;
        }
        return mid;
      }

      // if we found it, return
      if (elem->t == item.t) {
        return check;
      } else if (elem->t < item.t) {
        end = check; // if the searched for item is less than current item, set end
      } else {
        start = check;
        // otherwise, searched for item is more than current and we set start
      }
    }
    if (end < start) {
      start = end;
    }
    // handling the case where there is one element left
    // if you are leq, return start (index where elt is)
    // otherwise, return end (no element greater than you in the range)
    // printf("start = %d, end = %d, n = %d\n", start,end, N);

    if (elem->t <= items[start].t && !is_null(items[start])) {
      return start;
    }
    return end;
  }

  // find index of edge
  uint32_t find_index(DestID_ *elem_pointer) {
    DestID_ *array_start = items;
    uint32_t index = (elem_pointer - array_start);
    return index;
  }

  /*****************************************************************************
   *                                                                           *
   *   PCSR                                                                    *
   *                                                                           *
   *****************************************************************************/

  // when adjusting the list size, make sure you're still in the
  // density bound
  pair_double density_bound(int depth) {
    pair_double pair;

    // between 1/4 and 1/2
    // pair.x = 1.0/2.0 - (( .25*depth)/H);
    // between 1/8 and 1/4
    pair.x = low_0 + ((H - depth) * delta_low);;
    pair.y = up_0 - ((H - depth) * delta_up);
    return pair;
  }

  void clear() {
    int n = 0;
    free(items);
    N = 2 << bsr_word(n);
    logN = (1 << bsr_word(bsr_word(N) + 1));
    H = bsr_word(N / logN);
  }

  vector<tuple<uint32_t, uint32_t, uint32_t>> get_edges() {
    uint64_t n = get_n();
    vector<tuple<uint32_t, uint32_t, uint32_t>> output;

    for (int i = 0; i < n; i++) {
      uint32_t start = nodes[i].beginning;
      uint32_t end = nodes[i].end;
      for (int j = start + 1; j < end; j++) {
        if (!is_null(items[j])) {
          output.push_back(make_tuple(i, items[j].v, items[j].w));
        }
      }
    }
    return output;
  }

  uint64_t get_n() {
    return nodes.size();
  }

  uint64_t get_size() {
    uint64_t size = nodes.capacity() * sizeof(node_t);
    size += N * sizeof(DestID_);
    return size;
  }

  void print_array() {
    for (int i = 0; i < N; i++) {
      if (is_null(items[i])) {
        printf("%d-x ", i);
      } else if (is_sentinel(items[i])) {
        uint32_t value = items[i].w;
        if (value == UINT32_MAX) {
          value = 0;
        }
        printf("\n%d-s(%u):(%d, %d) ", i, value, nodes[value].beginning, nodes[value].end);
      } else {
        printf("%d-(%d, %llu, %u) ", i, items[i].v, items[i].t, items[i].w);
      }
    }
    printf("\n\n");
  }

  // fix pointer from node to moved sentinel
  void fix_sentinel(int32_t node_index, int in) {
    nodes[node_index].beginning = in;
    if (node_index > 0) {
      nodes[node_index - 1].end = in;
    }
    if (node_index == nodes.size() - 1) {
      nodes[node_index].end = N - 1;
    }
  }

  // Evenly redistribute elements in the ofm, given a range to look into
  // index: starting position in ofm structure
  // len: area to redistribute
  void redistribute_during_resize(int index, int len) {
    // printf("REDISTRIBUTE: \n");
    // print_array();
    // std::vector<DestID_> space(len); //
    DestID_ *space = (DestID_ *)malloc(len * sizeof(*(items)));
    int j = 0;

    // move all items in ofm in the range into
    // a temp array
    for (int i = index; i < index + len; i++) {
      space[j] = items[i];
      // counting non-null edges
      j += (!is_null(items[i]));
      // setting section to null
      items[i].w = 0;
      items[i].v = 0;
      items[i].t = 0;
    }

    // evenly redistribute for a uniform density
    // rebalance_counter[bsr_word(len/logN)] += 1;
    double index_d = index;
    double step = ((double)len) / j;
    for (int i = 0; i < j; i++) {
      int in = index_d;

      items[in] = space[i];
      if (is_sentinel(space[i])) {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = space[i].w;
        if (node_index == UINT32_MAX) {
          node_index = 0;
        }
        fix_sentinel(node_index, in);
      }
      index_d += step;
    }
    free(space);
  }

  // Evenly redistribute elements in the ofm, given a range to look into
  // index: starting position in ofm structure
  // len: area to redistribute
  void redistribute(int index, int len) {
    // printf("REDISTRIBUTE: \n");
    // print_array();
    // std::vector<DestID_> space(len); //
    DestID_ *space = (DestID_ *)malloc(len * sizeof(*(items)));
    int j = 0;

    // move all items in ofm in the range into
    // a temp array
    for (int i = index; i < index + len; i++) {
      space[j] = items[i];
      // counting non-null edges
      j += (!is_null(items[i]));
      // setting section to null
      items[i].w = 0;
      items[i].v = 0;
      items[i].t = 0;
    }

    // evenly redistribute for a uniform density
    double index_d = index;
    double step = ((double)len) / j;

    for (int i = 0; i < j; i++) {
      int in = index_d;

      items[in] = space[i];
      if (is_sentinel(space[i])) {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = space[i].w;
        if (node_index == UINT32_MAX) {
          node_index = 0;
        }
        fix_sentinel(node_index, in);
      }
      index_d += step;
    }
    free(space);
  }

  void double_list() {
    N *= 2;
    logN = (1 << bsr_word(bsr_word(N) + 1));
    H = bsr_word(N / logN);
    items = (DestID_ *) realloc(items, N * sizeof(*(items)));
    for (int i = N / 2; i < N; i++) {
      items[i].w = 0; // setting second half to null
      items[i].v = 0;  // setting second half to null
    }
    redistribute_during_resize(0, N);
  }

  void half_list() {
    assert(false && "half_list() should not be called!");
    N /= 2;
    logN = (1 << bsr_word(bsr_word(N) + 1));
    H = bsr_word(N / logN);
    DestID_ *new_array = (DestID_ *) malloc(N * sizeof(*(items)));
    int j = 0;
    for (int i = 0; i < N * 2; i++) {
      if (!is_null(items[i])) {
        new_array[j++] = items[i];
      }
    }
    free(items);
    items = new_array;
    redistribute_during_resize(0, N);
  }

  // index is the beginning of the sequence that you want to slide right.
  // notice that slide right does not not null the current spot.
  // this is ok because we will be putting something in the current index
  // after sliding everything to the right.
  int slide_right(int index) {
    int rval = 0;
    DestID_ el = items[index];
    items[index].v = 0;
    items[index].w = 0;
    items[index].t = 0;
    index++;
    while (index < N && !is_null(items[index])) {
      DestID_ temp = items[index];
      items[index] = el;
      if (!is_null(el) && is_sentinel(el)) {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = el.w;
        if (node_index == UINT32_MAX) {
          node_index = 0;
        }
        fix_sentinel(node_index, index);
      }
      el = temp;
      index++;
    }
    if (!is_null(el) && is_sentinel(el)) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = el.w;
      if (node_index == UINT32_MAX) {
        node_index = 0;
      }
      fix_sentinel(node_index, index);
    }
    // TODO There might be an issue with this going of the end sometimes
    if (index == N) {
      index--;
      slide_left(index);
      rval = -1;
      printf("slide off the end on the right, should be rare\n");
    }
    items[index] = el;

    return rval;
  }

  // only called in slide right if it was going to go off the edge
  // since it can't be full this doesn't need to worry about going off the other
  // end
  void slide_left(int index) {
    DestID_ el = items[index];
    items[index].v = 0;
    items[index].w = 0;
    items[index].t = 0;

    index--;
    while (index >= 0 && !is_null(items[index])) {
      DestID_ temp = items[index];
      items[index] = el;

      if (!is_null(el) && is_sentinel(el)) {
        // fixing pointer of node that goes to this sentinel
        uint32_t node_index = el.w;
        if (node_index == UINT32_MAX) {
          node_index = 0;
        }

        fix_sentinel(node_index, index);
      }
      el = temp;
      index--;
    }

    if (index == -1) {
      double_list();

      slide_right(0);
      index = 0;
    }
    if (!is_null(el) && is_sentinel(el)) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = el.w;
      if (node_index == UINT32_MAX) {
        node_index = 0;
      }
      fix_sentinel(node_index, index);
    }

    items[index] = el;
  }

  uint32_t find_value(uint32_t src, uint32_t dest) {
    assert(false && "find_value() should not be called!");
    DestID_ e;
    e.w = 0;
    e.v = dest;
    uint32_t loc = binary_search(&e, nodes[src].beginning + 1, nodes[src].end);
    if (!is_null(items[loc]) && items[loc].v == dest) {
      return items[loc].w;
    } else {
      return 0;
    }
  }

  // NOTE: potentially don't need to return the index of the element that was
  // inserted? insert elem at index returns index that the element went to (which
  // may not be the same one that you put it at)
  uint32_t insert(uint32_t index, DestID_ elem, uint32_t src) {
    int node_index = find_leaf(index);
    // printf("node_index = %d\n", node_index);
    int level = H;
    int len = logN;

    // always deposit on the left
    if (is_null(items[index])) {
      items[index].w = elem.w;
      items[index].v = elem.v;
      items[index].t = elem.t;
    } else {
      // if the edge already exists in the graph, update its value
      // do not make another edge
      // return index of the edge that already exists
      if (!is_sentinel(elem) && items[index].v == elem.v) {
        items[index].w = elem.w;
        items[index].t = elem.t;
        return index;
      }
      if (index == N - 1) {
        // when adding to the end double then add edge
        double_list();
        node_t node = nodes[src];
//        uint32_t loc_to_add = binary_search(&elem, node.beginning + 1, node.end);
        uint32_t loc_to_add = node.end;
        // note: the following way increase the number of rebalance in the higher level (near root) of the pma-tree;
        // and slightly increase the overall runtime
//        uint32_t loc_to_add;
//        if(is_null(items[node.end - 1])) loc_to_add = node.end - 1;
//        else loc_to_add = node.end;
        return insert(loc_to_add, elem, src);
      } else {
        if (slide_right(index) == -1) {
          index -= 1;
          slide_left(index);
        }
      }
      items[index].w = elem.w;
      items[index].v = elem.v;
      items[index].t = elem.t;
    }

    // get density of the leaf you are in
    double density = get_density(node_index, len);

    // get density boundary of the leaf you are in
    pair_double density_b = density_bound(level);

    // while density too high, go up the implicit tree
    // go up to the biggest node above the density bound
    while (density >= density_b.y) {
      len *= 2;
      if (len <= N) {
        level--;
        node_index = find_node(node_index, len);
        density_b = density_bound(level);
        density = get_density(node_index, len);
      } else {
        // if you reach the root, double the list
        double_list();

        // search from the beginning because list was doubled
        return index;
        // return find_elem_pointer(0, elem);
      }
    }

    // do the re-balance at the appropriate level
    // this always confirm the re-balance at least from the level-0
    redistribute(node_index, len);

    // return find_elem_pointer(node_index, elem);
    return index;
  }

  void print_graph_adj_matrix() {
    int num_vertices = nodes.size();
    for (int i = 0; i < num_vertices; i++) {
      // +1 to avoid sentinel
      int matrix_index = 0;

      for (uint32_t j = nodes[i].beginning + 1; j < nodes[i].end; j++) {
        if (!is_null(items[j])) {
          while (matrix_index < items[j].v) {
            printf("000 ");
            matrix_index++;
          }
          printf("%03d ", items[j].w);
          matrix_index++;
        }
      }
      for (uint32_t j = matrix_index; j < num_vertices; j++) {
        printf("000 ");
      }
      printf("\n");
    }
  }

  void print_graph_edge_list() {
    int num_vertices = nodes.size();
    for (int i = 0; i < num_vertices; i++) {
      printf("%d:", i);
      // +1 to avoid sentinel
      for (uint32_t j = nodes[i].beginning + 1; j < nodes[i].end; j++) {
        if (!is_null(items[j])) {
          printf(" %d", items[j].v);
        }
      }
      printf("\n");
    }
  }

  void print_pma_meta() {
    cout << "num_edges: " << num_edges_ << ", num_vertices: " << nodes.size() << ", elem_capacity: " << N << endl;
    cout << "segment_size: " << logN << ", segment_count: " << (N / logN) << ", tree_height: " << H << endl;
  }

  // add a node to the graph
  void add_node() {
    node_t node;
    int len = nodes.size();
    DestID_ sentinel;
    sentinel.v = UINT32_MAX; // placeholder
    sentinel.w = len;       // back pointer
    sentinel.t = num_edges_;

    // incrementing number of edges
    num_edges_ += 1;

    if (len > 0) {
      node.beginning = nodes[len - 1].end;
      node.end = node.beginning + 1;
    } else {
      node.beginning = 0;
      node.end = 1;
      sentinel.w = UINT32_MAX;
    }
    node.num_neighbors = 0;

    nodes.push_back(node);
    insert(node.beginning, sentinel, nodes.size() - 1);
  }

  void add_edge(uint32_t src, uint32_t dest, uint64_t value) {
    if (value != 0) {
      node_t node = nodes[src];
      nodes[src].num_neighbors++;

      DestID_ e;
      e.v = dest;
      e.w = value;
      e.t = num_edges_;

      // incrementing number of edges
      num_edges_ += 1;

//      uint32_t loc_to_add = binary_search(&e, node.beginning + 1, node.end);
      uint32_t loc_to_add = node.end;
      // note: the following way increase the number of rebalance in the higher level (near root) of the pma-tree;
      // and slightly increase the overall runtime
//      uint32_t loc_to_add;
//      if(is_null(items[node.end - 1])) loc_to_add = node.end - 1;
//      else loc_to_add = node.end;
      insert(loc_to_add, e, src);
    }
  }

 private:
  bool directed_;
  /* PMA constants */
  // Height-based (as opposed to depth-based) tree thresholds
  // Upper density thresholds
  const double up_h = 0.75; // root
  const double up_0 = 1.00; // leaves
  // Lower density thresholds
  const double low_h = 0.50; // root
  const double low_0 = 0.25; // leaves

  /* General PMA fields */
  double delta_up;        // Delta for upper density threshold
  double delta_low;       // Delta for lower density threshold

  // data members
  std::vector<node_t> nodes;
  int N;
  int H;
  int logN;
  DestID_ *items;
  int64_t num_edges_;
  uint32_t num_vertices;
};

#endif  // GRAPH_H_
