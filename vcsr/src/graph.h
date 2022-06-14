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


// Used to hold node & weight, with another node it makes a weighted edge
template <typename NodeID_, typename WeightT_, typename TimestampT_>
struct NodeWeight {
  NodeID_ v;
  WeightT_ w;
  TimestampT_ t;
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

//template <typename NodeID_, typename WeightT_, typename TimestampT_>
//std::ostream& operator<<(std::ostream& os,
//                         const NodeWeight<NodeID_, WeightT_, TimestampT_>& nw) {
//  os << nw.v << " " << nw.w << " " << nw.t;
//  return os;
//}
//
//template <typename NodeID_, typename WeightT_, typename TimestampT_>
//std::istream& operator>>(std::istream& is, NodeWeight<NodeID_, WeightT_, TimestampT_>& nw) {
//  is >> nw.v >> nw.w >> nw.t;
//  return is;
//}



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

// structure for the vertices
struct vertex_element{
  int64_t index;
  int32_t degree;
};

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
    OffsetT start_offset_;
    uint32_t degree_;
   public:
    Neighborhood(DestID_* g_index, OffsetT start_index, OffsetT start_offset, uint32_t degree) :
        g_index_(g_index), start_index_(start_index), start_offset_(0), degree_(degree) {
      OffsetT max_offset = end() - begin();
      start_offset_ = std::min(start_offset, max_offset);
    }
    typedef DestID_ *iterator;
    iterator begin() { return g_index_ + start_index_ + start_offset_; }
    iterator end()   { return g_index_ + start_index_ + degree_; }
  };

  void ReleaseResources() {
    if (edges_ != nullptr) delete[] edges_;
    if (vertices_ != nullptr) delete[] vertices_;
    if (segment_edges_actual != nullptr) delete[] segment_edges_actual;
    if (segment_edges_total != nullptr) delete[] segment_edges_total;
  }


 public:
  CSRGraph() : directed_(false), num_edges_(-1), num_vertices(-1), avg_degree(0),
               elem_capacity(-1), segment_size(-1), segment_count(-1), tree_height(-1),
               edges_(nullptr), vertices_(nullptr), segment_edges_actual(nullptr), segment_edges_total(nullptr) {}

  CSRGraph(CSRGraph&& other) : directed_(other.directed_), num_edges_(other.num_edges_),
                               num_vertices(other.num_vertices), avg_degree(other.avg_degree),
                               elem_capacity(other.elem_capacity), segment_size(other.segment_size),
                               segment_count(other.segment_count), tree_height(other.tree_height),
                               edges_(other.edges_), vertices_(other.vertices_),
                               segment_edges_actual(other.segment_edges_actual), segment_edges_total(other.segment_edges_total) {

    other.num_edges_ = -1;
    other.num_vertices = -1;
    other.avg_degree = 0;

    other.elem_capacity = -1;
    other.segment_size = -1;
    other.segment_count = -1;
    other.tree_height = -1;

    other.edges_ = nullptr;
    other.vertices_ = nullptr;
    other.segment_edges_actual = nullptr;
    other.segment_edges_total = nullptr;
  }

  ~CSRGraph() {
    ReleaseResources();
  }

  CSRGraph& operator=(CSRGraph&& other) {
    if (this != &other) {
      ReleaseResources();

      num_edges_ = other.num_edges_;
      num_vertices = other.num_vertices;
      avg_degree = other.avg_degree;

      elem_capacity = other.elem_capacity;
      segment_size = other.segment_size;
      segment_count = other.segment_count;
      tree_height = other.tree_height;

      edges_ = other.edges_;
      vertices_ = other.vertices_;
      segment_edges_actual = other.segment_edges_actual;
      segment_edges_total = other.segment_edges_total;

      other.num_edges_ = -1;
      other.num_vertices = -1;
      other.avg_degree = 0;

      other.elem_capacity = -1;
      other.segment_size = -1;
      other.segment_count = -1;
      other.tree_height = -1;

      other.edges_ = nullptr;
      other.vertices_ = nullptr;
      other.segment_edges_actual = nullptr;
      other.segment_edges_total = nullptr;
    }
    return *this;
  }

  CSRGraph(EdgeList &edge_list, bool is_directed, int64_t n_edges, int64_t n_vertices) {
    num_edges_ = n_edges;
    num_vertices = n_vertices;
    directed_ = is_directed;
    compute_capacity();

    // array-based compete tree structure
    segment_edges_actual = (int64_t *) malloc (sizeof(int64_t) * segment_count * 2);
    segment_edges_total = (int64_t *) malloc (sizeof(int64_t) * segment_count * 2);
    memset(segment_edges_actual, 0, sizeof(int64_t)*segment_count*2);
    memset(segment_edges_total, 0, sizeof(int64_t)*segment_count*2);

    tree_height = floor_log2(segment_count);
    delta_up = (up_0 - up_h) / tree_height;
    delta_low = (low_h - low_0) / tree_height;

    edges_ = (DestID_ *) malloc (sizeof(DestID_) * elem_capacity);
    vertices_ = (struct vertex_element *) calloc (num_vertices, sizeof(struct vertex_element));
    memset(edges_, 0, sizeof(DestID_)*elem_capacity);
    //memset(vertices_, 0, sizeof(struct vertex_element)*num_vertices);

    // assume all the edges of a vertex comes together.
    for (int i = 0; i < num_edges_; i++)
    {
      int32_t t_src = edge_list[i].u;
      int32_t t_dst = edge_list[i].v.v;

      int32_t t_degree = vertices_[t_src].degree;
      int32_t t_segment_id = t_src / segment_size;

      if (t_degree == 0) vertices_[t_src].index = i;

      edges_[i].v = t_dst;
      edges_[i].w = edge_list[i].v.w;
      edges_[i].t = i;
      vertices_[t_src].degree += 1;

      // update the actual edges in each segment of the tree
      int32_t j = t_segment_id + segment_count; //tree leaves
      while (j > 0)
      {
        segment_edges_actual[j] += 1;
        j /= 2;
      }
    }

    // correct the starting of the vertices with 0 degree in the base-graph
    for (int i = 1; i < num_vertices; i++) {
      if(vertices_[i].degree == 0) vertices_[i].index = vertices_[i-1].index + vertices_[i-1].degree;
    }

    // spread_weighted(0, elem_capacity, num_edges_, 0, num_vertices);
    spread_weighted_V1(0, num_vertices);

    if(debug) cout << ">>>>>>>> root-density: " << ((double) segment_edges_actual[1] / (double) segment_edges_total[1]) << endl;
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
    return vertices_[v].degree;
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return vertices_[v].degree;
  }

  Neighborhood out_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    return Neighborhood(edges_, vertices_[n].index, start_offset, vertices_[n].degree);
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return Neighborhood(edges_, vertices_[n].index, start_offset, vertices_[n].degree);
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_vertices << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_vertices << std::endl;
  }

//  void PrintTopology() const {
//    for (NodeID_ i=0; i < num_nodes_; i++) {
//      std::cout << i << ": ";
//      for (DestID_ j : out_neigh(i)) {
//        std::cout << j << " ";
//      }
//      std::cout << std::endl;
//    }
//  }
//
//  static DestID_** GenIndex(const pvector<SGOffset> &offsets, DestID_* neighs) {
//    NodeID_ length = offsets.size();
//    DestID_** index = new DestID_*[length];
//    #pragma omp parallel for
//    for (NodeID_ n=0; n < length; n++)
//      index[n] = neighs + offsets[n];
//    return index;
//  }

  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    pvector<SGOffset> offsets(num_vertices+1);
    for (NodeID_ n=0; n < num_vertices+1; n++)
        offsets[n] = vertices_[n].index - vertices_[0].index;
    return offsets;
  }

  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

  inline void print_vertices(){
    for (int i = 0; i < num_vertices; i++){
      printf("(%d)|%llu,%d| ", i, vertices_[i].index, vertices_[i].degree);
    }
    printf("\n");
  }

  inline void print_vertices(int32_t from, int32_t to) {
    for (int32_t i = from; i < to; i++){
      printf("(%d)|%llu,%d| ", i, vertices_[i].index, vertices_[i].degree);
    }
    printf("\n");
  }

  inline void print_vertices(int32_t segment_id) {
    int32_t from = (segment_id) * segment_size;
    int32_t to = (segment_id + 1) * segment_size;
    cout << from << " " << to << endl;
    print_vertices(from, to);
  }

  inline void print_edges(){
    for (int i = 0; i < elem_capacity; i++){
      printf("%u ", edges_[i].v);
    }
    printf("\n");
  }

  inline void print_segment(){
    for (int i = 0; i < segment_count * 2; i++){
      printf("(%d)|%llu / %llu| ", i, segment_edges_actual[i], segment_edges_total[i]);
    }
    printf("\n");
  }

  inline void print_pma_meta() {
    cout << "max_size: " << max_size << ", num_edges: " << num_edges_ << ", num_vertices: " << num_vertices << ", avg_degree: " << avg_degree << ", elem_capacity: " << elem_capacity << endl;
    cout << "segment_count: " << segment_count << ", segment_size: " << segment_size << ", tree_height: " << tree_height << endl;
  }

  inline void edge_list_boundary_sanity_checker() {
    for(int32_t curr_vertex = 1; curr_vertex<num_vertices; curr_vertex += 1) {
      if(vertices_[curr_vertex - 1].index + vertices_[curr_vertex - 1].degree > vertices_[curr_vertex].index) {
        cout << "**** Invalid edge-list boundary found at vertex-id: " << curr_vertex - 1 << " index: " << vertices_[curr_vertex - 1].index;
        cout << " degree: " << vertices_[curr_vertex - 1].degree << " next vertex start at: " << vertices_[curr_vertex].index << endl;
      }
      assert(vertices_[curr_vertex - 1].index + vertices_[curr_vertex - 1].degree <= vertices_[curr_vertex].index && "Invalid edge-list boundary found!");
    }
    assert(vertices_[num_vertices - 1].index + vertices_[num_vertices - 1].degree <= elem_capacity && "Invalid edge-list boundary found!");
  }

  /*****************************************************************************
   *                                                                           *
   *   PMA                                                                     *
   *                                                                           *
   *****************************************************************************/
  // double the size of the "edges_" array
  void resize() {
    if(debug) printf("[resize()] Called!\n");

    elem_capacity *= 2;
    int64_t gaps = elem_capacity - num_edges_;
    int64_t *new_indices = calculate_positions_V1(0, num_vertices, gaps, num_edges_);
//    int64_t *new_indices = calculate_positions(0, num_vertices, gaps);

    DestID_ *new_edges_ = (DestID_ *) malloc (sizeof(DestID_) * elem_capacity);

    for(int32_t curr_vertex = num_vertices - 1; curr_vertex >= 0; curr_vertex--) {
      for(int32_t i=0; i<vertices_[curr_vertex].degree; i+=1) {
        new_edges_[new_indices[curr_vertex] + i] = edges_[vertices_[curr_vertex].index + i];
      }
      vertices_[curr_vertex].index = new_indices[curr_vertex];
    }

    free(edges_);
    edges_ = nullptr;
    edges_ = new_edges_;

    recount_segment_total();
  }

  inline int32_t get_segment_id(int32_t vertex_id) {
    return (vertex_id / segment_size) + segment_count;
  }

  inline void update_segment_edge_total(int32_t vertex_id, int count) {
    int32_t j = get_segment_id(vertex_id);
    while (j > 0){
      segment_edges_total[j] += count;
      j /= 2;
    }
  }

  void recount_segment_total() {
    // count the size of each segment in the tree
    memset(segment_edges_total, 0, sizeof(int64_t)*segment_count*2);
    for (int i = 0; i < segment_count; i++){
      int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : vertices_[(i+1)*segment_size].index;
      int64_t segment_total_p = next_starter - vertices_[i*segment_size].index;
      int32_t j = i + segment_count;  //tree leaves
      while (j > 0){
        segment_edges_total[j] += segment_total_p;
        j /= 2;
      }
    }
  }

  void recount_segment_total(int32_t start_vertex, int32_t end_vertex) {
    int32_t start_seg = get_segment_id(start_vertex) - segment_count;
    int32_t end_seg = get_segment_id(end_vertex) - segment_count;

    for(int32_t i = start_seg; i<end_seg; i += 1) {
      int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : vertices_[(i+1)*segment_size].index;
      int64_t segment_total_p = next_starter - vertices_[i*segment_size].index;
      int32_t j = i + segment_count;  //tree leaves
      segment_total_p -= segment_edges_total[j];  // getting the absolute difference
      while (j > 0){
        segment_edges_total[j] += segment_total_p;
        j /= 2;
      }
    }
  }

  // check all the segment maintains it's integrity (i.e., total number of edges in a segment is larger or equal to the actual number of edges in that segment)
  void segment_sanity_check() {
    for (int i = 0; i < segment_count; i++){
      int32_t j = i + segment_count;  //tree leaves
      while (j > 0){
        assert(segment_edges_total[j] >= segment_edges_actual[j] && "Rebalancing (adaptive) segment boundary invalidate segment capacity!");
        j /= 2;
      }
    }
  }

  void insert(int32_t src, int32_t dst, int32_t value)
  {
    if(debug) printf("[insert(%u, %u)] Called!\n", src, dst);

    // find an empty slot to the right of the vertex
    // move the data from loc to the free slot
    int32_t current_segment = get_segment_id(src);
    // raqib: we now allow to look for space beyond the segment boundary
    //assert(segment_edges_total[current_segment] > segment_edges_actual[current_segment] && "There is no enough space in the current segment for performing this insert!");

    int64_t loc = vertices_[src].index + vertices_[src].degree;
    int64_t right_free_slot = -1;
    int64_t left_free_slot = -1;
    int32_t left_vertex = src, right_vertex = src;
    int32_t left_vertex_boundary, right_vertex_boundary;

    if(segment_edges_total[current_segment] > segment_edges_actual[current_segment]) {
      left_vertex_boundary = (src / segment_size) * segment_size;
      right_vertex_boundary = min(left_vertex_boundary + segment_size, num_vertices - 1);
    }
    else {
      int32_t curr_seg_size = segment_size, j = current_segment;
      while(j) {
        if(segment_edges_total[j] > segment_edges_actual[j]) break;
        j /= 2;
        curr_seg_size *= 2;
      }
      left_vertex_boundary = (src / curr_seg_size) * curr_seg_size;
      right_vertex_boundary = min(left_vertex_boundary + curr_seg_size, num_vertices - 1);
    }

    // search right side for a free slot
    // raqib: shouldn't we search within the pma leaf?
    for (int32_t i = src; i < right_vertex_boundary; i++) {
      if (vertices_[i + 1].index > (vertices_[i].index + vertices_[i].degree)) {
        right_free_slot = vertices_[i].index + vertices_[i].degree;  // we get a free slot here
        right_vertex = i;
        break;
      }
    }

    // in the last segment where we skipped the last vertex
    //if (right_free_slot == -1 && elem_capacity > (1 + vertices_[num_vertices - 1].index + vertices_[num_vertices - 1].degree))
    if (right_free_slot == -1 && right_vertex_boundary == (num_vertices - 1)
        && elem_capacity > (1 + vertices_[num_vertices - 1].index + vertices_[num_vertices - 1].degree)) {
      right_free_slot = vertices_[num_vertices - 1].index + vertices_[num_vertices - 1].degree;
      right_vertex = num_vertices - 1;
    }

    // if no space on the right side, search the left side
    if (right_free_slot == -1) {
      for (int32_t i = src; i > left_vertex_boundary; i--) {
        if (vertices_[i].index > (vertices_[i - 1].index + vertices_[i - 1].degree)) {
          left_free_slot = vertices_[i].index - 1;  // we get a free slot here
          left_vertex = i;
          break;
        }
      }
    }

    if(debug) cout << "left_free_slot: " << left_free_slot << ", right_free_slot: " << right_free_slot << endl;
    if(debug) cout << "left_free_vertex: " << left_vertex << ", right_free_vertex: " << right_vertex << endl;

    // found free slot on the right
    if (right_free_slot != -1)
    {
      if (right_free_slot >= loc)  // move elements to the right to get the free slot
      {
        for (int i = right_free_slot; i > loc; i--)
        {
          edges_[i].v = edges_[i-1].v;
          edges_[i].w = edges_[i-1].w;
          edges_[i].t = edges_[i-1].t;
        }

        for (int32_t i = src+1; i <= right_vertex; i++) {
          vertices_[i].index += 1;
        }

        //update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
        if(current_segment != get_segment_id(right_vertex)) {
          update_segment_edge_total(src, 1);
          update_segment_edge_total(right_vertex, -1);
        }
      }
      edges_[loc].v = dst;
      edges_[loc].w = value;
      edges_[loc].t = num_edges_;

      vertices_[src].degree += 1;
    }
    else if (left_free_slot != -1)
    {
      if (left_free_slot < loc)
      {
        for (int i = left_free_slot; i < loc - 1; i++)
        {
          edges_[i].v = edges_[i+1].v;
          edges_[i].w = edges_[i+1].w;
          edges_[i].t = edges_[i+1].t;
        }

        for (int32_t i = left_vertex; i <= src; i++) {
          vertices_[i].index -= 1;
        }

        //update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
        if(current_segment != get_segment_id(left_vertex)) {
          update_segment_edge_total(src, 1);
          update_segment_edge_total(left_vertex, -1);
        }
      }
      edges_[loc - 1].v = dst;
      edges_[loc - 1].w = value;
      edges_[loc - 1].t = num_edges_;

      vertices_[src].degree += 1;
    }
    else
    {
      assert(false && "Should not happen");
    }

    // we insert a new edge, increasing the degree of the whole subtree
    int32_t j = current_segment;
    if(debug) cout << "segment where insert happened: " << j << endl;
    while (j > 0){
      segment_edges_actual[j] += 1;
      j /= 2;
    }

    // whether we need to update segment_edges_total
    // raqib: I think we should do the re-balance first and then make the insertion.
    // This will always ensure the empty space for the insertion.

    num_edges_ += 1;

    rebalance_wrapper(src);
    // todo: we should also consider rebalancing the donner vertex's segment
  }


  // let's assume we always have 2^k vertices. The total number of vertices are fixed.
  // fix the number of vertices in each segment; adjust gaps instead
  void compute_capacity()
  {
    segment_size = ceil_log2(num_vertices); // Ideal segment size
    segment_count = ceil_div(num_vertices, segment_size); // Ideal number of segments

    // The number of segments has to be a power of 2, though.
    segment_count = hyperfloor(segment_count);
    // Update the segment size accordingly
    segment_size = ceil_div(num_vertices, segment_count);

    // correcting the number of vertices
    num_vertices = segment_count * segment_size;
    avg_degree = ceil_div(num_edges_, num_vertices);

    //elem_capacity = segment_size * segment_count * avg_degree * max_sparseness;
    elem_capacity = num_edges_ * max_sparseness;
  }


  // spread the elements (vertices) based on their degrees
  // here, end_vertex is excluded.
  void spread_weighted(int64_t from, int64_t to,
                       int64_t n_edges,
                       int32_t start_vertex,
                       int32_t end_vertex)
  {
    int64_t capacity = to - from;
    int64_t gaps = capacity - n_edges;

    int64_t total_degree = n_edges;

    int64_t read_index = from + n_edges - 1;
    int32_t curr_vertex = end_vertex - 1;
    int64_t curr_degree = vertices_[curr_vertex].degree;
    int64_t curr_gap = ceil_div(gaps * curr_degree, total_degree);
    int64_t write_index = to - 1 - curr_gap;
    while (write_index > read_index)
    {
      for (int i = 0; i < curr_degree; i++)
      {
        edges_[write_index] = edges_[read_index];
        //edges_[read_index].v = 0;
        write_index--;
        read_index--;
      }
      // Caution: If the last vertex have degree 0, then it will point to the outside of the elem_capacity range
      vertices_[curr_vertex].index += (write_index - read_index);

      // move to the previous vertex
      curr_vertex -= 1;
      curr_degree = vertices_[curr_vertex].degree;
      curr_gap = ceil_div(gaps * curr_degree, total_degree);
      write_index -= curr_gap;
    }

    // dong: write_index may stop before reaching to 0.
    //assert(write_index == 0 && "Write-index should reach to the 0-th index.");

    recount_segment_total();
  }

// spread the elements (vertices) based on their degrees
// here, end_vertex is excluded, and start_vertex is expected to be 0
  void spread_weighted_V1(int32_t start_vertex, int32_t end_vertex)
  {
    assert(start_vertex == 0 && "start-vertex is expected to be 0 here.");
    int64_t gaps = elem_capacity - num_edges_;
    int64_t * new_positions = calculate_positions_V1(start_vertex, end_vertex, gaps, num_edges_);
//    int64_t * new_positions = calculate_positions(start_vertex, end_vertex, gaps);

    if(debug) {
      for (int32_t curr_vertex = start_vertex; curr_vertex < end_vertex; curr_vertex += 1) {
        cout << "vertex-id: " << curr_vertex << ", index: " << vertices_[curr_vertex].index;
        cout << ", degree: " << vertices_[curr_vertex].degree << ", new position: " << new_positions[curr_vertex] << endl;
      }
    }

    int64_t read_index, write_index, curr_degree;
    for(int32_t curr_vertex=end_vertex-1; curr_vertex>start_vertex; curr_vertex-=1) {
      curr_degree = vertices_[curr_vertex].degree;
      read_index = vertices_[curr_vertex].index + curr_degree - 1;
      write_index = new_positions[curr_vertex] + curr_degree - 1;

      if(write_index < read_index) {
        cout << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << endl;
      }
      assert(write_index >= read_index && "index anomaly occurred while spreading elements");

      for (int i = 0; i < curr_degree; i++)
      {
        edges_[write_index] = edges_[read_index];
        //if(read_index < new_positions[curr_vertex]) edges_[read_index].v = 0;
        write_index--;
        read_index--;
      }

      vertices_[curr_vertex].index = new_positions[curr_vertex];
    }

    free(new_positions);
    new_positions = nullptr;
    recount_segment_total();
//    segment_sanity_check();
//    edge_list_boundary_sanity_checker();
  }

// check the balance feature and do the adjustments.
  void rebalance_wrapper(int32_t src)
  {
    if(debug) cout << "[rebalance_wrapper] Called" << endl;
    // We're using signed indices here now since we need to perform
    // relative indexing and the range checking is much easier and
    // clearer with signed integral types.
    int32_t height = 0;
    int32_t window = (src / segment_size) + segment_count;
    double density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
    double up_height = up_0 - (height * delta_up);
    double low_height = low_0 + (height * delta_low);

    if(debug) cout << "Window: " << window << ", density: " << density << ", up_height: " << up_height << ", low_height: " << low_height << endl;

    //while (window > 0 && (density < low_height || density >= up_height))
    while (window > 0 && (density >= up_height))
    {
      // Repeatedly check window containing an increasing amount of segments
      // Now that we recorded the number of elements and occupancy in segment_edges_total and segment_edges_actual respectively;
      // so, we are going to check if the current window can fulfill the density thresholds.
      // density = gap / segment-size

      // Go one level up in our conceptual PMA tree
      window /= 2;
      height += 1;

      up_height = up_0 - (height * delta_up);
      low_height = low_0 + (height * delta_low);

      density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
      if(debug) cout << "Window: " << window << ", density: " << density << ", up_height: " << up_height << ", low_height: " << low_height << endl;
    }

    if(!height) {
      // rebalance is not required in the single pma leaf
      return;
    }
    int32_t left_index, right_index;
    //if (density >= low_height && density < up_height)
    if (density < up_height)
    {
      // Found a window within threshold
      int32_t window_size = segment_size * (1 << height);
      left_index = (src / window_size) * window_size;
      right_index = min(left_index + window_size, num_vertices);

      // do degree-based distribution of gaps
      rebalance_weighted(left_index, right_index, window);
    }
    else {
      // Rebalance not possible without increasing the underlying array size.
      // need to resize the size of "edges_" array
      resize();
    }
  }

  // calculate starting index of each vertex for the range of vertices based on the degree of vertices
  int64_t *calculate_positions_V1(int32_t start_vertex, int32_t end_vertex, int64_t gaps, int64_t total_degree) {
    int32_t size = end_vertex - start_vertex;
    int64_t *new_index = (int64_t *) calloc(size, sizeof(int64_t));
    total_degree += size;

    double index_d = vertices_[start_vertex].index;
    double step = ((double) gaps) / total_degree;  //per-edge step
    for (int i = start_vertex; i < end_vertex; i++){
      new_index[i-start_vertex] = index_d;
      //cout << index_d << " " << new_index[i-start_vertex] << " " << vertices_[i-1].index << " " << vertices_[i-1].degree << endl;
      if(i > start_vertex) {
        //printf("v[%d] with degree %d gets actual space %ld\n", i-1, vertices_[i-1].degree, (new_index[i-start_vertex]-new_index[i-start_vertex-1]));
        assert(new_index[i-start_vertex] >= new_index[(i-1)-start_vertex] + vertices_[i-1].degree && "Edge-list can not be overlapped with the neighboring vertex!");
      }
//      index_d += (vertices_[i].degree + (step * vertices_[i].degree));
      index_d += (vertices_[i].degree + (step * (vertices_[i].degree + 1)));
    }

    return new_index;
  }

  void rebalance_weighted(int32_t start_vertex,
                          int32_t end_vertex,
                          int32_t pma_idx) {
    if(debug) printf("[rebalance_weighted] Called!\n");
    int64_t from = vertices_[start_vertex].index;
    int64_t to = (end_vertex >= num_vertices) ? elem_capacity : vertices_[end_vertex].index;
    assert(to > from && "Invalid range found while doing weighted rebalance");
    int64_t capacity = to - from;

    assert(segment_edges_total[pma_idx] == capacity && "Segment capacity is not matched with segment_edges_total");
    int64_t gaps = segment_edges_total[pma_idx] - segment_edges_actual[pma_idx];

    // calculate the future positions of the vertices_[i].index
    int32_t size = end_vertex - start_vertex;

    int64_t *new_index = calculate_positions_V1(start_vertex, end_vertex, gaps, segment_edges_actual[pma_idx]);
//    int64_t *new_index = calculate_positions(start_vertex, end_vertex, gaps);

    int64_t index_boundary = (end_vertex >= num_vertices) ? elem_capacity : vertices_[end_vertex].index;
    assert(new_index[size - 1] + vertices_[end_vertex - 1].degree <= index_boundary && "Rebalance (weighted) index calculation is wrong!");

    int32_t ii, jj;
    int32_t curr_vertex = start_vertex + 1, next_to_start;
    int64_t read_index, last_read_index, write_index;

    while (curr_vertex < end_vertex)
    {
      for (ii = curr_vertex; ii < end_vertex; ii++)
      {
        if(new_index[ii-start_vertex] <= vertices_[ii].index) break;
      }
      if(ii == end_vertex) ii -= 1;
      // assert(new_index[ii-start_vertex] <= vertices_[ii].index && "This should not happen!");
      next_to_start = ii + 1;
      if(new_index[ii-start_vertex] <= vertices_[ii].index) {
        // now it is guaranteed that, ii's new-starting index is less than or equal to it's old-starting index
        jj = ii;
        read_index = vertices_[jj].index;
        last_read_index = read_index + vertices_[jj].degree;
        write_index = new_index[jj - start_vertex];

        while (read_index < last_read_index) {
          edges_[write_index] = edges_[read_index];
          //edges_[read_index].v = 0;
          write_index++;
          read_index++;
        }
        // update the index to the new position
        vertices_[jj].index = new_index[jj - start_vertex];

        ii -= 1;
      }

      // from current_vertex to ii, the new-starting index is greater than to it's old-starting index
      for (jj=ii; jj>=curr_vertex; jj-=1)
      {
        read_index = vertices_[jj].index + vertices_[jj].degree - 1;
        last_read_index = vertices_[jj].index;
        write_index = new_index[jj-start_vertex] + vertices_[jj].degree - 1;

        while(read_index >= last_read_index)
        {
          edges_[write_index] = edges_[read_index];
          //edges_[read_index].v = 0;
          write_index--;
          read_index--;
        }

        // update the index to the new position
        vertices_[jj].index = new_index[jj-start_vertex];
      }
      // move current_vertex to the next position of ii
      curr_vertex = next_to_start;
    }

    free(new_index);
    new_index = nullptr;

    recount_segment_total(start_vertex, end_vertex);
  }

private:
  bool directed_;
  /* PMA constants */
  // Reserve 8 bits to allow for fixed point arithmetic.
  int64_t max_size = (1ULL << 56) - 1ULL;

  // Height-based (as opposed to depth-based) tree thresholds
  // Upper density thresholds
  static constexpr double up_h = 0.75; // root
  static constexpr double up_0 = 1.00; // leaves
  // Lower density thresholds
  static constexpr double low_h = 0.50; // root
  static constexpr double low_0 = 0.25; // leaves

  int8_t max_sparseness = 1.0 / low_0;
  int8_t largest_empty_segment = 1.0 * max_sparseness;

  /* General PMA fields */
  int64_t num_edges_ = 0; // Number of edges
  int32_t num_vertices = 0; // Number of vertices
  int64_t avg_degree = 0; // averge degree of the graph

  int64_t elem_capacity; // size of the edges_ array
  int32_t segment_size;  // size of a pma leaf segment
  int64_t segment_count; // number of pma leaf segments
  int32_t tree_height;   // height of the pma tree
  double delta_up;        // Delta for upper density threshold
  double delta_low;       // Delta for lower density threshold

  DestID_* edges_; // Underlying storage for edgelist
  struct vertex_element *vertices_; //underlying storage for vertex list
  int64_t *segment_edges_actual;
  int64_t *segment_edges_total;
};

#endif  // GRAPH_H_
