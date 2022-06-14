// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <assert.h>
#include <cstring>

#include "pvector.h"
#include "util.h"
using namespace std;

#define debug 0
#define BLOCK_SIZE 511

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
template <typename NodeID_, typename WeightT_>
struct NodeWeight {
  NodeID_ v;        // destination of this edge in the graph
  WeightT_ w;       // weight of the edge
  uint64_t t;    // timestamp when this edge inserted
  NodeWeight() {}
  NodeWeight(NodeID_ v) : v(v), w(1), t(1) {}
  NodeWeight(NodeID_ v, WeightT_ w) : v(v), w(w), t(1) {}
  NodeWeight(NodeID_ v, WeightT_ w, uint64_t t) : v(v), w(w), t(t) {}

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

template <typename NodeID_, typename WeightT_>
std::ostream& operator<<(std::ostream& os,
                         const NodeWeight<NodeID_, WeightT_>& nw) {
  os << nw.v << " " << nw.w;
  return os;
}

template <typename NodeID_, typename WeightT_>
std::istream& operator>>(std::istream& is, NodeWeight<NodeID_, WeightT_>& nw) {
  is >> nw.v >> nw.w;
  return is;
}



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
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;
typedef int32_t NodeID;
typedef int32_t WeightT;

// structure for the vertices
struct vertex_element {
  uint64_t head;
  uint64_t tail;
  uint32_t degree;
};

// blocks of edges
struct edge_block {
  struct NodeWeight<NodeID, WeightT> block[BLOCK_SIZE];     // edge-list segment
  uint64_t next;      // timestamp when this edge inserted
};

template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class CSRGraph {
  // Used for *non-negative* offsets within a neighborhood
  typedef std::make_unsigned<std::ptrdiff_t>::type OffsetT;
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    struct edge_block *curr_edge_block_;
    uint32_t degree_, curr_idx_;
    DestID_ *begin_ptr_;
    DestID_ *end_ptr_;
   public:
    Neighborhood(struct edge_block *curr_edge_block, OffsetT start_offset, uint32_t degree) :
        curr_edge_block_(curr_edge_block), degree_(degree), curr_idx_(start_offset) {
      if(start_offset >= degree) begin_ptr_ = nullptr;
      else begin_ptr_ = &(curr_edge_block_->block[start_offset]);

      end_ptr_ = nullptr;

      //cout << "neighborhood: " << g_index->v << endl;
    }

    class iterator {
    public:
      struct edge_block *curr_edge_block_;
      uint32_t curr_idx_, degree_;

      iterator() {
        g_index_ = nullptr;
        curr_edge_block_ = nullptr;
        curr_idx_ = 0;
        degree_ = 0;
      }

      iterator(DestID_ *g_index) {
        g_index_ = g_index;
        curr_edge_block_ = nullptr;
        curr_idx_ = 0;
        degree_ = 0;
      }

      iterator(DestID_ *g_index, struct edge_block *curr_edge_block, uint32_t curr_idx, uint32_t degree) {
        g_index_ = g_index;
        curr_edge_block_ = curr_edge_block;
        curr_idx_ = curr_idx;
        degree_ = degree;
      }

      iterator &operator++() {
        //cout << "++" << endl;
        curr_idx_ += 1;
        if(curr_idx_ == degree_) g_index_ = nullptr;
        else {
          if (curr_idx_ % BLOCK_SIZE == 0) curr_edge_block_ = (struct edge_block *) curr_edge_block_->next;
          g_index_ = &(curr_edge_block_->block[curr_idx_ % BLOCK_SIZE]);
        }
        return *this;
      }

      operator DestID_ *() const {
        //cout << "DestID_ *" << endl;
        return g_index_;
      }

      DestID_ *operator->() {
        //cout << "*operator->" << endl;
        return g_index_;
      }

      DestID_ &operator*() {
        //cout << "&operator*" << endl;
        return (*g_index_);
      }

      bool operator==(const iterator &rhs) const {
        //cout << "operator==(const iterator &rhs)" << endl;
        return g_index_ == rhs.g_index_;
      }

      bool operator!=(const iterator &rhs) const {
        //cout << "operator!=(const iterator &rhs)" << endl;
        return (g_index_ != rhs.g_index_);
      }

    private:
      DestID_ *g_index_;
    };

    iterator begin() { return iterator(begin_ptr_, curr_edge_block_, curr_idx_, degree_); }
    iterator end()   { return iterator(end_ptr_); }
  };

  void ReleaseResources() {
    for(NodeID_ i=0; i<num_nodes_; i+=1) {
      struct edge_block *head = (struct edge_block *) vertices_[i].head;
      while (head != nullptr) {
        struct edge_block *tmp = head;
        head = (struct edge_block *) head->next;
        delete[] tmp;
      }
    }

    if (vertices_ != nullptr) delete[] vertices_;
  }


 public:
  CSRGraph() : directed_(false), num_nodes_(-1), num_edges_(-1), vertices_(nullptr) {}

  CSRGraph(CSRGraph&& other) : directed_(other.directed_),
    num_nodes_(other.num_nodes_), num_edges_(other.num_edges_),
    vertices_(other.vertices_) {
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.vertices_ = nullptr;
  }

  ~CSRGraph() {
    ReleaseResources();
  }

  CSRGraph& operator=(CSRGraph&& other) {
    if (this != &other) {
      ReleaseResources();
      directed_ = other.directed_;
      num_edges_ = other.num_edges_;
      num_nodes_ = other.num_nodes_;
      vertices_ = other.vertices_;

      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.vertices_ = nullptr;
    }
    return *this;
  }

  CSRGraph(EdgeList &base_edge_list, bool is_directed, uint64_t n_edges, uint64_t n_vertices) {
    num_edges_ = n_edges;
    num_nodes_ = n_vertices;
    directed_ = is_directed;

    vertices_ = (struct vertex_element *) calloc(num_nodes_, sizeof(struct vertex_element));

    uint32_t t_src;
    for (int i = 0; i < num_edges_; i++)
    {
      t_src = base_edge_list[i].u;

      if(vertices_[t_src].degree == 0) {
        // initialize a new edge-list segment and update head/tail in the vertex structure
        struct edge_block *curr_block = (struct edge_block *) malloc(sizeof(struct edge_block));
        curr_block->next = 0;

        int32_t curr_idx = 0;
        curr_block->block[curr_idx].v = base_edge_list[i].v.v;
        curr_block->block[curr_idx].w = base_edge_list[i].v.w;
        curr_block->block[curr_idx].t = base_edge_list[i].v.t;

        vertices_[t_src].head = (uint64_t) curr_block;
        vertices_[t_src].tail = (uint64_t) curr_block;
      }
      else {
        if(vertices_[t_src].degree%BLOCK_SIZE == 0) {
          // it's time to create a new segment
          struct edge_block *curr_block = (struct edge_block *) malloc(sizeof(struct edge_block));
          curr_block->next = 0;

          int32_t curr_idx = 0;
          curr_block->block[curr_idx].v = base_edge_list[i].v.v;
          curr_block->block[curr_idx].w = base_edge_list[i].v.w;
          curr_block->block[curr_idx].t = base_edge_list[i].v.t;

          // linking current-block at the next pointer of the current tail
          ((struct edge_block *) vertices_[t_src].tail)->next = (uint64_t) curr_block;

          // update tail segment
          vertices_[t_src].tail = (uint64_t) curr_block;
        }
        else {
          // we have enough space in current segment
          struct edge_block *curr_block = (struct edge_block *) vertices_[t_src].tail;
          int32_t curr_idx = vertices_[t_src].degree%BLOCK_SIZE;

          curr_block->block[curr_idx].v = base_edge_list[i].v.v;
          curr_block->block[curr_idx].w = base_edge_list[i].v.w;
          curr_block->block[curr_idx].t = base_edge_list[i].v.t;
        }
      }
      vertices_[t_src].degree += 1;
    }
  }

  void insert(uint32_t src, uint32_t dst, uint32_t value)
  {
    if(debug) printf("[insert(%u, %u)] Called!\n", src, dst);

    if(vertices_[src].degree == 0) {
      // initialize a new edge-list segment and update head/tail in the vertex structure
      struct edge_block *curr_block = (struct edge_block *) malloc(sizeof(struct edge_block));
      curr_block->next = 0;

      int32_t curr_idx = 0;
      curr_block->block[curr_idx].v = dst;
      curr_block->block[curr_idx].w = value;
      curr_block->block[curr_idx].t = num_edges_;

      vertices_[src].head = (uint64_t) curr_block;
      vertices_[src].tail = (uint64_t) curr_block;
    }
    else {
      if(vertices_[src].degree%BLOCK_SIZE == 0) {
        // it's time to create a new segment
        struct edge_block *curr_block = (struct edge_block *) malloc(sizeof(struct edge_block));
        curr_block->next = 0;

        int32_t curr_idx = 0;
        curr_block->block[curr_idx].v = dst;
        curr_block->block[curr_idx].w = value;
        curr_block->block[curr_idx].t = num_edges_;

        // linking current-block at the next pointer of the current tail
        ((struct edge_block *) vertices_[src].tail)->next = (uint64_t) curr_block;

        // update tail segment
        vertices_[src].tail = (uint64_t) curr_block;
      }
      else {
        // we have enough space in current segment
        struct edge_block *curr_block = (struct edge_block *) vertices_[src].tail;
        int32_t curr_idx = vertices_[src].degree%BLOCK_SIZE;

        curr_block->block[curr_idx].v = dst;
        curr_block->block[curr_idx].w = value;
        curr_block->block[curr_idx].t = num_edges_;
      }
    }
    vertices_[src].degree += 1;
    num_edges_ += 1;
  }

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_nodes_;
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
    //cout << "degree: " << vertices_[n].degree << " " << vertices_[n].head << endl;
    return Neighborhood((struct edge_block *) vertices_[n].head, start_offset, vertices_[n].degree);
  }

  Neighborhood in_neigh(NodeID_ n, OffsetT start_offset = 0) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return Neighborhood((struct edge_block *) vertices_[n].head, start_offset, vertices_[n].degree);
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_nodes_ << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_nodes_ << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i=0; i < num_nodes_; i++) {
      std::cout << i << ": ";
      for (DestID_ j : out_neigh(i)) {
        std::cout << j << " ";
      }
      std::cout << std::endl;
    }
  }

  void PrintTopology(NodeID_ src) const {
    uint32_t j = 0;
    uint64_t curr_ptr = vertices_[src].head;
    std::cout << src << "(" << vertices_[src].degree << "): ";
    while(curr_ptr) {
      struct edge_block *curr_edge_block = (struct edge_block *) curr_ptr;
      cout << curr_edge_block->block[j%BLOCK_SIZE].v << " ";
      j += 1;
      if(j == vertices_[src].degree) break;
      if(j%BLOCK_SIZE == 0) curr_ptr = curr_edge_block->next;
    }
    cout << endl;

    std::cout << src << "(" << out_degree(src) << "): ";
    for (DestID_ j : out_neigh(src)) {
      std::cout << j.v << " ";
    }
    std::cout << std::endl << std::endl;
  }

  static DestID_** GenIndex(const pvector<SGOffset> &offsets, DestID_* neighs) {
    NodeID_ length = offsets.size();
    DestID_** index = new DestID_*[length];
    #pragma omp parallel for
    for (NodeID_ n=0; n < length; n++)
      index[n] = neighs + offsets[n];
    return index;
  }

  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    // note: keeing this for dummy purpose
    pvector<SGOffset> offsets(num_nodes_+1);
    return offsets;
  }

  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

 private:
  bool directed_;
  int64_t num_nodes_;
  int64_t num_edges_;
  struct vertex_element *vertices_;       //underlying storage for vertex list
};

#endif  // GRAPH_H_
