// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <type_traits>
#include <utility>

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "timer.h"
#include "util.h"


/*
GAP Benchmark Suite
Class:  BuilderBase
Author: Scott Beamer

Given arguements from the command line (cli), returns a built graph
 - MakeGraph() will parse cli and obtain edgelist and call
   MakeGraphFromEL(edgelist) to perform actual graph construction
 - edgelist can be from file (reader) or synthetically generated (generator)
 - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
*/


template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_, typename TimestampT_ = WeightT_, bool invert = true>
class BuilderBase {
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  const CLBase &cli_;
  bool symmetrize_;
  bool needs_weights_;
  int64_t num_nodes_ = -1;
  int64_t num_edges_ = 0;
  int64_t base_graph_num_edges_ = 0;

 public:
  explicit BuilderBase(const CLBase &cli) : cli_(cli) {
    symmetrize_ = cli_.symmetrize();
    needs_weights_ = !std::is_same<NodeID_, DestID_>::value;
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeID_> e) {
    return e.u;
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_, TimestampT_>> e) {
    return NodeWeight<NodeID_, WeightT_, TimestampT_>(e.u, e.v.w, e.v.t);
  }

  NodeID_ FindMaxNodeID(const EdgeList &el) {
    NodeID_ max_seen = 0;
    #pragma omp parallel for reduction(max : max_seen)
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;
      max_seen = std::max(max_seen, e.u);
      max_seen = std::max(max_seen, (NodeID_) e.v);
    }
    return max_seen;
  }

//  pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose) {
//    pvector<NodeID_> degrees(num_nodes_, 0);
//    #pragma omp parallel for
//    for (auto it = el.begin(); it < el.end(); it++) {
//      Edge e = *it;
//      if (symmetrize_ || (!symmetrize_ && !transpose))
//        fetch_and_add(degrees[e.u], 1);
//      if (symmetrize_ || (!symmetrize_ && transpose))
//        fetch_and_add(degrees[(NodeID_) e.v], 1);
//    }
//    return degrees;
//  }
//
//  static
//  pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees) {
//    pvector<SGOffset> sums(degrees.size() + 1);
//    SGOffset total = 0;
//    for (size_t n=0; n < degrees.size(); n++) {
//      sums[n] = total;
//      total += degrees[n];
//    }
//    sums[degrees.size()] = total;
//    return sums;
//  }
//
//  static
//  pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees) {
//    const size_t block_size = 1<<20;
//    const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
//    pvector<SGOffset> local_sums(num_blocks);
//    #pragma omp parallel for
//    for (size_t block=0; block < num_blocks; block++) {
//      SGOffset lsum = 0;
//      size_t block_end = std::min((block + 1) * block_size, degrees.size());
//      for (size_t i=block * block_size; i < block_end; i++)
//        lsum += degrees[i];
//      local_sums[block] = lsum;
//    }
//    pvector<SGOffset> bulk_prefix(num_blocks+1);
//    SGOffset total = 0;
//    for (size_t block=0; block < num_blocks; block++) {
//      bulk_prefix[block] = total;
//      total += local_sums[block];
//    }
//    bulk_prefix[num_blocks] = total;
//    pvector<SGOffset> prefix(degrees.size() + 1);
//    #pragma omp parallel for
//    for (size_t block=0; block < num_blocks; block++) {
//      SGOffset local_total = bulk_prefix[block];
//      size_t block_end = std::min((block + 1) * block_size, degrees.size());
//      for (size_t i=block * block_size; i < block_end; i++) {
//        prefix[i] = local_total;
//        local_total += degrees[i];
//      }
//    }
//    prefix[degrees.size()] = bulk_prefix[num_blocks];
//    return prefix;
//  }
//
//  // Removes self-loops and redundant edges
//  // Side effect: neighbor IDs will be sorted
//  void SquishCSR(const CSRGraph<NodeID_, DestID_, invert> &g, bool transpose,
//                 DestID_*** sq_index, DestID_** sq_neighs) {
//    pvector<NodeID_> diffs(g.num_nodes());
//    DestID_ *n_start, *n_end;
//    #pragma omp parallel for private(n_start, n_end)
//    for (NodeID_ n=0; n < g.num_nodes(); n++) {
//      if (transpose) {
//        n_start = g.in_neigh(n).begin();
//        n_end = g.in_neigh(n).end();
//      } else {
//        n_start = g.out_neigh(n).begin();
//        n_end = g.out_neigh(n).end();
//      }
//      std::sort(n_start, n_end);
//      DestID_ *new_end = std::unique(n_start, n_end);
//      new_end = std::remove(n_start, new_end, n);
//      diffs[n] = new_end - n_start;
//    }
//    pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
//    *sq_neighs = new DestID_[sq_offsets[g.num_nodes()]];
//    *sq_index = CSRGraph<NodeID_, DestID_>::GenIndex(sq_offsets, *sq_neighs);
//    #pragma omp parallel for private(n_start)
//    for (NodeID_ n=0; n < g.num_nodes(); n++) {
//      if (transpose)
//        n_start = g.in_neigh(n).begin();
//      else
//        n_start = g.out_neigh(n).begin();
//      std::copy(n_start, n_start+diffs[n], (*sq_index)[n]);
//    }
//  }
//
//  CSRGraph<NodeID_, DestID_, invert> SquishGraph(
//      const CSRGraph<NodeID_, DestID_, invert> &g) {
//    DestID_ **out_index, *out_neighs, **in_index, *in_neighs;
//    SquishCSR(g, false, &out_index, &out_neighs);
//    if (g.directed()) {
//      if (invert)
//        SquishCSR(g, true, &in_index, &in_neighs);
//      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
//                                                out_neighs, in_index,
//                                                in_neighs);
//    } else {
//      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
//                                                out_neighs);
//    }
//  }
//
//  /*
//  Graph Bulding Steps (for CSR):
//    - Read edgelist once to determine vertex degrees (CountDegrees)
//    - Determine vertex offsets by a prefix sum (ParallelPrefixSum)
//    - Allocate storage and set points according to offsets (GenIndex)
//    - Copy edges into storage
//  */
//  void MakeCSR(const EdgeList &el, bool transpose, DestID_*** index,
//               DestID_** neighs) {
//    pvector<NodeID_> degrees = CountDegrees(el, transpose);
//    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
//    *neighs = new DestID_[offsets[num_nodes_]];
//    *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
//    #pragma omp parallel for
//    for (auto it = el.begin(); it < el.end(); it++) {
//      Edge e = *it;
//      if (symmetrize_ || (!symmetrize_ && !transpose))
//        (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
//      if (symmetrize_ || (!symmetrize_ && transpose))
//        (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
//            GetSource(e);
//    }
//  }
//
//  CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el) {
//    DestID_ **index = nullptr, **inv_index = nullptr;
//    DestID_ *neighs = nullptr, *inv_neighs = nullptr;
//    Timer t;
//    t.Start();
//    if (num_nodes_ == -1)
//      num_nodes_ = FindMaxNodeID(el)+1;
//    if (needs_weights_)
//      Generator<NodeID_, DestID_, WeightT_, TimestampT_>::InsertWeights(el);
//    MakeCSR(el, false, &index, &neighs);
//    if (!symmetrize_ && invert)
//      MakeCSR(el, true, &inv_index, &inv_neighs);
//    t.Stop();
//    PrintTime("Build Time", t.Seconds());
//    if (symmetrize_)
//      return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs);
//    else
//      return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs,
//                                                inv_index, inv_neighs);
//  }

//  /*
//   * Graph Building Steps:
//   *  - Read edgelist once to determine vertex degrees (CountDegrees)
//   *      - CountDegrees will count the degrees of the base-graph only
//   *  - Determine vertex offsets by a prefix sum (ParallelPrefixSum)
//   *  - Allocate storage and set points according to offsets
//   *  - Insert edges into storage
//   * */
//  void MakePMemGraphFromEL(EdgeList &el, CSRGraph<NodeID_, DestID_, invert> &g) {
//    Timer t;
//    t.Start();
//    if (num_nodes_ == -1) num_nodes_ = FindMaxNodeID(el) + 1;
//
//
//    random_shuffle(el.begin(), el.end());
////    PrintEL(el);
//
////    pvector<NodeID_> out_degrees = CountDegrees(el, false);
////    AutoCorrectDegrees<NodeID_>(out_degrees);        // adding gap to degrees
////    pvector<SGOffset> out_offsets(out_degrees.size() + 1);
////    ParallelPrefixSum<NodeID_, SGOffset>(out_degrees, out_offsets, onseg_num_vertex_, onseg_max_neighbor_);
////    NodeID_ out_index_length = out_offsets.size();
//
//    t.Stop();
//    PrintTime("Aux. Data Build Time", t.Seconds());
//
//    t.Start();
//    // base graph insertion
//    auto end_it = el.begin() + base_graph_num_edges_;
//    for (auto it = el.begin(); it < end_it; it++) {
//      Edge e = *it;
//
//      // adding edge u->v
////      g.insert_edge(e.u, e.v);
////      if (symmetrize_) {
////        // adding edge v->u
////        g.insert_edge(static_cast<NodeID_>(e.v), GetSource(e));
////      }
//    }
//
//    t.Stop();
//    PrintTime("B-Graph Build Time", t.Seconds());
//
//    t.Start();
//    // dynamic insertion
//    auto begin_it = el.begin() + base_graph_num_edges_;
//    for (auto it = begin_it; it < el.end(); it++) {
//      Edge e = *it;
//
//      // adding edge u->v
////      g.insert_edge(e.u, e.v, true);
////      if (symmetrize_) {
////        // adding edge v->u
////        g.insert_edge(static_cast<NodeID_>(e.v), GetSource(e), true);
////      }
//    }
//
//    t.Stop();
//    PrintTime("D-Graph Build Time", t.Seconds());
//    cerr << t.Seconds() << endl;
////    fprintf(stderr, "%lf\n", t.Seconds());
//  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraph() {
    EdgeList el;
    Timer t;
    if (cli_.base_filename() != "") {
      Reader<NodeID_, DestID_, WeightT_, TimestampT_, invert> r(cli_.base_filename());
      el = r.ReadFile(needs_weights_);
    }
    else {
      printf("[%s]: graph input-file not exists, abort!!!\n", __FUNCTION__);
      exit(0);
    }

    base_graph_num_edges_ = el.size();
    num_nodes_ = FindMaxNodeID(el) + 1;

    if(symmetrize_) {
      for(int i=0; i<base_graph_num_edges_; i+=1) {
        el.push_back(EdgePair<NodeID_, DestID_>(static_cast<NodeID_>(el[i].v), GetSource(el[i])));
      }
      base_graph_num_edges_ *= 2;
    }

    std::sort(el.begin(), el.end(), [](Edge &a, Edge &b) {
      if(a.u != b.u) return a.u < b.u;
      return (a.v.t < b.v.t);
    });

    if (needs_weights_) Generator<NodeID_, DestID_, WeightT_, TimestampT_>::InsertWeights(el);
    CSRGraph<NodeID_, DestID_, invert> g(el, !symmetrize_, base_graph_num_edges_, num_nodes_);

    el.clear();

    if (cli_.dynamic_filename() != "") {
      Reader<NodeID_, DestID_, WeightT_, TimestampT_, invert> r(cli_.dynamic_filename());
      el = r.ReadFile(needs_weights_);
    }
    else {
      printf("[%s]: graph input-file not exists, abort!!!\n", __FUNCTION__);
      exit(0);
    }
    if (needs_weights_) Generator<NodeID_, DestID_, WeightT_, TimestampT_>::InsertWeights(el);
    size_t dynamic_edges = el.size();
    t.Start();
    for(uint32_t i=0; i<dynamic_edges; i+=1) {
      g.insert(el[i].u, el[i].v.v, el[i].v.w);
      if(symmetrize_) {
        g.insert(el[i].v.v, el[i].u, el[i].v.w);
      }
//      if(i && i % 10000000 == 0) cout << "inserted " << (i/1000000) << "M dynamic edges" << endl;
    }
    t.Stop();
    cout << "D-Graph Build Time: " << t.Seconds() << " seconds." << endl;

    return g;
  }

  // Relabels (and rebuilds) graph by order of decreasing degree
  static
  CSRGraph<NodeID_, DestID_, invert> RelabelByDegree(
      const CSRGraph<NodeID_, DestID_, invert> &g) {
    if (g.directed()) {
      std::cout << "Cannot relabel directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();
    typedef std::pair<int64_t, NodeID_> degree_node_p;
    pvector<degree_node_p> degree_id_pairs(g.num_nodes());
    #pragma omp parallel for
    for (NodeID_ n=0; n < g.num_nodes(); n++)
      degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
    std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
              std::greater<degree_node_p>());
    pvector<NodeID_> degrees(g.num_nodes());
    pvector<NodeID_> new_ids(g.num_nodes());
    #pragma omp parallel for
    for (NodeID_ n=0; n < g.num_nodes(); n++) {
      degrees[n] = degree_id_pairs[n].first;
      new_ids[degree_id_pairs[n].second] = n;
    }
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_* neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_** index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
    #pragma omp parallel for
    for (NodeID_ u=0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.out_neigh(u))
        neighs[offsets[new_ids[u]]++] = new_ids[v];
      std::sort(index[new_ids[u]], index[new_ids[u]+1]);
    }
    t.Stop();
    PrintTime("Relabel", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }
};

#endif  // BUILDER_H_
