# VCSR
This is the source code of VCSR, a new mutable CSR storage format that leverages the Packed Memory Array (PMA) via a new vertex-centric strategy to store temporal graphs efficiently. You can learn more about this in our [CCGrid 2022 paper](https://webpages.charlotte.edu/ddai/papers/dong-ccgrid-22.pdf).

This repo includes the source code of VCSR along with the competitor data structures we have used in our performance evaluation. We also added a few sample datasets and workloads used in our experiments/tests.

## Citing VCSR:

```
@inproceedings{islam2022vcsr,
  title={VCSR: Mutable CSR Graph Format Using Vertex-Centric Packed Memory Array},
  author={Islam, Abdullah Al Raqibul and Dai, Dong and Cheng, Dazhao},
  booktitle={2022 22th IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGRID)},
  year={2022},
  pages={71-80},
  doi={10.1109/CCGrid54584.2022.00016}
}
```

### Table of Contents
**[Quick Start](#quick-start)**<br>
**[Design and Implementations](#design-and-implementation)**<br>
**[Usage](#usage)**<br>
**[Contribution](#contribution)**<br>
**[Contact](#contact)**<br>
**[Reference](#reference)**<br>

## Quick Start
The code has been implemented in C++ and tested on Ubuntu 18.04 with CMake 3.13.4 and gcc 7.5.0.

### Prerequisites
* C++11 compiler
* OpenMP

### Build & Run
Clone the repository:
```
> git clone https://github.com/DIR-LAB/VCSR.git
```

Set the `VCSR root` directory:
```
> export VCSR_ROOT=$PWD/VCSR
```

To build any data structure, go to the corresponding subdirectory and run `make`. For example, to build vcsr:

```
> cd $VCSR_ROOT/vcsr
> make
```

After the build process, you will find four executables (`pr`, `bfs`, `sssp`, and `cc`). When you run one of them, it will first load and build the graph from the input provided and then run the corresponding graph analytic algorithm.

Please note that the current version of VCSR supports single thread graph insertion/build and parallel graph analysis. You can set the number of parallel threads for the graph analysis by overwriting the following `OpenMP` environment variable:
```
> export OMP_NUM_THREADS=32
```

Run a graph analysis in dynamic graph data structures (`vcsr`, `pcsr`, and `bal`):
```
./ALGO -B BASE_GRAPH_INPUT -D BASE_GRAPH_INPUT -r START_NODE -n NUMBER_OF_TRIALS -a
```

Run a graph analysis in static graph data structures (`csr`):
```
./ALGO -f GRAPH_INPUT -r START_NODE -n NUMBER_OF_TRIALS -a -s
```

Replace `ALGO` with one of the executables (`pr`, `bfs`, `sssp`, and `cc`) you compiled before. You can run the code with `-h` (i.e. `./pr -h`) to see all the available command-line arguments. Please check the data structure subdirectories for a detailed example of running instruction.

## Design and Implementations
This repository contains the implementation of the following data structures
* CSR: Baseline for graph analysis performance.
* VCSR: Our proposed mutable CSR storage format.
* Blocked Adjacency List (BAL): Fixed number of edges (e.g., 512 edges in our case) stored per block.
* [Packed CSR (PCSR)](https://github.com/wheatman/Packed-Compressed-Sparse-Row) [1]: Another mutable CSR format that also leverages PMA.

We evaluated all the data structures in the [GAP Benchmark Suite (GAPBS)](https://github.com/sbeamer/gapbs).

### Repository
At the high level, the repository structure looks like this:

```
.
├── bal
├── csr
├── data
├── pcsr
└── vcsr
```
* As the name suggests, `bal`, `csr`, `pcsr`, and `vcsr` contains the code for the corresponding data structures.
* The `data` directory contains two temporal graph input files (`sx-mathoverflow` and `fb-wall`) that we have used for performance evaluation in the paper.

### Input Graph Data

#### Graph Properties
* **Direction of the graph:** Currently, all the data structures (except `CSR`) only store the out-going edges of the graph. Some of the graph algorithms implemented in the [GAP Benchmark Suite (GAPBS)](https://github.com/sbeamer/gapbs) expect to access both the in and out-going edges.
    * We solve this problem by inserting the inverse edges for the directed graph datasets.
    * If your input graph is a directed graph, pass `-s` parameter so that the program will insert the inverse edges during the insertion procedure.
    * One way to solve this problem is to initiate two data structure instances to store both the in and out-going edges. We consider this as a future extension.
* **Weighted/Property of the graph:** Currently, all the graph data structures stores the weighted graph. Please check the [Supported Graph Data Format](#supported-graph-data-format) for details.

#### Supported Graph Data Format
All the data structures (implemented in this repository) expect the input graphs in the **edge-list format**. So, for example, the data will look like the following:

```
1015    1017
1017    1015
14736   14752
14752   14736
1080    1531
```

We used weighted versions of all graphs for a fair comparison with the publicly available version of [PCSR](https://github.com/wheatman/Packed-Compressed-Sparse-Row), which has a required field (e.g., value/weight) in its edge structure. So, we assign random integer weights in the range `[0,256)` if the input graph is unweighted.  The framework expects `.wel` and `.el` as the file extension for the weighted and unweighted input graphs, respectively.

#### Graph Data Partition

In our graph insertion evaluation, we initialize the data structures with the first X% of the graph. Later, we insert the rest of the graphs and evaluate the dynamic insertion performance. For this reason, we partition each of the graphs into two files (we call them `base` and `dynamic`). You can learn more if you look closely at the `"data"` directory.

The `data` directory looks like this:

```
.
├── fbwall
│   ├── b10
│   └── b30
└── mathoverflow
    ├── b10
    └── b30
```

* Directory `fbwall` and `mathoverflow` contains the data for `fb-wall` [3] and `sx-mathoverflow` [2] graph respectively.
* Subdirectory `b10` and `b30` contain the partitions of 10% and 30% base graph, respectively. Each of these subdirectories has files `base.wel` and `dynamic.wel` for the graph's base and dynamic partition, respectively.

Please check the run instructions below to understand how to use the `"data"` directory in your benchmark.

## Usage

One of the key motivations of this project is to allow others to reuse our VCSR implementation for performance comparison. You will get some idea about workload generation from the [Workloads](#workloads) section.

## Contribution

If you would like to contribute to this project, we would certainly appreciate your help! Here are some of the ways you can contribute:

* Bug fixes, whether for performance or correctness of the existing data structure implementation.
* Improvement of the existing documentation.
* Add additional data structure implementations for dynamic graphs; we are open to adding more data structures in this repo. However, please keep in mind that we will only accept new data structures if you either integrate them in the [GAP Benchmark Suite (GAPBS)](https://github.com/sbeamer/gapbs) or use the graph analysis code implemented at GAPBS. In this way, we will be able to make comparable performance analyses in the future.

Our future goal is to provide a set of portable, high-performance data structure baselines for dynamic graphs. For code contributions, please focus on code simplicity and readability. If you open a pull request, we will do a quick sanity check and respond to you as soon as possible.

## Contact
* [Abdullah Al Raqibul Islam (UNC Charlotte)](https://github.com/biqar)
* [Dong Dai (UNC Charlotte)](https://webpages.charlotte.edu/ddai)

## Reference
1. B. Wheatman and H. Xu, "Packed Compressed Sparse Row: A Dynamic Graph Representation," 2018 IEEE High Performance extreme Computing Conference (HPEC), 2018, pp. 1-7, doi: 10.1109/HPEC.2018.8547566.
2. A. Paranjape, A. R. Benson, and J. Leskovec, “Motifs in temporal networks,” in Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, ser. WSDM ’17. New York, NY, USA: Association for Computing Machinery, 2017, p. 601–610. [Online]. Available: https://doi.org/10.1145/3018661.3018731
3. M. Beladev, L. Rokach, G. Katz, I. Guy, and K. Radinsky, “Tdgraphembed: Temporal dynamic graph-level embedding,” in Proceedings of the 29th ACM International Conference on Information amp; Knowledge Management, ser. CIKM ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 55–64. [Online]. Available: https: //doi.org/10.1145/3340531.3411953