// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <cinttypes>
#include <string>

#include "timer.h"


/*
GAP Benchmark Suite
Author: Scott Beamer

Miscellaneous helpers that don't fit into classes
*/

/*****************************************************************************
 *                                                                           *
 *   APMA Utility functions                                                  *
 *                                                                           *
 *****************************************************************************/

// Returns the 1-based index of the last (most significant) bit set in x.
inline uint64_t last_bit_set(uint64_t x)
{
  return (sizeof(uint64_t) * 8 - __builtin_clzll(x)); // Linux
}

inline uint64_t floor_log2(uint64_t x)
{
  return (last_bit_set(x) - 1);
}

inline uint64_t ceil_log2(uint64_t x)
{
  assert(x > 0 && "x should not be 0 or less");
  return (last_bit_set(x - 1));
  // i.e. ceil_log2(13) = 4, ceil_log2(27) = 5, etc.
}

inline uint64_t floor_div(uint64_t x, uint64_t y)
{
  return (x / y);
}

inline uint64_t ceil_div(uint64_t x, uint64_t y)
{
  if(x == 0) return 0;
  return (1 + ((x - 1) / y));
}

// Returns the largest power of 2 not greater than x ($2^{\lfloor \lg x \rfloor}$).
inline uint64_t hyperfloor(uint64_t x)
{
  return (1ULL << floor_log2(x));
}

// Returns the smallest power of 2 not less than x ($2^{\lceil \lg x \rceil}$).
inline uint64_t hyperceil(uint64_t x)
{
  return (1ULL << ceil_log2(x));
}

static const int64_t kRandSeed = 27491095;


void PrintLabel(const std::string &label, const std::string &val) {
  printf("%-21s%7s\n", (label + ":").c_str(), val.c_str());
}

void PrintTime(const std::string &s, double seconds) {
  printf("%-21s%3.5lf\n", (s + ":").c_str(), seconds);
}

void PrintStep(const std::string &s, int64_t count) {
  printf("%-14s%14" PRId64 "\n", (s + ":").c_str(), count);
}

void PrintStep(int step, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5d%11" PRId64 "  %10.5lf\n", step, count, seconds);
  else
    printf("%5d%23.5lf\n", step, seconds);
}

void PrintStep(const std::string &s, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5s%11" PRId64 "  %10.5lf\n", s.c_str(), count, seconds);
  else
    printf("%5s%23.5lf\n", s.c_str(), seconds);
}

// Runs op and prints the time it took to execute labelled by label
#define TIME_PRINT(label, op) {   \
  Timer t_;                       \
  t_.Start();                     \
  (op);                           \
  t_.Stop();                      \
  PrintTime(label, t_.Seconds()); \
}


template <typename T_>
class RangeIter {
  T_ x_;
 public:
  explicit RangeIter(T_ x) : x_(x) {}
  bool operator!=(RangeIter const& other) const { return x_ != other.x_; }
  T_ const& operator*() const { return x_; }
  RangeIter& operator++() {
    ++x_;
    return *this;
  }
};

template <typename T_>
class Range{
  T_ from_;
  T_ to_;
 public:
  explicit Range(T_ to) : from_(0), to_(to) {}
  Range(T_ from, T_ to) : from_(from), to_(to) {}
  RangeIter<T_> begin() const { return RangeIter<T_>(from_); }
  RangeIter<T_> end() const { return RangeIter<T_>(to_); }
};

#endif  // UTIL_H_
