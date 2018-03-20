#ifndef UTIL_NULLSTREAM_HPP
#define UTIL_NULLSTREAM_HPP

#include <ostream>

class NullBuffer : public std::streambuf {
public:
  int overflow(int c) { return c; }
};

class NullStream : public std::ostream {
public:
  NullStream() : std::ostream(&nb_) {}

private:
  NullBuffer nb_;
};

#endif