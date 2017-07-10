#ifndef OPTIONAL_HPP
#define OPTIONAL_HPP

#include <utility>

template <typename U> class optional {
private:
  U value_;
  bool has_value_;

public:
  constexpr optional() noexcept : has_value_(false) {}
  constexpr optional(U &&value) : value_(std::move(value)){};
  constexpr optional(U &value) : value_(value){};
  constexpr optional(const optional &other)
      : value_(other.value_), has_value_(other.has_value_) {}

  constexpr explicit operator bool() const noexcept { return has_value_; }
  constexpr bool has_value() const noexcept { return has_value(); }

  U &value() { return value_; }
  const U &value() const { return value_; }
};

#endif