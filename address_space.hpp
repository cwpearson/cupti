#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <map>
#include <string>

class AddressSpace {
public:
  typedef uint64_t flag_t;

  static constexpr flag_t Unknown = 0x0;
  static constexpr flag_t Host = 0x1;
  static constexpr flag_t Cuda = 0x2;

  AddressSpace() {}
  AddressSpace(flag_t type) : type_(type) {}
  AddressSpace(const AddressSpace &other) : type_(other.type_) {}

  bool operator==(const AddressSpace &rhs) const { return type_ == rhs.type_; }
  bool operator<(const AddressSpace &rhs) const { return type_ < rhs.type_; }

  bool is_host() const { return type_ & Host; }
  bool is_cuda() const { return type_ & Cuda; }
  bool is_unknown() const { return type_ == Unknown; }

  bool maybe_equal(const AddressSpace &other) const {
    return other == *this || is_unknown() || other.is_unknown();
  }

  std::string json() const;

private:
  flag_t type_;
};

#endif
