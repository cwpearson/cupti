#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <map>
#include <string>

class AddressSpace {
public:
  enum class Type { Unknown, Host, Cuda, Invalid };
  AddressSpace() : type_(Type::Invalid) {}
  AddressSpace(const AddressSpace &other) : type_(other.type_) {}

private:
  AddressSpace(Type type) : type_(type) {}

public:
  bool operator==(const AddressSpace &rhs) const { return type_ == rhs.type_; }
  bool operator<(const AddressSpace &rhs) const { return type_ < rhs.type_; }

  bool is_valid() const { return type_ != Type::Invalid; }
  bool is_host() const { return type_ == Type::Host; }
  bool is_cuda() const { return type_ == Type::Cuda; }
  bool is_unknown() const { return type_ == Type::Unknown; }

  bool maybe_equal(const AddressSpace &other) const;

  std::string json() const;

  static AddressSpace Host() { return AddressSpace(AddressSpace::Type::Host); }
  static AddressSpace Cuda() { return AddressSpace(AddressSpace::Type::Cuda); }
  static AddressSpace Unknown() {
    return AddressSpace(AddressSpace::Type::Unknown);
  }

private:
  Type type_;
};

#endif
