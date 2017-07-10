#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <map>
#include <string>

class AddressSpace {
public:
  typedef uint64_t flag_t;

  static constexpr flag_t Host = 0x1;
  static constexpr flag_t CudaDevice = 0x2;
  static constexpr flag_t CudaUnified = 0x4;
  static constexpr flag_t CudaUnknown = 0x8;
  static constexpr flag_t CudaAny =
      Host | CudaDevice | CudaUnified | CudaUnknown;

  AddressSpace() {}
  AddressSpace(flag_t type) : type_(type) {}
  AddressSpace(const AddressSpace &other) : type_(other.type_) {}

  bool operator==(const AddressSpace &rhs) const { return type_ == rhs.type_; }

  bool is_host() const { return type_ & Host; }
  bool is_device() const { return type_ & CudaDevice; }
  bool is_unified() const { return type_ & CudaUnified; }
  bool is_host_accessible() const { return is_host() || is_unified(); }
  bool is_device_accessible() const { return is_device() || is_unified(); }
  bool overlaps(const AddressSpace &other) const { return type_ & other.type_; }

  std::string json() const;

private:
  static std::string to_string(const flag_t &f);
  flag_t type_;
};

#endif
