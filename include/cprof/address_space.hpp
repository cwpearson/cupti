#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <map>
#include <string>

class AddressSpace {
public:
  enum class Type {
    Unknown,
    Host,       ///< CUDA <4.0 host address space
    CudaDevice, ///< CUDA <4.0 address space for a single device
    CudaUVA,    ///< CUDA >4.0 unified virtual addressing
    Invalid
  };

private:
  AddressSpace(Type type, int device) : type_(type), device_(device) {}
  AddressSpace(Type type) : AddressSpace(type, -1) {}
  Type type_;
  int device_; ///< which device the address space is associated with

public:
  bool operator==(const AddressSpace &rhs) const { return type_ == rhs.type_; }
  bool operator<(const AddressSpace &rhs) const { return type_ < rhs.type_; }

  bool is_valid() const { return type_ != Type::Invalid; }
  bool is_host() const { return type_ == Type::Host; }
  bool is_cuda_device() const { return type_ == Type::CudaDevice; }
  bool is_cuda_uva() const { return type_ == Type::CudaUVA; }
  bool is_unknown() const { return type_ == Type::Unknown; }

  bool maybe_equal(const AddressSpace &other) const;

  std::string json() const;

  static AddressSpace Host() { return AddressSpace(AddressSpace::Type::Host); }
  static AddressSpace CudaDevice(int device) {
    return AddressSpace(AddressSpace::Type::CudaDevice, device);
  }
  static AddressSpace CudaUVA() {
    return AddressSpace(AddressSpace::Type::CudaUVA);
  }
  static AddressSpace Unknown() {
    return AddressSpace(AddressSpace::Type::Unknown);
  }
};

#endif
