#include <nlohmann/json.hpp>

#include "cprof/allocation.hpp"

using json = nlohmann::json;
using cprof::model::Location;
using cprof::model::Memory;

size_t AllocationRecord::next_val_ = 1;

std::string AllocationRecord::to_json_string() const {
  return to_json().dump();
}

json AllocationRecord::to_json() const {
  json j;
  j["allocation"]["id"] = uintptr_t(this);
  j["allocation"]["pos"] = lower_;
  j["allocation"]["size"] = upper_ - lower_;
  j["allocation"]["addrsp"] = address_space_.to_json();
  j["allocation"]["mem"] = cprof::model::to_json(memory_);
  j["allocation"]["loc"] = location_.to_json();
  return j;
}

cprof::Value AllocationRecord::new_value(uintptr_t pos, size_t size,
                                         const bool initialized) {
  val_ = next_val_++;
  val_initialized_ = initialized;
  val_size_ = size;
  cprof::Value newVal(val_, pos, val_size_, uintptr_t(this), address_space_,
                      val_initialized_);
  logging::atomic_out(newVal.to_json_string());
  return newVal;
}

/// \brief Get existing value at pos
cprof::Value AllocationRecord::value(const uintptr_t pos) const {
  return cprof::Value(val_, pos, val_size_, uintptr_t(this), address_space_,
                      val_initialized_);
}

uintptr_t Allocation::pos() const noexcept { return ar_->lower_; }
size_t Allocation::size() const noexcept { return ar_->upper_ - ar_->lower_; }
AddressSpace Allocation::address_space() const { return ar_->address_space_; }
Memory Allocation::memory() const { return ar_->memory_; }
Location Allocation::location() const { return ar_->location_; }
bool Allocation::freed() const noexcept { return ar_->freed_; }
void Allocation::free() { ar_->freed_ = true; }
size_t Allocation::id() const noexcept { return uintptr_t(ar_.get()); }
std::string Allocation::to_json_string() const { return ar_->to_json_string(); }

bool Allocation::operator!() const noexcept { return !ar_; }
Allocation::operator bool() const noexcept { return bool(ar_); }

bool Allocation::operator==(const Allocation &rhs) const noexcept {
  return ar_ == rhs.ar_;
}