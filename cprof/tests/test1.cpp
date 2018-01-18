#include <gtest/gtest.h>

#include "cprof/allocations.hpp"
#include "cprof/model/location.hpp"
#include "cprof/model/memory.hpp"

using cprof::model::Location;
using cprof::model::Memory;

// The fixture for testing class Foo.
class AllocationsTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  AllocationsTest() {
    // You can do set-up work for each test here.
  }

  virtual ~AllocationsTest() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

TEST_F(AllocationsTest, ctor) { cprof::Allocations as; }

TEST_F(AllocationsTest, find) {
  cprof::Allocations as;

  const auto AS = AddressSpace::Host();
  const auto M = Memory::Pageable;
  const auto L = Location::Host();

  const auto a1 = as.new_allocation(1, 1, AS, M, L);
  EXPECT_EQ(1, a1.pos());
  EXPECT_EQ(1, a1.size());
  EXPECT_EQ(1, as.size());

  const auto a2 = as.find(a1.pos(), a1.size(), AS);
  EXPECT_EQ(a1, a2);

  const auto a3 = as.find(a1.pos(), AS);
  EXPECT_EQ(a1, a3);
}

TEST_F(AllocationsTest, nofind) {
  cprof::Allocations as;

  const auto AS = AddressSpace::Host();
  const auto M = Memory::Pageable;
  const auto L = Location::Host();

  const auto a1 = as.new_allocation(100, 10, AS, M, L);
  const auto a2 = as.find(10, 1, AS);
  EXPECT_FALSE(a2);
}

TEST_F(AllocationsTest, free) {
  cprof::Allocations as;

  const auto AS = AddressSpace::Host();
  const auto M = Memory::Pageable;
  const auto L = Location::Host();
  auto a1 = as.new_allocation(1, 1, AS, M, L);
  EXPECT_TRUE(a1);

  auto count = as.free(a1.pos(), AS);
  EXPECT_EQ(true, count);
}

TEST_F(AllocationsTest, merge) {
  cprof::Allocations as;

  const auto AS = AddressSpace::Host();
  const auto M = Memory::Pageable;
  const auto L = Location::Host();

  const auto a1 = as.new_allocation(1, 3, AS, M, L);
  const auto a2 = as.new_allocation(2, 3, AS, M, L);
  std::cerr << a1.id() << " " << a2.id() << "\n";
  EXPECT_EQ(1, as.size());
  EXPECT_EQ(a1, a2);

  const auto a3 = as.find(2, AS);
  EXPECT_EQ(a3.pos(), 1);
  EXPECT_EQ(a3.size(), 4);

  const auto a4 = as.find(1, AS);
  EXPECT_EQ(a4.pos(), 1);
  EXPECT_EQ(a4.size(), 4);
}

TEST_F(AllocationsTest, newsame) {
  cprof::Allocations as;

  const auto AS = AddressSpace::Host();
  const auto M = Memory::Pageable;
  const auto L = Location::Host();

  const auto a1 = as.new_allocation(70366076463264, 200, AS, M, L);
  const auto a2 = as.find(70366076463264, 200, AS);
  EXPECT_EQ(a1, a2);

  const auto a3 = as.new_allocation(70366076463264, 200, AS, M, L);
  EXPECT_EQ(1, as.size());
  EXPECT_EQ(a1, a3);
  EXPECT_EQ(200, a3.size());
  EXPECT_EQ(70366076463264, a3.pos());
}