GTEST_DIR = $(CPROF_ROOT)/external/googletest

include $(GTEST_DIR)/Makefile

TEST_TARGETS += $(TEST_DIR)/test1

TEST_INC_ += -isystem$(GTEST_DIR)/include -Iinclude
TEST_LIB_ += $(GTEST_DIR)/lib/gtest_main.a -L$(LIBDIR) -lcprof $(LIB)

$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(TEST_INC_) $< -c -o $@

$(TEST_DIR)/test1: $(TEST_DIR)/test1.o $(LIBDIR)/libcprof.so $(GTEST_DIR)/lib/gtest_main.a
	$(CXX) $(CXXFLAGS) $^ -o $@ $(TEST_LIB_)

tests: $(TEST_TARGETS) 
	$(TEST_DIR)/test1

# Tell parent make how to clean up after us
CLEAN_TARGETS += $(TEST_TARGETS) $(TEST_DIR)/*.o