TEST_TARGETS += $(TEST_DIR)/test1

TEST_INC_ += -isystem$(GTEST_DIR)/include -Iinclude
TEST_LIB_ += $(GTEST_DIR)/lib/gtest.a -L$(LIBDIR) -lcprof

$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(TEST_INC_) $< -c -o $@

$(TEST_DIR)/test1: $(TEST_DIR)/test1.o $(LIBDIR)/libcprof.so $(GTEST_DIR)/lib/gtest.a
	$(CXX) $(CXXFLAGS) $^ -o $@ $(TEST_LIB_)

tests: $(TEST_TARGETS)
	$(TEST_DIR)/test1

# Tell parent make how to clean up after us
CLEAN_TARGETS += $(TEST_TARGETS) $(TEST_DIR)/*.o