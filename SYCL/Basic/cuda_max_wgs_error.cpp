// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// REQUIRE: cuda

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

const size_t lsize = 32;
const size_t max_x = (1ull << 31ull) - 1ull;
const size_t max_yz = 65535;
const std::string expected_msg = "Number of work-groups exceed limit for dimension ";

template <int N>
void check(sycl::range<N> global, sycl::range<N> local, bool expect_fail = false) {
  queue q;
  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<N>(global, local), [=](sycl::nd_item<N> item) {});
    });
  } catch (sycl::nd_range_error e) {
    if (expect_fail) {
      std::string msg = e.what();
      assert(msg.rfind(expected_msg, 0) == 0);
    } else {
      throw e;
    }
  }
}

int main() {
  return 0;
  check(sycl::range<1>(max_x * lsize), sycl::range<1>(lsize));
  check(sycl::range<1>((max_x + 1) * lsize), sycl::range<1>(lsize), true);

  check(sycl::range<2>(1, max_x * lsize), sycl::range<2>(1, lsize));
  check(sycl::range<2>(1, (max_x + 1) * lsize), sycl::range<2>(1, lsize), true);
  check(sycl::range<2>(max_yz * lsize, 1), sycl::range<2>(lsize, 1));
  check(sycl::range<2>((max_yz + 1) * lsize, 1), sycl::range<2>(lsize, 1), true);

  check(sycl::range<3>(1, 1, max_x * lsize), sycl::range<3>(1, 1, lsize));
  check(sycl::range<3>(1, 1, (max_x + 1) * lsize), sycl::range<3>(1, 1, lsize), true);
  check(sycl::range<3>(1, max_yz * lsize, 1), sycl::range<3>(1, lsize, 1));
  check(sycl::range<3>(1, (max_yz + 1) * lsize, 1), sycl::range<3>(1, lsize, 1), true);
  check(sycl::range<3>(max_yz * lsize, 1, 1), sycl::range<3>(lsize, 1, 1));
  check(sycl::range<3>((max_yz + 1) * lsize, 1, 1), sycl::range<3>(lsize, 1, 1), true);
}
