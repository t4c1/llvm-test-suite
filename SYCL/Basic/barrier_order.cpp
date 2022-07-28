// UNSUPPORTED: hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <stdlib.h>
#include <sycl/sycl.hpp>

int main() {
  sycl::device dev{sycl::default_selector{}};
  sycl::queue q{dev};

  unsigned long long *x = sycl::malloc_shared<unsigned long long>(1, q);

  int error = 0;

  // test barrier without arguments
  *x = 0;
  for (int i = 0; i < 64; i++) {
    q.parallel_for(1024, [=](cl::sycl::id<1> ID) {
      // do some busywork
      float y = *x;
      for (int j = 0; j < 10000; j++) {
        y = sycl::cos(y);
      }
      // update the value
      if (ID.get(0) == 0)
        *x *= 2;
    });
    q.ext_oneapi_submit_barrier();
    q.parallel_for(1024, [=](cl::sycl::id<1> ID) {
      if (ID.get(0) == 0)
        *x += 1;
    });
    q.ext_oneapi_submit_barrier();
  }

  q.wait_and_throw();
  error |= (*x != (unsigned long long)-1);

  // test barrier when events are passed arguments
  *x = 0;
  for (int i = 0; i < 64; i++) {
    sycl::event e = q.parallel_for(1024, [=](cl::sycl::id<1> ID) {
      // do some busywork
      float y = *x;
      for (int j = 0; j < 10000; j++) {
        y = sycl::cos(y);
      }
      // update the value
      if (ID.get(0) == 0)
        *x *= 2;
    });
    q.ext_oneapi_submit_barrier({e});
    e = q.parallel_for(1024, [=](cl::sycl::id<1> ID) {
      if (ID.get(0) == 0)
        *x += 1;
    });
    q.ext_oneapi_submit_barrier({e});
  }

  q.wait_and_throw();
  error |= (*x != (unsigned long long)-1);

  std::cout << (error ? "failed\n" : "passed\n");

  sycl::free(x, q);

  return error;
}
