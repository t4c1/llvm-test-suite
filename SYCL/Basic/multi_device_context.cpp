// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// REQUIRES: cuda

#include <sycl.hpp>
#include <stdlib.h>

sycl::event add(sycl::queue q, sycl::buffer<int> buff, int* usm, sycl::event e){
  	return q.submit([&](sycl::handler &cgh) {
      auto acc = buff.get_access<sycl::access::mode::read_write>(cgh);
      cgh.depends_on(e);
      cgh.single_task([=]() {
        acc[0] += *usm;
      });
    });
}

int main() {
  sycl::platform plat = sycl::platform::get_platforms()[0];
  auto devices = plat.get_devices();
  if(devices.size()<2){
    std::cout << "Need two devices for the test!" << std::endl;
    return 0;
  }

  sycl::device dev1 = devices[0];
  sycl::device dev2 = devices[1];

  sycl::context ctx{{dev1, dev2}};

  sycl::queue q1{ctx, dev1};
  sycl::queue q2{ctx, dev2};

  int a = 1;
  int b = 2;
  int c = 4;
  int d = 5;
  {
    sycl::buffer<int> buff1(&a,1);
    sycl::buffer<int> buff2(&b,1);

    int* usm1 = sycl::malloc_device<int>(1,q1);
    int* usm2 = sycl::malloc_device<int>(1,q2);
    sycl::event e1 = q1.memcpy(usm1, &c, 1);
    sycl::event e2 = q2.memcpy(usm2, &d, 1);
    *usm1 = 4;
    *usm2 = 5;

    sycl::event e3 = add(q1, buff1, usm1, e1);
    sycl::event e4 = add(q2, buff2, usm2, e2);
    sycl::event e5 =q1.memcpy(usm1, &d, 1, e3);
    sycl::event e6 =q2.memcpy(usm2, &c, 1, e4);
    add(q1, buff2, usm1, e5);
    add(q2, buff1, usm2, e6);
  }
  assert(a == 1+2*4);
  assert(b == 2+2*5);

  return 0;
}
