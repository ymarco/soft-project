#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  assert(argc == 2);
  int cluster_count = atoi(argv[0]);
  int samples_count = atoi(argv[1]);
  int dim = atoi(argv[2]);
  int max_iters = atoi(argv[2]);
  return 0;
}
