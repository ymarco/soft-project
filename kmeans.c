#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int cluster_count;
float *centroids;
int *cluster_size; /* amount of elements in each cluster */
int sample_count;
float *samples;
int *sample_cluster_index;
int dim;
int max_iters;

/*
 * fill the data in samples from CSV file stream IN.
 */
void read_data() {
  float f;
  char c;
  float *cent_arr = centroids;
  int cur_vector = 0;
  int cur_element = 0;

  while (scanf("%f%c", &f, &c) == 2) {
    cent_arr[cur_vector * dim + cur_element] = f;
    samples[cur_vector * dim + cur_element] = f;
    switch (c) {
    case ',':
      cur_vector++;
      break;
    case '\n':
      cur_element++;
      cur_vector = 0;
      break;
    default:
      fprintf(stderr, "Invalid input\n");
      exit(EXIT_FAILURE);
    }
    cent_arr = (cur_vector < cluster_count) ? centroids : samples;
  }
}

/*
 * Return the squared distance between v1 and v2, both assumed to be with length
 * dim.
 */
float squared_distance(float *v1, float *v2) {
  float res = 0;
  int i;
  for (i = 0; i < dim; i++) {
    res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return res;
}

/*
 * Return index of the closest centroid to samples[sample_i]
 */
int closest_centroid_index(int sample_i) {
  float min = squared_distance(&centroids[0], &samples[sample_i * dim]);
  int min_i = 0;
  float cur;
  int centroid_i;
  for (centroid_i = 1; centroid_i < cluster_count; centroid_i++) {
    cur = squared_distance(&centroids[centroid_i * dim],
                           &samples[sample_i * dim]);
    min_i = cur < min ? centroid_i : min_i;
    min = cur < min ? cur : min;
  }
  return min_i;
}

/*
 * sum += vec, by their components
 */
void vector_plus_equal(float *sum, float *vec) {
  int i;
  for (i = 0; i < dim; i++) {
    sum[i] += vec[i];
  }
}
/*
 * quotient /= denumenator, by quotient's compopnents
 */
void vector_divide_equal(float *quotient, float denumenator) {
  int i;
  for (i = 0; i < dim; i++) {
    quotient[i] /= denumenator;
  }
}

/*
 * Update centroid index for sample at index i.
 * Return true if the index changed since the last time.
 */
int update_centroid_for_sample(int i) {
  int c = closest_centroid_index(i);
  int res = c != sample_cluster_index[i];
  sample_cluster_index[i] = c;
  return res;
}

/*
 * Update the cluster indecies of all samples.
 * Return true if any index changed.
 */
int mass_cluster_centroid_index_update() {
  int i;
  int res = 0;
  for (i = 0; i < sample_count; i++) {
    res += res | update_centroid_for_sample(i);
  }
  return res;
}

void mass_centroid_update() {
  /* memset sums to 0 */
  int i;
  int sample_i;
  for (i = 0; i < dim * cluster_count; i++) {
    centroids[i] = 0;
  }
  /* memset size to 0 */
  for (i = 0; i < cluster_count; i++) {
    cluster_size[i] = 0;
  }
  /* sum up samples in centroids */
  for (sample_i = 0; sample_i < sample_count; sample_i++) {
    int centroid_i = sample_cluster_index[sample_i];
    cluster_size[centroid_i]++;
    vector_plus_equal(&centroids[centroid_i * dim], &samples[sample_i * dim]);
  }
  /* devide clusters by their size to get the average */
  for (i = 0; i < cluster_count; i++) {
    vector_divide_equal(&centroids[i * dim], cluster_size[i]);
  }
}

void print_vectors(float *vectors, int count) {
  float f;
  char c;
  int i;
  for (i = 0; i < dim * count; i++) {
    f = vectors[i];
    c = (i % dim == 1) ? '\n' : ',';
    printf("%.2f%c", f, c);
  }
}

int main(int argc, char *argv[]) {
  int i;
  assert(argc == 5);
  cluster_count = atoi(argv[1]);
  sample_count = atoi(argv[2]);
  dim = atoi(argv[3]);
  max_iters = atoi(argv[4]);

  /* printf("cluster_count: %d, sample_count:", ); */
  /* We might be able to chain all of these into one big malloc. Not sure its
   * worth it though. Standards after ANSI C just allow variable-sized arrays in
   * stack. */
  centroids = malloc(sizeof(*centroids) * dim * cluster_count);
  cluster_size = malloc(sizeof(*cluster_size) * cluster_count);
  samples = malloc(sizeof(*samples) * dim * sample_count);
  sample_cluster_index = malloc(sizeof(*sample_cluster_index) * sample_count);

  read_data();

  for (i = 0; i < max_iters; i++) {
    if (!mass_cluster_centroid_index_update()) /* no indecies changed */
      break;
    mass_centroid_update();
  }
  fprintf(stderr, "iters = %d\n", i);
  print_vectors(centroids, cluster_count);
  return 0;
}
