#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int cluster_count;
double *centroids;
int *cluster_size; /* amount of elements in each cluster */
int sample_count;
double *samples;
int *sample_cluster_index;
int dim;
int max_iters;

/*
 * fill the data in samples from CSV file stream IN.
 */
void read_data() {
  double f;
  char c;
  int cur_vector = 0;
  int cur_element = 0;

  while (scanf("%lf%c", &f, &c) == 2) {
    if (cur_element < cluster_count)
      centroids[cur_element * dim + cur_vector] = f;
    samples[cur_element * dim + cur_vector] = f;
    switch (c) {
    case ',':
      cur_vector++;
      if (cur_vector > dim)
        goto err;
      break;
    case '\n':
      cur_element++;
      if (cur_element > sample_count)
        goto err;
      cur_vector = 0;
      break;
    default:
      goto err;
    }
  }
  if (cur_element != sample_count)
    goto err;
  return;
err:
  fprintf(stderr, "Invalid input\n");
  exit(EXIT_FAILURE);
}

double squared(double d) { return d * d; }

/*
 * Return the squared distance between v1 and v2, both assumed to be of length
 * dim.
 */
double squared_distance(double *v1, double *v2) {
  double res = 0;
  int i;
  for (i = 0; i < dim; i++) {
    res += squared(v1[i] - v2[i]);
  }
  return res;
}

/*
 * Return index of the closest centroid to samples[sample_i]
 */
int closest_centroid_index(int sample_i) {
  double min = squared_distance(&centroids[0], &samples[sample_i * dim]);
  int min_i = 0;
  double cur;
  int centroid_i;
  for (centroid_i = 1; centroid_i < cluster_count; centroid_i++) {
    cur = squared_distance(&centroids[centroid_i * dim],
                           &samples[sample_i * dim]);
    min_i = (cur < min ? centroid_i : min_i);
    min = (cur < min ? cur : min);
  }
  return min_i;
}

/*
 * sum += vec, by their components
 */
void vector_plus_equal(double *sum, double *vec) {
  int i;
  for (i = 0; i < dim; i++) {
    sum[i] += vec[i];
  }
}
/*
 * quotient /= denominator, by quotient's compopnents
 */
void vector_divide_equal(double *quotient, double denominator) {
  int i;
  for (i = 0; i < dim; i++) {
    quotient[i] /= denominator;
  }
}

/*
 * Update centroid index for sample at index i.
 * Return true if the index changed since the last time.
 */
int update_cluster_for_sample(int i) {
  int c = closest_centroid_index(i);
  int changed = (c != sample_cluster_index[i]);
  sample_cluster_index[i] = c;
  return changed;
}

/*
 * Update the cluster indecies of all samples.
 * Return true if any index changed.
 */
int mass_cluster_indices_update() {
  int i;
  int changed = 0;
  for (i = 0; i < sample_count; i++) {
    changed |= update_cluster_for_sample(i);
  }
  return changed;
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
  /* divide clusters by their size to get the average */
  for (i = 0; i < cluster_count; i++) {
    vector_divide_equal(&centroids[i * dim], cluster_size[i]);
  }
}

void print_vectors(double *vectors, int count) {
  double f;
  char c;
  int i;
  for (i = 0; i < dim * count; i++) {
    f = vectors[i];
    c = ((i + 1) % dim == 0) ? '\n' : ',';
    printf("%.2f%c", f, c);
  }
}

static PyObject *set_dim(PyObject *self, PyObject *args) {
  int dim_;
  if (!PyArg_ParseTuple(args, "i", &dim_))
    return NULL;
  assert(dim > 0);
  dim = dim_;
  Py_INCREF(Py_None);
  return Py_None;
}

static int parse_py_list_to_vecs(PyObject *listObj, double *vecs) {
  int vec_count = PyList_Size(listObj);
  for (int i = 0; i < vec_count; i++) {
    double *vec = &vecs[dim * i];
    PyObject *vec_list_obj = PyList_GetItem(listObj, i);
    if (!PyList_Check(vec_list_obj)) {
      fprintf(stderr, "not a vec level list\n");
      return 0;
    }
    int vec_len = PyList_Size(vec_list_obj);
    if (vec_len != dim) {
      fprintf(stderr, "vec length isn't dim\n");
      return 0;
    }
    for (int j = 0; j < dim; j++) {
      PyObject *e = PyList_GetItem(vec_list_obj, j);
      if (!PyFloat_Check(e)) {
        fprintf(stderr, "vec item is not a float\n");
        return 0;
      }
      vec[j] = PyFloat_AsDouble(e);
    }
  }
  return 1;
}

static PyObject *set_centroids(PyObject *self, PyObject *args) {
  PyObject *listObj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj))
    return NULL;
  cluster_count = PyList_Size(listObj);
  centroids = malloc(sizeof(*centroids) * dim * cluster_count);
  cluster_size = malloc(sizeof(*cluster_size) * cluster_count);
  parse_py_list_to_vecs(listObj, centroids);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *set_samples(PyObject *self, PyObject *args) {
  PyObject *listObj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj))
    return NULL;
  sample_count = PyList_Size(listObj);
  samples = malloc(sizeof(*samples) * dim * sample_count);
  sample_cluster_index = malloc(sizeof(*sample_cluster_index) * sample_count);
  parse_py_list_to_vecs(listObj, samples);
  Py_INCREF(Py_None);
  return Py_None;
}
static PyObject *iterate(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "i", &max_iters))
    return NULL;
  if (max_iters <= 0) {
    fprintf(stderr, "max_iters <= 0\n");
    return NULL;
  };
  mass_cluster_indices_update();
  mass_centroid_update();
  int i;
  for (i = 0; i < max_iters; i++) {
    if (!mass_cluster_indices_update()) /* no indecies changed */
      break;
    mass_centroid_update();
  }
  fprintf(stderr, "iters = %d\n", i);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *print_centroids(PyObject *self, PyObject *args) {
  print_vectors(centroids, cluster_count);
  Py_INCREF(Py_None);
  return Py_None;
}

/*  define functions in module */
static PyMethodDef methods[] = {
    {"set_dim", set_dim, METH_VARARGS, "initialize dim"},
    {"set_centroids", set_centroids, METH_VARARGS, "initialize centroids"},
    {"set_samples", set_samples, METH_VARARGS, "initialize samples"},
    {"iterate", iterate, METH_VARARGS,
     "Perform n iterations of centroid optimization"},
    {"print_centroids", print_centroids, METH_VARARGS, "print centroids"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef mod_def = {PyModuleDef_HEAD_INIT, "mykmeanssp",
                                     "Some docs", -1, methods};

PyMODINIT_FUNC PyInit_mykmeanssp(void) { return PyModule_Create(&mod_def); }

int main(int argc, char *argv[]) {
  assert(argc == 5);

  assert(cluster_count < sample_count);
  /* printf("cluster_count: %d, sample_count:", ); */
  /* We might be able to chain all of these into one big malloc. Not sure its
   * worth it though. Standards after ANSI C just allow variable-sized arrays in
   * stack. */

  if (!(centroids || cluster_size || samples || sample_cluster_index)) {
    fprintf(stderr, "allocation failure\n");
    exit(EXIT_FAILURE);
  }
  read_data();
  /*
   * the first centroids aren't purely dependent on cluster indecies
   * so always do at least one iteration
   */
  free(centroids);
  free(cluster_size);
  free(samples);
  free(sample_cluster_index);

  return 0;
}
