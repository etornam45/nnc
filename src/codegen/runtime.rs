pub fn helper_init_tensor() -> &'static str {
r###"
void init_tensor(Tensor* tensor, int ndim, const int* shape_values) {
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) {
        exit(EXIT_FAILURE);
    }
    memcpy(tensor->shape, shape_values, ndim * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape_values[i];
    }

    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    if (tensor->data == NULL) {
        free(tensor->shape);
        exit(EXIT_FAILURE);
    }
}"###
}

pub fn helper_free_tensor() -> &'static str {
    r###"
void free_tensor(Tensor* tensor) {
    if (tensor->data != NULL) {
        free(tensor->data);
        tensor->data = NULL;
    }
    if (tensor->shape != NULL) {
        free(tensor->shape);
        tensor->shape = NULL;
    }
}"###
}

pub fn helper_reshape_tensor() -> &'static str {
    r###"
void reshape_tensor(Tensor* tensor, int ndim, const int* shape_values) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape_values[i];
    }

    if (tensor->data != NULL && tensor->size == size) {
        if (tensor->ndim != ndim) {
            free(tensor->shape);
            tensor->shape = (int*)malloc(ndim * sizeof(int));
            tensor->ndim = ndim;
        }
        memcpy(tensor->shape, shape_values, ndim * sizeof(int));
        return;
    }

    if (tensor->data != NULL) {
        free(tensor->data);
    }
    if (tensor->shape != NULL) {
        free(tensor->shape);
    }

    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) { exit(EXIT_FAILURE); }
    memcpy(tensor->shape, shape_values, ndim * sizeof(int));

    tensor->size = size;
    tensor->data = (float*)malloc(size * sizeof(float));
    if (tensor->data == NULL) { free(tensor->shape); exit(EXIT_FAILURE); }
}"###
}
