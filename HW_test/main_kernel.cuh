#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define AS_DATA_TYPE unsigned short
#define SAMPLE_RESOL 4096
#define INPUT_WIDTH 64
#define INPUT_HEIGHT 64

cudaError_t evaluateComponents(int *components, int *inputs, int *results, const int comp);
__global__ void evalKernel(const int *comps, int *inputs, int *results);