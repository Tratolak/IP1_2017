#include <iostream>

#include "main_kernel.cuh"
#include "Adders.cuh"

#define KER_CLEAN()\
cudaFree(dev_comps);\
cudaFree(dev_in);\
cudaFree(dev_out);\

cudaError_t evaluateComponents(int *components, int *inputs, int *results, const int numofcomps)
{
	int *dev_comps = 0; //device components
	int *dev_in = 0;	//device inputs
	int *dev_out = 0;	//device outputs  
	cudaError_t cudaStatus;
	
	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaSetDevice failed";
	}

	// Allocate GPU buffers.
	cudaStatus = cudaMalloc((void**)&dev_comps, numofcomps * sizeof(int));
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaMalloc failed";
		KER_CLEAN();
	}

	cudaStatus = cudaMalloc((void**)&dev_in, SAMPLE_RESOL * 2 * numofcomps * sizeof(int));
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaMalloc failed";
		KER_CLEAN();
	}

	cudaStatus = cudaMalloc((void**)&dev_out, SAMPLE_RESOL * numofcomps * sizeof(int));
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaMalloc failed";
		KER_CLEAN();
	}

	// Copy inputs from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_comps, components, numofcomps * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaMemcpy failed";
		KER_CLEAN();
	}

	cudaStatus = cudaMemcpy(dev_in, inputs, SAMPLE_RESOL * 2 * numofcomps * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "evalKernel: cudaMemcpy failed";
		KER_CLEAN();
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 Grid(2, 2, numofcomps);
	dim3 Block(32, 32, 1);

	// Launch a kernel on the GPU.
	cudaEventRecord(start);
	evalKernel << <Grid, Block >> >(dev_comps, dev_in, dev_out);
	cudaEventRecord(stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: launch failed" << cudaGetErrorString(cudaStatus) << std::endl;
		KER_CLEAN();
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaDeviceSynchronize returned error code " << cudaStatus;
		KER_CLEAN();
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_out, SAMPLE_RESOL * numofcomps * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		std::cerr << "evalKernel: cudaMemcpy failed" << cudaStatus;
		KER_CLEAN();
	}

	//Print execution time
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	printf("Execution time: %f ms\n", miliseconds);

	/*
	unsigned int total_dev = 0;
	for (int i = 0; i < 472; ++i)
	{
	total_dev = 0;
	for(int ii = 0; ii < (1 << 16); ++ii)
	{
	total_dev += abs(results[i*(1 << 16) + ii] - (ii % 256 + ii / 256));
	}
	printf("Adder %d average deviation: %f\n", i, float(total_dev)/float(1<<16));
	}*/

	
	// Free all alocated memory
	KER_CLEAN();

	return cudaStatus;
}

__global__ void evalKernel(const int *comps, int *inputs, int *results)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int ii = threadIdx.y + blockIdx.y*blockDim.y;

	// index for 1st input = offset for component + position 
	// index for 2nd input = offset for component + position + SAMPLE_RESOL
	int in1 = inputs[(blockIdx.z * 2 * SAMPLE_RESOL) + ii*INPUT_HEIGHT + i];
	int in2 = inputs[(blockIdx.z * 2 * SAMPLE_RESOL) + ii*INPUT_HEIGHT + i + SAMPLE_RESOL];

	AS_DATA_TYPE i0 = (in1 & 0x01);
	AS_DATA_TYPE i1 = (in1 >> 1) & 0x01;
	AS_DATA_TYPE i2 = (in1 >> 2) & 0x01;
	AS_DATA_TYPE i3 = (in1 >> 3) & 0x01;
	AS_DATA_TYPE i4 = (in1 >> 4) & 0x01;
	AS_DATA_TYPE i5 = (in1 >> 5) & 0x01;
	AS_DATA_TYPE i6 = (in1 >> 6) & 0x01;
	AS_DATA_TYPE i7 = (in1 >> 7) & 0x01;
	AS_DATA_TYPE i8 = (in2 & 0x01);
	AS_DATA_TYPE i9 = (in2 >> 1) & 0x01;
	AS_DATA_TYPE i10 = (in2 >> 2) & 0x01;
	AS_DATA_TYPE i11 = (in2 >> 3) & 0x01;
	AS_DATA_TYPE i12 = (in2 >> 4) & 0x01;
	AS_DATA_TYPE i13 = (in2 >> 5) & 0x01;
	AS_DATA_TYPE i14 = (in2 >> 6) & 0x01;
	AS_DATA_TYPE i15 = (in2 >> 7) & 0x01;

	AS_DATA_TYPE o0;
	AS_DATA_TYPE o1;
	AS_DATA_TYPE o2;
	AS_DATA_TYPE o3;
	AS_DATA_TYPE o4;
	AS_DATA_TYPE o5;
	AS_DATA_TYPE o6;
	AS_DATA_TYPE o7;
	AS_DATA_TYPE o8;

	switch (comps[blockIdx.z])
	{
	case 0:
		add1(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 1:
		add2(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 2:
		add3(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 3:
		add4(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 4:
		add5(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 5:
		add6(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 6:
		add7(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 7:
		add8(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 8:
		add9(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 9:
		add10(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 10:
		add11(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 11:
		add12(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 12:
		add13(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 13:
		add14(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 14:
		add15(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 15:
		add16(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 16:
		add17(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 17:
		add18(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 18:
		add19(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 19:
		add20(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 20:
		add21(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 21:
		add22(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 22:
		add23(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 23:
		add24(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 24:
		add25(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 25:
		add26(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 26:
		add27(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 27:
		add28(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 28:
		add29(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 29:
		add30(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 30:
		add31(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 31:
		add32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 32:
		add33(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 33:
		add34(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 34:
		add35(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 35:
		add36(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 36:
		add37(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 37:
		add38(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 38:
		add39(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 39:
		add40(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 40:
		add41(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 41:
		add42(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 42:
		add43(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 43:
		add44(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 44:
		add45(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 45:
		add46(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 46:
		add47(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 47:
		add48(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 48:
		add49(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 49:
		add50(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 50:
		add51(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 51:
		add52(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 52:
		add53(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 53:
		add54(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 54:
		add55(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 55:
		add56(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 56:
		add57(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 57:
		add58(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 58:
		add59(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 59:
		add60(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 60:
		add61(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 61:
		add62(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 62:
		add63(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 63:
		add64(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 64:
		add65(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 65:
		add66(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 66:
		add67(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 67:
		add68(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 68:
		add69(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 69:
		add70(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 70:
		add71(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 71:
		add72(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 72:
		add73(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 73:
		add74(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 74:
		add75(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 75:
		add76(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 76:
		add77(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 77:
		add78(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 78:
		add79(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 79:
		add80(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 80:
		add81(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 81:
		add82(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 82:
		add83(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 83:
		add84(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 84:
		add85(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 85:
		add86(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 86:
		add87(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 87:
		add88(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 88:
		add89(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 89:
		add90(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 90:
		add91(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 91:
		add92(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 92:
		add93(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 93:
		add94(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 94:
		add95(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 95:
		add96(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 96:
		add97(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 97:
		add98(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 98:
		add99(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 99:
		add100(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 100:
		add101(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 101:
		add102(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 102:
		add103(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 103:
		add104(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 104:
		add105(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 105:
		add106(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 106:
		add107(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 107:
		add108(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 108:
		add109(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 109:
		add110(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 110:
		add111(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 111:
		add112(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 112:
		add113(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 113:
		add114(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 114:
		add115(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 115:
		add116(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 116:
		add117(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 117:
		add118(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 118:
		add119(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 119:
		add120(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 120:
		add121(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 121:
		add122(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 122:
		add123(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 123:
		add124(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 124:
		add125(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 125:
		add126(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 126:
		add127(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 127:
		add128(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 128:
		add129(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 129:
		add130(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 130:
		add131(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 131:
		add132(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 132:
		add133(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 133:
		add134(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 134:
		add135(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 135:
		add136(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 136:
		add137(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 137:
		add138(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 138:
		add139(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 139:
		add140(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 140:
		add141(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 141:
		add142(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 142:
		add143(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 143:
		add144(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 144:
		add145(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 145:
		add146(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 146:
		add147(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 147:
		add148(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 148:
		add149(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 149:
		add150(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 150:
		add151(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 151:
		add152(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 152:
		add153(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 153:
		add154(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 154:
		add155(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 155:
		add156(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 156:
		add157(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 157:
		add158(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 158:
		add159(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 159:
		add160(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 160:
		add161(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 161:
		add162(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 162:
		add163(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 163:
		add164(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 164:
		add165(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 165:
		add166(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 166:
		add167(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 167:
		add168(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 168:
		add169(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 169:
		add170(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 170:
		add171(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 171:
		add172(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 172:
		add173(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 173:
		add174(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 174:
		add175(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 175:
		add176(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 176:
		add177(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 177:
		add178(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 178:
		add179(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 179:
		add180(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 180:
		add181(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 181:
		add182(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 182:
		add183(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 183:
		add184(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 184:
		add185(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 185:
		add186(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 186:
		add187(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 187:
		add188(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 188:
		add189(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 189:
		add190(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 190:
		add191(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 191:
		add192(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 192:
		add193(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 193:
		add194(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 194:
		add195(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 195:
		add196(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 196:
		add197(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 197:
		add198(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 198:
		add199(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 199:
		add200(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 200:
		add201(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 201:
		add202(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 202:
		add203(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 203:
		add204(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 204:
		add205(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 205:
		add206(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 206:
		add207(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 207:
		add208(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 208:
		add209(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 209:
		add210(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 210:
		add211(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 211:
		add212(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 212:
		add213(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 213:
		add214(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 214:
		add215(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 215:
		add216(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 216:
		add217(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 217:
		add218(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 218:
		add219(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 219:
		add220(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 220:
		add221(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 221:
		add222(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 222:
		add223(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 223:
		add224(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 224:
		add225(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 225:
		add226(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 226:
		add227(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 227:
		add228(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 228:
		add229(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 229:
		add230(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 230:
		add231(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 231:
		add232(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 232:
		add233(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 233:
		add234(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 234:
		add235(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 235:
		add236(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 236:
		add237(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 237:
		add238(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 238:
		add239(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 239:
		add240(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 240:
		add241(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 241:
		add242(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 242:
		add243(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 243:
		add244(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 244:
		add245(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 245:
		add246(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 246:
		add247(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 247:
		add248(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 248:
		add249(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 249:
		add250(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 250:
		add251(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 251:
		add252(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 252:
		add253(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 253:
		add254(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 254:
		add255(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 255:
		add256(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 256:
		add257(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 257:
		add258(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 258:
		add259(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 259:
		add260(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 260:
		add261(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 261:
		add262(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 262:
		add263(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 263:
		add264(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 264:
		add265(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 265:
		add266(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 266:
		add267(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 267:
		add268(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 268:
		add269(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 269:
		add270(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 270:
		add271(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 271:
		add272(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 272:
		add273(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 273:
		add274(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 274:
		add275(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 275:
		add276(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 276:
		add277(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 277:
		add278(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 278:
		add279(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 279:
		add280(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 280:
		add281(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 281:
		add282(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 282:
		add283(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 283:
		add284(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 284:
		add285(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 285:
		add286(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 286:
		add287(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 287:
		add288(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 288:
		add289(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 289:
		add290(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 290:
		add291(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 291:
		add292(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 292:
		add293(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 293:
		add294(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 294:
		add295(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 295:
		add296(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 296:
		add297(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 297:
		add298(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 298:
		add299(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 299:
		add300(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 300:
		add301(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 301:
		add302(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 302:
		add303(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 303:
		add304(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 304:
		add305(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 305:
		add306(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 306:
		add307(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 307:
		add308(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 308:
		add309(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 309:
		add310(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 310:
		add311(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 311:
		add312(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 312:
		add313(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 313:
		add314(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 314:
		add315(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 315:
		add316(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 316:
		add317(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 317:
		add318(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 318:
		add319(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 319:
		add320(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 320:
		add321(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 321:
		add322(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 322:
		add323(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 323:
		add324(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 324:
		add325(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 325:
		add326(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 326:
		add327(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 327:
		add328(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 328:
		add329(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 329:
		add330(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 330:
		add331(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 331:
		add332(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 332:
		add333(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 333:
		add334(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 334:
		add335(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 335:
		add336(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 336:
		add337(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 337:
		add338(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 338:
		add339(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 339:
		add340(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 340:
		add341(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 341:
		add342(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 342:
		add343(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 343:
		add344(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 344:
		add345(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 345:
		add346(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 346:
		add347(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 347:
		add348(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 348:
		add349(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 349:
		add350(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 350:
		add351(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 351:
		add352(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 352:
		add353(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 353:
		add354(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 354:
		add355(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 355:
		add356(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 356:
		add357(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 357:
		add358(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 358:
		add359(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 359:
		add360(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 360:
		add361(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 361:
		add362(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 362:
		add363(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 363:
		add364(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 364:
		add365(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 365:
		add366(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 366:
		add367(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 367:
		add368(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 368:
		add369(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 369:
		add370(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 370:
		add371(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 371:
		add372(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 372:
		add373(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 373:
		add374(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 374:
		add375(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 375:
		add376(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 376:
		add377(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 377:
		add378(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 378:
		add379(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 379:
		add380(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 380:
		add381(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 381:
		add382(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 382:
		add383(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 383:
		add384(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 384:
		add385(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 385:
		add386(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 386:
		add387(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 387:
		add388(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 388:
		add389(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 389:
		add390(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 390:
		add391(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 391:
		add392(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 392:
		add393(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 393:
		add394(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 394:
		add395(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 395:
		add396(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 396:
		add397(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 397:
		add398(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 398:
		add399(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 399:
		add400(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 400:
		add401(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 401:
		add402(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 402:
		add403(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 403:
		add404(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 404:
		add405(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 405:
		add406(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 406:
		add407(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 407:
		add408(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 408:
		add409(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 409:
		add410(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 410:
		add411(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 411:
		add412(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 412:
		add413(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 413:
		add414(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 414:
		add415(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 415:
		add416(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 416:
		add417(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 417:
		add418(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 418:
		add419(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 419:
		add420(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 420:
		add421(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 421:
		add422(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 422:
		add423(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 423:
		add424(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 424:
		add425(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 425:
		add426(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 426:
		add427(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 427:
		add428(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 428:
		add429(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 429:
		add430(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 430:
		add431(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 431:
		add432(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 432:
		add433(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 433:
		add434(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 434:
		add435(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 435:
		add436(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 436:
		add437(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 437:
		add438(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 438:
		add439(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 439:
		add440(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 440:
		add441(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 441:
		add442(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 442:
		add443(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 443:
		add444(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 444:
		add445(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 445:
		add446(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 446:
		add447(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 447:
		add448(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 448:
		add449(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 449:
		add450(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 450:
		add451(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 451:
		add452(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 452:
		add453(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 453:
		add454(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 454:
		add455(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 455:
		add456(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 456:
		add457(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 457:
		add458(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 458:
		add459(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 459:
		add460(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 460:
		add461(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 461:
		add462(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 462:
		add463(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 463:
		add464(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 464:
		add465(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 465:
		add466(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 466:
		add467(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 467:
		add468(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 468:
		add469(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 469:
		add470(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 470:
		add471(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 471:
		add472(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	case 472:
		add473(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8);
		break;
	}

	int out_index = (blockIdx.z * SAMPLE_RESOL) + ii*INPUT_HEIGHT + i;

	results[out_index] |= o0 & 0x01;
	results[out_index] |= (o1 & 0x01) << 1;
	results[out_index] |= (o2 & 0x01) << 2;
	results[out_index] |= (o3 & 0x01) << 3;
	results[out_index] |= (o4 & 0x01) << 4;
	results[out_index] |= (o5 & 0x01) << 5;
	results[out_index] |= (o6 & 0x01) << 6;
	results[out_index] |= (o7 & 0x01) << 7;
	results[out_index] |= (o8 & 0x01) << 8;
}