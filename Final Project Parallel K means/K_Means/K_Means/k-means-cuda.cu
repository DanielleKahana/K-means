
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "k-means.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


__device__ double calculate_distance_with_cuda(Position* p1, Position* p2)
{
	double calc_x, calc_y, calc_z, x0, y0, z0, x1, y1, z1;
	x0 = p1->x;
	y0 = p1->y;
	z0 = p1->z;

	x1 = p2->x;
	y1 = p2->y;
	z1 = p2->z;

	calc_x = pow(x1 - x0, 2);
	calc_y = pow(y1 - y0, 2);
	calc_z = pow(z1 - z0, 2);

	return sqrt(calc_x + calc_y + calc_z);

}

__global__ void update_position_by_time_gpu(Point* points, int points_size, double time)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < points_size)
	{
		points[index].position.x = points[index].init_position.x + (time*points[index].velocity.vx);
		points[index].position.y = points[index].init_position.y + (time*points[index].velocity.vy);
		points[index].position.z = points[index].init_position.z + (time*points[index].velocity.vz);
	}
}

__global__ void group_points_kernel(Point* points,int points_size, Cluster* clusters, int clusters_size, int* has_changed)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int i, new_cluster_id = 0; 
	double min_distance = DBL_MAX;
	double distance;

	if (index < points_size)
	{
		for (i = 0; i < clusters_size; i++)
		{
			distance = calculate_distance_with_cuda(&(points[index].position), &(clusters[i].centroid));
			if (distance < min_distance)
			{
				min_distance = distance;
				new_cluster_id = clusters[i].id;
			}
		}

		if (points[index].cluster_id != new_cluster_id)
		{
			*has_changed = 1;
			points[index].cluster_id = new_cluster_id;
		}
	}
}


void group_points_with_cuda(int* has_changed, Point* set_of_points, Cluster* k_clusters, int set_of_points_size, int k_clusters_size)
{
	cudaError_t cudaStatus;
	int num_of_blocks, num_of_threads;
	Point* dev_points;
	Cluster* dev_clusters;
	int* dev_has_changed;
	cudaDeviceProp device_prop;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//get the device properties
	cudaGetDeviceProperties(&device_prop, 0);

	
	num_of_threads = device_prop.maxThreadsPerBlock;

	// Formula was taken from CUDA tutorial - https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf, page 44
	num_of_blocks = (set_of_points_size + num_of_threads - 1) / num_of_threads;

	// Allocate GPU buffers for points array
	cudaStatus = cudaMalloc((void**)&dev_points, set_of_points_size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		cuda_error(dev_points);
	}


	cudaStatus = cudaMalloc((void**)&dev_clusters, k_clusters_size * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		cuda_error(dev_clusters);
	}


		cudaStatus = cudaMalloc((void**)&dev_has_changed, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
			cuda_error(dev_has_changed);
		}
	


	// Copy input array from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, set_of_points, set_of_points_size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#1 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	cudaStatus = cudaMemcpy(dev_clusters, k_clusters, k_clusters_size * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#1 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	cudaStatus = cudaMemcpy(dev_has_changed, has_changed, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#1 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	// Launch a kernel on the GPU with one thread for each element.
	group_points_kernel<<< num_of_blocks , num_of_threads >>>(dev_points, set_of_points_size, dev_clusters, k_clusters_size, dev_has_changed);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "group_points_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cuda_error(dev_points);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching group_points_kernel!\n", cudaStatus);
		cuda_error(dev_points);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(set_of_points, dev_points, set_of_points_size * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#2 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	cudaStatus = cudaMemcpy(has_changed, dev_has_changed, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#2 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	free_cuda_memory(dev_points, dev_clusters, dev_has_changed);

}



void set_position_by_time_with_cuda(Point* set_of_points, int set_of_points_size, double time)
{
	cudaError_t cudaStatus;
	int num_of_blocks, num_of_threads;
	Point* dev_points;
	cudaDeviceProp device_prop;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//get the device properties
	cudaGetDeviceProperties(&device_prop, 0);


	num_of_threads = device_prop.maxThreadsPerBlock;

	// Formula was taken from CUDA tutorial - https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf, page 44
	num_of_blocks = (set_of_points_size + num_of_threads - 1) / num_of_threads;


	// Allocate GPU buffers for points array
	cudaStatus = cudaMalloc((void**)&dev_points, set_of_points_size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		cuda_error(dev_points);
	}

	// Copy input array from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, set_of_points, set_of_points_size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#1 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}

	// Launch a kernel on the GPU with one thread for each element.
	update_position_by_time_gpu << < num_of_blocks, num_of_threads >> >(dev_points, set_of_points_size, time);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "group_points_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cuda_error(dev_points);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching group_points_kernel!\n", cudaStatus);
		cuda_error(dev_points);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(set_of_points, dev_points, set_of_points_size * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "#2 cudaMemcpy failed!\n");
		cuda_error(dev_points);
	}
}


void free_cuda_memory(Point* dev_points, Cluster* dev_clusters, int* dev_has_changed)
{
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	cudaFree(dev_has_changed);
}


void cuda_error(void* array) 
{
	cudaFree(array);
}
