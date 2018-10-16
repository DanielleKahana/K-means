#include "k-means.h"
#pragma warning(disable: 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include <mpi.h>



int main(int argc, char *argv[])
{
	int myid, numprocs;
	double start_time, end_time;
	int errorCode = MPI_ERR_COMM;

	//MPI init
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (numprocs == 1)
		MPI_Abort(MPI_COMM_WORLD, errorCode);


	//Start time
	start_time = MPI_Wtime();

	k_means_algo(myid, numprocs);

	//end time here
	end_time = MPI_Wtime();



	MPI_Barrier(MPI_COMM_WORLD);
	//printf("\n\nprocess #%d: finished!\n", myid);
	if (myid == MASTER)
	{
		printf("time = %lf", end_time - start_time);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;


}

void k_means_algo(int myid, int numprocs)
{
	File_info file_info;
	MPI_Status status;
	Point *set_of_points = NULL, *process_points = NULL;
	Cluster* k_clusters = NULL;
	MPI_Datatype File_info_MPI_type, Cluster_MPI_type, Velocity_MPI_type,
		Position_MPI_type, Point_MPI_type, Summary_MPI_type;
	int process_points_size = 0, finished = 0;
	int* all_clusters_points;
	double q, t;

	create_MPI_types(&File_info_MPI_type, &Cluster_MPI_type, &Velocity_MPI_type,
		&Point_MPI_type, &Position_MPI_type, &Summary_MPI_type);

	if (myid == MASTER)
	{
		//reads data from file
		set_of_points = read_from_file(&file_info);

		//initial the k cluster with the first k points
		k_clusters = init_k_clusters(set_of_points, file_info.K);
	}

	//Broadcast file information
	MPI_Bcast(&file_info, 1, File_info_MPI_type, MASTER, MPI_COMM_WORLD);

	all_clusters_points = (int*)malloc(sizeof(int)*file_info.K);

	if (myid != MASTER)
	{
		k_clusters = (Cluster*)malloc(sizeof(Cluster)*(file_info.K));
	}

	//Broadcast the first K clusters	
	MPI_Bcast(k_clusters, file_info.K, Cluster_MPI_type, MASTER, MPI_COMM_WORLD);

	//Scatter the points equaly between the N processes
	send_points_to_processes(&file_info, numprocs, myid, set_of_points, &process_points, &process_points_size, &Point_MPI_type, &status);

	for (t = 0; t < file_info.T && !finished; t += file_info.dT)
	{
		set_position_by_time(process_points, process_points_size, t);

		classify_points(k_clusters, &file_info, process_points, process_points_size);

		gather_all_points(process_points, process_points_size, k_clusters, set_of_points, numprocs, myid, &Point_MPI_type, &file_info, all_clusters_points);

		if (myid == MASTER)
		{
			q = evaluate_quality(k_clusters, file_info.K, set_of_points, file_info.N, all_clusters_points);

			if (q < file_info.QM)
				finished = 1;
		}
		MPI_Bcast(&finished, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	}

	if (myid == MASTER)
	{
		write_to_file(&file_info, k_clusters, t, q);
	}
}

Point* read_from_file(File_info* file_info)
{
	Point* set_of_points = NULL;
	FILE* f;
	int i;
	Position position = { 0,0,0 };
	Velocity velocity = { 0,0,0 };


	f = fopen(INPUT_FILE_NAME, "r");
	//char* name = "C:\\Users\\idan\\Desktop\\DanielleKahana_Motherfucker\\DanielleKahana_Motherfucker\\Danielle_KMeans\\Danielle_KMeans\\input.txt";
	//f = fopen(name, "r");


	if (f == NULL)
	{
		printf("Failed opening file. Exiting!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	fscanf(f, "%d %d %d %lf %lf %lf", &(file_info->N), &(file_info->K), &(file_info->T), &(file_info->dT), &(file_info->LIMIT), &(file_info->QM));

	//printf("N = %d, K = %d, T = %d, dT = %lf, LIMIT = %lf, QM = %lf\n", file_info->N, file_info->K, file_info->T, file_info->dT, file_info->LIMIT, file_info->QM);

	set_of_points = (Point*)malloc(sizeof(Point)*(file_info->N));

#pragma omp parallel for
	for (i = 0; i < file_info->N; i++)
	{
		fscanf(f, "%lf %lf %lf %lf %lf %lf", &position.x, &position.y, &position.z,
			&velocity.vx, &velocity.vy, &velocity.vz);

		set_of_points[i].init_position = position;
		set_of_points[i].position = position;
		set_of_points[i].velocity = velocity;
		set_of_points[i].cluster_id = -1;

	}
	return set_of_points;
}

void print_array(Point* array, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("%d - [%lf, %lf, %lf] , [%lf, %lf, %lf]\n", array[i].cluster_id, array[i].position.x, array[i].position.y, array[i].position.z, array[i].velocity.vx, array[i].velocity.vy, array[i].velocity.vz);
	}
}

void send_points_to_processes(File_info* file_info, int numprocs, int myid, Point* set_of_points, Point** process_points, int* process_points_size, MPI_Datatype* Point_MPI_type, MPI_Status* status)
{
	int chunk_size, remainder, i, offset;

	chunk_size = file_info->N / numprocs;
	remainder = file_info->N % numprocs;
	offset = chunk_size + remainder; //the first chunk goes to master

	if (myid == MASTER)
	{
		*process_points_size = offset;
		*process_points = (Point*)malloc(sizeof(Point)*(*process_points_size));
		memcpy(*process_points, set_of_points, offset * sizeof(Point));
		//printf("Process #%d: \n", myid);
		//print_array(*process_points, offset);

#pragma omp parallel for
		for (i = 1; i < numprocs; i++)
		{
			MPI_Send(set_of_points + offset, chunk_size, *Point_MPI_type, i, 0, MPI_COMM_WORLD);
			offset += chunk_size;
		}

	}
	else
	{
		*process_points_size = chunk_size;
		*process_points = (Point*)malloc(sizeof(Point)*chunk_size);
		MPI_Recv(*process_points, chunk_size, *Point_MPI_type, MASTER, 0, MPI_COMM_WORLD, status);
		//printf("Process #%d: \n", myid);
		//print_array(*process_points, chunk_size);
	}

}

double calculate_distance(Position p1, Position p2)
{
	double calc_x, calc_y, calc_z, x0, y0, z0, x1, y1, z1;
	x0 = p1.x;
	y0 = p1.y;
	z0 = p1.z;

	x1 = p2.x;
	y1 = p2.y;
	z1 = p2.z;

	calc_x = pow(x1 - x0, 2);
	calc_y = pow(y1 - y0, 2);
	calc_z = pow(z1 - z0, 2);

	return sqrt(calc_x + calc_y + calc_z);

}

Cluster* init_k_clusters(Point* set_of_points, int K)
{
	Cluster* k_clusters = (Cluster*)malloc(sizeof(Cluster)*K);
	int i;

	for (i = 0; i < K; i++)
	{
		k_clusters[i].centroid = set_of_points[i].position;
		k_clusters[i].number_of_points = 0;
		k_clusters[i].diameter = 0;
		k_clusters[i].id = i;
		k_clusters[i].sum.sum_x = 0;
		k_clusters[i].sum.sum_y = 0;
		k_clusters[i].sum.sum_z = 0;
	}

	return k_clusters;
}

void group_points_with_cuda(int* has_changed, Point* set_of_points, Cluster* k_clusters, int set_of_points_size, int k_clusters_size)
{
	//later
}

void group_points(int* has_changed, Point* set_of_points, Cluster* k_clusters, int set_of_points_size, int k_clusters_size)
{
	int i, j, cluster_id;
	double min_distance = DBL_MAX;
	double distance;
	//int has_changed = 0;
	int old_cluster_id;

	for (i = 0; i < set_of_points_size; i++)
	{
		for (j = 0; j < k_clusters_size; j++)
		{
			old_cluster_id = set_of_points[i].cluster_id;
			distance = calculate_distance(set_of_points[i].position, k_clusters[j].centroid);
			if (distance < min_distance)
			{
				min_distance = distance;
				cluster_id = k_clusters[j].id;
			}
		}

		if (set_of_points[i].cluster_id != cluster_id)
		{
			*has_changed = 1;

			if (set_of_points[i].cluster_id != -1)
			{
				k_clusters[old_cluster_id].number_of_points--;
				remove_point_from_sum(&k_clusters[old_cluster_id], set_of_points[i].position);
			}
			set_of_points[i].cluster_id = cluster_id;
			k_clusters[cluster_id].number_of_points++;
			add_point_to_sum(&k_clusters[cluster_id], set_of_points[i].position);
		}

		min_distance = DBL_MAX;
	}
}

void add_point_to_sum(Cluster* cluster, Position position)
{
	cluster->sum.sum_x += position.x;
	cluster->sum.sum_y += position.y;
	cluster->sum.sum_z += position.z;
}

void remove_point_from_sum(Cluster* old_cluster, Position position)
{
	old_cluster->sum.sum_x -= position.x;
	old_cluster->sum.sum_y -= position.y;
	old_cluster->sum.sum_z -= position.z;
}


void classify_points(Cluster* k_clusters, File_info* file_info, Point* set_of_points, int set_of_points_size)
{
	int  iter;
	int has_changed = 0, total_has_changed = 1;
	double q = 0;

	total_has_changed = 1;
	for (iter = 0; iter < file_info->LIMIT && total_has_changed > 0; iter++)
	{
		has_changed = 0;
		//group_points_with_cuda(&has_changed, set_of_points, k_clusters, file_info->N, file_info->K);
		group_points(&has_changed, set_of_points, k_clusters, set_of_points_size, file_info->K);
		recalculate_centroids(k_clusters, file_info->K);
		check_point_has_changed(&has_changed, &total_has_changed);
	}
}

void write_to_file(File_info* file_info, Cluster* k_clusters, double time, double quality)
{
	FILE* f;
	int i;

	//f = fopen(OUTPUT_FILE_NAME, "w");
	char* name = "C:\\Users\\idan\\Desktop\\DanielleKahana_Motherfucker\\DanielleKahana_Motherfucker\\Danielle_KMeans\\Danielle_KMeans\\output.txt";
	f = fopen(name, "w");

	if (f == NULL)
	{
		printf("Failed writing to file. Exiting!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fprintf(f, "First occurrence t = %lf  with q = %lf\n", time, quality);
	fprintf(f, "Centers of the clusters:\n");
	for (i = 0; i < file_info->K; i++)
	{
		fprintf(f, "[%lf, %lf, %lf]\n", k_clusters[i].centroid.x, k_clusters[i].centroid.y, k_clusters[i].centroid.z);
	}
	fclose(f);
}

void gather_all_points(Point* set_of_points, int set_of_points_size, Cluster* k_cluster, Point* all_points, int numprocs, int myid, MPI_Datatype* Point_MPI_type, File_info* file_info, int* all_cluster_points)
{
	int chunk_size, i, offset;
	MPI_Status status;

	if (myid == MASTER)
	{

		offset = set_of_points_size;
		chunk_size = file_info->N / numprocs;
		memcpy(all_points, set_of_points, set_of_points_size * sizeof(Point)); //copy master's points into the all points array

		for (i = 0; i < numprocs - 1; i++)
		{
			MPI_Recv(all_points + offset, chunk_size, *Point_MPI_type, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			offset += chunk_size;
		}

	}
	else
	{
		MPI_Send(set_of_points, set_of_points_size, *Point_MPI_type, MASTER, 0, MPI_COMM_WORLD);
	}

	for (i = 0; i < file_info->K; i++)
	{
		MPI_Reduce(&k_cluster[i].number_of_points, &all_cluster_points[i], 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
	}
}


void check_point_has_changed(int* has_changed, int* total_has_changed)
{
	MPI_Allreduce(has_changed, total_has_changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void recalculate_centroids(Cluster* k_clusters, int k_clusters_size)
{
	int num_of_points = 0, i;
	double sum_x = 0, sum_y = 0, sum_z = 0;

	//#pragma omp parallel for
	for (i = 0; i < k_clusters_size; i++)
	{
		//#pragma omp parallel reduction (+: num_of_points) 
		{
			MPI_Allreduce(&k_clusters[i].sum.sum_x, &sum_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&k_clusters[i].sum.sum_y, &sum_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&k_clusters[i].sum.sum_z, &sum_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			MPI_Allreduce(&k_clusters[i].number_of_points, &num_of_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			if (num_of_points != 0)
			{
				k_clusters[i].centroid.x = sum_x / num_of_points;
				k_clusters[i].centroid.y = sum_y / num_of_points;
				k_clusters[i].centroid.z = sum_z / num_of_points;
			}
		}
	}
}

void recalculate_centroids_serial(Cluster* k_clusters, int k_clusters_size, Point* set_of_points, int set_of_points_size)
{
	int i, j;
	int number_of_points;
	double sum_x = 0, sum_y = 0, sum_z = 0;

	for (i = 0; i < k_clusters_size; i++)
	{
		for (j = 0; j < set_of_points_size; j++)
		{
			if (k_clusters[i].id == set_of_points[j].cluster_id)
			{
				//all clusters point		
				sum_x += set_of_points[j].position.x;
				sum_y += set_of_points[j].position.y;
				sum_z += set_of_points[j].position.z;
			}
		}
		number_of_points = k_clusters[i].number_of_points;

		k_clusters[i].centroid.x = sum_x / number_of_points;
		k_clusters[i].centroid.y = sum_y / number_of_points;
		k_clusters[i].centroid.z = sum_z / number_of_points;

		sum_x = sum_y = sum_z = 0;
	}
}


void set_position_by_time(Point* set_of_points, int set_of_points_size, double time)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < set_of_points_size; i++)
	{
		set_of_points[i].position.x = set_of_points[i].init_position.x + (time*set_of_points[i].velocity.vx);
		set_of_points[i].position.y = set_of_points[i].init_position.y + (time*set_of_points[i].velocity.vy);
		set_of_points[i].position.z = set_of_points[i].init_position.z + (time*set_of_points[i].velocity.vz);
	}
}

double evaluate_quality(Cluster* k_cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size, int* all_clusters_points)
{
	int i, j, count = 0;
	double q = 0, distance;


	qsort(set_of_points, set_of_points_size, sizeof(Point), compare_points_by_cluster_id);

	calculate_diameter(k_cluster, k_cluster_size, set_of_points, set_of_points_size, all_clusters_points);

	for (i = 0; i < k_cluster_size; i++)
	{
		for (j = 0; j < k_cluster_size; j++)
		{
			if (i != j)
			{
				distance = calculate_distance(k_cluster[i].centroid, k_cluster[j].centroid);
				q += (k_cluster[i].diameter / distance);
				count++;
			}
		}
	}
	return q / count;
}

void calculate_diameter(Cluster* cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size, int* all_clusters_points)
{
	int i, j, k;
	double max_distance = 0, distance;
	int start = 0;


	for (k = 0; k < k_cluster_size; k++)
	{
#pragma omp parallel for
		for (i = start; i < all_clusters_points[k] + start; i++)
		{
			for (j = i + 1; j < all_clusters_points[k] + start; j++)
			{
				distance = calculate_distance(set_of_points[i].position, set_of_points[j].position);
				if (distance > max_distance)
					max_distance = distance;
			}
		}
		cluster[k].diameter = max_distance;
		start += all_clusters_points[k];
	}
}

int compare_points_by_cluster_id(const void * p1, const void * p2)
{
	return (((Point*)p1)->cluster_id - ((Point*)p2)->cluster_id);
}

void create_MPI_types(MPI_Datatype* File_info_MPI_type, MPI_Datatype* cluster_MPI_type, MPI_Datatype* Velocity_MPI_type, MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type)
{
	create_file_info_MPI_type(File_info_MPI_type);
	create_velocity_MPI_type(Velocity_MPI_type);
	create_position_MPI_type(Position_MPI_type);
	create_summary_MPI_type(Summary_MPI_type);
	create_point_MPI_type(Point_MPI_type, Position_MPI_type, Velocity_MPI_type);
	create_cluster_MPI_type(cluster_MPI_type, Position_MPI_type, Summary_MPI_type);
}

void create_file_info_MPI_type(MPI_Datatype* File_info_MPI_type)
{
	File_info file_info;
	MPI_Datatype type[FILE_INFO_MEMBERS] = { MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[FILE_INFO_MEMBERS] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[FILE_INFO_MEMBERS];

	disp[0] = (char *)&file_info.N - (char *)&file_info;
	disp[1] = (char *)&file_info.K - (char *)&file_info;
	disp[2] = (char *)&file_info.T - (char *)&file_info;
	disp[3] = (char *)&file_info.dT - (char *)&file_info;
	disp[4] = (char *)&file_info.LIMIT - (char *)&file_info;
	disp[5] = (char *)&file_info.QM - (char *)&file_info;
	MPI_Type_create_struct(FILE_INFO_MEMBERS, blocklen, disp, type, File_info_MPI_type);
	MPI_Type_commit(File_info_MPI_type);
}

void create_cluster_MPI_type(MPI_Datatype* Cluster_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type)
{
	Cluster cluster;
	MPI_Datatype type[CLUSTER_MEMBERS] = { MPI_INT, *Position_MPI_type , MPI_INT, MPI_DOUBLE, *Summary_MPI_type };
	int blocklen[CLUSTER_MEMBERS] = { 1, 1, 1, 1, 1 };
	MPI_Aint disp[CLUSTER_MEMBERS];

	disp[0] = (char *)&cluster.id - (char *)&cluster;
	disp[1] = (char *)&cluster.centroid - (char *)&cluster;
	disp[2] = (char *)&cluster.number_of_points - (char *)&cluster;
	disp[3] = (char *)&cluster.diameter - (char *)&cluster;
	disp[4] = (char *)&cluster.sum - (char *)&cluster;


	MPI_Type_create_struct(CLUSTER_MEMBERS, blocklen, disp, type, Cluster_MPI_type);
	MPI_Type_commit(Cluster_MPI_type);
}

void create_position_MPI_type(MPI_Datatype* Position_MPI_type)
{
	Position position;
	MPI_Datatype type[POSITION_MEMBERS] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[POSITION_MEMBERS] = { 1, 1, 1 };
	MPI_Aint disp[POSITION_MEMBERS];

	disp[0] = (char *)&position.x - (char *)&position;
	disp[1] = (char *)&position.y - (char *)&position;
	disp[2] = (char *)&position.z - (char *)&position;
	MPI_Type_create_struct(POSITION_MEMBERS, blocklen, disp, type, Position_MPI_type);
	MPI_Type_commit(Position_MPI_type);
}

void create_velocity_MPI_type(MPI_Datatype* Velocity_MPI_type)
{
	Velocity velocity;
	MPI_Datatype type[VELOCITY_MEMBERS] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[VELOCITY_MEMBERS] = { 1, 1, 1 };
	MPI_Aint disp[VELOCITY_MEMBERS];

	disp[0] = (char *)&velocity.vx - (char *)&velocity;
	disp[1] = (char *)&velocity.vy - (char *)&velocity;
	disp[2] = (char *)&velocity.vz - (char *)&velocity;
	MPI_Type_create_struct(VELOCITY_MEMBERS, blocklen, disp, type, Velocity_MPI_type);
	MPI_Type_commit(Velocity_MPI_type);
}

void create_point_MPI_type(MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Velocity_MPI_type)
{
	Point point;
	MPI_Datatype type[POINT_MEMBERS] = { *Position_MPI_type, *Position_MPI_type, *Velocity_MPI_type, MPI_INT };
	int blocklen[POINT_MEMBERS] = { 1, 1, 1, 1 };
	MPI_Aint disp[POINT_MEMBERS];

	disp[0] = (char *)&point.position - (char *)&point;
	disp[1] = (char *)&point.init_position - (char *)&point;
	disp[2] = (char *)&point.velocity - (char *)&point;
	disp[3] = (char *)&point.cluster_id - (char *)&point;

	MPI_Type_create_struct(POINT_MEMBERS, blocklen, disp, type, Point_MPI_type);
	MPI_Type_commit(Point_MPI_type);
}

void create_summary_MPI_type(MPI_Datatype* Summary_MPI_type)
{
	Summary sum;
	MPI_Datatype type[SUMMARY_MEMBERS] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[SUMMARY_MEMBERS] = { 1, 1, 1 };
	MPI_Aint disp[SUMMARY_MEMBERS];

	disp[0] = (char *)&sum.sum_x - (char *)&sum;
	disp[1] = (char *)&sum.sum_y - (char *)&sum;
	disp[2] = (char *)&sum.sum_z - (char *)&sum;


	MPI_Type_create_struct(SUMMARY_MEMBERS, blocklen, disp, type, Summary_MPI_type);
	MPI_Type_commit(Summary_MPI_type);
}

void print_cluster_info(Cluster* cluster, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("[Cluster #%d: \n", cluster[i].id);
		printf("center point: [%lf, %lf, %lf]\n", cluster[i].centroid.x, cluster[i].centroid.y, cluster[i].centroid.z);
		printf("number of points: %d\n", cluster[i].number_of_points);
		printf("diameter: %lf]\n", cluster[i].diameter);
	}
}

void print_cluster(Point* set_of_points, int points_size, Cluster* k_clusters, int clusters_size)
{
	int i, j;
	for (i = 0; i < clusters_size; i++)
	{
		printf("Cluster #%d: \n", i);
		for (j = 0; j < points_size; j++)
		{
			if (k_clusters[i].id == set_of_points[j].cluster_id)
			{
				printf("[%lf, %lf, %lf]\n", set_of_points[j].position.x, set_of_points[j].position.y, set_of_points[j].position.z);
			}
		}
	}
}

