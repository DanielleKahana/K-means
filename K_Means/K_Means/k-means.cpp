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


	//Start time
	if(myid == MASTER)
		start_time = MPI_Wtime();

	k_means_algo(myid, numprocs);

	//end time
	if (myid == MASTER)
	{
		end_time = MPI_Wtime();
		printf("K-Means done in %lf seconds\n", end_time - start_time);
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
	double quality, time;

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

	if (myid != MASTER)
	{
		k_clusters = (Cluster*)malloc(sizeof(Cluster)*(file_info.K));
	}

	//Broadcast the first K clusters	
	MPI_Bcast(k_clusters, file_info.K, Cluster_MPI_type, MASTER, MPI_COMM_WORLD);

	//Scatter the points equaly between the N processes
	send_points_to_processes(&file_info, numprocs, myid, set_of_points, &process_points, &process_points_size, &Point_MPI_type, &status);


	for (time = 0; time <= file_info.T && !finished; time += file_info.dT)
	{

		//set the points position by time - t
		set_position_by_time_with_cuda(process_points, process_points_size, time);

		//for each point assign the closest cluster - until no point has changed cluster
		classify_points(k_clusters, &file_info, process_points, process_points_size);

		//processes send to master their points
		gather_all_points(process_points, process_points_size, k_clusters, set_of_points, numprocs, myid, &Point_MPI_type, &file_info);


		if (myid == MASTER)
		{
			//calculate the quality
			quality = evaluate_quality(k_clusters, file_info.K, set_of_points, file_info.N);

			if (quality < file_info.QM)
				finished = 1;
		}
		//broadcasting to other processes if another iteration is needed
		MPI_Bcast(&finished, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	if (myid == MASTER)
	{
		//write to file the results
		write_to_file(&file_info, k_clusters,time - file_info.dT, quality); 
	}

	//free all dynamic memory
	free_memory(process_points, k_clusters, set_of_points);
}



Point* read_from_file(File_info* file_info)
{
	Point* set_of_points = NULL;
	FILE* f;
	int i;
	Position position = { 0,0,0 };
	Velocity velocity = { 0,0,0 };

	f = fopen(INPUT_FILE_NAME, "r");

	if (f == NULL)
	{
		printf("Failed opening file. Exiting!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	fscanf(f, "%d %d %lf %lf %lf %lf", &(file_info->N), &(file_info->K), &(file_info->T), &(file_info->dT), &(file_info->LIMIT), &(file_info->QM));

	if (file_info->N < MIN_POINTS || file_info->N > MAX_POINTS)
	{
		printf("Number of points should be between 10,000 to 3,000,000. try again!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	set_of_points = (Point*)malloc(sizeof(Point)*(file_info->N));


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
	}

}

double calculate_distance(Position* p1, Position* p2)
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

Cluster* init_k_clusters(Point* set_of_points, int K)
{
	Cluster* k_clusters = (Cluster*)malloc(sizeof(Cluster)*K);
	int i;

#pragma omp parallel for
	for (i = 0; i < K; i++)
	{
		k_clusters[i].id = i;
		k_clusters[i].centroid = set_of_points[i].position;
		k_clusters[i].number_of_points = 0;
		k_clusters[i].diameter = 0;
		k_clusters[i].sum.x = 0;
		k_clusters[i].sum.y = 0;
		k_clusters[i].sum.z = 0;
	}

	return k_clusters;
}


void classify_points(Cluster* k_clusters, File_info* file_info, Point* set_of_points, int set_of_points_size)
{
	double  iter;
	int has_changed = 0, total_has_changed = 1;

	
	for (iter = 0; iter <= file_info->LIMIT && total_has_changed > 0; iter++)
	{
		has_changed = 0;
		
		//for each points decide which cluster is the closest
		group_points_with_cuda(&has_changed, set_of_points, k_clusters, set_of_points_size, file_info->K);
	

		//add sum of points to the cluster
		update_clusters(k_clusters, file_info->K, set_of_points, set_of_points_size);

		//calculate the centers of each cluster 
		recalculate_centroids(k_clusters, file_info->K);

		//Check if any change was made in any cluster, if so iterations continues 
		update_has_changed(&has_changed, &total_has_changed);

	}
	
}

void write_to_file(File_info* file_info, Cluster* k_clusters, double time, double quality)
{
	FILE* f;
	int i;

	f = fopen(OUTPUT_FILE_NAME, "w");

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

void gather_all_points(Point* set_of_points, int set_of_points_size, Cluster* k_cluster, Point* all_points, int numprocs, int myid, MPI_Datatype* Point_MPI_type, File_info* file_info)
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
}


void update_has_changed(int* has_changed, int* total_has_changed)
{
	MPI_Allreduce(has_changed, total_has_changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void recalculate_centroids(Cluster* k_clusters, int k_clusters_size)
{
	int num_of_points = 0, i;
	double sum_x = 0, sum_y = 0, sum_z = 0;

	for (i = 0; i < k_clusters_size; i++)
	{
			MPI_Allreduce(&k_clusters[i].sum.x, &sum_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&k_clusters[i].sum.y, &sum_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&k_clusters[i].sum.z, &sum_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			MPI_Allreduce(&k_clusters[i].number_of_points, &num_of_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			
			k_clusters[i].number_of_points = num_of_points;

			if (num_of_points != 0)
			{
				k_clusters[i].centroid.x = sum_x / num_of_points;
				k_clusters[i].centroid.y = sum_y / num_of_points;
				k_clusters[i].centroid.z = sum_z / num_of_points;
			}
	}
}


double evaluate_quality(Cluster* k_cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size)
{
	int i, j = 0, num_of_clusters_to_divide = k_cluster_size * (k_cluster_size - 1);
	double quality = 0, distance = 0;
	
	calculate_diameter(k_cluster, k_cluster_size, set_of_points, set_of_points_size);


#pragma omp parallel for private(distance, j) reduction(+:quality)
	for (i = 0; i < k_cluster_size; i++)
	{
		for (j = 0; j < k_cluster_size; j++)
		{
			if (i != j)
			{
				distance = calculate_distance(&(k_cluster[i].centroid), &(k_cluster[j].centroid));
				
				quality += (k_cluster[i].diameter / distance);
				
			}
		}
	}
	
	return quality / num_of_clusters_to_divide;
}

void calculate_diameter(Cluster* cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size)
{
	int i, j, k;
	double distance, max_distance = 0;
	int start = 0;

	qsort(set_of_points, set_of_points_size, sizeof(Point), compare_points_by_cluster_id);

	for (k = 0; k < k_cluster_size; k++)
	{
		for (i = start; i < cluster[k].number_of_points + start; i++)
		{
			for (j = i + 1; j < cluster[k].number_of_points + start; j++)
			{
				distance = calculate_distance(&(set_of_points[i].position), &(set_of_points[j].position));
				if (distance > max_distance)
					max_distance = distance;
			}
		}
		cluster[k].diameter = max_distance;
		start += cluster[k].number_of_points;
		max_distance = 0;
	}
}

void reset_clusters(Cluster* clusters, int clusters_size)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < clusters_size; i++)
	{
		clusters[i].number_of_points = 0;
		clusters[i].sum.x = 0;
		clusters[i].sum.y = 0;
		clusters[i].sum.z = 0;
	}
}

void update_clusters(Cluster* clusters, int clusters_size, Point* points, int points_size)
{
	int i, index;

	//reset clusters
	reset_clusters(clusters, clusters_size);

	for (i = 0; i < points_size; i++)
	{
		index = points[i].cluster_id;
		add_point_to_cluster(&clusters[index], points[i].position);
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
	MPI_Datatype type[FILE_INFO_MEMBERS] = { MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
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

	disp[0] = (char *)&sum.x - (char *)&sum;
	disp[1] = (char *)&sum.y - (char *)&sum;
	disp[2] = (char *)&sum.z - (char *)&sum;


	MPI_Type_create_struct(SUMMARY_MEMBERS, blocklen, disp, type, Summary_MPI_type);
	MPI_Type_commit(Summary_MPI_type);
}

void add_point_to_cluster(Cluster* cluster, Position position)
{
	//sum up points
	cluster->sum.x += position.x;
	cluster->sum.y += position.y;
	cluster->sum.z += position.z;

	//add point to total
	cluster->number_of_points++;
}

void free_memory(Point* process_points, Cluster* k_clusters, Point* set_of_points)
{
	free(process_points);
	free(k_clusters);
	free(set_of_points);
}






