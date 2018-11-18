#pragma once
#include <mpi.h>

#define INPUT_FILE_NAME "C:\\Users\\idan\\Desktop\\input\\INPUT_FILE5.txt"
#define OUTPUT_FILE_NAME "C:\\Users\\idan\\Desktop\\input\\OUTPUT_FILE5.txt"
#define MASTER 0
#define VELOCITY_MEMBERS 3
#define POSITION_MEMBERS 3
#define SUMMARY_MEMBERS 3
#define POINT_MEMBERS 4
#define CLUSTER_MEMBERS 5
#define FILE_INFO_MEMBERS 6

//structs
typedef struct
{
	double x;
	double y;
	double z;
}Summary;

typedef struct
{
	double vx;
	double vy;
	double vz;
}Velocity;

typedef struct
{
	double x;
	double y;
	double z;
}Position;

typedef struct
{
	Position position;
	Position init_position;
	Velocity velocity;
	int cluster_id;
}Point;

typedef struct
{
	int id;
	Position centroid;
	int number_of_points;
	double diameter;
	Summary sum;
}Cluster;

typedef struct
{
	int N, K, T;
	double dT, LIMIT, QM;
}File_info;




//functions
void k_means_algo(int myid, int numprocs);
Point* read_from_file(File_info* file_info);
void write_to_file(File_info* file_info, Cluster* k_clusters, double time, double quality);
double calculate_distance(Position* p1, Position* p2);
Cluster* init_k_clusters(Point* set_of_points, int K);
void classify_points(Cluster* k_clusters, File_info* file_info, Point* set_of_points, int set_of_points_size);
void calculate_diameter(Cluster* cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size);
void recalculate_centroids(Cluster* k_clusters, int k_clusters_size);
int compare_points_by_cluster_id(const void * p1, const void * p2);
double evaluate_quality(Cluster* k_cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size);
void send_points_to_processes(File_info* file_info, int numprocs, int myid, Point* set_of_points, Point** process_points, int* process_points_size, MPI_Datatype* Point_MPI_type, MPI_Status* status);
void add_point_to_cluster(Cluster* cluster, Position position);
void update_has_changed(int* has_changed, int* total_has_changed);
void gather_all_points(Point* set_of_points, int set_of_points_size, Cluster* k_cluster, Point* all_points, int numprocs, int myid, MPI_Datatype* Point_MPI_type, File_info* file_info);
void update_clusters(Cluster* clusters, int clusters_size, Point* points, int points_size);
void reset_clusters(Cluster* clusters, int clusters_size);
void create_MPI_types(MPI_Datatype* File_info_MPI_type, MPI_Datatype* cluster_MPI_type, MPI_Datatype* Velocity_MPI_type, MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type);
void create_file_info_MPI_type(MPI_Datatype* File_info_MPI_type);
void create_cluster_MPI_type(MPI_Datatype* Cluster_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type);
void create_position_MPI_type(MPI_Datatype* Position_MPI_type);
void create_velocity_MPI_type(MPI_Datatype* Velocity_MPI_type);
void create_point_MPI_type(MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Velocity_MPI_type);
void create_summary_MPI_type(MPI_Datatype* Summary_MPI_type);

void free_memory(Point* process_points, Cluster* k_clusters, Point* set_of_points);

//cuda functions
void group_points_with_cuda(int* has_changed, Point* set_of_points, Cluster* k_clusters, int set_of_points_size, int k_clusters_size);
void set_position_by_time_with_cuda(Point* set_of_points, int set_of_points_size, double time);
void free_cuda_memory(Point* dev_points, Cluster* dev_clusters, int* dev_has_changed);
void cuda_error(void* array);




