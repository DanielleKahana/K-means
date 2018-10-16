#pragma once
#include <mpi.h>

#define INPUT_FILE_NAME "input.txt"
#define OUTPUT_FILE_NAME "output.txt"
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
	double sum_x;
	double sum_y;
	double sum_z;
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
double calculate_distance(Position p1, Position p2);
Cluster* init_k_clusters(Point* set_of_points, int K);
void classify_points(Cluster* k_clusters, File_info* file_info, Point* set_of_points, int set_of_points_size);
void set_position_by_time(Point* set_of_points, int set_of_points_size, double time);
void group_points(int* has_changed, Point* set_of_points, Cluster* k_clusters, int set_of_points_size, int k_clusters_size);
void recalculate_centroids_serial(Cluster* k_clusters, int k_clusters_size, Point* set_of_points, int set_of_points_size);
void recalculate_centroids(Cluster* k_clusters, int k_clusters_size);
int compare_points_by_cluster_id(const void * p1, const void * p2);
void calculate_diameter(Cluster* cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size, int* all_clusters_points);
double evaluate_quality(Cluster* k_cluster, int k_cluster_size, Point* set_of_points, int set_of_points_size, int* all_clusters_points);
void send_points_to_processes(File_info* file_info, int numprocs, int myid, Point* set_of_points, Point** process_points, int* process_points_size, MPI_Datatype* Point_MPI_type, MPI_Status* status);
void add_point_to_sum(Cluster* cluster, Position position);
void remove_point_from_sum(Cluster* old_cluster, Position position);
void check_point_has_changed(int* has_changed, int* total_has_changed);
void gather_all_points(Point* set_of_points, int set_of_points_size, Cluster* k_cluster, Point* all_points, int numprocs, int myid, MPI_Datatype* Point_MPI_type, File_info* file_info, int* all_clusters_points);

void create_MPI_types(MPI_Datatype* File_info_MPI_type, MPI_Datatype* cluster_MPI_type, MPI_Datatype* Velocity_MPI_type, MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type);
void create_file_info_MPI_type(MPI_Datatype* File_info_MPI_type);
void create_cluster_MPI_type(MPI_Datatype* Cluster_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Summary_MPI_type);
void create_position_MPI_type(MPI_Datatype* Position_MPI_type);
void create_velocity_MPI_type(MPI_Datatype* Velocity_MPI_type);
void create_point_MPI_type(MPI_Datatype* Point_MPI_type, MPI_Datatype* Position_MPI_type, MPI_Datatype* Velocity_MPI_type);
void create_summary_MPI_type(MPI_Datatype* Summary_MPI_type);

//print for debug
void print_cluster_info(Cluster* cluster, int size);
void print_cluster(Point* set_of_points, int points_size, Cluster* k_clusters, int clusters_size);
void print_array(Point* array, int size);