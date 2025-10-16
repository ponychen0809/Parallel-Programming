#include <iostream>
#include <cstdlib>  // For rand()
#include <ctime>    // For time()
#include <pthread.h>

using namespace std;

// Structure to hold data for each thread
struct ThreadData {
    long long int number_of_tosses;
    int number_in_circle;
};

int thread_count;  // Number of threads
long long int total_tosses;  // Total number of tosses across all threads
pthread_mutex_t mutex;  // Mutex to protect shared resources

void* monte_carlo(void* args) {
    // Extract thread data
    ThreadData* data = (ThreadData*)args;
    long long int tosses = data->number_of_tosses;
    int local_circle_count = 0;

    // Seed the random number generator for each thread
    unsigned int seed = time(NULL) ^ pthread_self();

    // Perform the tosses
    for (long long int toss = 0; toss < tosses; toss++) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) {
            local_circle_count++;  // Point is inside the circle
        }
    }
    // Protect shared resource using mutex before updating global count
    pthread_mutex_lock(&mutex);
    data->number_in_circle += local_circle_count;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char* argv[]) {

    thread_count = stoi(argv[1]);
    total_tosses = stoll(argv[2]);

    pthread_t* threads = new pthread_t[thread_count];
    ThreadData* thread_data = new ThreadData[thread_count];

    long long int tosses_per_thread = total_tosses / thread_count;
    pthread_mutex_init(&mutex, NULL);

    // Create threads
    for (int i = 0; i < thread_count; i++) {
        thread_data[i].number_of_tosses = tosses_per_thread;
        thread_data[i].number_in_circle = 0;
        pthread_create(&threads[i], NULL, monte_carlo, (void*)&thread_data[i]);
    }

    int total_in_circle = 0;
    // Join threads and collect results
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
        total_in_circle += thread_data[i].number_in_circle;
    }

    pthread_mutex_destroy(&mutex);

    double pi_estimate = 4 * (double)total_in_circle / (double)total_tosses;
    cout << pi_estimate << endl;

    // Free memory
    free(threads);
    free(thread_data);

    return 0;
}
