#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* * Helper: Count trailing zeros (portable version)
 * Used to calculate the ignore mask for non-power-of-two processes.
 */
int count_trailing_zeros(int x) {
    if (x == 0) return 0;
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
}

/* * Helper: Calculate ceiling of log2
 * Determines the height of the binomial tree.
 */
int get_ceil_log2(int p) {
    int r = 0;
    while ((1 << r) < p) {
        r++;
    }
    return r;
}

/*
 * The Sparbit Algorithm Implementation
 * Based on Algorithm 1 from the paper "Sparbit: a new logarithmic-cost..."
 */
int sparbit_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(recvtype, &lb, &extent);

    // --- STEP 1: Local Data Copy ---
    // In MPI_Allgather, every process places its own sendbuf into the 
    // rank-th block of the recvbuf.
    void *local_dest = (char *)recvbuf + (rank * recvcount * extent);
    
    // We use MPI_Sendrecv to safely copy data from sendbuf to recvbuf locally
    // handling potentially complex datatypes.
    if (sendbuf != MPI_IN_PLACE) {
        MPI_Sendrecv(sendbuf, sendcount, sendtype, rank, 0,
                     local_dest, recvcount, recvtype, rank, 0,
                     comm, MPI_STATUS_IGNORE);
    }

    // --- STEP 2: Sparbit Initialization ---
    int data = 1;      // Current number of blocks this process holds
    int ignore = 0;    // Flag to indicate if we should ignore sending in this step
    int ceil_log2_p = get_ceil_log2(p);
    
    // Distance 'd' starts at 2^(ceil(log2 p) - 1) and halves each step
    int d = 1 << (ceil_log2_p - 1); 

    // Calculate the 'ignore_steps' mask based on the paper's binary logic
    // This handles the non-power-of-two cases by identifying "virtual" tree nodes.
    int last_ignore = count_trailing_zeros(p);
    unsigned int p_shifted = (unsigned int)p >> last_ignore;
    // Formula from paper line 163 (Algorithm 1, line 5)
    unsigned int ignore_steps = ((-p_shifted) << 1) << last_ignore;

    // Pre-allocate request array (max 2 requests per block per step)
    // In the worst case, we might exchange 'p' blocks.
    MPI_Request *requests = malloc(sizeof(MPI_Request) * 2 * p); 
    
    // --- STEP 3: Main Loop (log2 p steps) ---
    for (int i = 0; i < ceil_log2_p; i++) {
        
        // Check if the current distance 'd' triggers the ignore flag
        if (d & ignore_steps) {
            ignore = 1;
        }

        int req_count = 0;
        // Number of blocks to exchange in this step
        int blocks_to_exchange = data - ignore;

        // Perform Parallel Sends and Receives for each block
        for (int j = 0; j < blocks_to_exchange; j++) {
            
            // Calculate peers using wrap-around modulo arithmetic
            int src_peer = (rank - d + p) % p;
            int dst_peer = (rank + d) % p;

            // Calculate which data block index to SEND (based on origin rank)
            // Paper formula: (rank - 2*j*d)
            int send_block_idx = (rank - 2 * j * d);
            send_block_idx = ((send_block_idx % p) + p) % p; // Handle negative C modulo

            // Calculate which data block index to RECEIVE (based on origin rank)
            // Paper formula: (rank - (2*j+1)*d)
            int recv_block_idx = (rank - (2 * j + 1) * d);
            recv_block_idx = ((recv_block_idx % p) + p) % p; // Handle negative C modulo

            // Calculate memory addresses
            void *send_ptr = (char *)recvbuf + (send_block_idx * recvcount * extent);
            void *recv_ptr = (char *)recvbuf + (recv_block_idx * recvcount * extent);

            // Issue Non-blocking Send and Receive
            MPI_Isend(send_ptr, recvcount, recvtype, dst_peer, 0, comm, &requests[req_count++]);
            MPI_Irecv(recv_ptr, recvcount, recvtype, src_peer, 0, comm, &requests[req_count++]);
        }

        // Wait for all communications in this step to complete
        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        // Update state for the next step
        d >>= 1;                      // Halve the distance
        data = (data << 1) - ignore;  // Update number of blocks held
        ignore = 0;                   // Reset ignore flag
    }

    free(requests);
    return MPI_SUCCESS;
}

/*
 * Structure to hold benchmark results
 */
typedef struct {
    double min_time;
    double max_time;
    double avg_time;
} Metrics;

/*
 * Benchmark Wrapper with WARMUPS
 */
Metrics benchmark_collective(void (*coll_func)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm),
                             const void *sendbuf, int count, MPI_Datatype type, 
                             void *recvbuf, MPI_Comm comm, int is_sparbit) {
    int iter = 50;
    int warmup = 5; // As specified in the paper 
    double start, end;
    double total_time = 0.0;
    
    int recvcount = count;

    // --- WARMUP PHASE ---
    // We run the algorithm 5 times without timing to settle the system.
    for(int i = 0; i < warmup; i++) {
        if (!is_sparbit) {
             MPI_Allgather(sendbuf, count, type, recvbuf, recvcount, type, comm);
        } else {
             sparbit_allgather(sendbuf, count, type, recvbuf, recvcount, type, comm);
        }
    }

    // Barrier ensures all processes finish warmup before the timer starts
    MPI_Barrier(comm);

    // --- MEASUREMENT PHASE ---
    double local_times[50];
    for(int i = 0; i < iter; i++) {
        start = MPI_Wtime();
        
        if (!is_sparbit) {
             MPI_Allgather(sendbuf, count, type, recvbuf, recvcount, type, comm);
        } else {
             sparbit_allgather(sendbuf, count, type, recvbuf, recvcount, type, comm);
        }
        
        end = MPI_Wtime();
        local_times[i] = end - start;
        total_time += local_times[i];
    }

    // Calculate local average
    double local_avg = total_time / iter;
    
    // Aggregate metrics across all processes (Min/Max/Avg of the averages)
    Metrics m;
    MPI_Reduce(&local_avg, &m.avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_avg, &m.min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_avg, &m.max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int size;
    MPI_Comm_size(comm, &size);
    m.avg_time /= size; // Calculate global average

    return m;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Configuration: 16KB integers (approx 64KB data)
    int count = 16384; 
    
    int *sendbuf = malloc(count * sizeof(int));
    int *recvbuf = malloc(size * count * sizeof(int));

    // Initialize data with dummy values
    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank + i;
    }

    if (rank == 0) {
        printf("--- Sparbit Benchmark vs MPI_Allgather ---\n");
        printf("Processes: %d\n", size);
        printf("Data size per process: %lu bytes\n", count * sizeof(int));
        printf("Warmup runs: 5\n");
        printf("Measured runs: 50\n");
        printf("------------------------------------------\n");
    }

    // 1. Benchmark Standard MPI_Allgather
    Metrics m_mpi = benchmark_collective(NULL, sendbuf, count, MPI_INT, recvbuf, MPI_COMM_WORLD, 0);

    // 2. Benchmark Sparbit Allgather
    Metrics m_spar = benchmark_collective(NULL, sendbuf, count, MPI_INT, recvbuf, MPI_COMM_WORLD, 1);

    if (rank == 0) {
        printf("%-15s | %-12s | %-12s | %-12s\n", "Algorithm", "Min (s)", "Avg (s)", "Max (s)");
        printf("-------------------------------------------------------------\n");
        printf("%-15s | %.6f     | %.6f     | %.6f\n", "MPI_Allgather", m_mpi.min_time, m_mpi.avg_time, m_mpi.max_time);
        printf("%-15s | %.6f     | %.6f     | %.6f\n", "Sparbit", m_spar.min_time, m_spar.avg_time, m_spar.max_time);
        
        double improvement = ((m_mpi.avg_time - m_spar.avg_time) / m_mpi.avg_time) * 100.0;
        printf("\nSparbit Improvement (Avg Time): %.2f%%\n", improvement);
    }

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}