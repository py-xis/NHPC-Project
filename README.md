# HPC Project - SPARBIT

## Prerequisites

- OpenMPI installed on your system
- C compiler (gcc/mpicc)

## Compilation

To compile the program, use the following command:

```bash
mpicc sparbit.c -o sparbit
```

## Running the Program

To run the program with OpenMPI, use:

```bash
mpirun -np <number-of-processes> ./sparbit
```

Replace `<number-of-processes>` with the desired number of processes.

### Example

```bash
# Compile the program
mpicc sparbit.c -o sparbit

# Run with 4 processes
mpirun -np 4 ./sparbit
```

## Project Files

- `sparbit.c` - Source code for parallel sparse matrix-vector multiplication
- `2109.08751v1-2.pdf` - Project reference paper
- `HPC Presentation.mp4` - Project presentation video
- `Project Presentation.pptx` - Project presentation slides

## Authors

- IMT2022036: Aryan Singhal
- IMT2022053: Pranav Kulkarni
- IMT2022122: Vansh Sinha
