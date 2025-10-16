CXX = mpicxx
CXXFLAGS = -O3 -std=c++11 -Wall

SRCS = main.cc construct_matrices.cc matrix_multiply.cc destruct_matrices.cc
OBJS = $(SRCS:.cc=.o)
DEPS = $(SRCS:.cc=.d)
EXEC = matmul

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(EXEC) $(OBJS) $(DEPS)

run:
	srun --mpi=pmix -N $(N) ./$(EXEC) $(FILE)