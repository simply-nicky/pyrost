# Makefile template for a shared library in C
# https://www.topbug.net/blog/2019/10/28/makefile-template-for-a-shared-library-in-c-with-explanations/

CC = /gpfs/cfel/cxi/scratch/user/nivanov/.conda/envs/pyrost/bin/x86_64-conda_cos6-linux-gnu-gcc  # C compiler
CFLAGS = -fPIC -Wall -Wextra -O2 -g -I /gpfs/cfel/cxi/scratch/user/nivanov/.conda/envs/pyrost/include \
-L /gpfs/cfel/cxi/scratch/user/nivanov/.conda/envs/pyrost/lib -lfftw3 -lfftw3_omp -lgomp -fopenmp -std=c99  # C flags
LDFLAGS = -shared   # linking flags
RM = rm -f   # rm command
TARGET_LIB = libimgutils.so  # target lib

SRCS = array.c fastsum.c fft_functions.c median.c pocket_fft.c pyrost.c routines.c # source files
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)