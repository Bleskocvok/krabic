
CFLAGS ?= -std=c11 -Wall -Wextra -pedantic -O2

NVCCFLAGS ?= -O3

NVCC = nvcc

TARGET = krabic
CUTARGET = cukrabic

LDLIBS += -lSDL2 -lSDL2main -lm

SRC = src/display.c \
	src/vector.c \
	src/main.c

OBJ = $(patsubst src/%,obj/%, $(patsubst %.c,%.o,$(SRC)))


all: $(CUTARGET) $(TARGET)


#
# CPU targets
#

$(TARGET): $(OBJ) obj/physics.o
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

obj/%.o: src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


#
# GPU targets
#

src/physics.cu: src/solve.cu ;

cuda: distclean $(CUTARGET)

$(CUTARGET): $(SRC) src/physics.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@


#
# Automatic header dependencies
#

DEPEND = $(OBJ:.o=.d)

%.o: CFLAGS += -MMD -MP

-include $(DEPEND)


#
# Utility targets
#

clean:
	$(RM) obj/*.o $(DEPEND)

distclean: clean
	$(RM) $(TARGET) $(CUTARGET)

.PHONY: all clean distclean cuda
