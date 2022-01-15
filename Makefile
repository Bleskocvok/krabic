
CFLAGS += -std=c99 -Wall -Wextra -pedantic

TARGET = krabic

SRC = src/display.c \
	  src/physics.c	\
	  src/vector.c \
	  src/main.c

OBJ = $(patsubst src/%,obj/%, $(patsubst %.c,%.o,$(SRC)))

DEPS = -lSDL2 -lSDL2main -lm


#
# Main targets
#

all: $(TARGET)

$(TARGET): LDLIBS += $(DEPS)
$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

obj/%.o: src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


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
	$(RM) $(OBJ) $(DEPEND)

distclean: clean
	$(RM) $(TARGET)

.PHONY: all clean distclean
