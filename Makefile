CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -ffast-math
INCLUDES = -Iinclude
LIBS = -lm

SRCDIR = src
BUILDDIR = build
TARGET = $(BUILDDIR)/ttt

SOURCES = ttt.c $(SRCDIR)/box_muller.c $(SRCDIR)/utils.c $(SRCDIR)/game.c $(SRCDIR)/neural_network.c

all: $(TARGET)

$(TARGET): $(SOURCES) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

debug: CFLAGS = -g -Wall -Wextra -std=c99 -DDEBUG
debug: $(TARGET)

.PHONY: all clean debug