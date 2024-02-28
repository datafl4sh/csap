PROGS=floatformat strtof hypot odometer cancellation quadratic bank epsilon floor
CFLAGS = -Wno-double-promotion -Wno-poison-system-directories

all: $(PROGS)

$(PROGS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< -lm


.PHONY: clean
clean:
	rm -rf $(PROGS)

