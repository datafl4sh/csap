
all: fact

fact.o: fact.s
	nasm -f elf64 -o $@ $<

fact: fact.o
	gcc -o $@ $<


.PHONY: clean
clean:
	rm -f fact.o fact
