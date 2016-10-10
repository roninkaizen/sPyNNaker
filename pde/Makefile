BUILDS = delay_pde neuron_pde
DIRS = $(BUILDS:%=builds/%)

all: $(DIRS)
	for d in $(DIRS); do (cd $$d; "$(MAKE)") || exit $$?; done

%.aplx: builds/%
	cd $*; "$(MAKE)"

clean: $(DIRS)
	for d in $(DIRS); do (cd $$d; "$(MAKE)" clean) || exit $$?; done
