SUBDIRS = deviceQuery vectorAdd

all: $(SUBDIRS)

.PHONY: clean

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir -f Makefile $@; \
	done


deviceQuery:
	$(MAKE) -c deviceQuery

