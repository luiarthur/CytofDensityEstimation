SHELL = /bin/bash

.PHONY: run getresults

SIMNAME = run0
RESULTS_DIR = $(SCRATCH_DIR)/cde/datastudy/$(SIMNAME)
AWS_BUCKET = s3://cytof-density-estimation/datastudy/$(SIMNAME)


run:
	rm -f $(RESULTS_DIR)/log.txt && mkdir -p $(RESULTS_DIR)
	julia run.jl $(RESULTS_DIR) $(AWS_BUCKET) &> $(RESULTS_DIR)/log.txt


getresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)
