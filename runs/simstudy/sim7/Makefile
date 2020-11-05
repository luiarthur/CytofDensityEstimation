SHELL = /bin/bash

.PHONY: run getresults sendresults

SIMNAME = sim7
RESULTS_DIR = $(SCRATCH_DIR)/cde/simstudy/$(SIMNAME)/results
AWS_BUCKET = s3://cytof-density-estimation/simstudy/$(SIMNAME)


run:
	rm -f $(RESULTS_DIR)/log.txt && mkdir -p $(RESULTS_DIR)
	julia sim.jl $(RESULTS_DIR) $(AWS_BUCKET) &> $(RESULTS_DIR)/log.txt &


getresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR) --exclude '*.bson'


getallresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)


sendresults:
	aws s3 sync $(RESULTS_DIR) $(AWS_BUCKET) --exclude '*.nfs*'
