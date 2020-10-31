SHELL = /bin/bash

.PHONY: run compile compile_and_run getresults sendresults

SIMNAME = sim4
RESULTS_DIR = $(SCRATCH_DIR)/cde/simstudy/$(SIMNAME)/results
AWS_BUCKET = s3://cytof-density-estimation/simstudy/$(SIMNAME)


compile:
	julia -e 'import Pkg; Pkg.activate("../../../"); using CytofDensityEstimation'


run:
	rm -f $(RESULTS_DIR)/log.txt && mkdir -p $(RESULTS_DIR)
	julia sim.jl $(RESULTS_DIR) $(AWS_BUCKET) &> $(RESULTS_DIR)/log.txt &


compile_and_run: compile run


getresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR) --exclude '*.bson'


getallresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)


sendresults:
	aws s3 sync $(RESULTS_DIR) $(AWS_BUCKET) --exclude '*.nfs*'
