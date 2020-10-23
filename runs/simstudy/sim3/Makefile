SHELL = /bin/bash

.PHONY: run compile compile_and_run getresults

SIMNAME = sim3
RESULTS_DIR = $(SCRATCH_DIR)/cde/simstudy/$(SIMNAME)/results
AWS_BUCKET = s3://cytof-density-estimation/simstudy/$(SIMNAME)


compile:
	julia -e 'import Pkg; Pkg.activate("../../../"); using CytofDensityEstimation'


run:
	. scripts/run.sh && runsim $(RESULTS_DIR) $(AWS_BUCKET)


compile_and_run: compile run

getresults:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)
