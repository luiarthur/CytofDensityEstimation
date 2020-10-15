SIMNAME = sim1
RESULTS_DIR = $(SCRATCH_DIR)/cde/simstudy/$(SIMNAME)/results
AWS_BUCKET = s3://cytof-density-estimation/simstudy/$(SIMNAME)


run:
	julia -e 'import Pkg; Pkg.activate("../../../"); using CytofDensityEstimation'
	for s in `seq 3`; do \
		resdir="$(RESULTS_DIR)/scenario$${s}"; \
		awsbucket="$(AWS_BUCKET)/scenario$${s}"; \
		rm -f $${resdir}/log.txt; \
		mkdir -p $${resdir}; \
		echo $${resdir}; \
		echo $${awsbucket}; \
		julia sim.jl $${resdir} $${awsbucket} $${s} &> $${resdir}/log.txt & \
	done;


