compile: .model.pkl

.model.pkl: model.stan
	time python3 compile_stan_model.py

sim-study: compile
	for method in advi nuts; do \
		if [ "$${method}" = "advi" ]; then stanseeds="1 2 3 4 5"; else stanseeds="1"; fi; \
		for stanseed in $${stanseeds}; do \
			for etaTK in 0.00 0.10 0.20 0.30 0.40 0.50; do \
				results_dir="results/test/etaTK_$${etaTK}-method_$${method}-stanseed_$${stanseed}"; \
				mkdir -p $${results_dir}; \
				echo $${results_dir}; \
				python3 sim_study.py $${results_dir} $$etaTK $$method $$stanseed &> \
				  $${results_dir}/log.txt &
			done; \
		done; \
	done

