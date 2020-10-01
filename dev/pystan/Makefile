compile: .model.pkl

.model.pkl: model.stan
	python3 compile_stan_model.py

sim-study:
	for method in advi nuts; do \
		for p in 0.95 0.05; do \
			results_dir="results/test/p_$${p}-method_$${method}"; \
			mkdir -p $${results_dir}; \
			echo $${results_dir}; \
			python3 sim_study.py $${results_dir} &> $${results_dir}/log.txt & \
		done; \
	done

