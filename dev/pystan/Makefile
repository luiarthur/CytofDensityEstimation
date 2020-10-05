SIMNAME="simstudy"
AWS_BUCKET="s3://cytof-density-estimation/$(simname)"
RESULTS_DIR="results/$(SIMNAME)"

compile: .model.pkl

.model.pkl: model.stan
	time python3 compile_stan_model.py

sim-study: compile
	for method in advi nuts; do \
		if [ "$${method}" = "advi" ]; then stanseeds="1 2 3 4 5"; else stanseeds="1"; fi; \
		for stanseed in $${stanseeds}; do \
			for etaTK in 0.00 0.10 0.20 0.30 0.40 0.50; do \
				results_dir="$(RESULTS_DIR)/etaTK_$${etaTK}-method_$${method}-stanseed_$${stanseed}"; \
				mkdir -p $${results_dir}; \
				echo $${results_dir}; \
				. venv/bin/activate && \
				  python3 sim_study.py $${results_dir} $$etaTK $$method $$stanseed &> \
				  $${results_dir}/log.txt & \
			done; \
		done; \
	done

send-results:
	aws s3 sync $(RESULTS_DIR) $(AWS_BUCKET)

get-results:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)

freeze:
	rm -f requirements.txt
	. venv/bin/activate && pip freeze > requirements.txt

