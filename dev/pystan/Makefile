SHELL = /bin/bash
SIMNAME = simstudy
AWS_BUCKET = s3://cytof-density-estimation/$(SIMNAME)
RESULTS_DIR = results/$(SIMNAME)

compile: .model.pkl

.model.pkl: model_reparameterized.stan
	# PYTHONPATH="" && . venv/bin/activate && time python compile_stan_model.py
	time python3 compile_stan_model.py

sim-study: compile
	for method in advi nuts; do \
		if [ "$${method}" = "advi" ]; then stanseeds="1 2 3 4 5"; else stanseeds="1"; fi; \
		for stanseed in $${stanseeds}; do \
			for etaTK in 0.00 0.10 0.20 0.30 0.40 0.50; do \
				simname="etaTK_$${etaTK}-method_$${method}-stanseed_$${stanseed}"; \
				results_dir="$(RESULTS_DIR)/$${simname}"; \
				mkdir -p $${results_dir}; \
				echo $${results_dir}; \
				PYTHONPATH="" && . venv/bin/activate && \
				  python sim_study.py $${results_dir} $$etaTK $$method $$stanseed &> \
				  $${results_dir}/log.txt && \
					aws s3 sync $${results_dir} $(AWS_BUCKET)/$${simname} & \
			done; \
		done; \
	done; \
  echo "Done submitting jobs.";\
  echo "Job submission time:";\
  date;\
  echo "Jobs are now running. A message will be printed and emailed when jobs are done.";\
  wait; \
  echo "Done!"


send-results:
	aws s3 sync $(RESULTS_DIR) $(AWS_BUCKET)

get-results:
	aws s3 sync $(AWS_BUCKET) $(RESULTS_DIR)

freeze:
	rm -f requirements.txt
	. venv/bin/activate && pip freeze > requirements.txt

