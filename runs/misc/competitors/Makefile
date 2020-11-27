SHELL = /bin/bash

resultsdir = $(SCRATCH_DIR)/cde/misc/competitors
awsbucket = "s3://cytof-density-estimation/misc/competitors"

getresults:
	aws s3 sync $(awsbucket) $(resultsdir) --exclude '*.jls'

getallresults:
	aws s3 sync $(awsbucket) $(resultsdir)

sendresults:
	aws s3 sync $(resultsdir) $(awsbucket) --exclude '*.nfs*'
