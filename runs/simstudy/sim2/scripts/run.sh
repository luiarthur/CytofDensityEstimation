#!/bin/bash

function runsim() {
  RESULTS_DIR=$1
  AWS_BUCKET=$2

  for scenario in `seq 5`; do
    resdir="${RESULTS_DIR}/scenario${scenario}"
    awsbucket="${AWS_BUCKET}/scenario${scenario}"
    rm -f ${resdir}/log.txt
    mkdir -p ${resdir}
    echo ${resdir}
    echo ${awsbucket}
    julia sim.jl ${resdir} ${awsbucket} ${scenario} &> ${resdir}/log.txt &
    sleep 200
  done
}
