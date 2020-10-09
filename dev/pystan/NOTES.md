# NOTES

- Weird data: Donor3, CD56. Seems to be truncated above 0?
- `G_T` and `G_C` need to be different. Otherwise, `p` could be low when
  the `G`'s are similar.
- For performance, 
    - I used the hurdle model representation and got a cleaner 
      and faster implementation.
    - The bottleneck is the `skew-t-lpdf` computation. Using a hierarchical model
      representation is faster, but mixing is slower.
    - I could try stan's map-reduce, but that seems a little tedious.
- Seems like advi/nuts gives you the same performance.

# TODO
- [X] fix `posterior_inference.py`
- [X] fix `simulate_data.py`
- [X] fix `pystan_util.py`
- [ ] fix `sim_study.py`
- [X] Rewrite model and send to Juhee
- [X] 3 components in simulation truth, in control, only 2 components.
- [ ] In simulation truth, set p=(0, 1) => beta=(0, 1)
- [ ] Rerun and send results to Juhee
- [X] Plot prior predictive.
- [X] Prior for nu should be lognormal
- [X] Compute beta posterior (double check)
- [ ] Posterior plots for posterior predictive should optionally be made with
      beta, and p.
- [ ] Posterior plots for all parameters
    - [ ] mu
    - [ ] sigma
    - [ ] nu
    - [ ] phi
    - [ ] `eta_C`, `eta_T_star`, `eta_T`
    - [ ] `gamma_C`, `gamma_T_star`, `gamma_T`
    - [ ] p
    - [ ] beta
- [ ] Full conditionals
- [ ] Sim study
    - [ ] a. `gammaC`=0.3, `gammaT`=0.2, `etaC`=(.5,.5,0), `etaT`=(.5,.2,.3)
    - [ ] b. `gammaC`=0.3, `gammaT`=0.3, `etaC`=(.5,.5,0), `etaT`=(.5,.2,.3)
    - [ ] c. `gammai`=0.3, `etaC`=(.5,.5,0)
    - with K=5. Focus on inference for beta and `F_i`.
