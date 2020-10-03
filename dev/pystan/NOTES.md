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
- [ ] Rewrite model and send to Juhee
- [ ] In simulation truth, set p=(0, 1) => beta=(0, 1)
- [ ] 3 components in simulation truth, in control, only 2 components.
- [ ] Rerun and send results to Juhee
- [X] Plot prior predictive.
- [X] Prior for nu should be lognormal
- [X] Compute beta posterior
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
