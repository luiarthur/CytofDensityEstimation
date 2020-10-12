# TODO

- [ ] Change implementation of `gibbs`
    - [ ] `nmcmc` -> `nsamps::Vector{Int}`, `thins` -> `thin::Int`
        - desired behavior: provide `nsamps::Vector{Int}` which needs to be 
          in descending order. `nsamps` needs to be same length as `monitors`.
          `thin` specifies how the first monitor should be thinned. So, if
          `nsamps=[1000, 10], thin=2, burn=20`, then total number number
          iterations is `burn + nsamps[1] * thin = 10 + 1000 * 2 = 2020`.
          After the burn-in period, thin accordingly for the second monitor so
          that in the end, 10 samples are collected.
- [ ] Implement `fit`
    - No need to keep `v` and `zeta` as the original `SkewT` parameters can be
      recovered from `psi` and `omega`.
- [ ] Simulate data
- [ ] Write Simulation Studies
- [ ] Methods for reading real data
- [ ] Methods for prior predictive
- [ ] Methods for plotting data
- [ ] Compare results to KS using `HypothesisTests.ApproximateTwoSampleKSTest`
- [ ] Methods for post processing
    - [ ] Test that `plots.gr()` works on server
    - [ ] Plot makers for each parameter
    - [ ] Credible interval for density estimate

