# README

This experiment has `Ni=10000` and simulated data is realistic.

The goal is to see if we recover simulation truth in this case.

# ISSUES
- implied prior (sig, phi) was bad. Fixed by adding tau.
- Marginalized update for lambda was actually bad. Tried changing 
  order of update so that (v, zeta) were updated right after lambda.
  Didn't help. But using the non-marginalized updates actually worked
  much better.
- Priors for nu. Use LogNormal(3, .5).
- Still use rep_aux=1.
