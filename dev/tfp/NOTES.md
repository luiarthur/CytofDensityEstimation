# NOTES

- Can inference (ADVI/HMC/NUTS) be done in tfp?
    - Yes
- What is required?
    - Implement `SkewT` distribution.
    - Then, it can be implemented as with the STAN model.
- Other issues
    - TFP doesn't have a progress bar. This is problematic as the run could take 
      several days with the full dataset. If there's a way to show any kind of progress
      I will implement the SkewT.
