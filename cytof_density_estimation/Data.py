import numpy as np

class Data:
    def __init__(self, yT, yC):
        self.yT = yT
        self.yC = yC
        self.yT_finite = yT[np.isfinite(yT)]
        self.yC_finite = yC[np.isfinite(yC)]
        self.NC = yC.shape[0]
        self.NT = yT.shape[0]
        self.NC_finite = self.yC_finite.shape[0]
        self.NT_finite = self.yT_finite.shape[0]
        self.ZC = self.NC - self.NC_finite
        self.ZT = self.NT - self.NT_finite