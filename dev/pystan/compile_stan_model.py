import pickle
import pystan

# sm = pystan.StanModel("model.stan")
sm = pystan.StanModel("model_reparameterized.stan")

with open('.model.pkl', 'wb') as f:
    pickle.dump(sm, f)
