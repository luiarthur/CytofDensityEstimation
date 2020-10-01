import time
import functools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tqdm import trange
dtype = np.float64
Root = tfd.JointDistributionCoroutine.Root

import sys
sys.path.append('../util')
import util

def compute_gamma_T_star(gamma_C, gamma_T, p):
    return p * gamma_T + (1 - p) * gamma_C

def compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star):
    numer = eta_T * (1 - gamma_T) * p + eta_C * (1 - p) * (1 - gamma_C)
    return numer / (1 - gamma_T_star)


def mix_T(gamma_C, gamma_T, eta_C, eta_T, p, loc, scale):
    gamma_T_star = compute_gamma_T_star(gamma_C, gamma_T, p)
    eta_T_star = compute_eta_T_star(gamma_C, gamma_T, eta_C, eta_T, p, gamma_T_star)
    return mix(gamma_T_star, eta_T_star, loc, scale)


def mix(eta, loc, scale):
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=eta),
        components_distribution=tfd.Normal(loc=loc, scale=scale))

# TEST:
K = 5
gamma = tfd.Beta(dtype(1), 1.).sample()
eta = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K).sample()
m = mix(gamma, eta, tf.zeros(K, dtype=dtype), tf.ones(K, dtype=dtype), dtype(-10))
s = m.sample(3)
m.log_prob(s)

# NOTE:
# - `Sample` and `Independent` resemble, respectively, `filldist` and `arraydist` in Turing.
def create_model(npos_C, npos_T, n0_C, n0_T, K, dtype=np.float64):
    n_C = npos_C + n0_C
    n_T = npos_T + n0_T
    return tfd.JointDistributionNamed(dict(
        p = tfd.Beta(dtype(1), dtype(1)),
        gamma_C = tfd.Beta(dtype(1), dtype(1)),
        gamma_T = tfd.Beta(dtype(1), dtype(1)),
        eta_C = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        eta_T = tfd.Dirichlet(tf.ones(K, dtype=dtype) / K),
        loc = tfd.Sample(tfd.Normal(dtype(0), dtype(1)),
                         sample_shape=K),
        sigma_sq = tfd.Sample(tfd.InverseGamma(dtype(3), dtype(2)),
                              sample_shape=K),
        n0_C = lambda gamma_C: tfd.Binomial(n_C, gamma_C),
        n0_T = lambda gamma_T: tfd.Binomial(n_C, gamma_C),
        y_C = lambda gamma_C, eta_C, loc, sigma_sq: tfd.Sample(
            mix(gamma_C, eta_C, loc, tf.sqrt(sigma_sq), dtype(neg_inf)), n_C),
        y_T = lambda gamma_C, gamma_T, eta_C, eta_T, p, loc, sigma_sq: tfd.Sample(
            mix_T(gamma_C, gamma_T, eta_C, eta_T, p, loc, tf.sqrt(sigma_sq),
                  dtype(neg_inf)), n_T)
    ))


# TEST
n_C=4
n_T=6
K=3
model = create_model(n_C=n_C, n_T=n_T, K=K)
s = model.sample(); s
model.log_prob(s)
# model.sample(2)  # FIXME

bijectors = [
    tfb.Sigmoid(),  # p
    tfb.Sigmoid(),  # gamma_C
    tfb.Sigmoid(),  # gamma_T
    tfb.SoftmaxCentered(),  # eta_C
    tfb.SoftmaxCentered(),  # eta_T
    tfb.Identity(),  # loc
    tfb.Exp()  # sigma_sq
]


d1 = util.read_data('../../data/TGFBR2/cytof-data/donor1.csv', 'CD16', 2000, 2)
model = create_model(n_C=d1['y_C'].shape[0], n_T=d1['y_T'].shape[0], K=5)
_ = model.sample()
def target_log_prob_fn(p, gamma_C, gamma_T, eta_C, eta_T, loc, sigma_sq):
    return model.log_prob(p=p, gamma_C=gamma_C, gamma_T=gamma_T, eta_C=eta_C, 
                          eta_T=eta_T, loc=loc, sigma_sq=sigma_sq, 
                          y_T=util.replace_inf(d1['y_T'], dtype(-10)),
                          y_C=util.replace_inf(d1['y_C'], dtype(-10)))

def generate_initial_state(K):
    return [
        tf.ones([], dtype, name='p') * 0.5,
        tf.ones([], dtype, name='gamma_C') * 0.5,
        tf.ones([], dtype, name='gamma_T') * 0.5,
        tf.ones(K, dtype, name='eta_C') / K,
        tf.ones(K, dtype, name='eta_T') / K,
        tf.zeros(K, dtype, name='loc'),
        tf.ones(K, dtype, name='sigma_sq'),
    ]

@tf.function(autograph=False, experimental_compile=True)
def nuts_sample(num_results, num_burnin_steps, current_state, pkr=None,
                max_tree_depth=10):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=current_state,
        previous_kernel_results=pkr,
        return_final_kernel_results=True,
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.NoUTurnSampler(
                     target_log_prob_fn=target_log_prob_fn,
                     max_tree_depth=max_tree_depth, step_size=0.01, seed=1),
                bijector=bijectors),
            num_adaptation_steps=num_burnin_steps,  # should be smaller than burn-in.
            target_accept_prob=0.8),
        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted)
        # trace_fn = lambda _, pkr: pkr)
        # trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted, pkr)



# Initial run
tf.random.set_seed(7)
current_state = generate_initial_state(K=5)  # generate initial values.
current_state, _, _ = nuts_sample(5, 5, current_state=current_state)

print_freq = 100
nburn = 400
nsamps = 400

def sample_with_progress(nburn, nsamps, print_freq, current_state, pkr=None):
    chunks = (nburn + nsamps) // print_freq
    chain_blocks = []
    print("Burn in ...")
    for n in trange(chunks):
        current_state = tf.nest.map_structure(lambda x: x[-1], current_state)
        if n < (nburn // print_freq):
            current_state, _, pkr = nuts_sample(1, print_freq - 1,
                                                current_state=current_state, pkr=pkr)
        else:
            current_state, _, pkr = nuts_sample(print_freq, 0,
                                                current_state=current_state, pkr=pkr)
            chain_blocks.append(current_state)

    print("Creating full chain ...")
    full_chain = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0),
                                       *chain_blocks)

    return full_chain, pkr

tic = time.time()
chain, pkr = sample_with_progress(nburn, nsamps, print_freq, current_state)
toc = time.time()
print(f'Time for blocks: {toc - tic}')
# 130 s


tic = time.time()
tf.random.set_seed(7)
current_state = generate_initial_state(K=5)  # generate initial values.
_ = nuts_sample(nsamps, nburn, current_state=current_state)
toc = time.time()
print(f'Time for single chain: {toc - tic}')
# 75 s

