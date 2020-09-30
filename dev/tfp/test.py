import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dist = tfd.Normal(0., 1.)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        dist.log_prob, step_size=0.1, num_leapfrog_steps=3),
    num_adaptation_steps=100)

summary_writer = tf.summary.create_file_writer('tmp/summary_chain', flush_millis=1000)

def trace_fn(state, results):
    with tf.summary.record_if(tf.equal(results.step % 100, 0)):
        step = tf.cast(results.step, tf.int64)
        str_step = str(step)
        tf.summary.text("", str_step, step=step)
    return ()

# XLA compilation not possible.
# @tf.function(autograph=False, experimental_compile=True)
def run():
    return tfp.mcmc.sample_chain(
        kernel=kernel, current_state=0., num_results=10000, trace_fn=trace_fn)
     
with summary_writer.as_default():
    chain, _ = run()
  
summary_writer.close()
