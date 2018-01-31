# This version of build_graph works with ballcatch9.py
# Deictic version...
# 
import tensorflow as tf
import tf_util_rob as U

def build_getq(make_obs_ph, q_func, num_actions, num_cascade, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions * num_cascade, scope="q_func")
        q_values_reshape = tf.reshape(q_values, shape=(-1,num_cascade,num_actions))
        getq = U.function(inputs=[observations_ph], outputs=q_values_reshape)
        return getq

def build_train_cascaded(make_obs_ph, 
                            make_target_ph,
                            q_func,
                            num_cascade,
                            num_actions,
                            optimizer, 
                            grad_norm_clipping=None, 
                            double_q=True, 
                            scope="deepq", 
                            reuse=None):

    getq_f = build_getq(make_obs_ph, q_func, num_actions, num_cascade, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        target_input = U.ensure_tf_input(make_target_ph("target"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), num_actions * num_cascade, scope="q_func", reuse=True)
        q_t = tf.reshape(q_t_raw, shape=(-1,num_cascade,num_actions))
        
        # calculate error
        td_error = q_t - tf.stop_gradient(target_input.get())
        errors = U.huber_loss(td_error)

        optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)
        
        targetTrain = U.function(
            inputs=[
                obs_t_input,
                target_input
            ],
            outputs=[td_error, obs_t_input.get(), target_input.get()],
            updates=[optimize_expr]
        )

    return getq_f, targetTrain
        