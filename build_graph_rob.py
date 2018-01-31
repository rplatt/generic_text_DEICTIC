#
# TThis is a fork off build_graph 
# 
# 
import tensorflow as tf
import tf_util_rob as U

# get q-values
#def build_getq_robtest(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
def build_getq(make_obs_ph, q_func, num_actions, scope="deepq", scope_q_func="q_func", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions, scope=scope_q_func)    
        getq = U.function(inputs=[observations_ph], outputs=q_values)
        return getq


def build_train_cascaded(make_obs_ph, 
                            make_target_ph,
                            make_actions_ph,
                            q_func,
                            num_cascade,
                            num_actions,
                            batch_size, 
                            num_deictic_patches, 
                            optimizer, 
                            gamma=1.0, 
                            grad_norm_clipping=None, 
                            double_q=True, 
                            scope="deepq", 
                            reuse=None):

    getq_f = build_getq(make_obs_ph, q_func, num_actions * num_cascade, scope=scope, scope_q_func="q_func", reuse=reuse)
#    getq_f_target = build_getq(make_obs_ph, q_func, num_actions * num_cascade, scope=scope, scope_q_func="target_q_func", reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        actions_input = U.ensure_tf_input(make_actions_ph("actions"))
        target_input = U.ensure_tf_input(make_target_ph("target"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), num_actions * num_cascade, scope="q_func", reuse=True)  # reuse parameters from act
        q_t = tf.reshape(q_t_raw,[batch_size*num_deictic_patches,num_cascade,num_actions])
        
        # q values for selected actions
        actionsTiled = tf.one_hot(actions_input.get(),num_actions)
        q_t_action_select = tf.reduce_sum(q_t * actionsTiled,2)
        
        # calculate error
        td_error = q_t_action_select - tf.stop_gradient(target_input.get())
        errors = U.huber_loss(td_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                errors,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

#        # update_target_fn will be called periodically to copy Q network to target Q network
#        update_target_expr = []
#        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
#                                   sorted(target_q_func_vars, key=lambda v: v.name)):
#            update_target_expr.append(var_target.assign(var))
#        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        targetTrain = U.function(
            inputs=[
                obs_t_input,
                actions_input,
                target_input
            ],
            outputs=[td_error, q_t_action_select, target_input.get()],
            updates=[optimize_expr]
        )

#        update_target = U.function([], [], updates=[update_target_expr])

#    return getq_f, getq_f_target, targetTrain, update_target
    return getq_f, targetTrain


