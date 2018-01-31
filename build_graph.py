#
# This version of build_graph contains a deictic implementation
#
"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
#import baselines.common.tf_util as U
import tf_util_rob as U

# get q-values
#def build_getq_robtest(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
def build_getq(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")    
        getq = U.function(inputs=[observations_ph], outputs=q_values)
        return getq


def build_train_deictic_min_streamlined(make_obs_ph, 
                                        q_func, 
                                        num_actions, 
                                        batch_size, 
                                        num_deictic_patches, 
                                        max_num_groups, 
                                        optimizer, 
                                        gamma=1.0, 
                                        grad_norm_clipping=None, 
                                        double_q=True, 
                                        scope="deepq", 
                                        reuse=None):

#    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    getq_f = build_getq(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        group_matching_ph = tf.placeholder(tf.int32, [None], name="group_matching")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done_mask")

        # Creating this placeholder to enable a tabular version of this code
        q_tp1_ph = tf.placeholder(tf.float32, [None,4], name="q_tp1")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
#        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        q_tp1 = q_tp1_ph
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_maxa = tf.reduce_max(q_tp1,1)

        # Calculate target = max_{a,d} Q(d,a). Size should be: Bx1
        q_tp1_maxa_reshape = tf.reshape(q_tp1_maxa, [batch_size,num_deictic_patches])
        q_tp1_target = tf.reduce_max(q_tp1_maxa_reshape,1)
        q_tp1_target = rew_t_ph + gamma * q_tp1_target
        q_tp1_target = (1.0 - done_mask_ph) * q_tp1_target

        # Calculate desc_2_state. Encodes which descriptors contained in each state.
        # Dimensions should be: B x max_num_groups
        group_matching_3dtensor = tf.reshape(group_matching_ph,[batch_size,num_deictic_patches])
        groups_onehot = tf.one_hot(group_matching_3dtensor,max_num_groups,axis=-1)
        desc_2_state = tf.reduce_max(groups_onehot,1)
        
        # Calculate target_min_per_D
        max_target = tf.reduce_max(q_tp1_target)
        q_tp1_target_tiled = tf.tile(tf.reshape(q_tp1_target,[batch_size,1]),[1,max_num_groups])
        target_min_per_D_expanded = desc_2_state * q_tp1_target_tiled + (1 - desc_2_state) * max_target
        target_min_per_D = tf.reduce_min(target_min_per_D_expanded,0)
        
        # Project target_min_per_D onto q_t
        D_2_DI = tf.one_hot(group_matching_ph,max_num_groups)
        
        # Both the next two lines produce the same results
        targets = tf.squeeze(tf.matmul(D_2_DI,tf.reshape(target_min_per_D,[max_num_groups,1])))
#        targets2 = tf.reduce_sum(D_2_DI * tf.tile(tf.reshape(target_min_per_D,[1,max_num_groups]),[batch_size*num_deictic_patches,1]),1)
        
        
#        tf.tile(tf.reshape(target_min_per_D,[max_num_groups,1]),[1,np.shape(q_t)[0]])
        
#        # expand target to BxD(I')
#        q_tp1_target_under_tiled = tf.tile(tf.reshape(q_tp1_target_under,[batch_size,1]),[1,num_deictic_patches])
#        q_tp1_target = tf.reshape(q_tp1_target_under_tiled,[batch_size*num_deictic_patches,1])
#        # Calculate min over groups
#        groups_onehot = tf.one_hot(group_matching_ph,max_num_groups+1)
#        q_tp1_target_tiled = tf.tile(q_tp1_target,[1,max_num_groups+1])
#        max_target = tf.reduce_max(q_tp1_target)
#        groups_target_tiled = groups_onehot*q_tp1_target_tiled + (1-groups_onehot)*max_target
#        groups_target = tf.reduce_min(groups_target_tiled,0)[:-1]
#        groups_target_reduced = groups_target[0:tf.minimum(tf.shape(group_actions_ph)[0],tf.shape(groups_target)[0])]

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute the error (potentially clipped)
#        td_error = q_group_selected - tf.stop_gradient(groups_target_reduced)
        td_error = q_t_selected - tf.stop_gradient(targets)
        errors = U.huber_loss(td_error)
        weighted_error = errors
        
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                group_matching_ph,
                done_mask_ph
            ],
            outputs=[targets],
            updates=[optimize_expr]
        )

        # Create callable functions
        trainWOUpdate = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                group_matching_ph,
                done_mask_ph,
                q_tp1_ph
            ],
            outputs=[q_tp1_target, desc_2_state, target_min_per_D, D_2_DI, targets]
        )

        update_target = U.function([], [], updates=[update_target_expr])
#        q_values = U.function([obs_t_input], q_t)

#        return getq_f, train, trainWOUpdate, update_target, {'q_values': q_values}
        return getq_f, train, trainWOUpdate, update_target
#        return getq_f, train, trainWOUpdate
#        return getq_f, trainWOUpdate, update_target


def build_train_deictic_min(make_obs_ph, make_match_ph, q_func, num_actions, batch_size, num_deictic_patches, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):

#    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    getq_f = build_getq(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        match_ph = tf.placeholder(tf.int32, [None], name="match")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        
        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1,1)

        q_tp1_best = tf.reshape(q_tp1_best, [batch_size,num_deictic_patches])
        q_tp1_best_reduced = tf.reduce_max(q_tp1_best,1)
        
        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_reduced
        q_t_selected_target_masked = (1.0 - done_mask_ph) * q_t_selected_target
        
        q_t_selected_target_tile2expand = tf.tile(tf.reshape(q_t_selected_target_masked,[batch_size,1]),[1,num_deictic_patches])
        q_t_selected_target_expanded = tf.reshape(q_t_selected_target_tile2expand,[batch_size*num_deictic_patches,])

        match_onehot = tf.transpose(tf.one_hot(match_ph,tf.size(match_ph)))
        q_t_selected_target_tiled_vert = tf.tile(tf.reshape(q_t_selected_target_expanded,[1,batch_size*num_deictic_patches]),[batch_size*num_deictic_patches,1])
        
        min_values_of_groups = tf.reduce_min(match_onehot * q_t_selected_target_tiled_vert + (1 - match_onehot) * 999999,1)

        q_t_selected_target_expanded_min = tf.reshape(tf.matmul(tf.transpose(match_onehot),tf.expand_dims(min_values_of_groups,1)),[batch_size*num_deictic_patches,])
                
        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target_expanded_min)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                match_ph,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[td_error, min_values_of_groups, match_onehot],
            updates=[optimize_expr]
        )

        # Create callable functions
        trainWOUpdate = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                match_ph,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[match_onehot, min_values_of_groups, q_t_selected_target_expanded_min, q_t_selected_target_expanded]
        )

        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

#        return act_f, train, update_target, {'q_values': q_values}
#        return getq_f, train, trainWOUpdate, update_target, {'q_values': q_values}
#        return getq_f, train, trainWOUpdate, {'q_values': q_values}
        return getq_f, train, trainWOUpdate, update_target, {'q_values': q_values}


def build_train_deictic(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):

#    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    getq_f = build_getq(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1,1)

        q_tp1_best = tf.reshape(q_tp1_best, [32,25])
        q_tp1_best_reduced = tf.reduce_max(q_tp1_best,1)
        
        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_reduced
        q_t_selected_target_masked = (1.0 - done_mask_ph) * q_t_selected_target
        
        q_t_selected_target_tiled = tf.tile(tf.reshape(q_t_selected_target_masked,[32,1]),[1,25])
        q_t_selected_target_expanded = tf.reshape(q_t_selected_target_tiled,[800,])

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target_expanded)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )

        # Create callable functions
        trainWOUpdate = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[q_t_selected_target_expanded, errors, td_error]
        )

        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

#        return act_f, train, update_target, {'q_values': q_values}
#        return getq_f, train, trainWOUpdate, update_target, {'q_values': q_values}
#        return getq_f, train, trainWOUpdate, {'q_values': q_values}
        return getq_f, train, trainWOUpdate, update_target, {'q_values': q_values}

    
def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}
