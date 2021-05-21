import functools

import tensorflow as tf
import os
from absl import logging
from absl import app
from tf_agents.agents.sac import sac_agent
from tf_agents.system.system_multiprocessing import handle_main
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver
from tf_agents.drivers import dynamic_episode_driver
from absl import flags
from new_SFC import SFC_run
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import time
import csv
import numpy as np
import copy
from tf_agents.agents.ddpg import critic_rnn_network, critic_network
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'root directory for writing summaries and checkpoints')
FLAGS = flags.FLAGS

###################### HYPER_params ####################################

# Params for collect
FLAGS.root_dir = "./PPO_test"
num_environment_steps = 300000000
collect_episodes_per_iteration = 6
num_parallel_environments = 16
replay_buffer_capacity = 2000  # Per-environment
# Params for train
num_epochs = 25
learning_rate = 3e-4
# Params for eval
num_eval_episodes = 40
eval_interval = 500

# Params for summaries and logging
checkpoint_interval = 500
log_interval = 50
summary_interval = 50
summaries_flush_secs = 1
use_tf_functions = True

fieldnames = ["episode", "critical_accuracy", "warning_accuracy", "normal_accuracy", "Proactive", "Reactive"]

target_update_tau = 0.005  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()


####################### DEFINING Networks ################################

def create_networks(tf_env):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        input_fc_layer_params=(512, 512,),

        lstm_size=(100, 100,),
        # rnn_construction_fn=(layers.LSTM(256 , return_sequences=True), layers.LSTM(256)),
        output_fc_layer_params=(256, 256,))

    value_net = critic_rnn_network.CriticRnnNetwork((tf_env.observation_spec(),tf_env.action_spec()),
                                                    joint_fc_layer_params=(512, 512,),

                                                    lstm_size=(100, 100,),
                                                    output_fc_layer_params=(256, 256,),

                                                    )


    return actor_net, value_net


def monitoring_policy(policy, py_environment, tf_environment, num_episodes=10):
    for _ in range(num_episodes):

        time_step = tf_environment.reset()
        py_environment.monitor_mode = True

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_environment.step(action_step.action)
            if time_step.is_last():
                print(time_step)
                py_environment.monitor()


####################### TRAIN_FUNC #################################

def train(root_dir):
    ######### DIRECTORY #################
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, "TRAIN")
    eval_dir = os.path.join(root_dir, "EVAL")
    saved_model_dir = os.path.join(root_dir, "Policy Saved Model")

    ########### summary And Metric Objects ################

    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    ######################################################################################3
    global_step = tf.compat.v1.train.get_or_create_global_step()

    with tf.summary.record_if(global_step % summary_interval == 0):

        ################# ENV_setup ###########################
        eval_py_env = SFC_run()

        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment([SFC_run] * num_parallel_environments))
        ################# AGENT ################################

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        actor_net, critic_net = create_networks(tf_env)

        tf_agent = sac_agent.SacAgent(time_step_spec=tf_env.time_step_spec(),
                                      action_spec=tf_env.action_spec(),
                                      actor_network=actor_net,
                                      critic_network=critic_net,
                                      actor_optimizer=optimizer,
                                      critic_optimizer=optimizer,
                                      alpha_optimizer=optimizer,
                                      target_update_tau=target_update_tau,
                                      target_update_period=target_update_period,
                                      td_errors_loss_fn=tf.math.squared_difference,
                                      gamma=gamma,
                                      reward_scale_factor=reward_scale_factor,
                                      )

        tf_agent.initialize()

        ################## TRAIN METRICS ##############################

        environment_step_metrics = tf_metrics.EnvironmentSteps()
        step_metrics = [tf_metrics.NumberOfEpisodes(),
                        environment_step_metrics, ]

        train_metrics = step_metrics + [
            tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),
        ]
        ################ POLICY ##########################################

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        ############### REPLAY_BUFFER ####################################

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity
        )
        ############# CHECKPOINTER_OBJECTS ################################

        train_checkpointer = common.Checkpointer(ckpt_dir=train_dir, agent=tf_agent,
                                                 global_step=global_step,
                                                 metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics")
                                                 )

        policy_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(root_dir, "POLICY"),
                                                  policy=eval_policy, global_step=global_step,
                                                  )

        saved_model = policy_saver.PolicySaver(policy=eval_policy, train_step=global_step)

        train_checkpointer.initialize_or_restore()

        ############## TRAINING_FUNCS ##########################################

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env, collect_policy,
                                                                     observers=[replay_buffer.add_batch] +
                                                                               train_metrics,
                                                                     num_episodes=collect_episodes_per_iteration)

        def train_step():
            trajectories = replay_buffer.gather_all()
            return tf_agent.train(experience=trajectories)

        def evaluate():
            metric_utils.eager_compute(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, global_step,
                                       eval_summary_writer, 'Metrics')

        def monitoring_policy(policy, py_environment, tf_environment, num_episodes=10, episode=global_step.numpy()):
            crit = 0
            warn = 0
            norm = 0
            k_crit = 0.0001
            k_warn = 0
            k_norm = 0
            reactive = 0
            proactive = 0
            fail = 0
            for _ in range(num_episodes):

                time_step = tf_environment.reset()
                state = policy.get_initial_state(tf_environment.batch_size)

                # py_environment.monitor_mode = True

                while not time_step.is_last():
                    st = copy.deepcopy(py_environment.VNF_state)
                    policy_step = policy.action(time_step, state)
                    state = policy_step.state
                    a = policy_step.action
                    action_space = np.array(a).reshape(py_environment.sfc_map.shape)
                    time_step = tf_environment.step(a)
                    for idx, val in np.ndenumerate(st):
                        if val == 2:
                            fail += 1
                            crit += 1
                            if -0.3<=action_space[idx] < 0.3:

                                k_crit += 1
                                if py_environment.backup_map[idx] != -1:
                                    proactive += 1
                                else:
                                    reactive += 1
                        elif val == 1:
                            warn += 1
                            if -1<=action_space[idx] < -0.3:
                                k_warn += 1
                        else:
                            norm += 1
                            if 0.3<= action_space[idx] <=1:
                                k_norm += 1
                    # py_environment.availability_ratio()
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                # "critical_accuracy", "warning_accuracy", "normal_accuracy", "Proactive", "Reactive"
                info = {
                    "episode": episode,
                    "critical_accuracy": k_crit / crit,
                    "warning_accuracy": k_warn / warn,
                    "normal_accuracy": k_norm / norm,
                    "Proactive": proactive / k_crit,
                    "Reactive": reactive / k_crit
                }

                csv_writer.writerow(info)

        if use_tf_functions:
            # TODO(b/123828980): Enable once the cause for slowdown was identified.
            collect_driver.run = common.function(collect_driver.run, autograph=True)
            tf_agent.train = common.function(tf_agent.train, autograph=True)
            train_step = common.function(train_step)

        ##################### TRAINING !!!!!!!!########################

        collect_time = 0
        train_time = 0
        time_at_step = global_step.numpy()

        while environment_step_metrics.result() < num_environment_steps:

            start_time = time.time()
            collect_driver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss = train_step()

            replay_buffer.clear()
            train_time = + time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

            global_step_val = global_step.numpy()

            if global_step_val % log_interval == 0:
                logging.info("step = %d , loss = %f", global_step_val, total_loss.loss)
                step_per_sec = ((global_step_val - time_at_step) / (collect_time + train_time))
                logging.info("%.3f steps/sec", step_per_sec)
                logging.info("collect time = {} , train time = {}".format(collect_time, train_time))

                with tf.summary.record_if(True):
                    tf.summary.scalar(name="global step per sec", data=step_per_sec, step=global_step)

                time_at_step = global_step_val
                collect_time = 0
                train_time = 0
            if global_step_val % eval_interval == 0 and global_step_val > 0:
                evaluate()

            if global_step_val % 50 == 0 and global_step_val > 0:
                monitoring_policy(eval_policy, eval_py_env, eval_tf_env, episode=global_step_val)

            # if global_step_val % 100 == 0 and global_step_val > 0:
            #     monitoring_policy(eval_policy,eval_py_env, eval_tf_env , episode=global_step_val)

            if global_step_val % checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)
                policy_checkpointer.save(global_step=global_step_val)
                saved_model_path = os.path.join(saved_model_dir, "Policy_" + ("%d" % global_step_val).zfill(9))
                saved_model.save(saved_model_path)

    evaluate()
    # monitoring_policy(eval_policy, eval_py_env, eval_tf_env)


def main(_):
    logging.set_verbosity(logging.INFO)

    if FLAGS.root_dir is None:
        raise AttributeError("train_eval requires a root dir")

    train(root_dir=FLAGS.root_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("root_dir")
    handle_main(
        functools.partial(app.run, main))
