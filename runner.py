from absl import logging
import numpy as np
import tensorflow as tf
from edited_SFC_functions import SFC_run
from tf_agents.environments import tf_py_environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies import tf_policy
from tf_agents.trajectories.policy_step import PolicyStep
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


num_episodes = 1

saved_policy = tf.saved_model.load("Policy_001226000")
policy = saved_policy
x = np.zeros(10)
backup = np.zeros(10)
env = SFC_run()
env_tf = tf_py_environment.TFPyEnvironment(env)
list=[]
for episode in range(num_episodes):
    print("Generating episode %d of %d" % (episode, num_episodes))


    time_step = env_tf.reset()
    s=0

    predict_step = 0
    backup_exist = 0
    state = policy.get_initial_state(env_tf.batch_size)
    print(state)

    env.monitor()
    while not time_step.is_last():
        print(env.time_step_spec().observation)

        policy_step: PolicyStep = policy.action(time_step, state)
        state = policy_step.state
        a =policy_step.action
        print(a)
        s +=1
        # print(policy_step)
        if any(fail ==1 for fail in np.nditer(env.failure)):
            predict_step +=1
            for idx , val in np.ndenumerate(env.failure):
                if val ==1:
                    if env.backup_map[idx]==1:
                        backup_exist +=1


            print("FAIL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        time_step = env_tf.step(policy_step.action)
        env.monitor()
        # print(env.th_store)


    print(env.cri)
    print(time_step)

    x[predict_step] +=1
    backup[predict_step] += 1

    print (x)
    print(backup)



    print(s)


logging.set_verbosity(logging.INFO)

u = np.arange(10)
v = x/num_episodes
y = backup /num_episodes
plt.bar(u +0.00 ,v , color="b" , width = 0.25 , alpha=0.5)
plt.bar(u+0.25 , y , color="r", width = 0.25, alpha=0.5)

plt.ylabel('Predicted Percent')
plt.xlabel("n'th failure in Episode")
plt.title('Failure Prediction')

plt.show()




