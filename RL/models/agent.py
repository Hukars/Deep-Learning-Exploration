# -*-coding:utf-8-*-
# from multiprocessing import Process, Queue
import multiprocessing
import sys
import time
from RL.utils import *


class Agent:
    def __init__(self, env, policy, running_state=None, render=False, num_threads=4):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.render = render
        self.num_thread = num_threads

    def collect_samples(self, pid, queue, total_steps):
        print(pid)
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        max_reward = -sys.maxsize
        min_reward = sys.maxsize
        log = dict()
        memory = Memory()

        while num_steps < total_steps:
            trajectory_reward = 0
            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            state = torch.tensor(state).float().unsqueeze(0)
            for t in range(5000):
                if self.render:
                    self.env.render()
                # select action
                with torch.no_grad():
                    if self.policy.is_disc_actions:
                        action = self.policy.select_action(state)
                    else:
                        action = self.policy.select_action(state) # [0].numpy().astype(np.float64)
                old_log_prob = self.policy.log_prob(state, action)
                next_state, reward, done, _ = self.env.step(action[0].numpy())

                trajectory_reward += reward
                if self.running_state is not None:
                    next_state = self.running_state(next_state)

                reward = torch.tensor([reward]).float()
                mask = torch.tensor([0.0] if done else [1.0])
                next_state = torch.tensor(next_state).float().unsqueeze(0)
                memory.push(state, action, reward, next_state, mask, old_log_prob)
                state = next_state

                if done:
                    break

            num_steps += (t+1)
            num_episodes += 1
            total_reward += trajectory_reward
            max_reward = max(max_reward, trajectory_reward)
            min_reward = min(min_reward, trajectory_reward)

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['avg_steps'] = num_steps / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward
        print('here', pid)
        if queue is not None:
            queue.put([pid, memory, log])
            print('queue')
        else:
            return memory, log

    def process_sample(self, batch_size):
        t_start = time.time()
        mini_batch_size = int(batch_size / self.num_thread)
        my_queue = multiprocessing.Queue()
        workers = []
        self.policy.to(torch.device('cpu'))

        # for i in range(self.num_thread):
        #     args = (i+1, queue, mini_batch_size)
        #     workers.append(Process(target=self.collect_samples, args=args))
        # print("Get here1!")
        # for p in workers:
        #     p.start()
        #
        # memories = [None] * self.num_thread
        # logs = [None] * self.num_thread
        # print("Get here2!")
        # for _ in range(self.num_thread):
        #     information = queue.get()
        #     memories[information[0]-1] = information[1]
        #     logs[information[0]-1] = information[2]
        # print("Get here3!")

        for i in range(self.num_thread - 1):
            args = (i+1, my_queue, mini_batch_size)
            workers.append(multiprocessing.Process(target=self.collect_samples, args=args))
        print(len(workers))
        for worker in workers:
            worker.start()
        memory, log = self.collect_samples(0, None, mini_batch_size)
        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = my_queue.get()
            print("Get here4!")
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        print("Get here3!")
        for worker_memory in worker_memories:
            memory.append(worker_memory)


        # combine all the information for returning
        # combined_memory = memories[0]
        # for j in range(len(memories)-1):
        #     combined_memory.append(memories[j+1])
        # batch = combined_memory.sample()
        batch = memory.sample()

        combined_log = dict()
        combined_log['num_step'] = sum([log['num_steps'] for log in logs])
        combined_log['num_episodes'] = sum([log['num_episodes'] for log in logs])
        combined_log['total_reward'] = sum([log['total_reward'] for log in logs])
        combined_log['avg_reward'] = combined_log['total_reward'] / combined_log['num_episodes']
        combined_log['avg_steps'] = combined_log['num_step'] / combined_log['num_episodes']
        combined_log['max_reward'] = max([log['max_reward'] for log in logs])
        combined_log['min_reward'] = min([log['min_reward'] for log in logs])

        self.policy.to(torch.device('cuda'))
        t_end = time.time()
        combined_log['sample_time'] = t_end - t_start
        return batch, combined_log




