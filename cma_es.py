import argparse
import cma
from datetime import datetime
import gym
from gym.wrappers import Monitor
import logging
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import os


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """CMA-ES wrapper."""
    def __init__(self,
                 num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 popsize=255,  # population size
                 weight_decay=0.01):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize,})

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(self.solutions,
                     (reward_table).tolist())  # convert minimizer to maximizer.

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


class Model(object):
    """NN model."""
    def __init__(self, config):
        self.W = []
        self.b = []
        self.activations = []
        self.num_params = 0
        input_size = config.input_size
        for layer_size in config.layers:
            self.W.append(np.random.randn(layer_size, input_size))
            self.b.append(np.zeros(layer_size))
            self.num_params += layer_size * (input_size + 1)
            input_size = layer_size
            if config.activation == 'relu':
                self.activations.append(lambda x: max(0, x))
            elif config.activation == 'tanh':
                self.activations.append(np.tanh)
            else:
                self.activations.append(lambda x: x)
        self.W.append(np.random.randn(config.output_size, input_size))
        self.b.append(np.zeros(config.output_size))
        self.num_params += config.output_size * (input_size + 1)
        self.activations.append(np.tanh)

    def forward(self, inputs):
        x = inputs
        for w, b, activation in zip(self.W, self.b, self.activations):
            x = activation(w.dot(x) + b)
        return x

    def get_params(self):
        params = [np.concatenate(x.copy().ravel(), y.copy())
                  for x, y in zip(self.W, self.b)]
        return np.concatenate(params)

    def set_params(self, params):
        n = len(self.W)
        offset = 0
        for i in range(n):
            o_size, i_size = self.W[i].shape
            param_size = o_size * i_size
            self.W[i] = params[offset:(offset + param_size)].reshape(
                [o_size, i_size])
            offset += param_size
            self.b[i] = params[offset:(offset + o_size)]
            offset += o_size


def play(env, model, n=1):
    """Rollouts with the given model."""
    rewards = []
    for _ in range(n):
        ob = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = model.forward(ob)
            ob, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                rewards.append(ep_reward)
    return rewards


def rollout(config, worker_id, downstream_q, upstream_q):
    """Do rollout."""
    config.logger.info('Worker {} started.'.format(worker_id))
    env = gym.make(config.env_name)
    env.seed(worker_id + 73)
    model = Model(config)
    try:
        while True:
            params, ix = downstream_q.get()
            model.set_params(params)
            if config.eval_flag:
                rewards = play(env, model)
            else:
                rewards = play(env, model, config.repeat_per_solution)
            upstream_q.put((np.mean(rewards), ix))
    except TypeError:
        config.logger.info('Worker {} quits.'.format(worker_id))
    finally:
        env.close()


def do_rollouts(workers, params, config, main_q, eval_flag=False):
    """Workers do rollouts."""
    total_rollouts = config.eval_rounds if eval_flag else config.population_size
    config.eval_flag = eval_flag
    # distribute work
    cnt = 0
    while cnt < total_rollouts:
        for _, worker_q in workers:
            worker_q.put((params[cnt], cnt))
            cnt += 1
            if cnt >= total_rollouts:
                break
    # collect fitness
    rewards = np.zeros(total_rollouts)
    cnt = 0
    while cnt < total_rollouts:
        r, ix = main_q.get()
        rewards[ix] = r
        cnt += 1
    return rewards


def test(config, workers, main_q, env):
    """Test an ES model."""
    logger = config.logger
    model = Model(config)
    model_file = os.path.join(config.job_dir, 'model.npz')
    data = np.load(model_file)
    if 'params' in data:
        params = data['params']
    else:
        params = data['arr_0']
    model.set_params(params)
    n = config.eval_rounds - config.n_video_episodes
    config.eval_rounds = n
    rewards = do_rollouts(workers, [params] * n, config, main_q, True).tolist()
    rewards += play(env, model, config.n_video_episodes)
    logger.info('TEST: r.len={}, r.mean={}, r.sd={}'.format(
        len(rewards), np.mean(rewards), np.std(rewards)))


def train(config, workers, main_q):
    """Train an ES model."""
    logger = config.logger
    model = Model(config)
    logger.info('#params={}'.format(model.num_params))
    solver = CMAES(num_params=model.num_params, popsize=config.population_size)
    for n_iter in range(config.max_iters):
        # rollouts
        params = solver.ask()
        rewards = do_rollouts(workers, params, config, main_q)
        # policy update
        solver.tell(rewards)
        logger.info(
            'n={}, r.len={}, r.mean={}, r.sd={}, r.max={}, r.min={}'.format(
                n_iter + 1, len(rewards), np.mean(rewards), np.std(rewards),
                np.max(rewards), np.min(rewards)))
        # evaluation
        if (n_iter + 1) % config.eval_interval == 0:
            best_param = solver.best_param()
            n = config.eval_rounds
            rewards = do_rollouts(
                workers, [best_param] * n, config, main_q, True)
            logger.info(
                'EVAL: r.len={}, r.mean={}, r.sd={}'.format(
                    len(rewards), np.mean(rewards), np.std(rewards)))
            params_file = os.path.join(config.job_dir, 'model.npz')
            np.savez(params_file, params=best_param)
            logger.info('Model parameters saved to {}.'.format(params_file))
            if np.mean(rewards) >= config.reward_target:
                break


def main(config):
    logger = get_logger(config)
    for k, v in config.__dict__.iteritems():
        logger.info('{}: {}'.format(k, v))
    config.logger = logger
    config.eval_flag = False

    env = gym.make(config.env_name)
    env.seed(config.seed)
    obv_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    config.input_size = obv_dim
    config.output_size = act_dim

    # initialize workers
    workers = []
    main_q = Queue()
    for i in range(config.num_workers):
        worker_q = Queue()
        worker = Process(target=rollout, args=(config, i, worker_q, main_q))
        worker.start()
        workers.append((worker, worker_q))

    try:
        if config.test:
            env = Monitor(
                env, config.job_dir, force=True, video_callable=lambda x: True)
            test(config, workers, main_q, env)
        else:
            train(config, workers, main_q)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
    finally:
        for _, worker_q in workers:
            worker_q.put(None)
        for worker, _ in workers:
            worker.join()
        env.close()


def get_logger(config):
    """Create a logger."""
    if not os.path.exists(config.job_dir):
        os.makedirs(config.job_dir)
    log_format = '%(asctime)s pid=%(process)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger('main')
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    if config.test:
        log_file = os.path.join(
            config.job_dir, 'test_{}.txt'.format(current_time))
    else:
        log_file = os.path.join(
            config.job_dir, 'train_{}.txt'.format(current_time))
    file_hdl = logging.FileHandler(log_file)
    formatter = logging.Formatter(fmt=log_format)
    file_hdl.setFormatter(formatter)
    logger.addHandler(file_hdl)
    return logger


def parse_args():
    """Parse command line argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='Job directory.',
        default='./es_results',
    )
    parser.add_argument(
        '--env-name',
        help='Name of gym environment.',
        default='BipedalWalkerHardcore-v2',
    )
    parser.add_argument(
        '--population-size',
        help='Size of population.',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--num-workers',
        help='Number of workers.',
        type=int,
        default=96,
    )
    parser.add_argument(
        '--activation',
        help='Activation unit.',
        choices={'relu', 'tanh', 'linear'},
        default='tanh',
    )
    parser.add_argument(
        '--layers',
        help='#Units in each layer.',
        default='40,40',
    )
    parser.add_argument(
        '--max-iters',
        help='Number of iterations to train.',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--eval-interval',
        help='Evaluation interval (in episodes).',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--repeat-per-solution',
        help='Number of rollouts for each candidate solution.',
        type=int,
        default=16
    )
    parser.add_argument(
        '--seed',
        help='Random seed for environments.',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--reward-target',
        help='Target reward.',
        type=float,
        default=300.0,
    )
    parser.add_argument(
        '--eval-rounds',
        help='Number of rounds to evaluate.',
        type=int,
        default=100
    )
    parser.add_argument(
        '--test',
        help='Whether to test a trained model.',
        action='store_true',
    )
    parser.add_argument(
        '--n-video-episodes',
        help='Number of test videos to record.',
        type=int,
        default=5,
    )
    args, _ = parser.parse_known_args()
    args.layers = [int(x) for x in args.layers.split(',')]
    return args


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
