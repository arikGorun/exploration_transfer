# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import timeit
import pprint

from copy import deepcopy
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.utils import get_batch, log, create_buffers, act
from src.init_models_and_states import init_models_and_states


def learn(actor_model,
          learner_model,
          exploration_model,
          state_embedding_model,
          inverse_dynamics_model,
          random_target_network,
          predictor_network,
          batch,
          initial_agent_state,
          initial_exploration_agent_state,
          optimizer,
          state_embedding_optimizer,
          inverse_dynamics_optimizer,
          predictor_optimizer,
          scheduler,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        episodic_bonus = batch['bonus_reward'][1:]
        intrinsic_rewards = episodic_bonus

        random_embedding_next = random_target_network(batch['frame'][1:].detach())
        predicted_embedding_next = predictor_network(batch['frame'][1:].detach())
        intrinsic_rewards_rnd = torch.norm(predicted_embedding_next.detach() - random_embedding_next.detach(),
                                                   dim=2, p=2)

        intrinsic_rewards_rnd *= intrinsic_rewards

        rnd_loss = flags.rnd_loss_coef * \
                   losses.compute_forward_dynamics_loss(predicted_embedding_next.detach(), random_embedding_next.detach())

        state_emb = state_embedding_model(batch['frame'][:-1].to(device=flags.device))
        next_state_emb = state_embedding_model(batch['frame'][1:].to(device=flags.device))

        pred_actions = inverse_dynamics_model(state_emb, next_state_emb)
        entropy_emb_actions = losses.compute_entropy_loss(pred_actions)


        intrinsic_reward_coef = flags.intrinsic_reward_coef
        intrinsic_rewards *= intrinsic_reward_coef

        inverse_dynamics_loss = flags.inverse_loss_coef * \
            losses.compute_inverse_dynamics_loss(pred_actions, batch['action'][1:])

        num_samples = flags.unroll_length * flags.batch_size
        actions_flat = batch['action'][1:].reshape(num_samples).cpu().detach().numpy()
        intrinsic_rewards_flat = intrinsic_rewards.reshape(num_samples).cpu().detach().numpy()

        exploration_logits = None
        if exploration_model is not None:
            with torch.no_grad():
                exploration_outputs, unused_state = exploration_model(batch, initial_exploration_agent_state)
                exploration_logits = exploration_outputs['policy_logits']
                # decay *= flags.exploration_model_decay TODO make scheduler for decay of exploration weight
        learner_outputs, unused_state = learner_model(batch, initial_agent_state, exploration_logits)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        elif flags.intrinsic_reward_coef > 0.:
            total_rewards = rewards + intrinsic_rewards
        else:
            total_rewards = rewards
        clipped_rewards = torch.clamp(total_rewards, -1, 1)

        discounts = (1 - batch['real_done'].float()).abs() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss + inverse_dynamics_loss + rnd_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_lengths = batch['episode_step'][batch['done']]
        episode_wins = batch['episode_win'][batch['done']]
        visited_states = batch['visited_states']
        stats = {
            'total_episodes': torch.sum(batch['done'].float()).item(),
            'episode_wins': torch.sum(episode_wins).item() / flags.batch_size,
            'visited_states': torch.max(visited_states).item(),
            'mean_episode_length': torch.mean(episode_lengths.float()).item(),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'inverse_dynamics_loss': inverse_dynamics_loss.item(),
            'rnd_loss': rnd_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
        }

        scheduler.step()
        optimizer.zero_grad()
        state_embedding_optimizer.zero_grad()
        inverse_dynamics_optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(learner_model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(state_embedding_model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(inverse_dynamics_model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
        optimizer.step()
        state_embedding_optimizer.step()
        inverse_dynamics_optimizer.step()
        predictor_optimizer.step()

        actor_model.load_state_dict(learner_model.state_dict())
        return stats


def train(flags):
    if flags.xpid is None:
        flags.xpid = 'RNDxE3B-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    models_and_states = init_models_and_states(flags)
    actor_model = models_and_states['actor_model']
    learner_model = models_and_states['learner_model']
    actor_exploration_model = models_and_states['actor_exploration_model']
    learner_exploration_model = models_and_states['learner_exploration_model']
    state_embedding_model = models_and_states['state_embedding_model']
    inverse_dynamics_model = models_and_states['inverse_dynamics_model']
    random_target_network = models_and_states['random_target_network']
    predictor_network = models_and_states['predictor_network']
    initial_agent_state_buffers = models_and_states['initial_agent_state_buffers']
    initial_exploration_agent_state_buffers = models_and_states['initial_exploration_agent_state_buffers']
    learner_model_optimizer = models_and_states['learner_model_optimizer']
    state_embedding_optimizer = models_and_states['state_embedding_optimizer']
    inverse_dynamics_optimizer = models_and_states['inverse_dynamics_optimizer']
    predictor_optimizer = models_and_states['predictor_optimizer']
    scheduler = models_and_states['scheduler']
    buffers = models_and_states['buffers']


    actor_processes = []
    ctx = mp.get_context(flags.mp_start)
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, actor_model, buffers,
                  dict(), dict(),
                  dict(), dict(),
                  initial_agent_state_buffers, flags,
                  actor_exploration_model, initial_exploration_agent_state_buffers))
        actor.start()
        actor_processes.append(actor)

    logger = logging.getLogger('logfile')

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state, exploration_agent_state = get_batch(free_queue, full_queue, buffers,
                initial_agent_state_buffers, flags, timings, initial_exploration_agent_state_buffers)
            stats = learn(actor_model, learner_model, learner_exploration_model,
                          state_embedding_model, inverse_dynamics_model, random_target_network,
                          predictor_network,
                          batch, agent_state, exploration_agent_state,
                          learner_model_optimizer,
                          state_embedding_optimizer, inverse_dynamics_optimizer,
                          predictor_optimizer,
                          scheduler, flags)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stats.keys()})
                plogger.log(to_log)
                frames += flags.unroll_length * flags.batch_size

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'actor_model_state_dict': actor_model.state_dict(),
            'state_embedding_model_state_dict': state_embedding_model.state_dict(),
            'inverse_dynamics_model_state_dict': inverse_dynamics_model.state_dict(),
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'actor_exploration_model_state_dict': actor_exploration_model.state_dict() if actor_exploration_model is not None else None,
            'learner_model_optimizer_state_dict': learner_model_optimizer.state_dict(),
            'state_embedding_optimizer_state_dict': state_embedding_optimizer.state_dict(),
            'inverse_dynamics_optimizer_state_dict': inverse_dynamics_optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)

            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''

            total_loss = stats.get('total_loss', float('inf'))
            if stats:
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f. \n Stats \n %s', \
                        frames, total_loss, fps, stats['mean_episode_return'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(frames)
    plogger.close()
