# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time

from copy import deepcopy
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.core import file_writer

import src.models as models

from src.env_utils import make_gym_env
from src.utils import get_batch, log, create_buffers

mp.set_sharing_strategy('file_system')

PolicyNet = models.PolicyNet
StateEmbeddingNet = models.StateEmbeddingNet
ForwardDynamicsNet = models.ForwardDynamicsNet
InverseDynamicsNet = models.InverseDynamicsNet

def init_models_and_states(flags):
    """Initialize models and LSTM states for all algorithms."""
    torch.manual_seed(flags.run_id)
    torch.cuda.manual_seed(flags.run_id)
    np.random.seed(flags.run_id)

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    # Only used for action_space and observation_space shapes
    env = make_gym_env(flags.env.split(',')[0], frame_stack=flags.frame_stack)
    frame_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()

    # Init models
    actor_model = PolicyNet(frame_shape, n_actions, flags.env) # No need for GPU since it is not updated
    learner_model = PolicyNet(frame_shape, n_actions, flags.env).to(device=flags.device)
    state_embedding_model = StateEmbeddingNet(frame_shape, flags.env).to(device=flags.device)
    forward_dynamics_model = ForwardDynamicsNet(n_actions).to(device=flags.device)
    inverse_dynamics_model = InverseDynamicsNet(n_actions).to(device=flags.device)
    random_target_network = StateEmbeddingNet(frame_shape, flags.env).to(device=flags.device)
    predictor_network = StateEmbeddingNet(frame_shape, flags.env).to(device=flags.device)

    actor_exploration_model = None
    learner_exploration_model = None
    if flags.checkpoint is not None and len(flags.checkpoint) > 0:
        if flags.model == 'vanilla':
            print('vanilla cannot load an exploration model')
            raise NotImplementedError

        print("loading model from", flags.checkpoint)
        checkpoint = torch.load(flags.checkpoint)
        if not flags.continue_learning:
            actor_exploration_model = PolicyNet(frame_shape, n_actions, flags.env, exploration_discount=flags.discount_exploration_start, discount_update=flags.discount_exploration_rate)
            actor_exploration_model.load_state_dict(checkpoint["actor_model_state_dict"])
            learner_exploration_model = deepcopy(actor_exploration_model).to(device=flags.device)
        else:  # When continuing to train transfer model
            actor_exploration_model = PolicyNet(frame_shape, n_actions, flags.env, exploration_discount=flags.discount_exploration_start, discount_update=flags.discount_exploration_rate)
            actor_exploration_model.load_state_dict(checkpoint["actor_exploration_model_state_dict"])
            learner_exploration_model = deepcopy(actor_exploration_model).to(device=flags.device)

    actor_model.share_memory()
    if actor_exploration_model:
        actor_exploration_model.share_memory()

    # Init LSTM states
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = actor_model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    initial_exploration_agent_state_buffers = None
    if actor_exploration_model:
        initial_exploration_agent_state_buffers = []
        for _ in range(flags.num_buffers):
            state = actor_exploration_model.initial_state(batch_size=1)
            for t in state:
                t.share_memory_()
            initial_exploration_agent_state_buffers.append(state)

    # Init optimizers
    learner_model_optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    state_embedding_optimizer = torch.optim.RMSprop(
        state_embedding_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    inverse_dynamics_optimizer = torch.optim.RMSprop(
        inverse_dynamics_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    forward_dynamics_optimizer = torch.optim.RMSprop(
        forward_dynamics_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    predictor_optimizer = torch.optim.RMSprop(
        predictor_network.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        x = np.maximum(flags.total_frames, 1e8)
        return 1 - min(epoch * flags.unroll_length * flags.batch_size, x) / x
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner_model_optimizer, lr_lambda)

    # policy_decay_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()

    if flags.checkpoint is not None and flags.continue_learning:
        checkpoint = torch.load(flags.checkpoint)
        if "actor_model_state_dict" in checkpoint:
            actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
            learner_model = deepcopy(actor_model).to(device=flags.device)
        if "state_embedding_model_state_dict" in checkpoint:
            state_embedding_model.load_state_dict(checkpoint["state_embedding_model_state_dict"])
        if "inverse_dynamics_model_state_dict" in checkpoint:
            inverse_dynamics_model.load_state_dict(checkpoint["inverse_dynamics_model_state_dict"])
        if "forward_dynamics_model_state_dict" in checkpoint:
            forward_dynamics_model.load_state_dict(checkpoint["forward_dynamics_model_state_dict"])
        if "random_target_network_state_dict" in checkpoint:
            random_target_network.load_state_dict(checkpoint["random_target_network_state_dict"])
        if "predictor_network_state_dict" in checkpoint:
            predictor_network.load_state_dict(checkpoint["predictor_network_state_dict"])
        if "learner_model_optimizer_state_dict" in checkpoint:
            learner_model_optimizer.load_state_dict(checkpoint["learner_model_optimizer_state_dict"])
        if "state_embedding_optimizer_state_dict" in checkpoint:
            state_embedding_optimizer.load_state_dict(checkpoint["state_embedding_optimizer_state_dict"])
        if "predictor_optimizer_state_dict" in checkpoint:
            predictor_optimizer.load_state_dict(checkpoint["predictor_optimizer_state_dict"])
        if "inverse_dynamics_optimizer_state_dict" in checkpoint:
            inverse_dynamics_optimizer.load_state_dict(checkpoint["inverse_dynamics_optimizer_state_dict"])
        if "forward_dynamics_optimizer_state_dict" in checkpoint:
            forward_dynamics_optimizer.load_state_dict(checkpoint["forward_dynamics_optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    buffers = create_buffers(frame_shape, learner_model.num_actions, flags)

    return dict(
        actor_model=actor_model,
        learner_model=learner_model,
        actor_exploration_model=actor_exploration_model,
        learner_exploration_model=learner_exploration_model,
        state_embedding_model=state_embedding_model,
        forward_dynamics_model=forward_dynamics_model,
        inverse_dynamics_model=inverse_dynamics_model,
        random_target_network=random_target_network,
        predictor_network=predictor_network,
        initial_agent_state_buffers=initial_agent_state_buffers,
        initial_exploration_agent_state_buffers=initial_exploration_agent_state_buffers,
        learner_model_optimizer=learner_model_optimizer,
        state_embedding_optimizer=state_embedding_optimizer,
        inverse_dynamics_optimizer=inverse_dynamics_optimizer,
        forward_dynamics_optimizer=forward_dynamics_optimizer,
        predictor_optimizer=predictor_optimizer,
        scheduler=scheduler,
        buffers=buffers,
        )
