# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import signal
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils.cuda_compat import check_cuda_runtime_compat
from legged_gym.utils import get_args, save_train_metadata, task_registry
import torch

def train(args, headless=True):
    check_cuda_runtime_compat()
    args.headless = headless
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    save_train_metadata(
        args=args,
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        env=env,
        runner=ppo_runner,
        actor_critic=getattr(ppo_runner.alg, "actor_critic", None),
        log_dir=getattr(ppo_runner, "log_dir", None),
    )
    def request_checkpoint_and_stop(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_checkpoint_and_stop)
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt as exc:
        iteration = int(getattr(ppo_runner, "current_learning_iteration", 0))
        checkpoint_path = os.path.join(ppo_runner.log_dir, f"model_{iteration}.pt")
        print(f"[graceful-stop] {exc}; saving {checkpoint_path}", flush=True)
        ppo_runner.save(checkpoint_path)
        if getattr(ppo_runner, "writer", None) is not None:
            ppo_runner.writer.flush()

if __name__ == '__main__':
    args = get_args()
    train(args, headless=True)
