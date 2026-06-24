## Learning-based Locomotion Control from OpenRobotLab
This repository contains learning-based locomotion control research from OpenRobotLab, currently including [Hybrid Internal Model](/projects/himloco/README.md) & [H-Infinity Locomotion Control](/projects/h_infinity/README.md).
## 🔥 News
- [2024-04] Code of HIMLoco is released.
- [2024-04] We release the [paper](https://arxiv.org/abs/2404.14405) of H-Infinity Locomotion Control. Please check the :point_right: [webpage](https://junfeng-long.github.io/HINF/) :point_left: and view our demos! :sparkler:
- [2024-01] HIMLoco is accepted by ICLR 2024.
- [2023-12] We release the [paper](https://arxiv.org/abs/2312.11460) of HIMLoco. Please check the :point_right: [webpage](https://junfeng-long.github.io/HIMLoco/) :point_left: and view our demos! :sparkler:

## 📝 TODO List
- \[x\] Release the training code of HIMLoco, please see `rsl_rl/rsl_rl/algorithms/him_ppo.py`.
- \[ \] Release deployment guidance of HIMLoco.
- \[ \] Release the training code of H-Infinity Locomotion Control.
- \[ \] Release deployment guidance of H-Infinity Locomotion Control.

## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

1. Create an environment and install PyTorch:

  - `conda create -n himloco python=3.7.16`
  - `conda activate himloco`
  - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`

3. Install optional web viewer dependencies:
  - `pip install flask imageio`

4. Clone this repository.

  - `git clone https://github.com/OpenRobotLab/HIMLoco.git`
  - `cd HIMLoco`


4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

**Note:** Please use legged_gym and rsl_rl provided in this repo, we have modefications on these repos.


### RTX 4090 / PyTorch compatibility

RTX 4090 GPUs require `sm_89` support. The original PyTorch `1.10.0+cu113` wheel can import on some systems but is not a reliable runtime for RTX 4090 training. Use an upgraded `hw` conda environment, preferably PyTorch `1.13.1+cu117` first, or PyTorch `2.0.1+cu118` if Isaac Gym is verified in that environment. The training and play entrypoints call `legged_gym.utils.cuda_compat.check_cuda_runtime_compat()` and fail early with an actionable message when RTX 4090 is paired with PyTorch 1.10. `TORCH_CUDA_ARCH_LIST` defaults to `8.9` for extension builds.

### Tutorial

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`
  - `python train.py --task go2w --headless --experiment_name go2w-stage0 --run_name 1 --stage 0 `

2. Tensorboard:

  - `cd /home/zhaozhuofan/test/HIMLoco-W/legged_gym`
  - `tensorboard --logdir logs`

3. Play and export the latest policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`
    - 默认会加载最新 checkpoint 并在 Isaac Gym 本地 viewer 中运行。
  - `python play.py --web`
    - 启动 Flask Web Viewer，在 `http://127.0.0.1:5000/` 打开浏览器查看。
    - 如果为无头机器，可使用 VS Code Live Preview 或浏览器访问此地址。
  - `python play.py --task go2w --load_run <run_name> --checkpoint <n>`
    - 直接加载指定训练目录与 checkpoint。
  - `python play.py --web --task go2w --load_run <run_name> --checkpoint <n>`
    - 以 web 模式播放指定任务的已有模型。

4. 如果需要导出 policy：

  - `play.py` 内置会在首次运行时导出 JIT 模型到 `logs/<experiment_name>/exported/policies`。


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{long2023him,
  title={Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response},
  author={Long, Junfeng and Wang, ZiRui and Li, Quanyi and Cao, Liu and Gao, Jiawei and Pang, Jiangmiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@misc{long2024hinf,
  title={Learning H-Infinity Locomotion Control}, 
  author={Junfeng Long and Wenye Yu and Quanyi Li and Zirui Wang and Dahua Lin and Jiangmiao Pang},
  year={2024},
  eprint={2404.14405},
  archivePrefix={arXiv},
}
```

## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements
- [legged_gym](https://github.com/leggedrobotics/legged_gym): Our codebase is built upon legged_gym.
