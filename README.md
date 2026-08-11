# unitree_rl_gym

Fork from [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)，基于官方仓库扩展了 **G1/H1 人形机器人** 与 **Go2 四足机器人** 的真机部署与 MuJoCo 仿真支持。

---

## 新增功能

| 功能 | 文件 | 说明 |
|---|---|---|
| Go2 真机 RL 部署 | `deploy/deploy_real/deploy_go2.py` | 48-dim 观测，unitree_go DDS，本地运行 |
| G1 真机部署 | `deploy/deploy_real/deploy_real.py` + `configs/g1.yaml` | 官方已有，已在 p100 + G1 真机验证 |
| Go2 MuJoCo 仿真 | `simulate_python/unitree_mujoco.py` + `go2_rl_controller.py` | 双进程 Sim2Sim，VNC 可视化 |
| MuJoCo 无头录制 | `deploy/deploy_mujoco/deploy_mujoco_headless.py` | EGL + cv2 录视频 |
| G1 多视角场景 | `resources/robots/g1_description/scene.xml.cam_*.xml` | track / panorama 视角 |

---

## 环境要求

### p100 (部署主机)
- Ubuntu 22.04
- Python 3.10 + conda
- miniconda3 env: `hupy3.8` (mujoco 3.2.3, torch 2.3.1, unitree_sdk2py)
- 6× Tesla P100 (训练用)

### G1 机器人
- NVIDIA Jetson Orin NX (aarch64)
- Ubuntu 20.04
- Python 3.8 + venv `hurl`
- 连接方式：网线 p100 eno2 ↔ G1 eth0 (192.168.123.x)

### Go2 机器人
- 连接方式：网线 p100 eno2 ↔ Go2 网口 (192.168.123.x)
- 需关闭机器人自身控制模式后 RL 才能接管

---

## G1 真机部署

### 硬件
- 网线：p100 eno2 ↔ G1 eth0
- G1 吊装，遥控器 L2+R2 进入 debug 模式（阻尼态）

### p100 启动
```bash
ssh asano@192.168.4.78
conda activate hupy3.8
cd ~/hu/unitree_rl_gym
python deploy/deploy_real/deploy_real.py eno2 g1.yaml
```

### 遥控器
| 阶段 | 按键 |
|---|---|
| 零力矩态 | 启动后自动 |
| 默认位姿 | **start** |
| 运动控制 | **A** → 原地踏步 → 稳定后放绳 |
| 退出 | **select** 或 `ctrl+c` |

---

## Go2 真机部署

### 硬件
- 网线：p100 eno2 ↔ Go2 网口
- Go2 地面放置，先趴下

### 关闭机器人自控制（关键！）
Go2 默认有自身控制固件运行，需先关闭，否则 RL LowCmd 会被覆盖：
- 方法因固件版本而异，常见为关闭 Web 控制界面上的运动控制开关

### p100 启动
```bash
ssh asano@192.168.4.78
source ~/miniconda3/bin/activate hupy3.8
python ~/hu/unitree_rl_gym/deploy/deploy_real/deploy_go2.py eno2
```

或用脚本：
```bash
bash ~/deploy_go2.sh
```

### 可选参数
```bash
# 换 checkpoint
python deploy_go2.py eno2 --ckpt /path/to/model.pt

# 改速度指令
python deploy_go2.py eno2 --cmd "0.3 0 0"     # 慢速前进
python deploy_go2.py eno2 --cmd "0 0 0.3"     # 转向
python deploy_go2.py eno2 --cmd "-0.5 0 0"    # 后退

# 改时长
python deploy_go2.py eno2 --duration 30
```

### 安全
- `ctrl+c` 随时中断 → 阻尼模式
- 策略输出大但电机不动 = 机器人固件未关闭，需先处理

---

## Go2 MuJoCo Sim2Sim 可视化

### 启动仿真窗口（VNC :2）
```bash
ssh asano@192.168.4.78
bash ~/start_go2_sim.sh
```
> 在 VNC Viewer (192.168.4.78:5902) 中打开终端运行，会弹出 MuJoCo 3D 窗口。

### 启动策略控制器
```bash
# VNC 里另开终端
bash ~/start_go2_controller.sh
```

### 查看训练策略
```bash
# 换 checkpoint
export GO2_CKPT="~/hu/unitree_rl_gym/logs/rough_go2/Jul23_18-04-08_go2_gpu2/model_1500.pt"
export GO2_DURATION="60"
export GO2_CMD="0.5 0 0"
python ~/hu/unitree_mujoco/simulate_python/go2_rl_controller.py
```

---

## G1 MuJoCo 仿真

```bash
ssh asano@192.168.4.78
conda activate hupy3.8
cd ~/hu/unitree_rl_gym

# 官方预训练策略
unset MUJOCO_GL
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# 或你自己的训练策略（需确认兼容性）
python deploy/deploy_mujoco/deploy_mujoco.py g1_trained.yaml
```

---

## 训练

基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 框架：

```bash
conda activate hupy3.8
cd ~/hu/unitree_rl_gym
python legged_gym/scripts/train.py --task=g1    # G1 训练
python legged_gym/scripts/train.py --task=go2   # Go2 训练
```

---

## 目录结构

```
deploy/
  deploy_mujoco/          # MuJoCo 仿真部署
    configs/
      g1.yaml
      g1_short.yaml
      g1_trained.yaml
      h1.yaml
      h1_2.yaml
    deploy_mujoco.py
    deploy_mujoco_headless.py
    deploy_mujoco_offscreen.py
  deploy_real/            # 真机部署
    configs/
      g1.yaml
      g1_local.yaml       # 机器人本地运行配置
    deploy_real.py        # G1/H1 真机部署
    deploy_go2.py         # Go2 真机部署（新增）
legged_gym/
  envs/
    g1/                   # G1 环境配置
    go2/                  # Go2 环境配置
  scripts/
    train.py
    play.py
logs/
  g1/                     # G1 训练日志
  rough_go2/              # Go2 训练日志
resources/
  robots/
    g1_description/       # G1 URDF/MJCF
    go2/                  # Go2 URDF/MJCF
```

---

## 已知问题

1. **g1_trained.yaml** 的 `policy_lstm_1.pt` 与 `torch.jit.load` 不兼容，需额外处理 LSTM 状态
2. **Go2 base_lin_vel**：当前 deploy_go2.py 中 base_lin_vel 为 0（LowState 无 velocity 字段），实际行走时建议通过 SportModeState 获取
3. **Go2 真机需关闭自控制**：机器人固件会覆盖外部 LowCmd，部署前必须确认

---

## License

沿用原仓库 License。
