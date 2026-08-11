import time
import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

# Motor order: policy/qpos = FL,FR,RL,RR ; LowCmd = FR,FL,RR,RL
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
DEFAULT_ANGLES = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                           0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)

LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
KP = 20.0
KD = 0.5

low_state = unitree_go_msg_dds__LowState_()
have_low = False

def lowstate_cb(msg):
    global low_state, have_low
    low_state = msg
    have_low = True

def create_zero_cmd(cmd):
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x00
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0

def create_damping_cmd(cmd):
    for i in range(12):
        cmd.motor_cmd[i].mode = 0x00
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 1.0

def send_cmd(cmd, pub):
    cmd.crc = CRC().Crc(cmd)
    pub.Write(cmd)

def move_to_default(cmd, pub, control_dt):
    total_time = 2.0
    num_step = int(total_time / control_dt)
    init_pos = np.zeros(12, dtype=np.float32)
    for i in range(12):
        init_pos[i] = low_state.motor_state[PERM[i]].q
    for step in range(num_step):
        alpha = step / num_step
        for j in range(12):
            motor_idx = PERM[j]
            cmd.motor_cmd[motor_idx].q = init_pos[j] * (1 - alpha) + DEFAULT_ANGLES[j] * alpha
            cmd.motor_cmd[motor_idx].dq = 0.0
            cmd.motor_cmd[motor_idx].kp = KP
            cmd.motor_cmd[motor_idx].kd = KD
            cmd.motor_cmd[motor_idx].tau = 0.0
        send_cmd(cmd, pub)
        time.sleep(control_dt)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, nargs="?", default="eno2", help="network interface")
    parser.add_argument("--ckpt", type=str,
                        default="/home/asano/hu/unitree_rl_gym/logs/rough_go2/exported/policies/policy_1.pt",
                        help="policy checkpoint path")
    parser.add_argument("--duration", type=float, default=60.0, help="run duration in seconds")
    parser.add_argument("--cmd", type=str, default="0.5 0 0", help="command: vx vy yaw")
    args = parser.parse_args()

    policy = torch.jit.load(args.ckpt, map_location="cpu")
    policy.eval()
    print(f"Policy loaded: {args.ckpt}")

    cmd = np.array([float(x) for x in args.cmd.split()], dtype=np.float32)
    print(f"Command: {cmd}")

    ChannelFactoryInitialize(0, args.net)
    print(f"DDS initialized on {args.net}")

    pub = ChannelPublisher("rt/lowcmd", LowCmdGo)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowStateGo)
    sub.Init(lowstate_cb, 10)
    print("Publisher/Subscriber ready")

    print("Waiting for LowState...")
    t0 = time.time()
    while not have_low and time.time() - t0 < 10:
        time.sleep(0.01)
    if not have_low:
        print("ERROR: No LowState received. Is the robot powered on?")
        return
    print("LowState received. Ready.")

    lowcmd = unitree_go_msg_dds__LowCmd_()
    lowcmd.head[0] = 0xFE
    lowcmd.head[1] = 0xEF
    lowcmd.level_flag = 0xFF
    lowcmd.gpio = 0

    # Phase 1: Zero torque
    print("Entering ZERO TORQUE state. Waiting 3 seconds...")
    t_start = time.time()
    try:
        while time.time() - t_start < 0.5:
            create_zero_cmd(lowcmd)
            send_cmd(lowcmd, pub)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted during zero torque")
        create_damping_cmd(lowcmd)
        send_cmd(lowcmd, pub)
        return

    # Phase 2: Move to default
    print("Moving to DEFAULT POSITION...")
    move_to_default(lowcmd, pub, 0.02)
    print("At default position.")
    
    # DEBUG: check motor positions after moving to default
    print("DEBUG motor pos after default:")
    for i in range(12):
        print(f"  motor[{i}].q = {low_state.motor_state[i].q:.3f}")
    
    # DEBUG: check quat
    print("DEBUG quat:", [f"{x:.3f}" for x in low_state.imu_state.quaternion])
    
    # Phase 3: Run policy
    print("Starting POLICY CONTROL. Press Ctrl+C to stop.")
    action = np.zeros(12, dtype=np.float32)
    obs = np.zeros(48, dtype=np.float32)
    last_print = 0.0
    t_run = time.time()

    try:
        while time.time() - t_run < args.duration:
            now = time.time() - t_run

            mq = np.zeros(12, dtype=np.float32)
            mdq = np.zeros(12, dtype=np.float32)
            for i in range(12):
                mq[i] = low_state.motor_state[PERM[i]].q
                mdq[i] = low_state.motor_state[PERM[i]].dq

            quat = np.array(low_state.imu_state.quaternion, dtype=np.float32)
            gyro = np.array(low_state.imu_state.gyroscope, dtype=np.float32)

            qw, qx, qy, qz = quat
            g = np.zeros(3)
            g[0] = 2 * (-qz * qx + qw * qy)
            g[1] = -2 * (qz * qy + qw * qx)
            g[2] = 1 - 2 * (qw * qw + qz * qz)

            base_lin_vel = np.zeros(3, dtype=np.float32)

            obs[0:3] = base_lin_vel * LIN_VEL_SCALE
            obs[3:6] = gyro * ANG_VEL_SCALE
            obs[6:9] = g
            obs[9:12] = cmd * CMD_SCALE
            obs[12:24] = (mq - DEFAULT_ANGLES) * 1.0
            obs[24:36] = mdq * DOF_VEL_SCALE
            obs[36:48] = action

            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            action = np.clip(action, -100.0, 100.0)

            target_motor = (action * ACTION_SCALE + DEFAULT_ANGLES)[PERM]
            for i in range(12):
                lowcmd.motor_cmd[i].q = float(target_motor[i])
                lowcmd.motor_cmd[i].kp = KP
                lowcmd.motor_cmd[i].kd = KD
                lowcmd.motor_cmd[i].dq = 0.0
                lowcmd.motor_cmd[i].tau = 0.0
                lowcmd.motor_cmd[i].mode = 0x01
            send_cmd(lowcmd, pub)

            if now - last_print >= 1.0:
                print(f"  t={now:.1f}s | action_mean={np.mean(np.abs(action)):.3f}")
                print(f"    DEBUG target_motor[0:4] = {target_motor[0:4]}")
                print(f"    DEBUG motor[0].q = {low_state.motor_state[0].q:.3f}")
                print(f"    DEBUG gravity = [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]")
                last_print = now

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Policy interrupted.")

    print("Entering DAMPING mode.")
    create_damping_cmd(lowcmd)
    for _ in range(100):
        send_cmd(lowcmd, pub)
        time.sleep(0.02)
    print("Exit.")

if __name__ == "__main__":
    main()
