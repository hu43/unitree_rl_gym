
import time
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os

os.environ["MUJOCO_GL"] = "egl"

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    print(f"Loading model: {xml_path}")
    print(f"Loading policy: {policy_path}")
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    policy = torch.jit.load(policy_path)
    print("Policy loaded successfully!")
    
    # Offscreen rendering
    renderer = mujoco.Renderer(m, 480, 480)
    
    start = time.time()
    sim_steps = 0
    last_print = 0
    
    while time.time() - start < simulation_duration:
        step_start = time.time()
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        
        # Render offscreen
        renderer.update_scene(d, camera="track")
        renderer.render()
        
        counter += 1
        sim_steps += 1
        
        if counter % control_decimation == 0:
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]
            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale
            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)
            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
        
        # Status print every 1000 steps (~2s sim time)
        if sim_steps - last_print >= 1000:
            elapsed = time.time() - start
            base_pos = d.qpos[0:3]
            print(f"  t={elapsed:.1f}s | pos=[{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}] | steps={sim_steps}")
            last_print = sim_steps
        
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    
    final_pos = d.qpos[0:3]
    elapsed = time.time() - start
    print(f"\n=== Sim2Sim COMPLETE ===")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Total steps: {sim_steps}")
    print(f"Final position: x={final_pos[0]:.2f}, y={final_pos[1]:.2f}, z={final_pos[2]:.2f}")
    print(f"G1 walked successfully in MuJoCo!")
    
    renderer.close()

