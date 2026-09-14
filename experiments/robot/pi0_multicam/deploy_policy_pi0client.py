"""deploy_policy.py — RoboTwin pi0 deploy via openpi WEBSOCKET CLIENT (decoupled from the sim env).
Replaces the original JAX in-process pi_model. The pi0.5 policy runs as a server (openpi env, port 8000,
with T3R/baseline hooks); this client (rtenv) sends the 3-camera obs and receives the action chunk.
"""
import numpy as np
import os, sys

from openpi_client import websocket_client_policy


class ClientPI0:
    def __init__(self, host="127.0.0.1", port=8000, pi0_step=50):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        self.pi0_step = int(pi0_step)
        self.instruction = None
        self.observation_window = None
        print(f"[client] connected to pi0.5 policy server {host}:{port}")

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"[client] instruction: {instruction}")

    def update_observation_window(self, img_arr, state):
        # img_arr order from encode_obs: [head_camera, right_camera, left_camera]
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        def chw(x):
            x = np.asarray(x)
            return np.transpose(x, (2, 0, 1)) if x.ndim == 3 and x.shape[-1] == 3 else x
        self.observation_window = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_high": chw(img_front),
                "cam_left_wrist": chw(img_left),
                "cam_right_wrist": chw(img_right),
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return np.asarray(self.client.infer(self.observation_window)["actions"])

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    host = usr_args.get("pi0_host", "127.0.0.1")
    port = usr_args.get("pi0_port", 8000)
    pi0_step = usr_args.get("pi0_step", 50)
    return ClientPI0(host=host, port=port, pi0_step=pi0_step)


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.pi0_step]
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
