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

from typing import List, Optional

import logging
import math
import threading

import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

try:
    import flask
except ImportError:
    flask = None

try:
    import imageio
    import isaacgym
    import isaacgym.torch_utils as torch_utils
    from isaacgym import gymapi
except ImportError:
    imageio = None
    isaacgym = None
    torch_utils = None
    gymapi = None


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        """
        if flask is None:
            raise ImportError("Flask is required for WebViewer. Install it with `pip install flask`.")
        if imageio is None:
            raise ImportError("imageio is required for WebViewer. Install it with `pip install imageio`.")

        self._app = flask.Flask(__name__)
        self._app.add_url_rule("/", view_func=self._route_index)
        self._app.add_url_rule("/_route_stream", view_func=self._route_stream)
        self._app.add_url_rule("/_route_stream_depth", view_func=self._route_stream_depth)
        self._app.add_url_rule("/_route_input_event", view_func=self._route_input_event, methods=["POST"])

        self._log = logging.getLogger('werkzeug')
        self._log.disabled = True
        self._app.logger.disabled = True

        self._image = None
        self._image_depth = None
        self._camera_id = 0
        self._camera_type = gymapi.IMAGE_COLOR
        self._notified = False
        self._wait_for_page = True
        self._pause_stream = False
        self._event_load = threading.Event()
        self._event_stream = threading.Event()
        self._event_stream_depth = threading.Event()

        # start server
        self._thread = threading.Thread(target=lambda: \
            self._app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
        self._thread.start()
        print(f"\nStarting web viewer on http://{host}:{port}/\n")

    def _route_index(self) -> 'flask.Response':
        """Render the web page"""
        file_path = os.path.join(os.path.dirname(__file__), "webviewer.html")
        with open(file_path, 'r', encoding='utf-8') as file:
            template = file.read()
        self._event_load.set()
        return flask.render_template_string(template)

    def _route_stream(self) -> 'flask.Response':
        """Stream the image to the web page"""
        return flask.Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_stream_depth(self) -> 'flask.Response':
        return flask.Response(self._stream_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_input_event(self) -> 'flask.Response':
        data = flask.request.get_json()
        key, mouse = data.get("key", None), data.get("mouse", None)
        dx, dy, dz = data.get("dx", None), data.get("dy", None), data.get("dz", None)

        transform = self._gym.get_camera_transform(self._sim,
                                                   self._envs[self._camera_id],
                                                   self._cameras[self._camera_id])

        if mouse == "wheel":
            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            r += 0.05 * dz
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif mouse == "left":
            dx *= 0.2 * math.pi / 180
            dy *= 0.2 * math.pi / 180
            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta -= dy
            phi -= dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif mouse == "right":
            dx *= -0.2 * math.pi / 180
            dy *= -0.2 * math.pi / 180
            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta += dy
            phi += dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif key == 219:  # prev
            self._camera_id = (self._camera_id-1) % self._env.num_envs
            return flask.Response(status=200)
        elif key == 221:  # next
            self._camera_id = (self._camera_id+1) % self._env.num_envs
            return flask.Response(status=200)
        elif key == 86:
            self._pause_stream = not self._pause_stream
            return flask.Response(status=200)
        elif key == 84:
            if self._camera_type == gymapi.IMAGE_COLOR:
                self._camera_type = gymapi.IMAGE_DEPTH
            elif self._camera_type == gymapi.IMAGE_DEPTH:
                self._camera_type = gymapi.IMAGE_COLOR
            return flask.Response(status=200)
        else:
            return flask.Response(status=200)

        return flask.Response(status=200)

    def _stream(self) -> bytes:
        while True:
            self._event_stream.wait()
            image = imageio.imwrite("<bytes>", self._image, format="JPEG")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            self._event_stream.clear()
            self._notified = False

    def _stream_depth(self) -> bytes:
        while self._env.cfg.depth.use_camera:
            self._event_stream_depth.wait()
            image = imageio.imwrite("<bytes>", self._image_depth, format="JPEG")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            self._event_stream_depth.clear()

    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 960
        camera_props.height = 540
        camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
        self._cameras.append(camera_handle)
        cam_pos = root_pos + np.array([0, 1, 0.5])
        self._gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

    def setup(self, env) -> None:
        self._gym = env.gym
        self._sim = env.sim
        self._envs = env.envs
        self._cameras = []
        self._env = env
        self.cam_pos_rel = np.array([0, 2, 1])
        for i in range(self._env.num_envs):
            root_pos = self._env.root_states[i, :3].cpu().numpy()
            self.attach_view_camera(i, self._envs[i], self._env.actor_handles[i], root_pos)

    def render(self,
               fetch_results: bool = True,
               step_graphics: bool = True,
               render_all_camera_sensors: bool = True,
               wait_for_page_load: bool = True) -> None:
        if self._wait_for_page:
            if wait_for_page_load:
                if not self._event_load.is_set():
                    print("Waiting for web page to begin loading...")
                self._event_load.wait()
                self._event_load.clear()
            self._wait_for_page = False

        if self._pause_stream:
            return

        if self._notified:
            return

        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)

        image = self._gym.get_camera_image(self._sim,
                                           self._envs[self._camera_id],
                                           self._cameras[self._camera_id],
                                           self._camera_type)
        if self._camera_type == gymapi.IMAGE_COLOR:
            self._image = image.reshape(image.shape[0], -1, 4)[..., :3]
        elif self._camera_type == gymapi.IMAGE_DEPTH:
            self._image = -image.reshape(image.shape[0], -1)
            minimum = 0 if np.isinf(np.min(self._image)) else np.min(self._image)
            maximum = 5 if np.isinf(np.max(self._image)) else np.max(self._image)
            self._image = np.clip(1 - (self._image - minimum) / (maximum - minimum), 0, 1)
            self._image = np.uint8(255 * self._image)
        else:
            raise ValueError("Unsupported camera type")

        if self._env.cfg.depth.use_camera:
            self._image_depth = self._env.depth_buffer[self._camera_id, -1].cpu().numpy() + 0.5
            self._image_depth = np.uint8(255 * self._image_depth)

        root_pos = self._env.root_states[self._camera_id, :3].cpu().numpy()
        cam_pos = root_pos + self.cam_pos_rel
        self._gym.set_camera_location(self._cameras[self._camera_id], self._envs[self._camera_id], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

        self._event_stream.set()
        if self._env.cfg.depth.use_camera:
            self._event_stream_depth.set()
        self._notified = True
