import numpy as np
import os, time
import pickle

from isaacgym import gymapi
from isaacgym import gymtorch

import torch
import torchvision

#from tasks.go2_terrain import Go2Terrain
from isaacgymenvs.tasks.go2_bipedal_amp import Go2BipedalAMP
import matplotlib.pyplot as plt

class Go2BipedalAMPRender(Go2BipedalAMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_buffer = []
        self.actions_hist = []

    def create_sim(self):
        super().create_sim()
        self.camera_height = 360 #540
        self.camera_width = 480 #960

        self.camera_offsets = [
                [0.3, -0.3, 0.1, 0.0, 0.0, -0.1],
                [0.4, 0.0, -0.05, 0.0, 0.0, -0.05],
                [0.0, 0.5, -0.05, 0.0, 0.0, -0.05],
                [0.3, 0.3, 1.0, 0.0, 0.0, 0.0],
        ]
        self.camera_handles = []
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_width
        camera_props.height = self.camera_height
        camera_props.enable_tensors = True
        for _ in self.camera_offsets:
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            assert camera_handle != -1, "The camera failed to be created"
            self.camera_handles.append(camera_handle)

    def write_video(self, video_frames, filepath, fps=50):
        if True:
            torchvision.io.write_video(filepath, video_frames, fps=fps, video_codec='libx264', options={'crf': '18'})
            return

        import cv2
        video_frames_np = video_frames.cpu().numpy() # Assuming you're using a CPU tensor
        height, width, _ = video_frames_np[0].shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height), isColor=True)

        for frame in video_frames_np:
            out.write(frame)
        out.release()

    def compute_constraints_cat(self):
        # Constraint to avoid having the robot upside-down
        cstr_upsidedown = self.projected_gravity[:, 2] > 0
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            base_height = self.root_states[:, 2]
        else:
            base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        cstr_base_height_min = torch.any(base_height < self.limits['base_height_min'])
        # Contacts causing early stopping
        cstr_termination_contacts = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.cstr_manager.add("upsidedown", cstr_upsidedown, max_p=1.0)
        self.cstr_manager.log_all(self.episode_sums)

        # Timeout if episodes have reached their maximum duration
        timeout = self.progress_buf >= self.max_episode_length - 1

        # Get final termination probability for each env from all constraints
        self.cstr_prob = self.cstr_manager.get_probs()

        # Probability of termination used to affect the discounted sum of rewards
        self.reset_buf = self.cstr_prob
        ##self.reset_env_buf = timeout | cstr_upsidedown | cstr_termination_contacts | cstr_base_height_min #/!\#
        allow_contact_steps = 30
        max_dof_change = 0.75
        any_contacts = torch.logical_and(
            torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1),
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(self.contact_forces[:, self.allow_initial_contact_indices, :], dim=-1) > 1., dim=1),
                self.progress_buf <= allow_contact_steps
            ))
        )
        stand_air_condition = torch.logical_and(
            torch.logical_and(self.progress_buf > 3, self.progress_buf <= allow_contact_steps),
            torch.any(self.foot_positions[:, self.fidx, 2] > 0.06, dim=-1)
        )
        abrupt_change_condition = torch.logical_and(
            torch.logical_and(self.progress_buf > 3, self.progress_buf <= allow_contact_steps),
            torch.any(torch.abs(self.dof_pos - self.last_dof_pos[:, :, 0]) > max_dof_change, dim=-1)
        )
        print(self.root_states[:, 2], any_contacts, timeout, stand_air_condition, abrupt_change_condition, self.progress_buf)
        self.reset_env_buf = timeout
        self.reset_env_buf |= any_contacts
        #self.reset_env_buf |= stand_air_condition
        #self.reset_env_buf |= abrupt_change_condition

    def post_physics_step(self):
        super().post_physics_step()

        self.actions_hist.append(self.actions.clone().cpu().squeeze().numpy())

        # Sample half of the time
        if self.common_step_counter % 1 == 0:
            # Manually put the camera at the desired location, as in PA's code
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            base_pos = (self.root_states[0, :3]).cpu().numpy()
            for i, camera_offset in enumerate(self.camera_offsets):
                cam_pos = gymapi.Vec3(base_pos[0] + camera_offset[0], base_pos[1] + camera_offset[1], base_pos[2] + camera_offset[2])
                cam_target = gymapi.Vec3(base_pos[0] + camera_offset[3], base_pos[1] + camera_offset[4], base_pos[2] + camera_offset[5])
                self.gym.set_camera_location(self.camera_handles[i], self.envs[0], cam_pos, cam_target)

            # Render simulation
            self.gym.step_graphics(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            # Access camera image
            self.gym.start_access_image_tensors(self.sim)
            images = []
            for i in range(len(self.camera_offsets)):
                image_ = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.camera_handles[i], gymapi.IMAGE_COLOR)
                image = gymtorch.wrap_tensor(image_)
                images.append(image)
            self.gym.end_access_image_tensors(self.sim)

            # Store image in replay buffer
            images = [image.view(self.camera_height, self.camera_width, 4)[:, :, :3].cpu() for image in images]
            if len(images) < 3:
                frame = torch.hstack(images)
            elif len(images) == 4:
                frame = torch.vstack([torch.hstack(images[:2]), torch.hstack(images[2:])])
            else:
                assert False, "Number of cameras not supported (yet)"
            self.video_buffer.append(frame)

        #self.compute_constraints_cat()

        if self.reset_env_buf[0] == 1.0 and len(self.video_buffer) > 0:

            print("====================")
            save_path = self.cfg['video_save_path']
            print(f"Saving {save_path} and reseting video_buffer")
            video = torch.stack(self.video_buffer, dim=0)
            self.write_video(video, save_path, fps=50.0)
            #torchvision.io.write_video(save_path, video, fps=50.0)
            print("video saved")
            self.video_buffer = []
            plt.figure(figsize=(10, 6))
            plt.plot(self.actions_hist[:50])
            plt.xlabel('Index of Vector in actions_hist')
            plt.ylabel('Feature Value')
            plt.title('Feature Curves from actions_hist')
            plt.legend()
            plt.grid(True)

            # Save the figure to disk
            plt.savefig('feature_curves.png')
            exit()



