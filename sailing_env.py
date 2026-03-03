import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

class ImprovedSailingEnv(gym.Env):
    """
    Ambiente di navigazione a vela migliorato.
    """
    def __init__(self, field_size=400, render_mode=None):
        super().__init__()
        
        self.field_size = field_size
        self.max_speed = 15.0
        self.max_wind = 25.0
        self.target_radius = 50.0
        self.dt = 1.0
        self.max_steps = 250
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4) 
        
        self.state = None
        self.target = None
        self.wind_direction = None
        self.wind_speed = None
        self.step_count = 0
        self.trajectory = []
        self.previous_distance = None
        self.best_distance = None
        self.fig = None
        self.ax = None

    def _get_polar_speed(self, apparent_wind_angle, wind_speed):
        angle_deg = np.abs(np.degrees(apparent_wind_angle) % 360)
        if angle_deg > 180: angle_deg = 360 - angle_deg
            
        if angle_deg < 40: speed_ratio = 0.0
        elif angle_deg < 50: speed_ratio = 0.2 + (angle_deg - 40) * 0.02
        elif angle_deg < 90: speed_ratio = 0.4 + (angle_deg - 50) * 0.0075
        elif angle_deg < 120: speed_ratio = 0.7
        elif angle_deg < 150: speed_ratio = 0.7 - (angle_deg - 120) * 0.003
        else: speed_ratio = 0.6 - (angle_deg - 150) * 0.005
            
        return min(speed_ratio * wind_speed, self.max_speed)

    def _normalize_angle(self, angle):
        return angle % (2 * np.pi)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.state = {
            'x': self.np_random.uniform(40, 120),
            'y': self.np_random.uniform(40, 120),
            'speed': 0.0,
            'heading': self.np_random.uniform(0, 2*np.pi)
        }
        
        self.target = np.array([
            self.np_random.uniform(self.field_size - 120, self.field_size - 40),
            self.np_random.uniform(self.field_size - 120, self.field_size - 40)
        ])
        
        target_angle = np.arctan2(
            self.target[1] - self.state['y'],
            self.target[0] - self.state['x']
        )
        self.wind_direction = target_angle + np.pi/2 + self.np_random.uniform(-np.pi/6, np.pi/6)
        self.wind_speed = self.np_random.uniform(10, 18)
        
        self.step_count = 0
        self.trajectory = [np.array([self.state['x'], self.state['y']])]
        
        pos = np.array([self.state['x'], self.state['y']])
        self.previous_distance = np.linalg.norm(pos - self.target)
        self.best_distance = self.previous_distance
        
        return self._get_obs(), {}

    def _get_obs(self):
        pos = np.array([self.state['x'], self.state['y']])
        dist_to_target = np.linalg.norm(pos - self.target)
        angle_to_target = np.arctan2(self.target[1] - pos[1], self.target[0] - pos[0])
        angle_to_target = self._normalize_angle(angle_to_target)
        
        obs = np.array([
            self.state['x'] / self.field_size,
            self.state['y'] / self.field_size,
            self.state['speed'] / self.max_speed,
            self.state['heading'] / (2 * np.pi),
            self.wind_direction / (2 * np.pi),
            self.wind_speed / self.max_wind,
            dist_to_target / (self.field_size * np.sqrt(2)),
            angle_to_target / (2 * np.pi)
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        self.step_count += 1
        pos = np.array([self.state['x'], self.state['y']])
        prev_dist = np.linalg.norm(pos - self.target)
        
        if action == 0: self.state['heading'] -= np.radians(15)
        elif action == 2: self.state['heading'] += np.radians(15)
        elif action == 3: self.state['heading'] += np.pi
            
        self.state['heading'] = self._normalize_angle(self.state['heading'])
        
        apparent_wind_angle = self.wind_direction - self.state['heading']
        self.state['speed'] = self._get_polar_speed(apparent_wind_angle, self.wind_speed)
        
        displacement = self.state['speed'] * 0.514 * self.dt
        self.state['x'] += displacement * np.cos(self.state['heading'])
        self.state['y'] += displacement * np.sin(self.state['heading'])
        
        self.trajectory.append(np.array([self.state['x'], self.state['y']]))
        
        pos = np.array([self.state['x'], self.state['y']])
        dist_to_target = np.linalg.norm(pos - self.target)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        distance_delta = prev_dist - dist_to_target
        reward += distance_delta * 2.0
        
        if dist_to_target < 200: reward += 10.0
        if dist_to_target < 150: reward += 15.0
        if dist_to_target < 100: reward += 25.0
        if dist_to_target < 75:  reward += 40.0
        if dist_to_target < 60:  reward += 60.0
            
        if dist_to_target > 100: reward -= 0.2
        elif dist_to_target > 60: reward -= 0.1
        else: reward -= 0.05
            
        if distance_delta > 0: reward += self.state['speed'] * 0.15
            
        angle_to_target = np.arctan2(self.target[1] - pos[1], self.target[0] - pos[0])
        heading_error = abs(self._normalize_angle(angle_to_target - self.state['heading']))
        if heading_error > np.pi: heading_error = 2 * np.pi - heading_error
        heading_alignment = 1.0 - (heading_error / np.pi)
        
        proximity_factor = 1.0
        if dist_to_target < 100: proximity_factor = 2.0
        if dist_to_target < 60:  proximity_factor = 3.0
        reward += heading_alignment * 1.5 * proximity_factor
        
        if dist_to_target < self.best_distance:
            improvement = self.best_distance - dist_to_target
            reward += improvement * 1.0
            self.best_distance = dist_to_target
            
        if dist_to_target < self.target_radius:
            efficiency_bonus = max(0, 250 - self.step_count) * 2
            reward += 2000.0 + efficiency_bonus
            terminated = True
            
        if (self.state['x'] < 0 or self.state['x'] > self.field_size or
            self.state['y'] < 0 or self.state['y'] > self.field_size):
            reward -= 100.0
            terminated = True
            
        if dist_to_target > prev_dist + 25: reward -= 5.0
            
        if self.step_count >= self.max_steps:
            truncated = True
            if self.best_distance < 60: reward += 200.0
            elif self.best_distance < 100: reward += 100.0
            elif self.best_distance < 150: reward += 50.0
                
        obs = self._get_obs()
        info = {
            'distance_to_target': dist_to_target,
            'speed': self.state['speed'],
            'steps': self.step_count,
            'best_distance': self.best_distance
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
            
    def _render_frame(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.field_size)
        ax.set_ylim(0, self.field_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        target_circle = plt.Circle(self.target, self.target_radius, 
                                  color='red', alpha=0.3, label='Target')
        ax.add_patch(target_circle)
        
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2)
            
        boat_size = 15
        boat_points = np.array([[boat_size, 0], [-boat_size/2, boat_size/2], [-boat_size/2, -boat_size/2]])
        rotation_matrix = np.array([
            [np.cos(self.state['heading']), -np.sin(self.state['heading'])],
            [np.sin(self.state['heading']), np.cos(self.state['heading'])]
        ])
        boat_points = boat_points @ rotation_matrix.T
        boat_points += np.array([self.state['x'], self.state['y']])
        
        boat = patches.Polygon(boat_points, closed=True, color='green', 
                              edgecolor='darkgreen', linewidth=2)
        ax.add_patch(boat)
        
        dist = np.linalg.norm(np.array([self.state['x'], self.state['y']]) - self.target)
        ax.set_title(
            f"Step: {self.step_count} | Speed: {self.state['speed']:.1f} kts | "
            f"Distance: {dist:.0f}m (Best: {self.best_distance:.0f}m)",
            fontsize=10, weight='bold'
        )
        
        fig.canvas.draw()
        try:
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        except AttributeError:
             image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
             image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
             
        plt.close(fig)
        return image
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
