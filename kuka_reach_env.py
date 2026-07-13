import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data 


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class KukaReachEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 関節角度(7) + 関節角速度(7) + EE位置(3) + EE姿勢(4)
        # + ターゲット位置(3) + 相対位置ベクトル(3) + 直前の行動(7)
        obs_dim = 7 + 7 + 3 + 4 + 3 + 3 + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 関節数と、エンドエフェクタ
        self.num_joints = 7
        self.end_effector_index = 6

        # 内部の初期化
        self.kuka_id = None
        self.target_pos = None
        # 前回の情報
        self.previous_action = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.physics_client)

        # ドメインランダム化
        gravity_z = np.random.uniform(-10.1, -9.5)
        p.setGravity(0, 0, gravity_z, physicsClientId=self.physics_client)
        

        p.loadURDF("plane.urdf", physicsClientId = self.physics_client)
        self.kuka_id = p.loadURDF(
            "kuka_iiwa/model.urdf", [0, 0, 0], physicsClientId = self.physics_client
        )


        # ターゲット位置をランダムに決定
        self.target_pos = np.array([
            np.random.uniform(0.3, 0.6),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.6),
        ])

        # 直前の行動をゼロベクトルにリセット
        self.previous_action = np.zeros(self.num_joints, dtype=np.float32)

        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        #ラジアン単位に変換
        max_angle = np.pi
        target_angles = action * max_angle

        for i in range(self.num_joints):
            p.setJointMotorControl2(
            bodyUniqueId=self.kuka_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angles[i],
            physicsClientId=self.physics_client,
        )
        
        p.stepSimulation(physicsClientId=self.physics_client)
        self.previous_action = action.astype(np.float32)

        # 新しい観測の取得
        observation = self._get_observation()

        #報酬計算
        end_effector_pos = observation[14:17]   # observationの中の該当部分を再利用
        target_pos = observation[21:24]
        distance = np.linalg.norm(target_pos - end_effector_pos)

        #報酬
        reward = -distance
        success_threshold = 0.05
        terminated = bool(distance < success_threshold)
        if terminated:
            #成功ボーナス
            reward += 10.0 

        truncated = False
        info = {"distance": distance}
        return observation, reward, terminated, truncated, info

    # pybulletから情報取得と組み立て
    def _get_observation(self):

        #複数の関節の角度・角速度の取得
        joint_states = p.getJointStates(
            self.kuka_id, range(self.num_joints), physicsClientId=self.physics_client
        )
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        link_state = p.getLinkState(
        self.kuka_id, self.end_effector_index, physicsClientId=self.physics_client)
        end_effector_pos = np.array(link_state[0], dtype=np.float32)      # (x, y, z)
        end_effector_orn = np.array(link_state[1], dtype=np.float32)      # (x, y, z, w) クォータニオン






        # ターゲット位置(3)
        target_pos = self.target_pos.astype(np.float32)

        #相対位置ベクトル(3)
        relative_pos = target_pos - end_effector_pos

        # 直前の行動（7）
        previous_action = self.previous_action.astype(np.float32)

        # 全部つなげて1本の34次元ベクトルにする
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            end_effector_pos,
            end_effector_orn,
            target_pos,
            relative_pos,
            previous_action,])

        return observation

        

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


if __name__ == "__main__":
    env = KukaReachEnv()
    obs, info = env.reset()
    print("観測ベクトルの形：", obs.shape)

    for step_num in range(10):
        action = env.action_space.sample()  # ランダムな行動を生成
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step_num}: reward={reward:.4f}, distance={info['distance']:.4f}, terminated={terminated}")
        if terminated or truncated:
            print("エピソード終了")
            break

    env.close()


