# Gym环境类
## 1.box
observation space: 离散空间，12维  x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
action space: 离散空间，4维  u_f, tau_phi, tau_theta, tau_psi = forces

## 2.reset


## 3.step
```python
    def step(self, action):    # 必须实现的函数，输入动作，输出下一步的状态、奖励、结束与否、其他可选info
        tmp_reward = reward_func(action) # 根据环境自定义的reward值
        self.reward += tmp_reward
        return self.state, self.reward, done,  truncated, info # done=True表示任务结束，truncated=True表示截断任务

```
### 3.1 reward
reward = reward_func(state, action, next_state, done)
### 3.2 done
### 3.3 info

## 4.render

# 注册环境
