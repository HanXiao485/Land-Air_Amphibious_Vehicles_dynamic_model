### PID控制器数学公式

#### 外环（位置控制）
计算期望加速度：
$$
\begin{aligned}
a_{x,\text{des}} &= K_{p,x} (x_{\text{des}} - x) + K_{i,x} \int (x_{\text{des}} - x) \, dt + K_{d,x} \frac{d}{dt}(x_{\text{des}} - x), \\
a_{y,\text{des}} &= K_{p,y} (y_{\text{des}} - y) + K_{i,y} \int (y_{\text{des}} - y) \, dt + K_{d,y} \frac{d}{dt}(y_{\text{des}} - y), \\
a_{z,\text{des}} &= K_{p,z} (z_{\text{des}} - z) + K_{i,z} \int (z_{\text{des}} - z) \, dt + K_{d,z} \frac{d}{dt}(z_{\text{des}} - z).
\end{aligned}
$$
通过小角度近似计算期望姿态角：
$$
\phi_{\text{des}} = \frac{a_{y,\text{des}}}{g}, \quad \theta_{\text{des}} = \frac{a_{x,\text{des}}}{g}.
$$

---

#### 中间环（姿态控制）
计算期望角速率：
$$
\begin{aligned}
p_{\text{des}} &= K_{p,\phi} (\phi_{\text{des}} - \phi) + K_{i,\phi} \int (\phi_{\text{des}} - \phi) \, dt + K_{d,\phi} \frac{d}{dt}(\phi_{\text{des}} - \phi), \\
q_{\text{des}} &= K_{p,\theta} (\theta_{\text{des}} - \theta) + K_{i,\theta} \int (\theta_{\text{des}} - \theta) \, dt + K_{d,\theta} \frac{d}{dt}(\theta_{\text{des}} - \theta), \\
r_{\text{des}} &= K_{p,\psi} (\psi_{\text{des}} - \psi) + K_{i,\psi} \int (\psi_{\text{des}} - \psi) \, dt + K_{d,\psi} \frac{d}{dt}(\psi_{\text{des}} - \psi).
\end{aligned}
$$

---

#### 内环（角速率控制）
计算控制力矩：
$$
\begin{aligned}
\tau_\phi &= K_{p,p} (p_{\text{des}} - p) + K_{i,p} \int (p_{\text{des}} - p) \, dt + K_{d,p} \frac{d}{dt}(p_{\text{des}} - p), \\
\tau_\theta &= K_{p,q} (q_{\text{des}} - q) + K_{i,q} \int (q_{\text{des}} - q) \, dt + K_{d,q} \frac{d}{dt}(q_{\text{des}} - q), \\
\tau_\psi &= K_{p,r} (r_{\text{des}} - r) + K_{i,r} \int (r_{\text{des}} - r) \, dt + K_{d,r} \frac{d}{dt}(r_{\text{des}} - r).
\end{aligned}
$$

---

#### 总升力计算
$$
u_f = m \cdot (g + a_{z,\text{des}}).
$$

---

### 控制框图
```
+----------------+       +----------------+       +----------------+       +----------------+
|                |       |                |       |                |       |                |
|  位置误差       |       |  姿态误差       |      | 角速率误差      |       | 控制力矩        |
| (x,y,z)        +------>| (φ,θ,ψ)        +------>| (p,q,r)        +------>| (τ_φ,τ_θ,τ_ψ)  |
|                |  PID  |                |  PID  |                |  PID  |                |
+--------+-------+       +--------+-------+       +--------+-------+       +----------------+
         |                        |                        |
         |                        |                        |
         v                        v                        v
     加速度生成             期望角速率生成             实际角速率反馈
     (a_x, a_y, a_z)         (p_des, q_des, r_des)       (p, q, r)
         |                        |                        |
         |                        |                        |
         v                        v                        v
     小角度近似转换             姿态反馈                 升力计算
     (φ_des, θ_des)            (φ, θ, ψ)                u_f = m(g + a_z)
```

#### 说明
1. **外环（位置控制）**：通过位置误差计算期望加速度 $a_{x,\text{des}}, a_{y,\text{des}}, a_{z,\text{des}}$，其中 $a_{x,\text{des}}, a_{y,\text{des}}$ 转换为期望姿态角 $\phi_{\text{des}}, \theta_{\text{des}}$。
2. **中间环（姿态控制）**：根据姿态误差计算期望角速率 $p_{\text{des}}, q_{\text{des}}, r_{\text{des}}$。
3. **内环（角速率控制）**：根据角速率误差生成控制力矩 $\tau_\phi, \tau_\theta, \tau_\psi$。
4. **总升力**：通过 $a_{z,\text{des}}$ 计算垂直方向的总升力 $u_f$。

--- 

此设计为级联PID结构，外环输出作为中间环的输入，中间环输出作为内环的输入，形成层级控制。