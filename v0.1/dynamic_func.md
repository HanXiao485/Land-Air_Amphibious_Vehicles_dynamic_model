$$
\begin{cases}
m \left[ \dot{u} - vr + wq - x_G (q^2 + r^2) + y_G (pq - \dot{r}) + z_G (pr + \dot{q}) \right] = X_\Sigma \\
m \left[ \dot{v} - wp + ur - y_G (r^2 + p^2) + z_G (qr - \dot{p}) + x_G (qp + \dot{r}) \right] = Y_\Sigma \\
m \left[ \dot{w} - uq + vp - z_G (p^2 + q^2) + x_G (rp - \dot{q}) + y_G (rq + \dot{p}) \right] = Z_\Sigma \\
I_{xx} \dot{p} + I_{yz} \dot{q} + I_{zx} \dot{r} + (I_{xy} p + I_{yy} q + I_{yz} r) p - (I_{xx} p + I_{xy} q + I_{zx} r) q + m \left[ x_G (\dot{v} + vp - up) - y_G (\dot{u} + ur - wp) \right] = N_\Sigma \\
I_{xx} \dot{p} + I_{xy} \dot{q} + I_{zx} \dot{r} + (I_{xx} p + I_{yz} q + I_{zx} r) q - (I_{xy} p + I_{yy} q + I_{yz} r) r + m \left[ y_G (\dot{w} + vp - up) - z_G (\dot{v} + ur - wp) \right] = K_\Sigma \\
I_{xy} \dot{p} + I_{yy} \dot{q} + I_{yz} \dot{r} + (I_{xx} p + I_{xy} q + I_{zx} r) r - (I_{xy} p + I_{yz} q + I_{zz} r) p + m \left[ z_G (\dot{u} + vp - up) - x_G (\dot{w} + ur - wp) \right] = M_\Sigma \\
\dot{x}_0 = u \cos \psi \cos \theta + v (\cos \psi \sin \theta \sin \varphi - \sin \psi \cos \varphi) + w (\cos \psi \sin \theta \cos \varphi + \sin \psi \sin \varphi) \\
\dot{y}_0 = u \sin \psi \cos \theta + v (\sin \psi \sin \theta \sin \varphi + \cos \psi \cos \varphi) + w (\sin \psi \sin \theta \cos \varphi - \cos \psi \sin \varphi) \\
\dot{z}_0 = -u \sin \theta + v \cos \theta \sin \varphi + w \cos \theta \cos \varphi \\
\dot{\varphi}_0 = p + q \tan \theta \sin \varphi + r \tan \theta \cos \varphi \\
\dot{\theta}_0 = q \cos \varphi - r \sin \varphi \\
\dot{\psi}_0 = q \sin \varphi / \cos \theta + r \cos \varphi / \cos \theta
\end{cases}
$$

$$
\begin{align*}
\dot{x}_0 &= u \cos \psi \cos \theta + v (\cos \psi \sin \theta \sin \varphi - \sin \psi \cos \varphi) + w (\cos \psi \sin \theta \cos \varphi + \sin \psi \sin \varphi) \\
\dot{y}_0 &= u \sin \psi \cos \theta + v (\sin \psi \sin \theta \sin \varphi + \cos \psi \cos \varphi) + w (\sin \psi \sin \theta \cos \varphi - \cos \psi \sin \varphi) \\
\dot{z}_0 &= -u \sin \theta + v \cos \theta \sin \varphi + w \cos \theta \cos \varphi \\
\dot{\varphi}_0 &= p + q \tan \theta \sin \varphi + r \tan \theta \cos \varphi \\
\dot{\theta}_0 &= q \cos \varphi - r \sin \varphi \\
\dot{\psi}_0 &= q \sin \varphi / \cos \theta + r \cos \varphi / \cos \theta
\end{align*}
$$













用python求解这个微分方程组，已知量为作用在三个坐标轴上的力和绕三个坐标轴的力矩，初始初始重心位置、速度、加速度、欧拉角均为0，转动惯量为diag([0.029618 0.069585 0.042503])，重量为3.18，求刚体在世界坐标系下的位置坐标、沿三个坐标轴的线速度、线加速度、绕三个坐标轴的角速度、角加速度，刚体在刚体坐标系下沿三个坐标轴的线速度、线加速度、绕三个坐标轴的角速度、角加速度，刚体相对于初始姿态的旋转矩阵，刚体相对于初始姿态的欧拉角。最后将输出数据绘制成曲线图表示出来。同时实时绘制出无人机的动态轨迹。并标注出无人机初始位置的坐标系和固定于无人机的动态坐标系，三个轴分别用不同颜色的线表示，要求动态坐标轴反应无人机当前时刻的姿态，  
$$
\begin{cases} 
\ddot{x} = \frac{1}{m} \left[ (cos\phi cos\psi sin\theta + sin\phi sin\psi) u_f - k_t \dot{x} \right] \\
\ddot{y} = \frac{1}{m} \left[ (cos\phi sin\theta sin\psi - cos\psi sin\phi) u_f - k_t \dot{y} \right] \\
\ddot{z} = \frac{1}{m} \left[ (cos\theta cos\phi) u_f - mg - k_t \dot{z} \right] \\
\dot{p} = \frac{1}{I_{xx}} \left[ -k_r p - qr (I_{zz} - I_{yy}) + \tau_p \right] \\
\dot{q} = \frac{1}{I_{yy}} \left[ -k_r q - pr (I_{xx} - I_{zz}) + \tau_q \right] \\
\dot{r} = \frac{1}{I_{zz}} \left[ -k_r r - pq (I_{yy} - I_{xx}) + \tau_r \right] \\
\dot{\phi} = p + q sin\phi tan\theta + r cos\phi tan\theta \\
\dot{\theta} = q cos\phi - r sin\phi \\
\dot{\psi} = \frac{1}{cos\theta} \left[ q sin\phi + r cos\phi \right]
\end{cases}
$$
​                                                                   其中$u_f$为总推力，$\tau_p, \tau_q, \tau_r$分别为绕x, y, z 轴的力矩，$k_t， k_r$为阻尼系数