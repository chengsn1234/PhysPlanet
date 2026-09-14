"""ManagerBasedRLEnv 子类，负责在 reset 之后触发软土预压。

只覆写 `_reset_idx`，`step()` 保持不变——逐物理步的力由 SoftSoilAction 在
`ActionManager.apply_action()` 里施加。
"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.soft_soil import SoftSoilAction


class SoftSoilNavEnv(ManagerBasedRLEnv):
    """带 Bekker-Wong 软土支撑的导航环境。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._soft_soil = next(
            (t for t in self.action_manager._terms.values() if isinstance(t, SoftSoilAction)), None)
        if self._soft_soil is None:
            raise RuntimeError(
                "SoftSoilNavEnv requires a SoftSoilAction term in the action config.")

    def _refresh_kinematics(self) -> None:
        """刷新 body_pos_w 等惰性缓存。

        `sim.forward()` 只在 fabric 可用时才更新关节链位姿，headless 训练下不保证，
        所以直接调 physics view；再 `scene.update` 推进缓存时间戳令其失效重取。
        """
        self.scene.write_data_to_sim()
        if self.sim.physics_sim_view is not None:
            self.sim.physics_sim_view.update_articulations_kinematic()
        self.scene.update(dt=self.physics_dt)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # 出生位姿由 event（V0/V1）或 command term（V2+）写入，两者都排在这之前。
        self._refresh_kinematics()
        self._soft_soil.settle(env_ids)
        self._refresh_kinematics()
