import numpy as np


class PoseResidual:
    """Unary pose residual given absolute pose measurement in SE2/SE3."""

    def __init__(self, T_obs, stiffness):
        self.T_obs = T_obs
        self.stiffness = stiffness
        self.obstype = type(T_obs)

    def evaluate(self, params, compute_jacobians=None):
        T_est = params[0]
        # 在计算residual时用SE3相乘再转换到se3, 而非直接用se3相减, 这样做应当是避免两次对数映射
        # self.stiffness是对角矩阵, 乘以se3得到residual (相当于stiffness中没有考虑不同变量间的关系?类比方差与协方差)
        # 不过self.stiffness也可以设定为非对角的矩阵
        residual = np.dot(self.stiffness,
                          self.obstype.log(T_est.dot(self.T_obs.inv())))

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            if compute_jacobians[0]:
                # jacobians[0]是一个dof * dof的对角方阵, jacobian是pose_residual关于se3的jacobian
                jacobians[0] = np.dot(self.stiffness,
                                      np.identity(self.obstype.dof))

            return residual, jacobians

        return residual
