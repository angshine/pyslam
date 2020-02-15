import numpy as np


class PoseToPoseResidual:
    """Binary pose-to-pose residual given relative pose mesurement in SE2/SE3."""

    def __init__(self, T_2_1_obs, stiffness):
        self.T_2_1_obs = T_2_1_obs  # observation为两帧间的相对位姿
        self.stiffness = stiffness
        self.obstype = type(T_2_1_obs)

    def evaluate(self, params, compute_jacobians=None):
        # 注意params使用的是李群而非李代数, 但是返回的jacobians是关于李代数的
        # 在Problem.solve_one_iter中计算的也是关于李代数的dx
        # 通过调用SOMatrixBase / SEMatrixBase的perturb方法来使用李代数的dx对李群进行更新
        T_1_0_est = params[0]
        T_2_0_est = params[1]

        # 注意这里的李群的命名非常方便, T_a_b时b到a的变换矩阵, 0为world-frame.
        # 这样命名可以防止计算出错
        residual = np.dot(self.stiffness,
                          self.obstype.log(
                              T_2_0_est.dot(T_1_0_est.inv().dot(self.T_2_1_obs.inv()))))

        if compute_jacobians:
            # jacobians[i]为residual关于第i个待优化变量的jacobian
            # 本例中, 每个待优化变量都是李群, 不过导数是关于李代数定义的
            jacobians = [None for _ in enumerate(params)]

            # 这里的jacobian是如何推导的? 用I来近似左右雅克比?
            # 为何关于params[1]的jacobians为stiffness *　identity?
            if compute_jacobians[0]:
                jacobians[0] = np.dot(self.stiffness, -T_2_0_est.dot(T_1_0_est.inv()).adjoint())

            if compute_jacobians[1]:
                jacobians[1] = np.dot(
                    self.stiffness, np.identity(self.obstype.dof))

            return residual, jacobians

        return residual
