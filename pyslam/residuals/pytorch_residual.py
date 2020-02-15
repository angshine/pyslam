import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from pyslam.torch_utils import calc_jacobian
from liegroups.torch import SE3, SO3


class TorchResidual:
    """ General class using pytorch to calculate residual and utilize pytorch's automatic
    differentiation to calculate the jacobian matrix.
    """

    def __init__(self, stiffness):
        """
        Args:
            stiffness: numpy.ndarray | stiffness matrix of residuals
                       i.e. sqrt(inv(CovMatrix))
        """
        self.stiffness = stiffness

    def eval_func(self, params):
        """ Evaluation of residual value using ``params``
        Args:
            params (list of torch.Tensor): list of parameters to be optimized
        Returns:
            residual (torch.Tensor): residual vector
        """
        raise NotImplementedError

    def evaluate(self, params, compute_jacobians=None, callbacks=None):
        # TODO: allow registration of callbacks to do logging or other things
        #       before & after residual evaluation
        torch_params = [torch.tensor(param, dtype=torch.float32, requires_grad=True) for param in params]
        # torch_params = torch.tensor(params[0], dtype=torch.float32, requires_grad=True)
        torch_res = self.eval_func(torch_params)
        np_res = torch_res.detach().numpy()

        if compute_jacobians:
            jacobians = [None for _ in enumerate(params)]

            # TODO: calculate jacobians of all params once instead of looping might be faster
            for i, param in enumerate(torch_params):
                if compute_jacobians[i]:
                    n_param = param.shape[0]
                    n_res = torch_res.shape[0]
                    torch_jac = calc_jacobian(torch_res, param, False)  # [n_res, n_param]
                    np_jac = torch_jac.detach().numpy()
                    assert np_jac.shape == (n_res, n_param)
                    jacobians[i] = np.dot(self.stiffness, np_jac)

            return np_res, jacobians

        return np_res


class PhotometricReisdual(TorchResidual):
    def __init__(self, prev_frame, post_frame, stiffness,
                 pose_init=("pred", "pred"), optimize=(False, True)):
        super(PhotometricReisdual, self).__init__(stiffness)
        self.optimize = optimize
        self.prev_frame = prev_frame
        self.post_frame = post_frame
        self.K = prev_frame["K"].float()
        self.kp3ds = self.prev_frame["kpt_3d"].T.float()
        self.n_kps = self.kp3ds.shape[1]
        self.prev_pose = self._init_pose(prev_frame, pose_init[0]).requires_grad_()
        self.post_pose = self._init_pose(post_frame, pose_init[1]).requires_grad_()
        self.residual_sum = None

    def _init_pose(self, frame, pose_init):
        if pose_init == "identity":
            pose_se3 = SE3(SO3(torch.eye(3)), torch.zeros(1, 3)).log()
        elif pose_init == "pred":
            pose_se3 = SE3(SO3(frame["pose_pred"][:, :3].float()), frame["pose_pred"][:, 3].float()).log()
        elif pose_init == "gt":
            pose_se3 = SE3(SO3(frame["pose_gt"][:, :3].float()), frame["pose_gt"][:, 3].float()).log()
        elif pose_init == "ones":
            pose_se3 = torch.ones(6).float()
        else:
            raise NotImplementedError(f"pose_init = {pose_init} not supported yet.")
        return pose_se3

    def eval_func(self, params):
        if all(self.optimize):
            self.prev_pose = params[0]
            self.post_pose = params[1]
        else:
            if self.optimize[0]:
                self.prev_pose = params[0]
            elif self.optimize[1]:
                self.post_pose = params[0]
            else:
                raise ValueError("`PhotometricResidual.optimize` should contain at least 1 True.")
        photometric_error = self.calc_photo_err()
        self.residual_sum = (photometric_error**2).sum() / 2
        return photometric_error

    def calc_photo_err(self):
        """

        """
        prev_kps_rgb = self.get_kps_rgb(SE3.exp(self.prev_pose), self.prev_frame["torch_img"])
        post_kps_rgb = self.get_kps_rgb(SE3.exp(self.post_pose), self.post_frame["torch_img"])
        error = (prev_kps_rgb - post_kps_rgb).view(-1)  # [3*9]
        return error

    def get_kps_rgb(self, pose_SE3, torch_img):
        """
        Given pose_SE3 and an image(in torch format), calculate keypoints' corresponding rgb values with self.kp3ds.
        """
        kp2ds_pixel = self.get_pose_kp2ds(pose_SE3)  # kp2ds in pixel coordinate
        x_coord = kp2ds_pixel[0, :]  #
        y_coord = kp2ds_pixel[1, :]
        C, H, W = torch_img.shape
        # transform the coordinate origin to center and rescale the range to [-1, 1]
        x_norm = 2 * x_coord / (W - 1) - 1  # [9]
        y_norm = 2 * y_coord / (H - 1) - 1
        _pixel_coords = torch.stack([x_norm, y_norm], dim=0)  # [2, 9]
        pixel_coords = _pixel_coords.transpose(0, 1)[None, None, ..., ...]  # [1, 1, 9, 2]
        kps_rgb = nn.functional.grid_sample(torch_img.unsqueeze(0), pixel_coords,
                                            align_corners=True)  # [1, 3, 1, 9]# align_corners?
        #         import pdb; pdb.set_trace()  # 检查输出的维度与结果
        return kps_rgb.squeeze_(0).squeeze_(1)  # [3, 9]  (3 for r, g, b)

    def get_pose_kp2ds(self, pose_SE3, other_kp3d=None):
        """
        Given pose_SE3 and possiblely kp3ds, calculate the corresponding kp2ds
        """
        pose_kp3d_norm = self.get_kp3d_norm(pose_SE3, other_kp3d)
        _pose_kp3d_norm = pose_kp3d_norm.clone()
        _pose_kp3d_norm[0, :] = pose_kp3d_norm[0, :] / pose_kp3d_norm[-1, :]
        _pose_kp3d_norm[1, :] = pose_kp3d_norm[1, :] / pose_kp3d_norm[-1, :]
        return _pose_kp3d_norm[:2, :]  # [2, 9]

    def get_kp3d_norm(self, pose_SE3, other_kp3d=None):  # function name might be inappropriate
        """ Transform self.kp3ds to normalized camera coordinate frame with pose_SE3 and self.K
        """
        kp3ds = self.kp3ds if other_kp3d is None else other_kp3d
        pose_kp3d = pose_SE3.rot.as_matrix() @ kp3ds + pose_SE3.trans.unsqueeze(-1)
        pose_kp3d_norm = self.K @ pose_kp3d  # [3, 9]
        return pose_kp3d_norm

    def calc_ADD(self, which="post"):
        """ Not actually the ADD metric, calculate (l2-norm)^2 instead of l1-norm.
        """

        if which == "prev":
            pose_SE3 = SE3.exp(self.prev_pose.detach())
            frame = self.prev_frame
        elif which == "post":
            pose_SE3 = SE3.exp(self.post_pose.detach())
            frame = self.post_frame
        else:
            raise ValueError(f"which=={which} not supported!")

        kp2ds_pose = self.get_pose_kp2ds(pose_SE3)
        pose_gt_SE3 = SE3(SO3(frame["pose_gt"][:, :3].float()), frame["pose_gt"][:, 3].float())
        kp2ds_gt_pose = self.get_pose_kp2ds(pose_gt_SE3)
        metric = ((kp2ds_pose - kp2ds_gt_pose) ** 2).sum()
        return metric

