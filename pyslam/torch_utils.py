from torch import autograd

# jacobian计算方法1 (使用for循环)
# 思路: 对于output中的每一个dim(每一个dim需要是一个scalar),分别计算其关于input的导数,组成jacobian矩阵
# 缺点: 串行计算,无法利用GPU
def jacobian_loop(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def calc_jacobian(y, x, create_graph=False):
    if y.shape == ():  # scalar loss & vector jacobian
        j = autograd.grad(y, [x], create_graph=create_graph)[0][None, ...]  # [1, n]
    else:  # vector loss & matrix jaobian
        j = jacobian_loop(y, x, create_graph=create_graph)  # [m, n]
    # print(f"jacobian shape: {j.shape}")
    return j

