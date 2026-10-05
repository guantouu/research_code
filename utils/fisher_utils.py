import torch
import torch.nn.functional as F
from modules.conv import QuantBnConv2d

def compute_fisher_diag(model, params, dataloader, criterion, fisher_type='true', mc_samples=0, seed=0):
    """
    Diagonal of the Fisher of the conv weights in params, averaged over every sample of dataloader (the
    same calibration subset as the Hutchinson pass, dataset.get_hessian_loader). Only the diagonal is
    computed — not the full Fisher Information Matrix, which is intractable at this parameter count.
        fisher_type 'true' (default): labels from the model's own predictive distribution,
            F_ii = 1/N sum_n sum_c p(c|x_n) (d log p(c|x_n) / d theta_i)^2,
            exactly over all classes (mc_samples 0) or with mc_samples labels drawn per sample.
        fisher_type 'empirical': diagonal empirical Fisher — the standard tractable approximation used in
            the FIM/EWC literature, not the full Fisher Information Matrix — with the dataset labels,
            F_ii = 1/N sum_n (d loss(x_n, y_n) / d theta_i)^2. It vanishes on data the model fits (e.g. the
            CIFAR training images of the calibration subset), so the true Fisher is the default.
    The squares are taken per sample, not of a batch gradient (which would give the squared mean gradient,
    ~0 for a trained model). Per-sample gradients come from one batched backward pass per label (C passes
    per batch when exact): conv hooks combine each sample's unfolded input with its output gradient. For a
    QuantBnConv2d the gradient reaches conv.weight through the straight-through quantizer and the BN fold
    factor. The model runs in eval mode (BN running statistics), so samples are independent.
    criterion is kept for the interface: 'empirical' uses the log-likelihood of the dataset label, i.e. the
    cross-entropy loss. Returns a list of tensors matching params.
    """
    convs = _conv_modules(model, params)
    fisher = {id(p): torch.zeros_like(p) for p in params}
    state = {'weights': None, 'inputs': {}}

    def fwd_hook(module, inputs, output):
        x = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
        state['inputs'][module] = x.detach()
        if output.requires_grad:
            output.register_hook(lambda g, module=module: accumulate(module, g))

    def accumulate(module, grad_out):
        p, conv, fold = convs[module]
        x = state['inputs'][module]
        cols = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding, conv.stride)       # [N, I*kh*kw, P]
        g = torch.einsum('nop,nqp->noq', grad_out.flatten(2), cols)                          # [N, O, I*kh*kw]
        if fold is not None:
            g = g * fold.view(1, -1, 1)
        fisher[id(p)] += torch.einsum('n,noq->oq', state['weights'], g ** 2).view_as(p)

    handles = [m.register_forward_hook(fwd_hook) for m in convs]
    was_training = model.training
    model.eval()
    gen = torch.Generator(device='cuda').manual_seed(seed)
    num_data = 0
    try:
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(0), targets.cuda(0)
            logp = F.log_softmax(model(inputs), dim=1)
            n = inputs.size(0)
            passes = []    # (per-sample weights, per-sample log-likelihood to differentiate)
            if fisher_type == 'empirical':
                passes.append((torch.ones(n, device=inputs.device), logp.gather(1, targets.view(-1, 1)).squeeze(1)))
            elif fisher_type == 'true' and mc_samples == 0:
                prob = logp.detach().exp()
                passes += [(prob[:, c], logp[:, c]) for c in range(logp.size(1))]
            elif fisher_type == 'true':
                labels = torch.multinomial(logp.detach().exp(), mc_samples, replacement=True, generator=gen)
                passes += [(torch.full((n,), 1.0 / mc_samples, device=inputs.device),
                            logp.gather(1, labels[:, k:k + 1]).squeeze(1)) for k in range(mc_samples)]
            else:
                raise ValueError('Unknown fisher_type: {}'.format(fisher_type))
            for i, (weights, loglik) in enumerate(passes):
                state['weights'] = weights
                torch.autograd.grad(loglik.sum(), params, retain_graph=i < len(passes) - 1)
            num_data += n
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return [fisher[id(p)] / num_data for p in params]

def _conv_modules(model, params):
    """
    {module: (param, conv, BN fold factor or None)} of the conv modules whose forward uses params:
    a QuantBnConv2d for its conv.weight (its inner Conv2d is never called), else the Conv2d itself.
    """
    wanted = {id(p): p for p in params}
    found = {}
    for m in model.modules():
        if isinstance(m, QuantBnConv2d) and id(m.conv.weight) in wanted:
            if m.conv.groups != 1:
                raise ValueError('Grouped convolutions are not supported')
            fold = (m.bn.weight / torch.sqrt(m.bn.running_var + m.bn.eps)).detach()
            found[id(m.conv.weight)] = (m, (m.conv.weight, m.conv, fold))
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and id(m.weight) in wanted and id(m.weight) not in found:
            if m.groups != 1:
                raise ValueError('Grouped convolutions are not supported')
            found[id(m.weight)] = (m, (m.weight, m, None))
    missing = [i for i in wanted if i not in found]
    if missing:
        raise ValueError('{} of the params are not the weight of a conv module'.format(len(missing)))
    return dict(found.values())

def fisher_distance(fisher_diag_a, fisher_diag_b):
    """
    Sum over all parameters of (a-b)**2, flattened across the tensor list:
    ||F_a - F_b||_F^2 restricted to the diagonal.
    """
    return sum(((a - b) ** 2).sum().item() for a, b in zip(fisher_diag_a, fisher_diag_b))

def fisher_norm(fisher_diag):
    """
    ||F||_F^2 of a diagonal Fisher, to make fisher_distance relative (comparable across models).
    """
    return sum((f ** 2).sum().item() for f in fisher_diag)
