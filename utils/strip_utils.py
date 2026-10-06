import torch
import numpy as np
from modules.quantizer import weight_to_strips, symmetric_linear_quantization_params, SymmetricQuantFunction

def get_modules_list(model, known_modules):
    modules = []
    for module in model.modules():
        classname = module.__class__.__name__
        if classname in known_modules:
            modules.append(module)
    return modules

def simplify_attribute_path(attribute_path):
    parts = attribute_path.split('.')
    if(len(parts) == 3):
        simplified_parts = [parts[1], parts[2]]
    elif(len(parts) == 4):
        simplified_parts = [parts[1]]
    elif(len(parts) == 5):
        simplified_parts = [parts[1], parts[2], parts[3]]
    else:
        simplified_parts = [parts[1], parts[2], parts[4]]
    simplified_path = '.'.join(simplified_parts)
    
    return simplified_path

def hessian_vector_product(model, params, dataloader, criterion, v):
    """
    Hv of the loss averaged over the whole dataloader (batches weighted by size).
    """
    THv = [torch.zeros_like(p) for p in params]
    num_data = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        loss = criterion(model(inputs), targets)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        Hv = torch.autograd.grad(grads, params, grad_outputs=v)
        bs = inputs.size(0)
        THv = [THv1 + Hv1.detach() * bs for THv1, Hv1 in zip(THv, Hv)]
        num_data += bs
    return [THv1 / num_data for THv1 in THv]

def get_bn_fold_scale(model, conv):
    """
    Per-output-channel factor gamma / sqrt(var + eps) of the BatchNorm2d registered right after conv
    (ConvBlock .conv/.bn in ResNet, convN/bnN in VGG); ones if the conv has no BN.
    QuantBnConv2d quantizes the folded weight conv.weight * factor.
    """
    modules = list(model.modules())
    nxt = modules[modules.index(conv) + 1] if modules.index(conv) + 1 < len(modules) else None
    if isinstance(nxt, torch.nn.BatchNorm2d) and nxt.num_features == conv.out_channels:
        return (nxt.weight / torch.sqrt(nxt.running_var + nxt.eps)).detach()
    return torch.ones(conv.out_channels, device=conv.weight.device)

def strip_quant_error(conv, fold_scale, bit):
    """
    Squared quantization error ||Q_b(w_s) - w_s||^2 of every strip, measured in the space of conv.weight.
    Quantization follows QuantBnConv2d (per-strip symmetric on the BN-folded weight); the
    error on the folded weight is mapped back through the fold factor.
    """
    with torch.no_grad():
        O, _, kH, kW = conv.weight.shape
        folded = conv.weight * fold_scale.view(-1, 1, 1, 1)
        strips = weight_to_strips(folded)
        bits = [bit] * strips.size(0)
        scale = symmetric_linear_quantization_params(bits, strips.min(dim=1).values, strips.max(dim=1).values, True)
        dequant = weight_to_strips(SymmetricQuantFunction.apply(folded, bits, scale)) * scale.view(-1, 1)
        strip_fold = fold_scale.repeat_interleave(kH * kW).view(-1, 1)
        err = torch.where(strip_fold.abs() > 1e-12, (dequant - strips) / strip_fold, torch.zeros_like(strips))
        return (err ** 2).sum(dim=1)

def compute_strip_importances(model, dataloader, criterion, bits, saliency='quant_perturbation',
                              max_iters=100, min_iters=10, tol=0.05, seed=0, log=True, return_traces=False,
                              return_variance=False):
    """
    Hutchinson estimate of tr(H_ss) for every crossbar strip s = weight[o, :, kh, kw] of every Conv2d:
        tr(H_ss) = E_v[ v_s^T (Hv)_s ],  v ~ Rademacher
    Sampling stops after min_iters once the relative standard error of the strip-trace
    vector, sqrt(sum_s Var_s / t) / ||mean||, drops below tol, or at max_iters.

    Saliency of each strip (higher = keep at high bitwidth):
        'quant_perturbation': the HAWQ-v2 quantization-perturbation sensitivity, extended from layer to
            strip (ReRAM crossbar column) granularity; this is the formula used for all reported results:
            1/2 * tr(H_ss)/n_s * (||Q_low(w_s) - w_s||^2 - ||Q_high(w_s) - w_s||^2)
        'weight_norm': a Hessian-based pruning-style sensitivity (OBD/OBS-style), the loss of zeroing the
            strip, tr(H_ss)/n_s * ||w_s||^2; kept only for ablation - not used for reported results.

    With return_traces=True a third dict {module: [tr(H_ss) per strip]} is returned as well
    (see aggregate_layer_trace), and with return_variance=True also {module: [variance of that estimate]}
    (sample variance / number of samples, see shrink_traces).
    """
    importances = {}
    strip_importances_per_layer = {}
    strip_traces_per_layer = {}
    strip_trace_vars_per_layer = {}
    known_modules = {"Conv2d"}

    was_training = model.training
    model.eval()

    modules = get_modules_list(model, known_modules)
    params = [m.weight for m in modules]

    # Welford running mean / M2 of per-strip v_s^T (Hv)_s, one vector per layer
    mean = [torch.zeros(weight_to_strips(p).size(0), device=p.device) for p in params]
    m2 = [torch.zeros_like(mu) for mu in mean]
    gen = torch.Generator(device=params[0].device).manual_seed(seed)

    for t in range(1, max_iters + 1):
        v = [torch.randint(0, 2, p.shape, generator=gen, device=p.device).float() * 2 - 1 for p in params]
        Hv = hessian_vector_product(model, params, dataloader, criterion, v)

        for k, (Hv_k, v_k) in enumerate(zip(Hv, v)):
            sample = (weight_to_strips(Hv_k) * weight_to_strips(v_k)).sum(dim=1)
            delta = sample - mean[k]
            mean[k] += delta / t
            m2[k] += delta * (sample - mean[k])

        if t >= 2:
            all_mean = torch.cat(mean)
            all_var = torch.cat(m2) / (t - 1)
            rel_se = (all_var.sum() / t).sqrt() / all_mean.norm().clamp(min=1e-12)
            done = (t >= min_iters and rel_se < tol) or t == max_iters
            if log and (t % 10 == 0 or done):
                print('=> Hutchinson iter {}: relative std error {:.4f}, negative strips {:.2%}'.format(
                    t, rel_se.item(), (all_mean < 0).float().mean().item()))
            if done:
                break

    model.train(was_training)

    for k, m in enumerate(modules):
        strip_weight = weight_to_strips(m.weight.detach())
        avg_trace = mean[k] / strip_weight.size(1)
        if saliency == 'quant_perturbation':
            fold_scale = get_bn_fold_scale(model, m)
            gain = strip_quant_error(m, fold_scale, bits['insensitive']) - strip_quant_error(m, fold_scale, bits['highly_sensitive'])
            strip_saliency = 0.5 * avg_trace * gain
        elif saliency == 'weight_norm':
            strip_saliency = avg_trace * strip_weight.norm(dim=1) ** 2
        else:
            raise ValueError("Unknown saliency: {}".format(saliency))
        tmp = strip_saliency.cpu().tolist()
        strip_importances_per_layer[m] = tmp
        importances[m] = (tmp, len(tmp))
        strip_traces_per_layer[m] = mean[k].cpu().tolist()
        strip_trace_vars_per_layer[m] = (m2[k] / max(t - 1, 1) / t).cpu().tolist()

    if return_traces and return_variance:
        return importances, strip_importances_per_layer, strip_traces_per_layer, strip_trace_vars_per_layer
    if return_traces:
        return importances, strip_importances_per_layer, strip_traces_per_layer
    return importances, strip_importances_per_layer

def shrink_traces(traces, variances):
    """
    Empirical-Bayes shrinkage of noisy per-strip Hutchinson traces toward their layer mean. Within a layer the
    strip traces are modeled as true values spread with variance tau^2 around the layer mean mu, each observed
    with its own estimation noise sigma_s^2 (variances, from compute_strip_importances(return_variance=True)):
        tau^2   = max(Var_s(trace_s) - mean_s(sigma_s^2), 0)
        trace_s <- mu + tau^2 / (tau^2 + sigma_s^2) * (trace_s - mu)
    so an estimate dominated by noise is pulled to the layer mean and a precise one is kept; a layer whose spread
    is all noise gets its mean for every strip. Unconverged Hutchinson estimates (relative std error ~0.6 after
    200 samples) make many strip traces negative, which ranks those strips as the least sensitive.
    traces, variances: {key: [value per strip]}. Returns {key: [shrunk trace per strip]}.
    """
    shrunk = {}
    for k in traces:
        t = torch.tensor(traces[k], dtype=torch.float64)
        v = torch.tensor(variances[k], dtype=torch.float64)
        mu = t.mean()
        tau2 = (t.var(unbiased=True) - v.mean()).clamp(min=0) if t.numel() > 1 else torch.tensor(0.0, dtype=torch.float64)
        weight = torch.where(tau2 + v > 0, tau2 / (tau2 + v), torch.ones_like(v))
        shrunk[k] = (mu + weight * (t - mu)).tolist()
    return shrunk

def strip_fisher_traces(model, modules, dataloader, criterion, fisher_type='true', mc_samples=0, seed=0):
    """
    tr(F_ss) per strip of the Fisher diagonal of the conv weights (utils.fisher_utils.compute_fisher_diag,
    true Fisher by default): a positive semi-definite (Gauss-Newton) approximation of tr(H_ss) at a minimum,
    never negative and free of Hutchinson sampling noise. Returns {module: [trace per strip]}.
    """
    from utils.fisher_utils import compute_fisher_diag
    fisher = compute_fisher_diag(model, [m.weight for m in modules], dataloader, criterion,
                                 fisher_type=fisher_type, mc_samples=mc_samples, seed=seed)
    return {m: weight_to_strips(f).sum(dim=1).cpu().tolist() for m, f in zip(modules, fisher)}

def saliency_from_traces(model, bits, modules, traces):
    """
    'quant_perturbation' saliency of every strip from given per-strip traces:
        1/2 * trace_s / n_s * (||Q_low(w_s) - w_s||^2 - ||Q_high(w_s) - w_s||^2)
    traces: {module: [trace per strip]}. Returns {module: [saliency per strip]}.
    """
    saliency = {}
    for m in modules:
        fold_scale = get_bn_fold_scale(model, m)
        gain = strip_quant_error(m, fold_scale, bits['insensitive']) - strip_quant_error(m, fold_scale, bits['highly_sensitive'])
        trace = torch.tensor(traces[m], dtype=gain.dtype, device=gain.device)
        saliency[m] = (0.5 * trace / m.in_channels * gain).cpu().tolist()
    return saliency

def strip_traces_from_saliency(model, bits, modules, strip_saliency_per_layer):
    """
    Recover tr(H_ss) of every strip from an existing 'quant_perturbation' saliency (no new Hutchinson run):
        tr(H_ss) = 2 * n_s * saliency_s / (||Q_low(w_s) - w_s||^2 - ||Q_high(w_s) - w_s||^2)
    The quantization errors only depend on the weights. Strips whose error gain is ~0 (e.g. all-zero
    strips) carry no trace information and get 0. Returns ({module: [trace per strip]}, number of such strips).
    """
    traces, num_unknown = {}, 0
    for m in modules:
        fold_scale = get_bn_fold_scale(model, m)
        gain = (strip_quant_error(m, fold_scale, bits['insensitive']) -
                strip_quant_error(m, fold_scale, bits['highly_sensitive'])).double()
        saliency = torch.tensor(strip_saliency_per_layer[m], dtype=torch.float64, device=gain.device)
        known = gain.abs() > 1e-12 * gain.abs().max().clamp(min=1e-30)
        trace = torch.where(known, 2 * m.in_channels * saliency / torch.where(known, gain, torch.ones_like(gain)),
                            torch.zeros_like(gain))
        num_unknown += int((~known).sum())
        traces[m] = trace.cpu().tolist()
    return traces, num_unknown

def aggregate_layer_trace(strip_importances_per_layer, modules):
    """Sum the already-computed per-strip Hutchinson trace estimates back up to one
    trace-sum per Conv2d layer. Reuses the existing per-strip samples — does NOT run a
    new Hessian/Hutchinson estimation.

    strip_importances_per_layer: {module: [tr(H_ss) per strip]}, i.e. the per-strip traces returned by
    compute_strip_importances(..., return_traces=True) or strip_traces_from_saliency (not the saliencies).
    The strips partition the layer, so the sum is the Hutchinson estimate of tr(H) of the whole layer.
    """
    return {m: float(sum(strip_importances_per_layer[m])) for m in modules}

def layer_quant_perturbation(model, bits, modules, layer_trace_sum):
    """HAWQ-v2 formula at full-layer granularity:
    0.5 * (layer_trace_sum[m] / weight.numel()) *
          (||Q_low(W_m) - W_m||^2 - ||Q_high(W_m) - W_m||^2)
    for each Conv2d module m. Reuse get_bn_fold_scale()/strip_quant_error() logic, applied to
    the whole weight tensor instead of per-strip slices.

    The quantizer is the deployed one (QuantBnConv2d, per-strip scales), so ||Q_b(W_m) - W_m||^2 is the
    sum of the strip errors; only the trace is averaged over the whole layer instead of per strip.
    """
    scores = {}
    for m in modules:
        fold_scale = get_bn_fold_scale(model, m)
        gain = (strip_quant_error(m, fold_scale, bits['insensitive']).sum() -
                strip_quant_error(m, fold_scale, bits['highly_sensitive']).sum()).item()
        scores[m] = 0.5 * layer_trace_sum[m] / m.weight.numel() * gain
    return scores
