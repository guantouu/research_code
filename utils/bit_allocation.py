import math
import torch

# Bit allocation algorithms: name -> fn(saliency, bits, ratio, **kwargs) -> {layer_name: [bit per strip]}
#   saliency: {layer_name: {'strip_len': I, 'saliency': [score per strip], 'trace': [tr(H_ss) per strip],
#              'layer_saliency': HAWQ-v2 layer score}}, as saved by strip_wise_hessian_trace.py
#              ('trace' / 'layer_saliency' only in files written since the hawqv2_layer baseline,
#              or back-filled with scripts/add_layer_saliency.py)
#   bits:     {'highly_sensitive': 8, 'insensitive': 4}
#   ratio:    fraction of the WEIGHTS (over the whole network, each strip counted by its length) at the
#             low bitwidth, so the same ratio means the same compression for every allocator and granularity
# New algorithms are added with @register_allocator('name').
ALLOCATORS = {}

def register_allocator(name):
    def wrap(fn):
        ALLOCATORS[name] = fn
        return fn
    return wrap

def allocate_bits(method, saliency, bits, ratio, log=False, **kwargs):
    if method not in ALLOCATORS:
        raise ValueError("Unknown allocator: {} (available: {})".format(method, list(ALLOCATORS)))
    bit_config = ALLOCATORS[method](saliency, bits, ratio, **kwargs)
    if log:
        for name, layer_bits in bit_config.items():
            print("{}: high/low bit strips {}/{}".format(
                name, layer_bits.count(bits['highly_sensitive']), layer_bits.count(bits['insensitive'])))
    return bit_config

def low_weight_budget(total_weights, ratio):
    return int(math.floor(ratio * total_weights + 1e-9))

def lowest_within_budget(scores, sizes, ratio):
    """
    Mask of the units (strips or layers) at the low bitwidth: units are taken in increasing score order
    while the number of weights taken stays within ratio * (total weights). Stopping at the first unit that
    does not fit keeps the low-bit set nested in the ratio (a larger ratio only adds units).
    """
    order = torch.sort(scores, stable=True).indices
    taken = torch.cumsum(sizes[order], dim=0) <= low_weight_budget(int(sizes.sum()), ratio)
    low = torch.zeros(scores.numel(), dtype=torch.bool)
    low[order[taken]] = True
    return low

def global_topk(scores, bits, ratio, strip_len):
    """
    Give the high bitwidth to the top strips of the whole network ranked by scores ({layer_name: tensor});
    the lowest-ranked strips holding (up to) ratio of all weights get the low bitwidth.
    strip_len: {layer_name: weights per strip}.
    """
    names = list(scores)
    flat = torch.cat([scores[n] for n in names])
    sizes = torch.cat([torch.full((scores[n].numel(),), strip_len[n], dtype=torch.int64) for n in names])
    low = lowest_within_budget(flat, sizes, ratio)
    bit_config, start = {}, 0
    for n in names:
        size = scores[n].numel()
        bit_config[n] = [bits['insensitive'] if l else bits['highly_sensitive'] for l in low[start:start + size].tolist()]
        start += size
    return bit_config

@register_allocator('saliency')
def saliency_allocator(saliency, bits, ratio, **kwargs):
    """
    Hessian saliency ranking with one global threshold.
    """
    return global_topk({n: torch.tensor(s['saliency'], dtype=torch.float64) for n, s in saliency.items()}, bits, ratio,
                       {n: s['strip_len'] for n, s in saliency.items()})

@register_allocator('random')
def random_allocator(saliency, bits, ratio, seed=0, **kwargs):
    """
    Baseline: the same number of high-bit strips, chosen at random.
    """
    g = torch.Generator().manual_seed(seed)
    return global_topk({n: torch.rand(len(s['saliency']), generator=g, dtype=torch.float64) for n, s in saliency.items()}, bits, ratio,
                       {n: s['strip_len'] for n, s in saliency.items()})

@register_allocator('hawqv2_layer')
def hawqv2_layer_allocator(saliency, bits, ratio, **kwargs):
    """Layer-granularity HAWQ-v2 baseline: rank LAYERS (not strips) by
    layer_quant_perturbation, then assign every strip in a selected layer the low bit.
    Must emit a bit_config in the SAME per-strip output format as saliency_allocator /
    random_allocator (one bit value per strip index) — this makes it a drop-in for
    ratio_sweep.py, hardware_eval.py, and utee/hook.py with zero changes to any of them.

    Same estimator and formula as the strip allocator, only the granularity differs. Whole layers
    rarely add up to exactly ratio, so the achieved low-bit weight fraction is <= ratio: compare
    allocators at their achieved cost (avg_weight_bits / low_weight_fraction), not at the nominal ratio.
    """
    missing = [n for n, s in saliency.items() if 'layer_saliency' not in s]
    if missing:
        raise ValueError("No 'layer_saliency' for {} layers (e.g. {}): re-run strip_wise_hessian_trace.py or "
                         "back-fill the file with scripts/add_layer_saliency.py".format(len(missing), missing[0]))
    names = list(saliency)
    scores = torch.tensor([saliency[n]['layer_saliency'] for n in names], dtype=torch.float64)
    sizes = torch.tensor([len(saliency[n]['saliency']) * saliency[n]['strip_len'] for n in names], dtype=torch.int64)
    low = lowest_within_budget(scores, sizes, ratio).tolist()
    return {n: [bits['insensitive'] if l else bits['highly_sensitive']] * len(saliency[n]['saliency'])
            for n, l in zip(names, low)}

def bit_config_cost(bit_config, saliency, bits):
    """
    Hardware cost proxies: average weight bitwidth (weighted by strip length, i.e. cells), high-bit strip
    fraction and the achieved low-bit weight fraction (the quantity ratio targets).
    """
    total_bits = sum(sum(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    total_weights = sum(len(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    num_high = sum(b.count(bits['highly_sensitive']) for b in bit_config.values())
    num_strips = sum(len(b) for b in bit_config.values())
    low_weights = sum(b.count(bits['insensitive']) * saliency[n]['strip_len'] for n, b in bit_config.items())
    return {'avg_weight_bits': total_bits / total_weights, 'high_strip_fraction': num_high / num_strips,
            'low_weight_fraction': low_weights / total_weights}

def pareto_front(records, acc_key='acc1', cost_key='avg_weight_bits'):
    """
    Flags records that no other record beats on both accuracy (higher) and cost (lower).
    """
    flags = []
    for r in records:
        dominated = any(o[acc_key] >= r[acc_key] and o[cost_key] <= r[cost_key] and
                        (o[acc_key] > r[acc_key] or o[cost_key] < r[cost_key]) for o in records)
        flags.append(not dominated)
    return flags
