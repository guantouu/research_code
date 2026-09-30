import math
import torch

# Bit allocation algorithms: name -> fn(saliency, bits, ratio, **kwargs) -> {layer_name: [bit per strip]}
#   saliency: {layer_name: {'strip_len': I, 'saliency': [score per strip]}}, as saved by strip_wise_hessian_trace.py
#   bits:     {'highly_sensitive': 8, 'insensitive': 4}
#   ratio:    fraction of strips (over the whole network) kept at the low bitwidth
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

def num_high_strips(total, ratio):
    return total - int(math.floor(ratio * total + 1e-9))

def global_topk(scores, bits, ratio):
    """
    Give the high bitwidth to the top strips of the whole network ranked by scores ({layer_name: tensor}).
    """
    names = list(scores)
    flat = torch.cat([scores[n] for n in names])
    high = torch.zeros(flat.numel(), dtype=torch.bool)
    k = num_high_strips(flat.numel(), ratio)
    if k > 0:
        high[torch.topk(flat, k).indices] = True
    bit_config, start = {}, 0
    for n in names:
        size = scores[n].numel()
        bit_config[n] = [bits['highly_sensitive'] if h else bits['insensitive'] for h in high[start:start + size].tolist()]
        start += size
    return bit_config

@register_allocator('saliency')
def saliency_allocator(saliency, bits, ratio, **kwargs):
    """
    Hessian saliency ranking with one global threshold.
    """
    return global_topk({n: torch.tensor(s['saliency'], dtype=torch.float64) for n, s in saliency.items()}, bits, ratio)

@register_allocator('random')
def random_allocator(saliency, bits, ratio, seed=0, **kwargs):
    """
    Baseline: the same number of high-bit strips, chosen at random.
    """
    g = torch.Generator().manual_seed(seed)
    return global_topk({n: torch.rand(len(s['saliency']), generator=g, dtype=torch.float64) for n, s in saliency.items()}, bits, ratio)

def bit_config_cost(bit_config, saliency, bits):
    """
    Hardware cost proxies: average weight bitwidth (weighted by strip length, i.e. cells) and high-bit strip fraction.
    """
    total_bits = sum(sum(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    total_weights = sum(len(b) * saliency[n]['strip_len'] for n, b in bit_config.items())
    num_high = sum(b.count(bits['highly_sensitive']) for b in bit_config.values())
    num_strips = sum(len(b) for b in bit_config.values())
    return {'avg_weight_bits': total_bits / total_weights, 'high_strip_fraction': num_high / num_strips}

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
