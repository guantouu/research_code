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

@register_allocator('hawqv2_ilp')
def hawqv2_ilp_allocator(saliency, bits, ratio, **kwargs):
    """
    HAWQ-v2/V3 layer allocation as in the HAWQ release (github.com/Zhen-Dong/HAWQ, ILP.ipynb), on this
    pipeline's layer scores: choose the layers at the low bitwidth by an integer linear program
        minimize   sum_l x_l * layer_saliency_l            (x_l = 1: layer l at the low bitwidth)
        subject to sum_l x_l * weights_l >= ratio * total weights   (model size: avg bits within the limit)
    with the first conv layer fixed at the high bitwidth, as HAWQ keeps the first (and last) layer at 8 bits
    (the last layer here is the FC, which is not part of the strip bit config), and every residual-path conv
    tied to the first conv of its block (they read the same input), as ILP.ipynb ties the downsampling layers
    (residual_partner). Solved exactly with
    scipy.optimize.milp (HiGHS) instead of HAWQ's PuLP + GLPK. Unlike hawqv2_layer (layers taken in score
    order until the first one that does not fit, achieved ratio <= ratio), the achieved low-bit fraction
    is >= ratio, the choice is optimal for the scores, and it need not grow monotonically with the ratio.
    Scores below 0 are clipped to 0, and a 1e-9 relative penalty on low-bit weights picks, among equal
    scores, the fewest low-bit weights (so ratio 0 is all high). If the ratio cannot be reached without the
    first layer, every other layer is low-bit.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    missing = [n for n, s in saliency.items() if 'layer_saliency' not in s]
    if missing:
        raise ValueError("No 'layer_saliency' for {} layers (e.g. {}): re-run strip_wise_hessian_trace.py or "
                         "back-fill the file with scripts/add_layer_saliency.py".format(len(missing), missing[0]))
    names = list(saliency)
    free = names[1:]
    weights = np.array([len(saliency[n]['saliency']) * saliency[n]['strip_len'] for n in free], dtype=float)
    total = weights.sum() + len(saliency[names[0]]['saliency']) * saliency[names[0]]['strip_len']
    target = low_weight_budget(int(total), ratio)
    if target > weights.sum():
        low = np.ones(len(free), dtype=bool)
    elif target <= 0:
        low = np.zeros(len(free), dtype=bool)
    else:
        scores = np.array([max(saliency[n]['layer_saliency'], 0.0) for n in free])
        cost = scores + 1e-9 * max(scores.max(), 1e-30) * weights / weights.max()
        constraints = [LinearConstraint(weights[None, :], lb=target, ub=np.inf)]
        for i, n in enumerate(free):
            partner = residual_partner(n, free)
            if partner is not None:
                tie = np.zeros((1, len(free)))
                tie[0, i], tie[0, free.index(partner)] = 1, -1
                constraints.append(LinearConstraint(tie, lb=0, ub=0))
        res = milp(cost, constraints=constraints, integrality=np.ones(len(free)), bounds=Bounds(0, 1))
        if not res.success:
            raise RuntimeError('hawqv2_ilp: MILP failed at ratio {}: {}'.format(ratio, res.message))
        low = res.x > 0.5
    config = {names[0]: [bits['highly_sensitive']] * len(saliency[names[0]]['saliency'])}
    for n, l in zip(free, low):
        config[n] = [bits['insensitive'] if l else bits['highly_sensitive']] * len(saliency[n]['saliency'])
    return config

def residual_partner(name, names):
    """
    The conv whose bitwidth HAWQ ties to a residual-path conv: the first conv of the same block
    (CIFAR ResNet 'stageS.unitU.identity_conv' -> 'stageS.unitU.conv1', torchvision
    'layerL.B.downsample.0' -> 'layerL.B.conv1'); None for other layers or if it is not in names.
    """
    if name.endswith('.identity_conv'):
        partner = name[:-len('identity_conv')] + 'conv1'
    elif '.downsample.' in name:
        partner = name.split('.downsample.')[0] + '.conv1'
    else:
        return None
    return partner if partner in names else None

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

def snap_to_crossbar_capacity(bit_config, scores, bits, num_col_sub_array, column_bit, cell_bit):
    """
    Deterministic post-processing, called AFTER an r* has been selected and its bit_config
    built via the existing allocator path — NOT part of fim_pareto_search.py's search loop.

    utee/hook.py maps a b-bit strip to b / column_bit column groups of ceil(column_bit / cell_bit) cells,
    and NeuroSIM tiles each layer's columns contiguously into subarrays of num_col_sub_array columns, so
    a layer leaves its last subarray partly empty unless its cell-column count is a multiple of
    num_col_sub_array (8/2 with cellBit 1: an 8-bit strip is 8 columns, a 2-bit strip 2, so a layer of
    n strips with q high-bit ones has 2 (n + 3q) columns). For each mixed layer, q moves to the nearest
    count that fills its subarrays (ties: more high-bit strips), flipping the strips nearest the cut of
    scores ({layer: [score per strip]}, higher = keep high bits): the best low-bit strips are raised or
    the worst high-bit strips lowered. Uniform layers are left as they are, so layer-granularity
    configs stay per layer; a layer with no such count is left as it is.
    Returns (new bit_config, number of flipped strips).
    """
    high, low = bits['highly_sensitive'], bits['insensitive']
    def cells(b):
        return (b // column_bit) * math.ceil(column_bit / cell_bit)
    snapped, flipped = {}, 0
    for name, layer_bits in bit_config.items():
        n, q = len(layer_bits), layer_bits.count(high)
        snapped[name] = list(layer_bits)
        if q in (0, n):
            continue
        ok = [k for k in range(n + 1) if (n * cells(low) + k * (cells(high) - cells(low))) % num_col_sub_array == 0]
        if not ok or q in ok:
            continue
        target = min(ok, key=lambda k: (abs(k - q), -k))
        s = torch.tensor(scores[name], dtype=torch.float64)
        is_high = torch.tensor([b == high for b in layer_bits])
        if target > q:
            cand = torch.nonzero(~is_high).flatten()
            pick = cand[torch.sort(s[cand], descending=True, stable=True).indices[:target - q]]
            new = high
        else:
            cand = torch.nonzero(is_high).flatten()
            pick = cand[torch.sort(s[cand], stable=True).indices[:q - target]]
            new = low
        for i in pick.tolist():
            snapped[name][i] = new
        flipped += pick.numel()
    return snapped, flipped

def dual_crossbar_cost(num_strips, num_high, strip_len, bits, geometry):
    """
    Crossbars of one layer when its high-bit and low-bit strips sit in separate arrays (dual crossbar).
    A b-bit strip takes ceil(b / cell_bit) columns (NeuroSIM's numColPerSynapse), columns are tiled
    contiguously into crossbars of geometry['cols'] columns, and a strip of strip_len rows spans
    ceil(strip_len / geometry['rows']) crossbars vertically. num_high may be an int or an integer tensor.
    """
    cols_high = math.ceil(bits['highly_sensitive'] / geometry['cell_bit'])
    cols_low = math.ceil(bits['insensitive'] / geometry['cell_bit'])
    row_tiles = math.ceil(strip_len / geometry['rows'])
    num_high = torch.as_tensor(num_high)
    col_tiles = (torch.div(num_high * cols_high + geometry['cols'] - 1, geometry['cols'], rounding_mode='floor') +
                 torch.div((num_strips - num_high) * cols_low + geometry['cols'] - 1, geometry['cols'], rounding_mode='floor'))
    return row_tiles * col_tiles

def bit_config_crossbars(bit_config, saliency, bits, geometry):
    """
    Total dual-crossbar count of a bit config.
    """
    return sum(int(dual_crossbar_cost(len(b), b.count(bits['highly_sensitive']), saliency[n]['strip_len'], bits, geometry))
               for n, b in bit_config.items())

def crossbar_options(saliency, bits, geometry, layer_cost=None):
    """
    Per layer, the efficient (cost, gain, q) choices of a dual-crossbar allocation: q = number of high-bit
    strips, taken in decreasing saliency within the layer; gain = their saliency sum, each strip counted at
    least eps (eps = 1e-6 of the largest saliency). The floor matters because unconverged Hutchinson
    estimates make some saliencies negative: they stay last in the order, but a strip never lowers the gain,
    so free room in a crossbar is always filled with high-bit strips and the largest budget gives all-high.
    Choices that a cheaper (or equal-cost) choice matches on gain are dropped; at equal cost and gain the
    one with more high-bit strips is kept.
    layer_cost(name, num_strips, q) -> integer cost tensor for q = 0..num_strips; default dual_crossbar_cost
    (crossbars). crossbar_optimal_allocation solves the knapsack for any such integer cost.
    """
    eps = 1e-6 * max(max(s['saliency']) for s in saliency.values())
    options = {}
    for name, s in saliency.items():
        scores = torch.sort(torch.tensor(s['saliency'], dtype=torch.float64), descending=True, stable=True).values
        gain = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(scores.clamp(min=eps), 0)])
        q = torch.arange(scores.numel() + 1)
        cost = (dual_crossbar_cost(scores.numel(), q, s['strip_len'], bits, geometry) if layer_cost is None
                else layer_cost(name, scores.numel(), q))
        order = sorted(range(q.numel()), key=lambda i: (int(cost[i]), -float(gain[i]), -int(q[i])))
        kept, best = [], -math.inf
        for i in order:
            if gain[i] > best:
                kept.append((int(cost[i]), float(gain[i]), int(q[i])))
                best = float(gain[i])
        options[name] = kept
    return options

def crossbar_optimal_allocation(saliency, bits, budget, geometry, options=None):
    """
    Bit config with at most budget crossbars (dual crossbar, see dual_crossbar_cost) that maximizes the
    total saliency of its high-bit strips: a multiple-choice knapsack (one choice of q per layer, the q
    strips of highest saliency in the layer are high-bit), solved exactly by dynamic programming over the
    crossbar count. Saliency is the loss saved by keeping a strip at the high bitwidth, so this is the
    allocation of least estimated loss for that hardware (saliencies floored at a small eps, see
    crossbar_options). Returns (bit_config, crossbars used).
    """
    options = options or crossbar_options(saliency, bits, geometry)
    names = list(saliency)
    min_total = sum(opts[0][0] for opts in options.values())
    if budget < min_total:
        raise ValueError('Budget {} is below the minimum of {} crossbars'.format(budget, min_total))
    dp = torch.full((budget + 1,), -math.inf, dtype=torch.float64)
    dp[0] = 0.0
    choices = []
    for name in names:
        new = torch.full_like(dp, -math.inf)
        pick = torch.full((budget + 1,), -1, dtype=torch.int32)
        for j, (cost, gain, _) in enumerate(options[name]):
            if cost > budget:
                break
            cand = torch.full_like(dp, -math.inf)
            cand[cost:] = dp[:budget + 1 - cost] + gain
            better = cand > new
            new[better] = cand[better]
            pick[better] = j
        dp = new
        choices.append(pick)
    k = int(torch.argmax(dp))
    used = k
    bit_config = {}
    for name, pick in zip(reversed(names), reversed(choices)):
        cost, _, q = options[name][int(pick[k])]
        k -= cost
        order = torch.sort(torch.tensor(saliency[name]['saliency'], dtype=torch.float64), descending=True, stable=True).indices
        high = torch.zeros(len(saliency[name]['saliency']), dtype=torch.bool)
        high[order[:q]] = True
        bit_config[name] = [bits['highly_sensitive'] if h else bits['insensitive'] for h in high.tolist()]
    return {n: bit_config[n] for n in names}, used

def dual_column_reads(num_strips, num_high, positions, bits, geometry):
    """
    Column reads of one layer in a dual-crossbar chip: every output position reads each used column once,
    and a b-bit strip has ceil(b / cell_bit) columns. NeuroSIM's energy follows the column reads, not the
    number of crossbars (see utils.hardware_proxy).
    """
    cols_high = math.ceil(bits['highly_sensitive'] / geometry['cell_bit'])
    cols_low = math.ceil(bits['insensitive'] / geometry['cell_bit'])
    return positions * (num_high * cols_high + (num_strips - num_high) * cols_low)

def bit_config_column_reads(bit_config, net_structure, bits, geometry):
    """
    Total dual-crossbar column reads of a bit config; net_structure = {layer: output positions}.
    """
    return sum(dual_column_reads(len(b), b.count(bits['highly_sensitive']), net_structure[n], bits, geometry)
               for n, b in bit_config.items())

def energy_optimal_allocation(saliency, bits, budget, net_structure, geometry):
    """
    Bit config with at most budget column reads (dual crossbar, see dual_column_reads) that maximizes the
    total saliency of its high-bit strips (floored at a small eps, as in crossbar_options). Raising a strip
    costs a fixed number of extra column reads in its layer, so this is a knapsack with additive costs:
    strips are raised in decreasing saliency per extra column read until the budget is reached. This is
    optimal for the fractional problem and within one strip of the integer optimum; within a layer the
    order is the saliency order, and the high-bit set only grows with the budget.
    Returns (bit_config, column reads used).
    """
    eps = 1e-6 * max(max(s['saliency']) for s in saliency.values())
    names = list(saliency)
    base = sum(dual_column_reads(len(saliency[n]['saliency']), 0, net_structure[n], bits, geometry) for n in names)
    if budget < base:
        raise ValueError('Budget {} is below the {} column reads of all low-bit strips'.format(budget, base))
    gain = torch.cat([torch.tensor(saliency[n]['saliency'], dtype=torch.float64).clamp(min=eps) for n in names])
    extra = torch.cat([torch.full((len(saliency[n]['saliency']),),
                                  float(dual_column_reads(1, 1, net_structure[n], bits, geometry) -
                                        dual_column_reads(1, 0, net_structure[n], bits, geometry)), dtype=torch.float64)
                       for n in names])
    order = torch.sort(gain / extra, descending=True, stable=True).indices
    taken = base + torch.cumsum(extra[order], 0) <= budget
    high = torch.zeros(gain.numel(), dtype=torch.bool)
    high[order[taken]] = True
    bit_config, start = {}, 0
    for n in names:
        size = len(saliency[n]['saliency'])
        bit_config[n] = [bits['highly_sensitive'] if h else bits['insensitive'] for h in high[start:start + size].tolist()]
        start += size
    return bit_config, int(base + extra[order[taken]].sum())

def dual_adc_reads(num_strips, num_high, positions, strip_len, bits, geometry):
    """
    ADC conversions of one layer in a dual-crossbar chip: dual_column_reads times the row tiles, since a strip
    longer than geometry['rows'] spans several subarrays whose columns are converted separately. This is what
    the dual-crossbar NeuroSIM energy follows (scripts/fit_dual_energy.py: R^2 0.999 on ResNet20 8/2, against
    0.98 without the row tiles and 0.92 with the crossbar count alone).
    """
    return dual_column_reads(num_strips, num_high, positions, bits, geometry) * math.ceil(strip_len / geometry['rows'])

def bit_config_adc_reads(bit_config, net_structure, saliency, bits, geometry):
    """
    Total dual-crossbar ADC conversions of a bit config.
    """
    return sum(dual_adc_reads(len(b), b.count(bits['highly_sensitive']), net_structure[n], saliency[n]['strip_len'],
                              bits, geometry) for n, b in bit_config.items())

def dual_energy_model(bit_config, net_structure, saliency, bits, geometry, model):
    """
    Modeled dual-crossbar energy (uJ, before the merge cost): model['per_crossbar_uJ'] * crossbars +
    model['per_adc_read_uJ'] * ADC conversions + model.get('offset_uJ', 0), fitted to NeuroSIM by
    scripts/fit_dual_energy.py.
    """
    return (model.get('offset_uJ', 0.0) + model['per_crossbar_uJ'] * bit_config_crossbars(bit_config, saliency, bits, geometry) +
            model['per_adc_read_uJ'] * bit_config_adc_reads(bit_config, net_structure, saliency, bits, geometry))

def adc_energy_allocation(saliency, bits, budget, net_structure, geometry, model, units=20000):
    """
    Bit config with modeled energy (dual_energy_model, without the offset) at most budget uJ that maximizes the
    total saliency of its high-bit strips (floored at eps as in crossbar_options).
      per_crossbar_uJ == 0: the cost is additive per strip (extra ADC conversions of its layer), so strips are
          raised in decreasing saliency per extra uJ: optimal for the fractional knapsack, within one strip of
          the integer optimum, and the high-bit set grows with the budget.
      per_crossbar_uJ > 0: the crossbar term makes the cost a step function of each layer's high-bit count, so
          the exact multiple-choice knapsack DP of crossbar_optimal_allocation is used on the cost rounded to
          units steps over the full range (optimal up to that rounding).
    Returns (bit_config, modeled energy used in uJ).
    """
    alpha, beta = model['per_crossbar_uJ'], model['per_adc_read_uJ']
    names = list(saliency)
    if alpha == 0:
        eps = 1e-6 * max(max(s['saliency']) for s in saliency.values())
        base = sum(beta * dual_adc_reads(len(saliency[n]['saliency']), 0, net_structure[n], saliency[n]['strip_len'], bits, geometry)
                   for n in names)
        gain = torch.cat([torch.tensor(saliency[n]['saliency'], dtype=torch.float64).clamp(min=eps) for n in names])
        extra = torch.cat([torch.full((len(saliency[n]['saliency']),),
                                      beta * (dual_adc_reads(1, 1, net_structure[n], saliency[n]['strip_len'], bits, geometry) -
                                              dual_adc_reads(1, 0, net_structure[n], saliency[n]['strip_len'], bits, geometry)),
                                      dtype=torch.float64) for n in names])
        order = torch.sort(gain / extra, descending=True, stable=True).indices
        taken = base + torch.cumsum(extra[order], 0) <= budget * (1 + 1e-12)
        high = torch.zeros(gain.numel(), dtype=torch.bool)
        high[order[taken]] = True
        bit_config, start = {}, 0
        for n in names:
            size = len(saliency[n]['saliency'])
            bit_config[n] = [bits['highly_sensitive'] if h else bits['insensitive'] for h in high[start:start + size].tolist()]
            start += size
        return bit_config, float(base + extra[order[taken]].sum())

    def layer_energy(name, num_strips, q):
        s = saliency[name]
        return (alpha * dual_crossbar_cost(num_strips, q, s['strip_len'], bits, geometry).double() +
                beta * dual_adc_reads(num_strips, q, net_structure[name], s['strip_len'], bits, geometry).double())
    full = sum(float(layer_energy(n, len(s['saliency']), torch.tensor(len(s['saliency'])))) for n, s in saliency.items())
    unit = full / units
    options = crossbar_options(saliency, bits, geometry,
                               layer_cost=lambda name, num_strips, q: torch.round(layer_energy(name, num_strips, q) / unit).long())
    min_units = sum(o[0][0] for o in options.values())
    bit_config, _ = crossbar_optimal_allocation(saliency, bits, max(min_units, round(budget / unit)), geometry, options)
    used = sum(float(layer_energy(n, len(b), torch.tensor(b.count(bits['highly_sensitive'])))) for n, b in bit_config.items())
    return bit_config, used
