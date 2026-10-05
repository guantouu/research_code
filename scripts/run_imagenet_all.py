"""
Overnight ImageNet PTQ (8/2) pipeline for several networks, in priority order:
convert (check published accuracy) -> Hessian saliency -> ratio sweep (saliency + random seeds 0-2, hawqv2_layer in seed 0)
-> hardware evaluation (NeuroSIM, CPU) launched in the background while the next network uses the GPU.
Every step is skipped if its output exists, so the script can be re-run after an interruption.
Progress: log/runs/imagenet_status.txt; per-step logs: log/runs/imagenet_<net>_<step>.out.
Run from the repo root (logs stay in log/runs, which is not tracked):
    setsid nohup python scripts/run_imagenet_all.py > log/runs/imagenet_orchestrator.out 2>&1 < /dev/null &
"""
import csv
import json
import os
import subprocess
import time

NETS = ['resnet18', 'resnet50', 'vgg16', 'resnet34', 'vgg11', 'resnet101', 'vgg13', 'vgg19', 'resnet152']
HESSIAN_BATCH = {'resnet18': 64, 'resnet34': 64, 'resnet50': 32, 'resnet101': 32, 'resnet152': 16,
                 'vgg11': 16, 'vgg13': 16, 'vgg16': 16, 'vgg19': 16}
EVAL_BATCH = {net: (128 if net.startswith('vgg') else 256) for net in NETS}
KNEE_DROPS = (10.0, 25.0)   # hardware ratios: largest ratio whose saliency accuracy is within each drop of all-8-bit
CFG = 'configs/imagenet/8_2'
LOG = 'log/runs'
STATUS = os.path.join(LOG, 'imagenet_status.txt')

def status(msg):
    line = '{} {}'.format(time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')

def run(net, step, cmd):
    t = time.time()
    out = os.path.join(LOG, 'imagenet_{}_{}.out'.format(net, step))
    with open(out, 'w') as f:
        ok = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0
    status('[{}] {} {} ({:.0f} min){}'.format(net, step, 'done' if ok else 'FAILED', (time.time() - t) / 60,
                                             '' if ok else ', see ' + out))
    return ok

def config(name, **updates):
    """
    Config derived from the ResNet18 one (configs/imagenet/8_2/<name with resnet18>.json).
    """
    with open(os.path.join(CFG, name.format(net='resnet18'))) as f:
        c = json.load(f)
    c.update(updates)
    return c

def write(c, path):
    with open(path, 'w') as f:
        json.dump(c, f, indent=4)
    return path

def sweep_results(net, seed):
    return 'log/{}/imagenet/sweep/ratio_sweep_8_2_seed{}/results.csv'.format(net, seed)

def choose_ratios(net):
    rows = [r for r in csv.DictReader(open(sweep_results(net, 0))) if r['allocator'] == 'saliency']
    acc = {float(r['ratio']): float(r['acc1']) for r in rows}
    ratios = set()
    for drop in KNEE_DROPS:
        ok = [r for r, a in acc.items() if 0 < r < 1 and a >= acc[0.0] - drop]
        ratios.add(max(ok) if ok else min(r for r in acc if r > 0))
    status('[{}] saliency sweep {} -> hardware ratios {} (drops {})'.format(
        net, {r: round(a, 1) for r, a in sorted(acc.items())}, sorted(ratios), KNEE_DROPS))
    return sorted(ratios)

def main():
    os.makedirs(LOG, exist_ok=True)
    status('=== start: {}'.format(NETS))
    hardware = []
    for net in NETS:
        saliency = 'saliency/{}_imagenet_quant_perturbation_8_2.json'.format(net)
        if not os.path.exists('log/{}/imagenet/best.pth'.format(net)):
            if not run(net, 'convert', ['python', 'convert_pretrained.py', '--net', net, '--dataset', 'imagenet']):
                continue
        if not os.path.exists(saliency):
            c = config('hessian_trace_{net}.json', net=net, batch_size=HESSIAN_BATCH[net])
            path = write(c, os.path.join(CFG, 'hessian_trace_{}.json'.format(net)))
            if not run(net, 'hessian', ['python', 'strip_wise_hessian_trace.py', '--config', path]):
                continue
        elif 'layer_saliency' not in next(iter(json.load(open(saliency)).values())):
            # saliency from before the hawqv2_layer baseline: add the layer scores without a new Hessian run
            if not run(net, 'layer_saliency', ['python', 'scripts/add_layer_saliency.py', '--saliency_file', saliency,
                                               '--config', os.path.join(CFG, 'hessian_trace_{}.json'.format(net))]):
                continue
        failed = False
        for seed in [0, 1, 2]:
            if os.path.exists(sweep_results(net, seed)):
                continue
            c = config('ratio_sweep_{net}_seed%d.json' % seed, net=net, batch_size=EVAL_BATCH[net], saliency_file=saliency)
            path = write(c, os.path.join(CFG, 'ratio_sweep_{}_seed{}.json'.format(net, seed)))
            if not run(net, 'sweep_seed{}'.format(seed), ['python', 'ratio_sweep.py', '--config', path]):
                failed = True
                break
        if failed:
            continue
        if os.path.exists('log/{}/imagenet/hardware/ptq_8_2/summary.csv'.format(net)):
            continue
        designs = [{'name': 'all8', 'uniform': 8}]
        for r in choose_ratios(net):
            designs += [{'name': 'saliency_{}'.format(r), 'allocator': 'saliency', 'ratio': r},
                        {'name': 'random_{}'.format(r), 'allocator': 'random', 'ratio': r},
                        {'name': 'hawqv2_layer_{}'.format(r), 'allocator': 'hawqv2_layer', 'ratio': r}]
        designs.append({'name': 'all2', 'uniform': 2})
        c = config('hardware_eval_{net}.json', net=net, saliency_file=saliency, designs=designs,
                   batch_size=EVAL_BATCH[net] // 4, max_parallel=len(designs))
        path = write(c, os.path.join(CFG, 'hardware_eval_{}.json'.format(net)))
        out = open(os.path.join(LOG, 'imagenet_{}_hardware.out'.format(net)), 'w')
        status('[{}] hardware evaluation started in the background: {}'.format(net, [d['name'] for d in designs]))
        hardware.append((net, time.time(), subprocess.Popen(['python', 'hardware_eval.py', '--config', path],
                                                            stdout=out, stderr=subprocess.STDOUT)))
    for net, t, proc in hardware:
        ok = proc.wait() == 0
        status('[{}] hardware {} ({:.0f} min since start)'.format(net, 'done' if ok else 'FAILED', (time.time() - t) / 60))
    status('=== all done')

if __name__ == '__main__':
    main()
