import torch
import sys
import os

def to_cpu(x):
    if isinstance(x, dict):
        return { k : to_cpu(v) for k, v in x.items() }
    elif isinstance(x, list):
        return [ to_cpu(v) for v in x ]
    elif isinstance(x, torch.Tensor):
        return x.cpu()
    else:
        return x

def model2numpy(model):
    return { k : v.cpu().numpy() for k, v in model.state_dict().items() }

def activation2numpy(output):
    if isinstance(output, dict):
        return { k : activation2numpy(v) for k, v in output.items() }
    elif isinstance(output, list):
        return [ activation2numpy(v) for v in output ]
    elif isinstance(output, Variable):
        return output.data.cpu().numpy()

def count_size(x):
    if isinstance(x, dict):
        return sum([ count_size(v) for k, v in x.items() ])
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum([ count_size(v) for v in x ])
    elif isinstance(x, torch.Tensor):
        return x.nelement() * x.element_size()
    else:
        return sys.getsizeof(x)

def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result

def get_mem_usage():
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s\t" % (mem2str(mem.available))
    result += "used: %s\t" % (mem2str(mem.used))
    result += "free: %s\t" % (mem2str(mem.free))
    # result += "active: %s\t" % (mem2str(mem.active))
    # result += "inactive: %s\t" % (mem2str(mem.inactive))
    # result += "buffers: %s\t" % (mem2str(mem.buffers))
    # result += "cached: %s\t" % (mem2str(mem.cached))
    # result += "shared: %s\t" % (mem2str(mem.shared))
    # result += "slab: %s\t" % (mem2str(mem.slab))
    return result


def accumulate(all_y, y):
    if all_y is None:
        all_y = dict()
        for k, v in y.items():
            if isinstance(v, list):
                all_y[k] = [ [vv] for vv in v ]
            else:
                all_y[k] = [v]
    else:
        for k, v in all_y.items():
            if isinstance(y[k], list):
                for vv, yy in zip(v, y[k]):
                    vv.append(yy)
            else:
                v.append(y[k])

    return all_y

def combine(all_y):
    output = dict()
    for k, v in all_y.items():
        if isinstance(v[0], list):
            output[k] = [ torch.cat(vv) for vv in v ]
        else:
            output[k] = torch.cat(v)

    return output

def concatOutput(loader, nets, condition=None):
    outputs = [None] * len(nets)

    use_cnn = nets[0].use_cnn

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if not use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()

            outputs = [ accumulate(output, to_cpu(net(x))) for net, output in zip(nets, outputs) ]
            if condition is not None and not condition(i):
               break

    return [ combine(output) for output in outputs ]


