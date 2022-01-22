import common_utils
import re
import os
import glob
import torch

# resnet check. 
matcher = re.compile(r"encoder.layer([0-9]).([0-9]).conv([0-9]).weight")
def get_entropy(model):
    res = dict()
    for k, v in model["online_network_state_dict"].items():
      key = None
      if k == "encoder.conv1.weight":
        key = "conv1"
      else:  
        m = matcher.match(k) 
        if m:
          key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" 

      if key is not None:
        # Compute the sparsity of each filter. 
        v = v.view(v.size(0), -1).abs()
        v = v / (v.sum(dim=1, keepdim=True) + 1e-8)
        entropies = - (v * (v + 1e-8).log()).sum(dim=1)
        res["h_" + key] = entropies.mean().item()

    return res

def edge_energy(f):
    dx = f[1:, :, :] - f[:-1, :, :]
    dy = f[:, 1:, :] - f[:, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean()) / 2


def check_edge_stats(subfolder):
    model_files = glob.glob(os.path.join(subfolder, "checkpoints/model_*.pth"))
    # Find the latest.
    model_files = [ (os.path.getmtime(f), f) for f in model_files ]
    all_model_files = sorted(model_files, key=lambda x: x[0])

    config = common_utils.MultiRunUtil.load_full_cfg(subfolder)

    if len(all_model_files) == 0:
      return None

    last_model_file = all_model_files[-1][1]
    model = torch.load(last_model_file, map_location=torch.device('cpu'))

    avg_edge_strength = 0

    # res = get_entropy(model)
    ws = model["online_network_state_dict"]["encoder.conv1.weight"]
    for k in range(ws.size(0)):
      w = ws[k,:].permute(1, 2, 0)    
      edge_strength = edge_energy(w)
      avg_edge_strength += edge_strength

    avg_edge_strength /= ws.size(0)

    return dict(edge_strength=avg_edge_strength.item() * 1000)

_result_matcher = [
  {
    "match": re.compile(r"Epoch (\d+): best_acc: ([\d\.]+)"),
    "action": [
      [ "acc", "float(m.group(2))" ]
    ]
  }
]

_attr_multirun = {
  "check_result" : {
    "default": lambda subfolder: common_utils.MultiRunUtil.load_regex(subfolder, _result_matcher),
    "entropy": check_edge_stats
  },
  "default_metrics": [ "acc" ],
  "specific_options": dict(acc={}),
  "common_options" : dict(topk_mean=1, topk=10, descending=True),
}
