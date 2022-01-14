import common_utils
import re

_result_matcher = [
  {
    "match": re.compile(r"Epoch (\d+): best_acc: ([\d\.]+)"),
    "action": [
      [ "acc", "float(m.group(2))" ]
    ]
  }
]

_attr_multirun = {
    "check_result": lambda x: common_utils.MultiRunUtil.load_regex(x, _result_matcher),
    "common_options" : dict(topk_mean=1, topk=10, descending=True),
    "metrics": dict(acc={}),
}
