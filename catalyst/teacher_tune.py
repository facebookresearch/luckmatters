import torch
import utils

def tune_teacher(eval_loader, teacher):
    # Tune the bias of the teacher so that their activation/inactivation is approximated 0.5/0.5
    num_hidden = teacher.num_hidden_layers()
    for t in range(num_hidden):
        output = utils.concatOutput(eval_loader, [teacher])
        estimated_bias = output[0]["post_lins"][t].median(dim=0)[0]
        teacher.ws_linear[t].bias.data[:] -= estimated_bias.cuda() 
      
    # double check
    output = utils.concatOutput(eval_loader, [teacher])
    for t in range(num_hidden):
        activate_ratio = (output[0]["post_lins"][t] > 0).float().mean(dim=0)
        print(f"{t}: {activate_ratio}")
