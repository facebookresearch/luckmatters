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

def tune_teacher_last_layer(eval_loader, teacher):
    output = utils.concatOutput(eval_loader, [teacher])

    # Tune the final linear layer to make output balanced as well. 
    y = output[0]["y"]
    y_mean = y.mean(dim=0).cuda()
    y_std = y.std(dim=0).cuda()
    
    teacher.final_w.weight.data /= y_std[:, None]
    teacher.final_w.bias.data -= y_mean
    teacher.final_w.bias.data /= y_std

    # double check
    output = utils.concatOutput(eval_loader, [teacher])
    y = output[0]["y"]
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)

    print(f"Final layer: y_mean: {y_mean}, y_std: {y_std}")

