from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_max_corr_alpha(stats, teacher_thres=0.2, student_thres=0.2):
    s = stats[0][-1]
    num_layer = len(s["train_corrs"])
    iterations = [0, 10, 20, 30, 39]
    
    plt.figure(figsize=(20, 10))
    count = 0
    
    beta_limits = [0.4, 0.4, 0.6, 0.8]

    for layer in range(num_layer - 1, -1, -1):
    #for layer in range(num_layer):
        teacher_act_freq = (s["train_teacher_h"][layer] > 1e-5).float().mean(dim=0)
        teacher_sel = (teacher_act_freq - 0.5).abs() < teacher_thres
        # teacher_sel = teacher_act_freq > 0
        # teacher_sel = [0]
        
        print(f"{layer}: student/teacher: {s['train_corrs'][layer].size()}, good teacher ratio: {teacher_sel.float().mean():#.3f}")

        for it in iterations:
            count += 1
            ax = plt.subplot(num_layer, len(iterations), count)

            s = stats[0][it]
            train_corrs = s["train_corrs"][layer]
            alphas = s["train_betas_s"][layer][:-1,:-1]
            betas = s["train_betas"][layer][:-1, :-1].diag()
            
            student_act_freq = (s["train_student_h"][layer] > 1e-5).float().mean(dim=0)
            student_filter = (student_act_freq - 0.5).abs() < student_thres
            
            # sorted_energy, sorted_energy_indices = s["train_student_h"][layer].mean(dim=0).sort(descending=True)        
            # student_filter = sorted_energy_indices[:sorted_energy_indices.size(0) // 3]
            
            alphas = alphas[student_filter, :][:, teacher_sel]
            betas = betas[student_filter]
            train_corrs = train_corrs[student_filter, :][:, teacher_sel]
            
            student_usefulness, best_matched_teacher_indices = train_corrs.max(dim=1)
            student_fanout_coeffs = alphas.gather(1, best_matched_teacher_indices.view(-1, 1))
            
            student_ratio = student_filter.float().mean()
            
            # print(f"{student_fanout_coeffs.size()}, {student_usefulness.size()}")
            
            # plt.scatter(student_usefulness.numpy(), student_fanout_coeffs.numpy(), alpha=0.2)
            plt.scatter(student_usefulness.numpy(), betas.sqrt().numpy(), alpha=0.2)
            
            if it == 0:
                plt.ylabel("$\\sqrt{\\mathbb{E}_{\\mathbf{x}}\\left[\\beta_{kk}(\\mathbf{x})\\right]}$")
            else:
                ax.set_yticklabels([])
                
            if layer == 0:
                plt.xlabel("Max correlation among teacher")

            plt.axis([-0.05, 1.05, -0.001, beta_limits[layer]])
            
            if layer == 3:
                plt.title(f"Epoch {it}")
        # plt.legend()
            
    plt.savefig(f"max_corr_alpha-teacher{teacher_thres}-student{student_thres}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('root', type=str, help="root directory")

    args = parser.parse_args()

    stats = load_stats(args.root)
    plot_max_corr_alpha(stats, teacher_thres=0.2, student_thres=0.2)

