from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_multilayer_l_shape(stats, epoch_split=5):
    s = stats[0][-1]
    num_layer = len(s["train_corrs"])

    total_epoch = len(stats[0]) - 1
    epochs = [ int(i * total_epoch / (epoch_split - 1)) for i in range(epoch_split) ]
    
    plt.figure(figsize=(20, 10))
    count = 0

    beta_limits = [1, 1, 1, 1]

    for layer in range(num_layer - 1, -1, -1):
        print(f"{layer}: student/teacher: {s['train_corrs'][layer].size()}")

        for it in epochs:
            count += 1
            ax = plt.subplot(num_layer, len(epochs), count)

            s = stats[0][it]
            train_corrs = s["train_corrs"][layer]
            alphas = s["train_betas_s"][layer][:-1,:-1]
            betas = s["train_betas"][layer][:-1, :-1].diag()
            
            student_usefulness, best_matched_teacher_indices = train_corrs.max(dim=1)
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
            
    plt.savefig(f"multilayer_l_shape.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('root', type=str, help="root directory")

    args = parser.parse_args()

    stats = load_stats(args.root)
    plot_multilayer_l_shape(stats)

