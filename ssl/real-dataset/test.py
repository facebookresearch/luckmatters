import hydra
import torch

@hydra.main(config_path="./config", config_name="test.yaml")
def main(args):
    print(f"Seed: {args.seed}, Cuda #device: {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()
