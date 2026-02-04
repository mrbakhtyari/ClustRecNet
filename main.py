from src.training.train_pipeline import train_pipeline_main
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training pipeline with configurable model, seed and dataset.")
    parser.add_argument('--data-path', type=str, default="data/", help='Path to dataset')
    parser.add_argument('--model-name', type=str, default="resnet", choices=['resnet', 'baseline_cnn', 'no_cnn', 'no_res', 'no_att'],
                        help="Type of model to train (e.g., baseline_cnn, resnet, no_cnn, no_res, no_att)")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--run-name', type=str, default="", help="name for logging purposes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_pipeline_main(  
        data_path=args.data_path,
        seed=args.seed,
        model_name=args.model_name,
        epochs=args.epochs,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
