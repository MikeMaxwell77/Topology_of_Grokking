"""Compatibility module for dataset creation."""

from grokking.data import ModularArithmeticDataset, make_dataloaders

__all__ = ["ModularArithmeticDataset", "make_dataloaders"]


def main() -> None:
    train_loader, validation_loader = make_dataloaders(
        p=113, c=1, r=113, train_fraction=0.3, batch_size=512
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(validation_loader.dataset)}")


if __name__ == "__main__":
    main()
