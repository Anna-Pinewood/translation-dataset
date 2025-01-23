from datasets import Dataset
from huggingface_hub import HfApi
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class HFPublisher:
    """Handles publishing datasets to HuggingFace Hub."""

    def __init__(self, token: str, repo_id: str, save_path: str):
        """
        Initialize publisher with HF credentials and repository.

        Args:
            token: HuggingFace API token
            repo_id: Target repository ID (e.g., 'username/repo-name')
        """
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.save_path = save_path

    def prepare_dataset(self, manager) -> Tuple[Dataset, Path]:
        """
        Convert manager's DataFrame to HuggingFace Dataset and save locally.

        Args:
            manager: TranslationManager instance

        Returns:
            Tuple of (Dataset object, Path to saved dataset)
        """
        logger.info("Converting DataFrame to Dataset format")
        dataset = Dataset.from_pandas(manager._df)

        # Save dataset locally for inspection
        save_path = Path("prepared_dataset")
        dataset.save_to_disk(self.save_path)
        logger.info(f"Dataset saved locally at {save_path}")

        return dataset, save_path

    def publish(self, dataset_path: Path, branch: str = "main", private: bool = True) -> None:
        """
        Publish prepared dataset to HuggingFace Hub.

        Args:
            dataset_path: Path to saved dataset
            branch: Repository branch name
            private: Whether to create private repository
        """
        logger.info(f"Publishing dataset to {self.repo_id}")

        try:
            # Create repository
            self.api.create_repo(
                repo_id=self.repo_id,
                private=private,
                repo_type="dataset",
                exist_ok=True
            )

            # Upload dataset
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=dataset_path,
                repo_type="dataset",
                branch=branch
            )
            logger.info(f"Successfully published dataset to {self.repo_id}")

        except Exception as e:
            logger.error(f"Failed to publish dataset: {e}")
            raise
