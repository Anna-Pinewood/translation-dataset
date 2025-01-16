from typing import List, Dict, Optional
from datasets import Dataset
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TranslationEntry:
    """Represents a single translation entry."""

    def __init__(self, data: dict, idx: int):
        self._data = data.copy()
        self._idx = idx
        self._mutable_fields = {'prompt_ru', 'response_ru'}

    def set_translation(self, prompt_ru: str | None, response_ru: str | None) -> None:
        """Set Russian translations for both prompt and response."""
        if not prompt_ru and not response_ru:
            raise ValueError("At least one of the fields must be provided")
        if not prompt_ru or not response_ru:
            logger.warning("Only one of the fields is provided")
        logger.info("Setting translation for entry %d", self._idx)
        if prompt_ru:
            self._data['prompt_ru'] = prompt_ru
        if response_ru:
            self._data['response_ru'] = response_ru

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def is_translated(self) -> bool:
        return bool(self._data['prompt_ru'] and self._data['response_ru'])

    def to_dict(self) -> dict:
        return self._data.copy()

    def display(self) -> None:
        """Display entry in formatted view."""
        print(f"\nEntry [{self._idx}]")
        is_response_unsecure = "UNSECURE" if self._data['is_response_unsecure'] else "SECURE"
        print(f"{'*'*20}Response is {is_response_unsecure}{'*'*20}")
        print("=" * 80)
        print("PROMPT (ENG):")
        print(self._data['prompt'])
        print("\nPROMPT (RU):")
        print(self._data['prompt_ru'] or "[NOT TRANSLATED]")
        print("\nRESPONSE (ENG):")
        print(self._data['response'])
        print("\nRESPONSE (RU):")
        print(self._data['response_ru'] or "[NOT TRANSLATED]")
        print("=" * 80)


class TranslationManager:
    """Manages translation process using pandas DataFrame."""

    def __init__(self, path: Path):
        """Initialize manager with either Dataset or DataFrame path."""
        logger.info("Initializing TranslationManager with path: %s", path)
        self._path = path

        if str(path).endswith('.pkl'):
            self._df = pd.read_pickle(path)
        else:
            # Assuming it's a Dataset path
            logger.info("Converting Dataset to DataFrame")
            self._df = self._convert_dataset_to_df(path)
            # Save as DataFrame for future use
            logger.info("Saving DataFrame to pickle, path: %s",
                        path.parent / (path.name + '.pkl'))
            df_path = path.parent / (path.name + '.pkl')
            self._df.to_pickle(df_path)
            logger.info(
                f"Converted Dataset to DataFrame and saved to {df_path}")

        self._validate_structure()

    @staticmethod
    def _convert_dataset_to_df(dataset_path: Path) -> pd.DataFrame:
        """Convert HF dataset to DataFrame."""
        dataset = Dataset.load_from_disk(str(dataset_path))
        return pd.DataFrame(dataset)

    def convert_to_dataset(self, save_path: Optional[Path] = None) -> Dataset:
        """Convert current DataFrame to Dataset format."""
        dataset = Dataset.from_pandas(self._df)
        if save_path:
            dataset.save_to_disk(str(save_path))
        return dataset

    def _validate_structure(self) -> None:
        """Validate DataFrame structure."""
        required_fields = {'prompt', 'response',
                           'prompt_ru', 'response_ru', 'idx'}
        missing_fields = required_fields - set(self._df.columns)
        if missing_fields:
            raise ValueError(f"Data missing required fields: {missing_fields}")

    def get_entry(self, idx: int) -> TranslationEntry:
        """Get specific entry by index."""
        logger.debug("Retrieving entry %d", idx)
        row = self._df.iloc[idx].to_dict()
        return TranslationEntry(row, idx)

    def get_next_untranslated(self) -> Optional[TranslationEntry]:
        """Get next untranslated entry."""
        mask = (self._df['prompt_ru'].isna()) | (self._df['prompt_ru'] == '') | \
               (self._df['response_ru'].isna()) | (
                   self._df['response_ru'] == '')
        if mask.any():
            idx = self._df[mask].index[0]
            return self.get_entry(idx)
        logger.info("No untranslated entries found")
        return None

    def get_untranslated_entries(self, limit: int = 10) -> List[TranslationEntry]:
        """Get multiple untranslated entries."""
        mask = (self._df['prompt_ru'].isna()) | (self._df['prompt_ru'] == '') | \
               (self._df['response_ru'].isna()) | (
                   self._df['response_ru'] == '')
        indices = self._df[mask].index[:limit]
        entries = [self.get_entry(idx) for idx in indices]
        logger.debug("Retrieved %d untranslated entries", len(entries))
        return entries

    def update_entry(self, entry: TranslationEntry) -> None:
        """Update DataFrame with translated entry."""
        logger.debug("Updating entry %d in DataFrame", entry.idx)
        self._validate_entry(entry)

        # Update only translation fields
        new_data = entry.to_dict()
        self._df.at[entry.idx, 'prompt_ru'] = new_data['prompt_ru']
        self._df.at[entry.idx, 'response_ru'] = new_data['response_ru']
        self.save()

    def _validate_entry(self, entry: TranslationEntry) -> None:
        """Ensure only translation fields were modified."""
        original = self._df.iloc[entry.idx].to_dict()
        new_data = entry.to_dict()

        for key, value in original.items():
            if key not in {'prompt_ru', 'response_ru'}:
                if pd.isna(value) and pd.isna(new_data[key]):
                    continue
                assert new_data[key] == value, f"Field {key} was modified"

    def save(self) -> None:
        """Save DataFrame to pickle."""
        save_path = self._path if str(self._path).endswith(
            '.pkl') else Path(str(self._path) + '.pkl')
        logger.info("Saving DataFrame to %s", save_path)
        self._df.to_pickle(save_path)

    def get_progress(self) -> Dict[str, int]:
        """Get translation progress statistics."""
        total = len(self._df)
        mask = (self._df['prompt_ru'].notna()) & (self._df['prompt_ru'] != '') & \
               (self._df['response_ru'].notna()) & (
                   self._df['response_ru'] != '')
        translated = mask.sum()

        stats = {
            'total': total,
            'translated': translated,
            'remaining': total - translated
        }
        logger.info("Progress: %d/%d entries translated", translated, total)
        return stats

    def display_entries(self, entries: List[TranslationEntry], compact: bool = False) -> None:
        """Display multiple entries."""
        if compact:
            for entry in entries:
                data = entry.to_dict()
                print(f"[{entry.idx}] ", end="")
                print(f"Prompt: {data['prompt'][:50]}...")
                print(
                    f"Trans: {data['prompt_ru'][:50] or '[NOT TRANSLATED]'}...")
                print("-" * 40)
        else:
            for i, entry in enumerate(entries):
                if i > 0:
                    print("\n")
                entry.display()

    def display_range(self, start_idx: int, count: int = 5, compact: bool = False) -> None:
        """Display a range of entries."""
        end_idx = min(start_idx + count, len(self._df))
        entries = [self.get_entry(i) for i in range(start_idx, end_idx)]
        self.display_entries(entries, compact=compact)
