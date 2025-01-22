from typing import List, Dict, Optional
from datasets import Dataset
import pandas as pd
import logging
from pathlib import Path
import json
import litellm
from config import CONFIG
from excel_exporter import ExcelExporter
from llm_interface import LLMInterface
from models import TranslationEntry

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translation process using pandas DataFrame."""

    # NEW: Translation format template stored as class variable
    TRANSLATION_FORMAT = '''{
    "translations": [
        {
            "idx": <original_idx>,
            "prompt_ru": "translated prompt text",
            "response_ru": "translated response text"
        }
    ]
}'''

    def __init__(
            self,
            path: Path,
            llm_config: Optional[dict] = None,
            exporter: Optional[ExcelExporter] = None):
        """Initialize manager with either Dataset or DataFrame path."""
        logger.info("Initializing TranslationManager with path: %s", path)
        self._path = path

        # Initialize exporter
        self.exporter = exporter or ExcelExporter()

        if str(path).endswith('.pkl'):
            self._df = pd.read_pickle(path)
        else:
            self._df = self._convert_dataset_to_df(path)
            df_path = path.parent / (path.name + '.pkl')
            self._df.to_pickle(df_path)
            logger.info(
                f"Converted Dataset to DataFrame and saved to {df_path}")

        self._validate_structure()

        # Initialize LLM if config provided
        self.llm = None
        if llm_config:
            self.llm = LLMInterface(**llm_config)

    def export_to_excel(
            self,
            output_path: Path,
            entries: List[TranslationEntry]) -> None:
        logger.info(
            f"Exporting {len(entries)} selected entries to Excel: {output_path}")
        self.exporter.export_to_excel(
            entries=entries,
            output_path=output_path,
            translated_only=False
        )

    def update_from_excel(self, excel_path: Path) -> Dict[str, any]:
        """
        Update entries from Excel file.

        Parameters
        ----------
        excel_path : Path
            Path to Excel file with updated translations

        Returns
        -------
        Dict[str, any]
            Update results
        """
        # Create dictionary of current entries
        current_entries = {
            idx: self.get_entry(idx)
            for idx in range(len(self._df))
        }

        # Get updated entries from Excel
        updated_entries = self.exporter.update_from_excel(
            excel_path=excel_path,
            current_entries=current_entries
        )

        # Update entries in DataFrame using existing method
        return self.update_entries(updated_entries)

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

    # MODIFIED: Split into separate methods for translation and updating
    def get_llm_translations(self, entries: List[TranslationEntry]) -> Dict:
        """
        Get translations from LLM and map them to original indices.

        Args:
            entries: List of TranslationEntry objects to translate

        Returns:
            Dictionary with mapped translations in format:
            {
                "translations": [
                    {
                        "idx": original_idx,
                        "prompt_ru": "translated text",
                        "response_ru": "translated text"
                    },
                    ...
                ]
            }
        """
        if not self.llm:
            raise ValueError(
                "LLM not initialized. Provide llm_config during initialization.")

        prompt = self._format_batch_for_translation(entries)
        response = self.llm.send_request(prompt=prompt)
        raw_translations = self.llm.get_response_content(response)

        if not isinstance(raw_translations, list) or len(raw_translations) != len(entries):
            raise ValueError(f"Invalid translation format received from LLM")

        # Map translations back to original indices
        translations = []
        for entry, trans in zip(entries, raw_translations):
            translations.append({
                "idx": entry.idx,
                "prompt_ru": trans["prompt_ru"],
                "response_ru": trans["response_ru"]
            })

        return {"translations": translations}

    def prepare_entries_for_update(self, translation_data: Dict) -> List[TranslationEntry]:
        """
        Create TranslationEntry objects from LLM translations.

        Args:
            translation_data: Dictionary with translations and mapped indices
        """
        prepared_entries = []
        for trans in translation_data["translations"]:
            entry = self.get_entry(trans["idx"])
            entry.set_translation(
                prompt_ru=trans["prompt_ru"],
                response_ru=trans["response_ru"]
            )
            prepared_entries.append(entry)
        return prepared_entries

    def update_entries(self, entries: List[TranslationEntry]) -> Dict[str, any]:
        """
        Update DataFrame with multiple translated entries.

        Args:
            entries: List of TranslationEntry objects to update

        Returns:
            Dictionary with update results
        """
        updated_indices = []
        for entry in entries:
            try:
                self._validate_entry(entry)
                new_data = entry.to_dict()
                self._df.at[entry.idx, 'prompt_ru'] = new_data['prompt_ru']
                self._df.at[entry.idx, 'response_ru'] = new_data['response_ru']
                updated_indices.append(entry.idx)
            except Exception as e:
                logger.error(f"Failed to update entry {entry.idx}: {str(e)}")

        self.save()
        return {
            "success": True,
            "updated_indices": updated_indices,
            "total_updated": len(updated_indices)
        }

    def _format_batch_for_translation(self, entries: List[TranslationEntry]) -> str:
        """Format entries for LLM translation without exposing original indices."""
        prompt_parts = [
            "I am working on LLM security project, translating training dataset entries",
            " to improve future russian models security.\n Help me –",
            "Translate the following entries from English to Russian.\n",
            "For each entry, translate both the prompt and the response.\n",
            "Return translations as a JSON list in the following format:\n",
            '''[
    {{
        "prompt_ru": "translated prompt text",
        "response_ru": "translated response text"
    }},
    // ... more entries in same order as provided
]\n''',
            "\nEntries to translate:\n"
        ]

        for i, entry in enumerate(entries, 1):
            data = entry.to_dict()
            prompt_parts.extend([
                f"\n=== Entry {i} ===",
                "PROMPT (ENG):",
                data['prompt'],
                "\nRESPONSE (ENG):",
                data['response'],
                "\n" + "="*40
            ])

        return "\n".join(prompt_parts)

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
            for i, entry in enumerate(entries):
                data = entry.to_dict()
                print(f"{i}. [{entry.idx}] ", end="")
                print(f"Prompt: {data['prompt'][:50]}...")
                print(
                    f"Trans: {data['prompt_ru'][:50] or '[NOT TRANSLATED]'}...")
                print("-" * 40)
        else:
            for i, entry in enumerate(entries):
                if i > 0:
                    print("\n")
                print("List index:", i)
                entry.display()

    def display_range(self, start_idx: int, count: int = 5, compact: bool = False) -> None:
        """Display a range of entries."""
        end_idx = min(start_idx + count, len(self._df))
        entries = [self.get_entry(i) for i in range(start_idx, end_idx)]
        self.display_entries(entries, compact=compact)

    # def translate_batch(self, start_idx: int, count: int = 5) -> Dict[str, any]:
    #     entries = [self.get_entry(i) for i in range(start_idx, start_idx + count)
    #                if not self._df.iloc[i]['prompt_ru'] or not self._df.iloc[i]['response_ru']]

    #     if not entries:
    #         logger.info("No untranslated entries found in specified range")
    #         return {"success": False, "message": "No untranslated entries found"}

    #     try:
    #         translations = self.get_llm_translations(entries)
    #         prepared_entries = self.prepare_entries_for_update(translations)
    #         result = self.update_entries(prepared_entries)

    #         return result

    #     except Exception as e:
    #         logger.error(f"Translation batch failed: {str(e)}")
    #         return {
    #             "success": False,
    #             "error": str(e),
    #             "message": "Failed to process translation batch"
    #         }
