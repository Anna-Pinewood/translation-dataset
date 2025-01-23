import json
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
from translation_manager import TranslationManager
from models import TranslationEntry
from openai import OpenAI
from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes large-scale translation tasks using OpenAI's batch API.
    Handles creation and monitoring of translation batches.
    """

    # Constants with clearer naming
    ENTRIES_PER_MESSAGE = 10  # Number of TranslationEntries in one model prompt
    MESSAGES_PER_BATCH = 100  # Number of messages in one OpenAI batch request
    TOTAL_BATCHES = 100  # Total number of OpenAI batch requests to process
    TRANSLATION_START_IDX = 100000  # Starting index for translations

    def __init__(self, manager: TranslationManager):
        """Initialize BatchProcessor with necessary components and paths."""
        logger.info("Initializing BatchProcessor")
        self.manager = manager
        self.openai_client = OpenAI(
            api_key=CONFIG.llm_openai.api_key,
            base_url=CONFIG.llm_openai.base_url
        )

        # Setup batch processing directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_output_dir = Path("batches") / self.timestamp
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory at {self.batch_output_dir}")

        self.batch_records: List[Dict[str, Any]] = []

    def create_openai_message(
        self,
        entries: List[TranslationEntry],
        batch_num: int,
        message_num: int
    ) -> Dict[str, Any]:
        """
        Create a single OpenAI API message request for a group of translation entries.

        Args:
            entries: List of TranslationEntry objects to translate
            batch_num: Current OpenAI batch number
            message_num: Message number within the current batch

        Returns:
            Dictionary containing the OpenAI API request configuration
        """
        logger.debug(
            f"Creating message {message_num} for batch {batch_num} with {len(entries)} entries")

        prompt = self.manager._format_batch_for_translation(entries)
        message_id = f"batch_{batch_num}_msg_{message_num}_{self.timestamp}"

        return {
            "custom_id": message_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "response_format": {"type": "json_object"}
            }
        }

    def process_translation_batch(self, start_idx: int, batch_num: int) -> Dict[str, Any]:
        """
        Process a batch of translations through OpenAI's batch API.

        Args:
            start_idx: Starting index for translation entries
            batch_num: Current batch number

        Returns:
            Dictionary containing batch processing information
        """
        logger.info(
            f"Processing translation batch {batch_num} starting at index {start_idx}")

        # Calculate range for this batch
        entries_per_batch = self.ENTRIES_PER_MESSAGE * self.MESSAGES_PER_BATCH
        end_idx = start_idx + entries_per_batch

        # Get all entries for this batch
        entries = [self.manager.get_entry(i)
                   for i in range(start_idx, end_idx)]
        logger.info(f"Retrieved {len(entries)} entries for processing")

        # Create individual message requests
        batch_messages = []
        for msg_idx in range(0, len(entries), self.ENTRIES_PER_MESSAGE):
            entry_group = entries[msg_idx:msg_idx + self.ENTRIES_PER_MESSAGE]
            message = self.create_openai_message(
                entry_group,
                batch_num,
                msg_idx // self.ENTRIES_PER_MESSAGE
            )
            batch_messages.append(message)

        # Save batch messages to JSONL file
        batch_file = self.batch_output_dir / \
            f"batch_{batch_num}_messages.jsonl"
        with open(batch_file, "w", encoding="utf-8") as f:
            for msg in batch_messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        logger.info(f"Saved batch messages to {batch_file}")

        # Create OpenAI batch request
        with open(batch_file, "rb") as f:
            batch_file_obj = self.openai_client.files.create(
                file=f,
                purpose="batch"
            )

        # Submit batch to OpenAI
        batch = self.openai_client.batches.create(
            input_file_id=batch_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"translation batch {batch_num}",
                "entries_range": f"{start_idx}-{end_idx}"
            }
        )

        batch_info = batch.model_dump()
        logger.info(f"Successfully submitted batch {batch_num} to OpenAI")
        return batch_info

    def check_batch_status(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the status of a submitted batch.

        Args:
            batch_info: Batch information dictionary

        Returns:
            Updated status information
        """
        logger.debug(f"Checking status for batch {batch_info['id']}")
        status = self.openai_client.batches.retrieve(batch_info["id"])
        return status.model_dump()

    def _save_batch_records(self) -> None:
        """Save current batch processing records to JSON file."""
        info_file = self.batch_output_dir / "batch_processing_records.json"
        record_data = {
            "timestamp": self.timestamp,
            "total_batches": self.TOTAL_BATCHES,
            "entries_per_message": self.ENTRIES_PER_MESSAGE,
            "messages_per_batch": self.MESSAGES_PER_BATCH,
            "batches": self.batch_records
        }

        with open(info_file, "w") as f:
            json.dump(record_data, f, indent=2)
        logger.debug("Updated batch processing records")

    def run(self) -> None:
        """
        Run the batch processing pipeline.
        Processes all translation batches and monitors their status.
        """
        logger.info("Starting batch processing pipeline")

        for batch_num in range(self.TOTAL_BATCHES):
            start_idx = self.TRANSLATION_START_IDX + (
                batch_num * self.ENTRIES_PER_MESSAGE * self.MESSAGES_PER_BATCH
            )

            try:
                # Process batch
                batch_info = self.process_translation_batch(
                    start_idx, batch_num)
                self.batch_records.append(batch_info)
                self._save_batch_records()

                # Wait and check status
                time.sleep(30)
                status_info = self.check_batch_status(batch_info)
                logger.info(
                    f"Batch {batch_num + 1} status: {status_info['status']}")

                # Update status record
                batch_info.update({"status_after_30s": status_info['status']})
                self._save_batch_records()

            except Exception as e:
                logger.error(
                    f"Failed to process batch {batch_num + 1}: {str(e)}", exc_info=True)
                continue

    def collect_batch_results(self, batch_id: str) -> List[TranslationEntry]:
        """
        Collect and process results from a completed OpenAI batch.

        Args:
            batch_id: OpenAI batch identifier

        Returns:
            List of TranslationEntry objects with translations
        """
        logger.info(f"Collecting results for batch {batch_id}")

        # Get batch status and output files
        batch_status = self.openai_client.batches.retrieve(batch_id)
        if batch_status.status != "completed":
            logger.error(
                f"Batch {batch_id} not completed. Status: {batch_status.status}")
            return []

        # Download and process output files
        translation_entries: List[TranslationEntry] = []
        try:
            # Get completed jobs from the batch
            jobs = self.openai_client.batches.jobs.list(batch_id=batch_id)

            # Process each job's output
            for job in jobs.data:
                if job.status != "completed":
                    logger.warning(
                        f"Skipping incomplete job {job.id} in batch {batch_id}")
                    continue

                # Extract message ID to get original entry indices
                # format: batch_{num}_msg_{msg_num}_{timestamp}
                msg_id = job.custom_id
                msg_num = int(msg_id.split('_')[3])

                # Calculate entry indices for this message
                start_idx = self.TRANSLATION_START_IDX + \
                    (int(msg_id.split('_')[1]) * self.ENTRIES_PER_MESSAGE * self.MESSAGES_PER_BATCH) + \
                    (msg_num * self.ENTRIES_PER_MESSAGE)

                # Parse translation results
                try:
                    result = json.loads(job.output)
                    translations = result.get("translations", [])

                    # Create TranslationEntry objects
                    for i, trans in enumerate(translations):
                        entry_idx = start_idx + i
                        entry = self.manager.get_entry(entry_idx)
                        entry.set_translation(
                            prompt_ru=trans["prompt_ru"],
                            response_ru=trans["response_ru"]
                        )
                        translation_entries.append(entry)

                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse JSON from job {job.id}", exc_info=True)
                except Exception as e:
                    logger.error(
                        f"Error processing job {job.id}: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(
                f"Failed to collect batch results: {str(e)}", exc_info=True)

        logger.info(
            f"Collected {len(translation_entries)} translated entries from batch {batch_id}")
        return translation_entries

    def process_and_save_batch(self, batch_id: str) -> Dict[str, Any] | None:
        """
        Process batch results and save translations to the manager.

        Args:
            batch_id: OpenAI batch identifier

        Returns:
            Dictionary with processing results or None if failed
        """
        entries = self.collect_batch_results(batch_id)
        if not entries:
            return None

        try:
            result = self.manager.update_entries(entries)
            logger.info(
                f"Saved {result['total_updated']} translations from batch {batch_id}")
            return result
        except Exception as e:
            logger.error(
                f"Failed to save batch translations: {str(e)}", exc_info=True)
            return None


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_processing.log'),
            logging.StreamHandler()
        ]
    )
    manager = TranslationManager(
            path=Path('data/pandas_dfs/main_translation_df.pkl'))
    processor = BatchProcessor(manager=manager)
    processor.run()


if __name__ == "__main__":
    main()
