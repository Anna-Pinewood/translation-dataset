import logging

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
        print(f"Entry [{self._idx}]")
        is_response_unsecure = self._data['is_response_unsecure']
        response_string_status = "UNSECURE" if is_response_unsecure else "SECURE"
        print(f"{'*'*15} Response security: {response_string_status} {'*'*15}")
        print("=" * 80)
        print("PROMPT (ENG):")
        print(self._data['prompt'])
        print("\nPROMPT (RU):")
        print(self._data['prompt_ru'] or "[NOT TRANSLATED]")
        print("\nRESPONSE (ENG):")
        print(self._data['response'])
        print("\nRESPONSE (RU):")
        print(self._data['response_ru'] or "[NOT TRANSLATED]")
        print("=" * 80, "\n")
