from __future__ import annotations

from .event import get_event_ranges, Event, EventType


class Tokenizer:
    def __init__(self):
        """Fixed vocabulary tokenizer."""
        self._offset = 3
        self._event_ranges = get_event_ranges()

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding."""
        return -100

    @property
    def sos_id(self) -> int:
        """[SOS] token for start-of-sequence."""
        return 0

    @property
    def eos_id(self) -> int:
        """[EOS] token for end-of-sequence."""
        return 1

    @property
    def time_step_id(self) -> int:
        """[STEP] token for a single time step."""
        return 2

    def decode(self, id: int) -> Event:
        """Converts token ids into Event objects."""
        offset = self._offset
        for er in self._event_ranges:
            if offset <= id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"id {id} is not mapped to any event")

    def encode(self, event: Event) -> int:
        """Converts Event objects into token ids."""
        offset = self._offset
        for er in self._event_ranges:
            if event.type is er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"event value {event.value} is not within range "
                        f"[{er.min_value}, {er.max_value}] for event type {event.type}"
                    )
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"unknown event type: {event.type}")

    def event_type_range(self, event_type: EventType) -> tuple[int, int]:
        """Get the token id range of each Event type."""
        offset = self._offset
        for er in self._event_ranges:
            if event_type is er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event_type}")

    def vocab_size(self) -> int:
        """Get the total number of token ids."""
        return self._offset + sum(
            er.max_value - er.min_value + 1 for er in self._event_ranges
        )
