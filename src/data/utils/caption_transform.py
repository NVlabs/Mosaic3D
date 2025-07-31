from typing import List, Dict, Optional
from collections import Counter
import argparse


class FilterCaptionEmpty:
    """Filter out empty or whitespace-only captions."""

    def __call__(self, captions: List[str]) -> List[bool]:
        return [bool(caption.strip()) for caption in captions]


class FilterCaptionNumPoints:
    """Filter captions based on number of points."""

    def __init__(self, min_num_points: int = 5, max_num_points: Optional[int] = None):
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points

    def __call__(self, point_indices: List[List[int]]) -> List[bool]:
        if self.max_num_points is None:
            return [self.min_num_points <= len(indices) for indices in point_indices]
        else:
            return [
                self.min_num_points <= len(indices) <= self.max_num_points
                for indices in point_indices
            ]


class FilterCaptionWordCount:
    """Filter captions based on word count.

    Args:
        min_words (int): Minimum number of words required
        max_words (int): Maximum words allowed
    """

    def __init__(self, min_words: int = 3, max_words: int = 50):
        self.min_words = min_words
        self.max_words = max_words

    def preprocess_caption(self, caption: str) -> str:
        """Preprocess caption to handle special formats."""
        # Replace colons and commas with spaces
        caption = caption.replace(":", " ").replace(",", " ")
        # Remove multiple spaces
        caption = " ".join(caption.split())
        return caption

    def __call__(self, captions: List[str]) -> List[bool]:
        """Return boolean mask for valid captions."""
        word_counts = [len(self.preprocess_caption(caption).split()) for caption in captions]
        return [self.min_words <= count <= self.max_words for count in word_counts]


class FilterCaptionLetterRatio:
    """Filter captions based on ratio of letters to total characters.

    Args:
        min_letter_ratio (float): Minimum required ratio of letters to total characters
    """

    def __init__(self, min_letter_ratio: float = 0.5):
        self.min_letter_ratio = min_letter_ratio

    def __call__(self, captions: List[str]) -> List[bool]:
        valid_flags = []
        for caption in captions:
            letter_count = sum(1 for c in caption if c.isalpha())
            total_count = sum(1 for c in caption if not c.isspace())
            valid_flags.append(
                total_count > 0 and letter_count / total_count >= self.min_letter_ratio
            )
        return valid_flags


class FilterCaptionWordRepetition:
    """Filter captions with excessive word repetition.

    Args:
        max_repetition_ratio (float): Maximum allowed ratio of most common word
    """

    def __init__(self, max_repetition_ratio: float = 0.4):
        self.max_repetition_ratio = max_repetition_ratio

    def __call__(self, captions: List[str]) -> List[bool]:
        valid_flags = []
        for caption in captions:
            # skip for captions with fewer than 3 words
            words = caption.split()
            if len(words) < 3:
                valid_flags.append(True)
                continue

            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            valid_flags.append(most_common_count / len(words) <= self.max_repetition_ratio)
        return valid_flags


class FilterCaptionConsecutiveRepeats:
    """Filter captions with excessive consecutive word repetitions.

    Args:
        max_consecutive (int): Maximum allowed consecutive repetitions
    """

    def __init__(self, max_consecutive: int = 3):
        self.max_consecutive = max_consecutive

    def __call__(self, captions: List[str]) -> List[bool]:
        valid_flags = []
        for caption in captions:
            words = caption.split()
            if len(words) < 3:
                valid_flags.append(True)
                continue

            consecutive_count = 1
            prev_word = words[0]
            is_valid = True

            for word in words[1:]:
                if word == prev_word:
                    consecutive_count += 1
                    if consecutive_count > self.max_consecutive:
                        is_valid = False
                        break
                else:
                    consecutive_count = 1
                prev_word = word

            valid_flags.append(is_valid)
        return valid_flags


class FilterCaptionPhraseRepeats:
    """Filter captions with repeating multi-word phrases.

    Args:
        max_consecutive (int): Maximum allowed phrase repetitions
        max_phrase_length (int): Maximum length of phrases to check
    """

    def __init__(self, max_consecutive: int = 3, max_phrase_length: int = 4):
        self.max_consecutive = max_consecutive
        self.max_phrase_length = max_phrase_length

    def __call__(self, captions: List[str]) -> List[bool]:
        valid_flags = []
        for caption in captions:
            words = caption.split()
            if len(words) < 4:  # Need at least 4 words for phrase repetition
                valid_flags.append(True)
                continue

            is_valid = True
            for phrase_len in range(2, self.max_phrase_length):
                if len(words) >= phrase_len * 2:
                    phrases = [
                        " ".join(words[i : i + phrase_len])
                        for i in range(len(words) - phrase_len + 1)
                    ]
                    phrase_counts = Counter(phrases)
                    if any(count > self.max_consecutive for count in phrase_counts.values()):
                        is_valid = False
                        break

            valid_flags.append(is_valid)
        return valid_flags


class CaptionFilter:
    """Combined caption filter that applies filters in optimal order.

    Args:
        min_words (int): Minimum words required
        max_words (int): Maximum words allowed
        min_letter_ratio (float): Minimum letter ratio
        max_repetition_ratio (float): Maximum word repetition ratio
        min_unique_ratio (float): Minimum unique word ratio
        max_consecutive (int): Maximum consecutive repeats
    """

    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 50,
        min_letter_ratio: float = 0.5,
        max_repetition_ratio: float = 0.4,
        max_consecutive: int = 3,
        min_num_points: Optional[int] = None,
        max_num_points: Optional[int] = None,
        **kwargs,
    ):
        if min_num_points is not None or max_num_points is not None:
            self.point_filters = [
                FilterCaptionNumPoints(min_num_points, max_num_points),
            ]
        else:
            self.point_filters = []

        # Create filters in order of computational complexity
        self.caption_filters = [
            FilterCaptionEmpty(),  # Fastest: just string operations
            FilterCaptionWordCount(min_words, max_words),  # Fast: simple splitting
            FilterCaptionLetterRatio(min_letter_ratio),  # Medium: character counting
            FilterCaptionWordRepetition(max_repetition_ratio),  # Medium: word counting
            FilterCaptionConsecutiveRepeats(max_consecutive),  # Slower: sequential scan
            FilterCaptionPhraseRepeats(max_consecutive),  # Slowest: phrase analysis
        ]

    def __call__(
        self, captions: List[str], point_indices: Optional[List[List[int]]] = None
    ) -> List[bool]:
        """Apply all filters in sequence, failing fast."""
        valid_flags = [True] * len(captions)

        if point_indices is not None and len(self.point_filters) > 0:
            for filter_fn in self.point_filters:
                point_flags = filter_fn(point_indices)
                valid_flags = [
                    flag and point_flag for flag, point_flag in zip(valid_flags, point_flags)
                ]

        # Apply each filter only to captions that passed previous filters
        for filter_fn in self.caption_filters:
            # Only process captions that are still valid
            current_captions = [cap for cap, flag in zip(captions, valid_flags) if flag]
            if not current_captions:
                break

            # Get results for remaining valid captions
            current_flags = filter_fn(current_captions)

            # Update master flags
            idx = 0
            for i in range(len(valid_flags)):
                if valid_flags[i]:
                    valid_flags[i] = current_flags[idx]
                    idx += 1

        return valid_flags

    def filter_captions(
        self, captions: List[str], scene_names: List[str] = None
    ) -> tuple[List[str], List[str]]:
        """Filter captions and optionally their corresponding scene names."""
        valid_flags = self(captions)

        filtered_captions = [cap for cap, valid in zip(captions, valid_flags) if valid]

        if scene_names is not None:
            filtered_scenes = [scene for scene, valid in zip(scene_names, valid_flags) if valid]
            return filtered_captions, filtered_scenes

        return filtered_captions, None

    def debug_caption(self, caption: str) -> Dict[str, bool]:
        """Debug which filters pass/fail for a specific caption.

        Returns:
            Dict[str, bool]: Dictionary of filter names and their results (True = passed)
        """
        # Preprocess caption
        preprocessed = caption.replace(":", " ").replace(",", " ")
        preprocessed = " ".join(preprocessed.split())

        results = {}

        # Test each filter individually
        for filter_fn in self.caption_filters:
            filter_name = filter_fn.__class__.__name__
            result = filter_fn([preprocessed])[0]
            results[filter_name] = result
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=str, required=True)
    args = parser.parse_args()
    DEFAULT_FILTER_PARAMS = {
        "min_words": 1,
        "max_words": 100,
        "min_letter_ratio": 0.3,
        "max_repetition_ratio": 0.5,
        "max_consecutive": 4,
    }
    caption_filter = CaptionFilter(**DEFAULT_FILTER_PARAMS)
    print(caption_filter.debug_caption(args.caption))
