"""
DDNet RL Training Curriculum System

Provides map and segment-based curriculum learning for training RL agents
on DDNet maps of progressively increasing difficulty.

Map Curriculum:
- Orders maps by difficulty (category -> stars -> world record time)
- Progresses to harder maps as agent improves
- Supports diverse training batches for generalization

Segment Curriculum:
- Trains on checkpoint-to-checkpoint segments before full maps
- Progressive chaining: 1 segment -> 2 segments -> ... -> full map
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import random


class MapCategory(IntEnum):
    """DDNet map difficulty categories in order."""
    NOVICE = 0
    MODERATE = 1
    BRUTAL = 2
    INSANE = 3
    DUMMY = 4
    DDMAX = 5
    OLDSCHOOL = 6
    SOLO = 7
    RACE = 8


# Category name to enum mapping
CATEGORY_MAP = {
    'novice': MapCategory.NOVICE,
    'moderate': MapCategory.MODERATE,
    'brutal': MapCategory.BRUTAL,
    'insane': MapCategory.INSANE,
    'dummy': MapCategory.DUMMY,
    'ddmax': MapCategory.DDMAX,
    'oldschool': MapCategory.OLDSCHOOL,
    'solo': MapCategory.SOLO,
    'race': MapCategory.RACE,
}


@dataclass
class MapInfo:
    """Information about a DDNet map."""
    name: str
    path: str
    category: MapCategory
    stars: int = 0
    world_record_time: float = float('inf')  # Seconds
    checkpoints: int = 0

    @property
    def difficulty_score(self) -> float:
        """Compute a difficulty score for ordering maps."""
        # Category weight is most important
        category_weight = self.category.value * 10000
        # Stars within category
        star_weight = self.stars * 1000
        # World record time as tiebreaker (longer = harder)
        wr_weight = min(self.world_record_time, 600)  # Cap at 10 min
        return category_weight + star_weight + wr_weight


@dataclass
class Segment:
    """A segment between two checkpoints."""
    start_pos: Tuple[float, float]  # World coordinates
    end_pos: Tuple[float, float]    # World coordinates
    start_checkpoint: int           # -1 for spawn
    end_checkpoint: int             # Checkpoint index or -1 for finish


class MapCurriculum:
    """
    Curriculum manager that orders maps by difficulty and tracks progress.

    Usage:
        curriculum = MapCurriculum(maps_dir='/path/to/ddnet-maps')
        curriculum.load_maps()

        # Get current training map
        map_path = curriculum.get_current_map()

        # Update with episode results
        curriculum.record_episode(success=True, time=45.2)

        # Check for progression
        if curriculum.should_progress():
            curriculum.advance()
    """

    def __init__(
        self,
        maps_dir: str,
        ddnet_db_path: Optional[str] = None,
        starting_category: str = 'novice',
        success_threshold: float = 0.8,
        min_episodes: int = 100,
        max_stars: int = 5,
    ):
        """
        Initialize the map curriculum.

        Args:
            maps_dir: Path to ddnet-maps repository clone
            ddnet_db_path: Optional path to ddnet.sqlite for world record times
            starting_category: Starting difficulty category
            success_threshold: Success rate required to progress (0.0 to 1.0)
            min_episodes: Minimum episodes before considering progression
            max_stars: Maximum star rating to include
        """
        self.maps_dir = Path(maps_dir)
        self.ddnet_db_path = ddnet_db_path
        self.starting_category = CATEGORY_MAP.get(starting_category.lower(), MapCategory.NOVICE)
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes
        self.max_stars = max_stars

        # Map database
        self.maps: Dict[str, MapInfo] = {}
        self.difficulty_order: List[str] = []

        # Progress tracking
        self.current_index = 0
        self.episode_results: List[bool] = []
        self.episode_times: List[float] = []

    def load_maps(self):
        """Load and analyze all maps from the maps directory."""
        self.maps.clear()

        if not self.maps_dir.exists():
            raise FileNotFoundError(f"Maps directory not found: {self.maps_dir}")

        # Load world record times if database available
        wr_times = {}
        if self.ddnet_db_path and os.path.exists(self.ddnet_db_path):
            wr_times = self._load_world_records()

        # Scan maps directory structure
        # DDNet maps are organized as: types/<category>/<mapname>.map
        types_dir = self.maps_dir / 'types'
        if types_dir.exists():
            for category_dir in types_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name.lower()
                if category_name not in CATEGORY_MAP:
                    continue

                category = CATEGORY_MAP[category_name]

                for map_file in category_dir.glob('*.map'):
                    map_name = map_file.stem

                    # Get star rating from filename or default
                    stars = self._get_star_rating(map_name, category)
                    if stars > self.max_stars:
                        continue

                    map_info = MapInfo(
                        name=map_name,
                        path=str(map_file),
                        category=category,
                        stars=stars,
                        world_record_time=wr_times.get(map_name, float('inf')),
                    )
                    self.maps[map_name] = map_info

        # Sort maps by difficulty
        self._compute_difficulty_order()

        # Find starting position based on category
        self._set_starting_position()

        print(f"Loaded {len(self.maps)} maps, starting at index {self.current_index}")

    def _load_world_records(self) -> Dict[str, float]:
        """Load world record times from DDNet database."""
        wr_times = {}
        try:
            conn = sqlite3.connect(self.ddnet_db_path)
            cursor = conn.cursor()

            # Query for best times per map
            cursor.execute("""
                SELECT Map, MIN(Time) as BestTime
                FROM race
                GROUP BY Map
            """)

            for row in cursor.fetchall():
                map_name, best_time = row
                if best_time is not None:
                    wr_times[map_name] = float(best_time)

            conn.close()
        except Exception as e:
            print(f"Warning: Could not load world records: {e}")

        return wr_times

    def _get_star_rating(self, map_name: str, category: MapCategory) -> int:
        """Get star rating for a map (simplified heuristic)."""
        # In practice, you'd want to parse this from map metadata or a database
        # For now, use category as a rough proxy
        category_stars = {
            MapCategory.NOVICE: 1,
            MapCategory.MODERATE: 2,
            MapCategory.BRUTAL: 3,
            MapCategory.INSANE: 4,
            MapCategory.DDMAX: 4,
            MapCategory.SOLO: 3,
            MapCategory.RACE: 2,
        }
        return category_stars.get(category, 3)

    def _compute_difficulty_order(self):
        """Sort maps by difficulty score."""
        sorted_maps = sorted(
            self.maps.values(),
            key=lambda m: m.difficulty_score
        )
        self.difficulty_order = [m.name for m in sorted_maps]

    def _set_starting_position(self):
        """Set starting index based on starting category."""
        for i, map_name in enumerate(self.difficulty_order):
            map_info = self.maps[map_name]
            if map_info.category >= self.starting_category:
                self.current_index = i
                return
        self.current_index = 0

    def get_current_map(self) -> str:
        """Get the current training map path."""
        if not self.difficulty_order:
            raise RuntimeError("No maps loaded. Call load_maps() first.")

        map_name = self.difficulty_order[self.current_index]
        return self.maps[map_name].path

    def get_current_map_info(self) -> MapInfo:
        """Get info about the current training map."""
        if not self.difficulty_order:
            raise RuntimeError("No maps loaded. Call load_maps() first.")

        map_name = self.difficulty_order[self.current_index]
        return self.maps[map_name]

    def get_training_batch(self, batch_size: int, diversity_range: int = 5) -> List[str]:
        """
        Get a diverse batch of maps at similar difficulty for training.

        Args:
            batch_size: Number of maps to return
            diversity_range: Range of indices around current to sample from

        Returns:
            List of map paths
        """
        if not self.difficulty_order:
            return []

        # Sample from maps around current difficulty
        start_idx = max(0, self.current_index - diversity_range // 2)
        end_idx = min(len(self.difficulty_order), self.current_index + diversity_range)

        available = self.difficulty_order[start_idx:end_idx]

        if len(available) <= batch_size:
            selected = available
        else:
            selected = random.sample(available, batch_size)

        return [self.maps[name].path for name in selected]

    def record_episode(self, success: bool, time: float = 0.0):
        """
        Record the result of a training episode.

        Args:
            success: True if agent completed the map
            time: Time taken (if successful)
        """
        self.episode_results.append(success)
        if success and time > 0:
            self.episode_times.append(time)

        # Keep only recent history
        max_history = self.min_episodes * 2
        if len(self.episode_results) > max_history:
            self.episode_results = self.episode_results[-max_history:]
        if len(self.episode_times) > max_history:
            self.episode_times = self.episode_times[-max_history:]

    def get_success_rate(self) -> float:
        """Get recent success rate."""
        if not self.episode_results:
            return 0.0
        recent = self.episode_results[-self.min_episodes:]
        return sum(recent) / len(recent)

    def should_progress(self) -> bool:
        """Check if agent should advance to harder maps."""
        if len(self.episode_results) < self.min_episodes:
            return False

        success_rate = self.get_success_rate()
        return success_rate >= self.success_threshold

    def advance(self) -> bool:
        """
        Advance to the next difficulty level.

        Returns:
            True if successfully advanced, False if at max difficulty
        """
        if self.current_index >= len(self.difficulty_order) - 1:
            return False

        self.current_index += 1
        self.episode_results.clear()
        self.episode_times.clear()

        new_map = self.get_current_map_info()
        print(f"Advanced to: {new_map.name} (category: {new_map.category.name}, "
              f"stars: {new_map.stars})")
        return True

    def regress(self) -> bool:
        """
        Move back to an easier map (if struggling).

        Returns:
            True if successfully regressed, False if at min difficulty
        """
        if self.current_index <= 0:
            return False

        self.current_index -= 1
        self.episode_results.clear()
        self.episode_times.clear()
        return True


class SegmentCurriculum:
    """
    Curriculum for training on map segments (checkpoint to checkpoint).

    This allows training on long maps by breaking them into smaller,
    more manageable segments that can be mastered individually.

    Usage:
        segment_curr = SegmentCurriculum()
        segment_curr.load_checkpoints(map_checkpoints)

        # Get segment to train on
        start, goal = segment_curr.get_current_segment()

        # After mastering segment, chain more
        segment_curr.extend_chain()
    """

    def __init__(
        self,
        success_threshold: float = 0.9,
        min_episodes: int = 50,
    ):
        """
        Initialize segment curriculum.

        Args:
            success_threshold: Success rate required to extend chain
            min_episodes: Minimum episodes before considering extension
        """
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes

        # Checkpoint data
        self.spawn_pos: Tuple[float, float] = (0, 0)
        self.finish_pos: Tuple[float, float] = (0, 0)
        self.checkpoints: List[Tuple[float, float]] = []

        # Segment chain state
        self.chain_length = 1  # Start with 1 segment
        self.chain_start = 0   # Starting checkpoint index (-1 for spawn)

        # Episode tracking
        self.episode_results: List[bool] = []

    def load_checkpoints(
        self,
        checkpoints: List[Tuple[float, float]],
        spawn_pos: Tuple[float, float],
        finish_pos: Tuple[float, float],
    ):
        """
        Load checkpoint positions from map analysis.

        Args:
            checkpoints: List of (x, y) checkpoint positions in order
            spawn_pos: Spawn/start position
            finish_pos: Finish position
        """
        self.checkpoints = list(checkpoints)
        self.spawn_pos = spawn_pos
        self.finish_pos = finish_pos

        # Reset chain state
        self.chain_length = 1
        self.chain_start = -1  # Start from spawn
        self.episode_results.clear()

        print(f"Loaded {len(checkpoints)} checkpoints for segment training")

    def get_num_segments(self) -> int:
        """Get total number of segments (spawn->cp1, cp1->cp2, ..., cpN->finish)."""
        return len(self.checkpoints) + 1

    def get_current_segment(self) -> Segment:
        """Get the current training segment."""
        # Calculate end of chain
        chain_end = self.chain_start + self.chain_length

        # Determine start position
        if self.chain_start < 0:
            start_pos = self.spawn_pos
            start_cp = -1
        else:
            start_pos = self.checkpoints[self.chain_start]
            start_cp = self.chain_start

        # Determine end position
        if chain_end >= len(self.checkpoints):
            end_pos = self.finish_pos
            end_cp = -1  # Finish
        else:
            end_pos = self.checkpoints[chain_end]
            end_cp = chain_end

        return Segment(
            start_pos=start_pos,
            end_pos=end_pos,
            start_checkpoint=start_cp,
            end_checkpoint=end_cp,
        )

    def get_all_segments_in_chain(self) -> List[Segment]:
        """Get all individual segments in the current chain."""
        segments = []

        for i in range(self.chain_length):
            seg_start = self.chain_start + i
            seg_end = seg_start + 1

            if seg_start < 0:
                start_pos = self.spawn_pos
            else:
                start_pos = self.checkpoints[seg_start]

            if seg_end >= len(self.checkpoints):
                end_pos = self.finish_pos
            else:
                end_pos = self.checkpoints[seg_end]

            segments.append(Segment(
                start_pos=start_pos,
                end_pos=end_pos,
                start_checkpoint=seg_start,
                end_checkpoint=seg_end if seg_end < len(self.checkpoints) else -1,
            ))

        return segments

    def record_episode(self, success: bool):
        """Record segment completion result."""
        self.episode_results.append(success)

        # Keep recent history
        max_history = self.min_episodes * 2
        if len(self.episode_results) > max_history:
            self.episode_results = self.episode_results[-max_history:]

    def get_success_rate(self) -> float:
        """Get recent success rate on current segment chain."""
        if not self.episode_results:
            return 0.0
        recent = self.episode_results[-self.min_episodes:]
        return sum(recent) / len(recent)

    def should_extend(self) -> bool:
        """Check if should extend the segment chain."""
        if len(self.episode_results) < self.min_episodes:
            return False
        return self.get_success_rate() >= self.success_threshold

    def extend_chain(self) -> bool:
        """
        Extend the chain by one segment.

        Returns:
            True if extended, False if already at full map
        """
        max_segments = self.get_num_segments()
        if self.chain_length >= max_segments:
            return False

        self.chain_length += 1
        self.episode_results.clear()

        print(f"Extended segment chain to {self.chain_length} segments")
        return True

    def is_full_map(self) -> bool:
        """Check if training on the full map (all segments)."""
        return self.chain_length >= self.get_num_segments()

    def reset_to_segment(self, segment_idx: int):
        """Reset to train on a specific segment."""
        if segment_idx < 0:
            self.chain_start = -1
        else:
            self.chain_start = min(segment_idx, len(self.checkpoints) - 1)

        self.chain_length = 1
        self.episode_results.clear()


class CombinedCurriculum:
    """
    Combines map and segment curriculum for comprehensive training.

    Strategy:
    1. Start with easy maps, segment by segment
    2. Master full map before advancing to harder maps
    3. On harder maps, optionally use segment training again
    """

    def __init__(
        self,
        maps_dir: str,
        use_segment_training: bool = True,
        segment_threshold: int = 3,  # Use segments if map has >= N checkpoints
        **kwargs
    ):
        """
        Initialize combined curriculum.

        Args:
            maps_dir: Path to ddnet-maps
            use_segment_training: Whether to use segment training on long maps
            segment_threshold: Min checkpoints to trigger segment training
            **kwargs: Additional args passed to MapCurriculum
        """
        self.map_curriculum = MapCurriculum(maps_dir, **kwargs)
        self.segment_curriculum = SegmentCurriculum()

        self.use_segment_training = use_segment_training
        self.segment_threshold = segment_threshold

        # State
        self.using_segments = False

    def load_maps(self):
        """Load maps and initialize curriculum."""
        self.map_curriculum.load_maps()

    def setup_for_map(self, checkpoints: List[Tuple[float, float]],
                      spawn_pos: Tuple[float, float],
                      finish_pos: Tuple[float, float]):
        """
        Setup segment curriculum when loading a new map.
        Call this after map is loaded and analyzed.
        """
        if self.use_segment_training and len(checkpoints) >= self.segment_threshold:
            self.segment_curriculum.load_checkpoints(checkpoints, spawn_pos, finish_pos)
            self.using_segments = True
        else:
            self.using_segments = False

    def get_current_map(self) -> str:
        """Get current map path."""
        return self.map_curriculum.get_current_map()

    def get_segment(self) -> Optional[Segment]:
        """Get current training segment (if using segment training)."""
        if self.using_segments:
            return self.segment_curriculum.get_current_segment()
        return None

    def record_episode(self, success: bool, time: float = 0.0):
        """Record episode result."""
        if self.using_segments:
            self.segment_curriculum.record_episode(success)

            # Check for segment extension
            if self.segment_curriculum.should_extend():
                if not self.segment_curriculum.extend_chain():
                    # Full map mastered, record for map curriculum
                    self.map_curriculum.record_episode(True, time)
                    self.using_segments = False
        else:
            self.map_curriculum.record_episode(success, time)

    def check_progression(self) -> bool:
        """
        Check and handle progression.

        Returns:
            True if progressed to new map/segment
        """
        if self.using_segments:
            if self.segment_curriculum.should_extend():
                return self.segment_curriculum.extend_chain()
        else:
            if self.map_curriculum.should_progress():
                return self.map_curriculum.advance()
        return False

    def get_status(self) -> Dict:
        """Get current curriculum status."""
        map_info = self.map_curriculum.get_current_map_info()

        status = {
            'map': map_info.name,
            'category': map_info.category.name,
            'stars': map_info.stars,
            'map_index': self.map_curriculum.current_index,
            'total_maps': len(self.map_curriculum.difficulty_order),
            'using_segments': self.using_segments,
        }

        if self.using_segments:
            status['segment_chain'] = self.segment_curriculum.chain_length
            status['total_segments'] = self.segment_curriculum.get_num_segments()
            status['segment_success_rate'] = self.segment_curriculum.get_success_rate()
        else:
            status['map_success_rate'] = self.map_curriculum.get_success_rate()

        return status


if __name__ == "__main__":
    # Test curriculum system
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", type=str, required=True,
                        help="Path to ddnet-maps repository")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to ddnet.sqlite")
    args = parser.parse_args()

    print("Testing MapCurriculum...")
    curriculum = MapCurriculum(
        maps_dir=args.maps_dir,
        ddnet_db_path=args.db_path,
        starting_category='novice',
    )
    curriculum.load_maps()

    print(f"\nTotal maps: {len(curriculum.maps)}")
    print(f"Current map: {curriculum.get_current_map_info().name}")

    # Show first 10 maps in order
    print("\nFirst 10 maps by difficulty:")
    for i, name in enumerate(curriculum.difficulty_order[:10]):
        info = curriculum.maps[name]
        print(f"  {i}: {info.name} ({info.category.name}, {info.stars}*)")

    # Test batch
    batch = curriculum.get_training_batch(5)
    print(f"\nTraining batch (5 maps): {[Path(p).stem for p in batch]}")
