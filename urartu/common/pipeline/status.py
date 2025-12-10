"""
Pipeline status display.

This module handles displaying pipeline execution status with progress tracking.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from urartu.utils.logging import get_logger

from .pipeline_action import PipelineAction

logger = get_logger(__name__)


class PipelineStatusDisplay:
    """Handles displaying pipeline execution status."""

    def __init__(self, pipeline_config, loopable_actions: list, loop_iterations: list, run_dir: Optional[Path] = None):
        """
        Initialize the status display.

        Args:
            pipeline_config: The pipeline configuration
            loopable_actions: List of loopable action names
            loop_iterations: List of loop iteration dictionaries
            run_dir: Optional run directory path for writing status file
        """
        self.pipeline_config = pipeline_config
        self.loopable_actions = loopable_actions
        self.loop_iterations = loop_iterations
        self.run_dir = run_dir
        self.status_file = None
        if run_dir:
            self.status_file = Path(run_dir) / "pipeline_status.txt"

    def print_status(self, all_actions: List[PipelineAction], statuses: Dict[str, Dict], current_index: int):
        """
        Print a clean, colorful status view of pipeline progress.

        Args:
            all_actions: List of all pipeline actions
            statuses: Dict mapping action names to their status
            current_index: Index of currently completed action (1-based)
        """
        # ANSI color codes
        GREEN = '\033[92m'  # Bright green for completed
        BLUE = '\033[94m'  # Bright blue for cached
        GRAY = '\033[90m'  # Gray for pending
        BOLD = '\033[1m'  # Bold text
        RESET = '\033[0m'  # Reset to default

        # Helper function to calculate display width
        def display_width(text):
            """Calculate display width accounting for ANSI codes and emoji width"""
            # Strip ANSI codes
            ansi_pattern = re.compile(r'\033\[[0-9;]*m')
            clean_text = ansi_pattern.sub('', text)

            # Try to use wcwidth library if available
            try:
                import wcwidth

                return wcwidth.wcswidth(clean_text)
            except ImportError:
                pass

            # Fallback: Our specific emojis (all are 2 columns wide in terminal)
            wide_chars = {
                '✅',  # U+2705 White Heavy Check Mark
                '✨',  # U+2728 Sparkles
                '💾',  # U+1F4BE Floppy Disk
                '🚀',  # U+1F680 Rocket
                '📊',  # U+1F4CA Bar Chart
                '⭕',  # U+2B55 Hollow Red Circle
                '🔄',  # U+1F504 Counterclockwise Arrows Button
            }

            # Calculate width
            width = 0
            for char in clean_text:
                if char in wide_chars:
                    width += 2
                else:
                    width += 1
            return width

        # Get experiment name
        experiment_name = self.pipeline_config.get('experiment_name', 'Pipeline')
        total_actions = len(all_actions)

        # Clear line
        print("\r" + " " * 100 + "\r", end='')

        # Horizontal separator
        separator = "═" * 80

        # Title with experiment name
        title_text = f"🚀 {experiment_name}"
        title_width = display_width(title_text)
        title_padding_left = (80 - title_width) // 2
        title_padding_right = 80 - title_width - title_padding_left
        title_line = f"{' ' * title_padding_left}{BOLD}{title_text}{RESET}{' ' * title_padding_right}"

        # Progress counter
        progress_text = f"📊 Progress: {current_index}/{total_actions} actions"
        progress_width = display_width(progress_text)
        progress_padding_left = (80 - progress_width) // 2
        progress_padding_right = 80 - progress_width - progress_padding_left
        progress_line = f"{' ' * progress_padding_left}{progress_text}{' ' * progress_padding_right}"

        status_lines = [
            f"\n{BOLD}{separator}{RESET}",
            title_line,
            progress_line,
            f"{BOLD}{separator}{RESET}",
        ]

        # Action status lines
        for i, action in enumerate(all_actions):
            # Handle loopable actions block - expand it to show individual actions
            if action.name == "__loopable_actions__":
                # Show loopable actions with indentation
                if self.loopable_actions:
                    num_iterations = len(self.loop_iterations)

                    # Check if all iterations are complete for all loopable actions
                    all_complete = True
                    for loopable_action_name in self.loopable_actions:
                        # Count how many iterations of this action have completed
                        completed_iterations = sum(
                            1 for key in statuses.keys() if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                        )
                        if completed_iterations < num_iterations:
                            all_complete = False
                            break

                    if all_complete and num_iterations > 0:
                        # All loopable actions completed - show them as a group with ✅
                        status_lines.append(
                            f"  {GREEN}✅ Loopable Actions ({len(self.loopable_actions)} actions × {num_iterations} iterations){RESET}"
                        )
                        for loopable_action_name in self.loopable_actions:
                            # Count cached vs executed iterations for this action
                            cached_count = 0
                            executed_count = 0
                            action_found = False

                            for key in statuses.keys():
                                if key.startswith(loopable_action_name + "_"):
                                    action_status = statuses[key]
                                    if action_status.get('completed', False):
                                        action_found = True
                                        if action_status.get('cached', False):
                                            cached_count += 1
                                        else:
                                            executed_count += 1

                            if action_found:
                                # Build status string with counts
                                status_parts = []
                                if executed_count > 0:
                                    status_parts.append(f"{executed_count} executed")
                                if cached_count > 0:
                                    status_parts.append(f"{cached_count} from cache")

                                status_str = f"({', '.join(status_parts)})" if status_parts else ""

                                # Choose emoji and color based on whether all were cached
                                if cached_count > 0 and executed_count == 0:
                                    # All cached - blue with disk emoji
                                    line = f"    {BLUE}  ✅ {loopable_action_name} 💾 {status_str}{RESET}"
                                elif executed_count > 0:
                                    # Some or all executed - green with sparkles
                                    line = f"    {GREEN}  ✅ {loopable_action_name} ✨ {status_str}{RESET}"
                                else:
                                    # Fallback
                                    line = f"    {GREEN}  ✅ {loopable_action_name} ✨ {status_str}{RESET}"
                            else:
                                # Fallback: show as completed
                                line = f"    {GREEN}  ✅ {loopable_action_name} ✨{RESET}"
                            status_lines.append(line)
                    else:
                        # Loopable actions in progress
                        # Count completed iterations for display
                        total_completed = 0
                        for loopable_action_name in self.loopable_actions:
                            completed_iterations = sum(
                                1 for key in statuses.keys() if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                            )
                            total_completed += completed_iterations

                        # Calculate progress: total completed iterations / (actions × iterations)
                        total_expected = len(self.loopable_actions) * num_iterations
                        progress_info = f"{total_completed}/{total_expected}" if total_expected > 0 else ""

                        # Check if all iterations are complete to decide on emoji
                        all_complete_for_header = True
                        for loopable_action_name_check in self.loopable_actions:
                            completed_check = sum(
                                1
                                for key in statuses.keys()
                                if key.startswith(loopable_action_name_check + "_") and statuses[key].get('completed', False)
                            )
                            if completed_check < num_iterations:
                                all_complete_for_header = False
                                break

                        # Use ✅ if all complete, otherwise ⭕
                        header_emoji = "✅" if all_complete_for_header and num_iterations > 0 else "⭕"
                        header_color = GREEN if all_complete_for_header and num_iterations > 0 else GRAY
                        status_lines.append(
                            f"  {header_color}{header_emoji} Loopable Actions ({len(self.loopable_actions)} actions × {num_iterations} iterations) {progress_info}{RESET}"
                        )
                        for loopable_action_name in self.loopable_actions:
                            # Count completed iterations for this specific action
                            completed_iterations = sum(
                                1 for key in statuses.keys() if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                            )

                            # Count cached vs executed for completed iterations
                            cached_count = sum(
                                1
                                for key in statuses.keys()
                                if key.startswith(loopable_action_name + "_")
                                and statuses[key].get('completed', False)
                                and statuses[key].get('cached', False)
                            )
                            executed_count = completed_iterations - cached_count

                            # Check if action is currently running by checking iteration-specific statuses
                            is_running = False
                            running_count = 0
                            for key in statuses.keys():
                                if key.startswith(loopable_action_name + "_") and statuses[key].get('running', False):
                                    is_running = True
                                    running_count += 1

                            # Build status string with counts (including running)
                            status_parts = []
                            if running_count > 0:
                                status_parts.append(f"{running_count} running")
                            if executed_count > 0:
                                status_parts.append(f"{executed_count} executed")
                            if cached_count > 0:
                                status_parts.append(f"{cached_count} from cache")

                            status_str = f"({', '.join(status_parts)})" if status_parts else ""

                            if is_running:
                                # Currently running - show as in progress with running indicator
                                if completed_iterations > 0:
                                    # Some completed, some running
                                    if cached_count > 0 and executed_count == 0:
                                        line = f"    {GREEN}  🔄 {loopable_action_name} ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                    else:
                                        line = f"    {GREEN}  🔄 {loopable_action_name} ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                else:
                                    # Just starting, nothing completed yet
                                    line = f"    {GREEN}  🔄 {loopable_action_name} (running...){RESET}"
                            elif completed_iterations > 0:
                                # In progress - show with ✅ and progress
                                if completed_iterations < num_iterations:
                                    # Some iterations done, but not all
                                    if cached_count > 0 and executed_count == 0:
                                        line = (
                                            f"    {BLUE}  ✅ {loopable_action_name} 💾 ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                        )
                                    else:
                                        line = (
                                            f"    {GREEN}  ✅ {loopable_action_name} ✨ ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                        )
                                else:
                                    # All iterations complete
                                    if cached_count > 0 and executed_count == 0:
                                        line = (
                                            f"    {BLUE}  ✅ {loopable_action_name} 💾 ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                        )
                                    else:
                                        line = (
                                            f"    {GREEN}  ✅ {loopable_action_name} ✨ ({completed_iterations}/{num_iterations}) {status_str}{RESET}"
                                        )
                            else:
                                # Not started yet
                                line = f"    {GRAY}  ⭕ {loopable_action_name}{RESET}"
                            status_lines.append(line)
                else:
                    # No loopable actions configured, just show placeholder
                    if i < current_index:
                        line = f"  {GREEN}✅ {action.name} ✨ (executed){RESET}"
                    else:
                        line = f"  {GRAY}⭕ {action.name} (pending){RESET}"
                    status_lines.append(line)
            else:
                # Regular action
                if i < current_index:
                    # Completed action
                    status = statuses.get(action.name, {})
                    if status.get('cached', False):
                        # Loaded from cache - blue with disk emoji
                        line = f"  {BLUE}✅ {action.name} 💾 (from cache){RESET}"
                    else:
                        # Freshly executed - green with sparkles
                        line = f"  {GREEN}✅ {action.name} ✨ (executed){RESET}"
                else:
                    # Pending - gray
                    line = f"  {GRAY}⭕ {action.name} (pending){RESET}"

                status_lines.append(line)

        status_lines.append(f"{BOLD}{separator}{RESET}")

        # Print all status lines to console
        print("\n".join(status_lines))
        print()  # Empty line for spacing

        # Also write to status file (without ANSI codes for readability)
        self._write_status_to_file(all_actions, statuses, current_index)

    def _write_status_to_file(self, all_actions: List[PipelineAction], statuses: Dict[str, Dict], current_index: int):
        """
        Write pipeline status to a file without ANSI color codes.

        Args:
            all_actions: List of all pipeline actions
            statuses: Dict mapping action names to their status
            current_index: Index of currently completed action (1-based)
        """
        if not self.status_file:
            return

        try:
            # Get experiment name
            experiment_name = self.pipeline_config.get('experiment_name', 'Pipeline')
            total_actions = len(all_actions)

            # Horizontal separator
            separator = "═" * 80

            # Title with experiment name
            title_text = f"🚀 {experiment_name}"
            title_padding_left = (80 - len(title_text)) // 2
            title_padding_right = 80 - len(title_text) - title_padding_left
            title_line = f"{' ' * title_padding_left}{title_text}{' ' * title_padding_right}"

            # Progress counter
            progress_text = f"📊 Progress: {current_index}/{total_actions} actions"
            progress_padding_left = (80 - len(progress_text)) // 2
            progress_padding_right = 80 - len(progress_text) - progress_padding_left
            progress_line = f"{' ' * progress_padding_left}{progress_text}{' ' * progress_padding_right}"

            status_lines = [
                separator,
                title_line,
                progress_line,
                separator,
            ]

            # Action status lines (without ANSI codes)
            for i, action in enumerate(all_actions):
                # Handle loopable actions block
                if action.name == "__loopable_actions__":
                    if self.loopable_actions:
                        num_iterations = len(self.loop_iterations)

                        # Check if all iterations are complete
                        all_complete = True
                        for loopable_action_name in self.loopable_actions:
                            completed_iterations = sum(
                                1 for key in statuses.keys() if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                            )
                            if completed_iterations < num_iterations:
                                all_complete = False
                                break

                        if all_complete and num_iterations > 0:
                            status_lines.append(f"  ✅ Loopable Actions ({len(self.loopable_actions)} actions × {num_iterations} iterations)")
                            for loopable_action_name in self.loopable_actions:
                                cached_count = 0
                                executed_count = 0
                                action_found = False

                                for key in statuses.keys():
                                    if key.startswith(loopable_action_name + "_"):
                                        action_status = statuses[key]
                                        if action_status.get('completed', False):
                                            action_found = True
                                            if action_status.get('cached', False):
                                                cached_count += 1
                                            else:
                                                executed_count += 1

                                if action_found:
                                    status_parts = []
                                    if executed_count > 0:
                                        status_parts.append(f"{executed_count} executed")
                                    if cached_count > 0:
                                        status_parts.append(f"{cached_count} from cache")
                                    status_str = f"({', '.join(status_parts)})" if status_parts else ""
                                    status_lines.append(f"    ✅ {loopable_action_name} {status_str}")
                                else:
                                    status_lines.append(f"    ✅ {loopable_action_name}")
                        else:
                            total_completed = 0
                            for loopable_action_name in self.loopable_actions:
                                completed_iterations = sum(
                                    1
                                    for key in statuses.keys()
                                    if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                                )
                                total_completed += completed_iterations

                            total_expected = len(self.loopable_actions) * num_iterations
                            progress_info = f"{total_completed}/{total_expected}" if total_expected > 0 else ""
                            # Check if all complete to decide on emoji
                            all_complete_for_header = True
                            for loopable_action_name_check in self.loopable_actions:
                                completed_check = sum(
                                    1
                                    for key in statuses.keys()
                                    if key.startswith(loopable_action_name_check + "_") and statuses[key].get('completed', False)
                                )
                                if completed_check < num_iterations:
                                    all_complete_for_header = False
                                    break

                            header_emoji = "✅" if all_complete_for_header and num_iterations > 0 else "⭕"
                            status_lines.append(
                                f"  {header_emoji} Loopable Actions ({len(self.loopable_actions)} actions × {num_iterations} iterations) {progress_info}"
                            )

                            for loopable_action_name in self.loopable_actions:
                                completed_iterations = sum(
                                    1
                                    for key in statuses.keys()
                                    if key.startswith(loopable_action_name + "_") and statuses[key].get('completed', False)
                                )

                                # Count cached vs executed
                                cached_count = sum(
                                    1
                                    for key in statuses.keys()
                                    if key.startswith(loopable_action_name + "_")
                                    and statuses[key].get('completed', False)
                                    and statuses[key].get('cached', False)
                                )
                                executed_count = completed_iterations - cached_count

                                # Check if running
                                is_running = any(
                                    key.startswith(loopable_action_name + "_") and statuses[key].get('running', False) for key in statuses.keys()
                                )
                                running_count = sum(
                                    1 for key in statuses.keys() if key.startswith(loopable_action_name + "_") and statuses[key].get('running', False)
                                )

                                # Build status string
                                status_parts = []
                                if running_count > 0:
                                    status_parts.append(f"{running_count} running")
                                if executed_count > 0:
                                    status_parts.append(f"{executed_count} executed")
                                if cached_count > 0:
                                    status_parts.append(f"{cached_count} from cache")
                                status_str = f"({', '.join(status_parts)})" if status_parts else ""

                                if is_running:
                                    # Currently running
                                    if completed_iterations > 0:
                                        status_lines.append(f"    🔄 {loopable_action_name} ({completed_iterations}/{num_iterations}) {status_str}")
                                    else:
                                        status_lines.append(f"    🔄 {loopable_action_name} (running...)")
                                elif completed_iterations > 0:
                                    # Show ✅ for in-progress actions with completed iterations
                                    if cached_count > 0 and executed_count == 0:
                                        status_lines.append(
                                            f"    ✅ {loopable_action_name} 💾 ({completed_iterations}/{num_iterations}) {status_str}"
                                        )
                                    else:
                                        status_lines.append(
                                            f"    ✅ {loopable_action_name} ✨ ({completed_iterations}/{num_iterations}) {status_str}"
                                        )
                                else:
                                    status_lines.append(f"    ⭕ {loopable_action_name}")
                    else:
                        if i < current_index:
                            status_lines.append(f"  ✅ {action.name} (executed)")
                        else:
                            status_lines.append(f"  ⭕ {action.name} (pending)")
                else:
                    # Regular action
                    if i < current_index:
                        status = statuses.get(action.name, {})
                        if status.get('cached', False):
                            status_lines.append(f"  ✅ {action.name} 💾 (from cache)")
                        else:
                            status_lines.append(f"  ✅ {action.name} ✨ (executed)")
                    else:
                        status_lines.append(f"  ⭕ {action.name} (pending)")

            status_lines.append(separator)

            # Write to file
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(status_lines) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write status to file {self.status_file}: {e}")
