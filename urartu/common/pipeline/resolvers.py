"""
Data resolvers for pipeline configuration.

This module contains resolvers that handle special value resolution in configs,
such as action output references and loop context references.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import ActionOutput

logger = logging.getLogger(__name__)


class DataResolver(ABC):
    """Abstract base class for data resolvers that handle special value resolution in configs."""

    @abstractmethod
    def can_resolve(self, value: str) -> bool:
        """Check if this resolver can handle the given value."""
        pass

    @abstractmethod
    def resolve(self, value: str, context: Dict[str, Any]) -> Any:
        """Resolve the value using the provided context."""
        pass


class ActionOutputResolver(DataResolver):
    """Resolver for action output references in the format {{actions.action_name.output_key}}"""

    def can_resolve(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith("{{actions.") and value.endswith("}}")

    def resolve(self, value: str, context: Dict[str, Any]) -> Any:
        # Parse reference like {{actions.construct_samples.data_files}}
        parts = value[10:-2].split(".")  # Remove {{actions. and }}
        if len(parts) < 2:
            raise ValueError(f"Invalid action reference: {value}")

        action_name = parts[0]
        output_key = ".".join(parts[1:])

        action_outputs = context.get("action_outputs", {})

        # Check if this is a loopable action stored in nested dict structure
        action_output = None
        if action_name in action_outputs:
            entry = action_outputs[action_name]
            # Check if it's a dict of iterations (loopable action) or a direct ActionOutput (regular action)
            if isinstance(entry, dict) and not isinstance(entry, ActionOutput):
                # This is a loopable action with multiple iterations
                loop_context = context.get("loop_context", {})
                if loop_context:
                    # Build iteration ID from loop context
                    iteration_id = None
                    if len(loop_context) == 1:
                        first_key, first_value = next(iter(loop_context.items()))
                        sanitized = str(first_value).replace('/', '_').replace(':', '_').replace(' ', '_')
                        iteration_id = f"{first_key}_{sanitized}"
                    elif loop_context:
                        # Multiple loop params - create composite ID
                        parts_list = [f"{k}_{str(v).replace('/', '_').replace(':', '_').replace(' ', '_')}" for k, v in loop_context.items()]
                        iteration_id = "_".join(parts_list)

                    if iteration_id and iteration_id in entry:
                        action_output = entry[iteration_id]
                        logger.debug(f"Using iteration '{iteration_id}' for loopable action '{action_name}'")
                    else:
                        # Fallback: use first iteration if available
                        if entry:
                            first_iteration_id = next(iter(entry.keys()))
                            action_output = entry[first_iteration_id]
                            logger.debug(f"Using first iteration '{first_iteration_id}' for loopable action '{action_name}'")
                        else:
                            raise ValueError(f"Action '{action_name}' has no iterations executed yet")
                else:
                    # No loop context - use first iteration or raise error
                    if entry:
                        first_iteration_id = next(iter(entry.keys()))
                        action_output = entry[first_iteration_id]
                        logger.debug(f"No loop context, using first iteration '{first_iteration_id}' for '{action_name}'")
                    else:
                        raise ValueError(f"Action '{action_name}' has no iterations executed yet")
            else:
                # Regular action - direct ActionOutput
                action_output = entry
        else:
            # Try backward compatibility: iteration-specific name (e.g., action_name_iteration_id)
            loop_context = context.get("loop_context", {})
            if loop_context:
                iteration_id = None
                if len(loop_context) == 1:
                    first_key, first_value = next(iter(loop_context.items()))
                    sanitized = str(first_value).replace('/', '_').replace(':', '_').replace(' ', '_')
                    iteration_id = f"{first_key}_{sanitized}"
                elif loop_context:
                    parts_list = [f"{k}_{str(v).replace('/', '_').replace(':', '_').replace(' ', '_')}" for k, v in loop_context.items()]
                    iteration_id = "_".join(parts_list)

                if iteration_id:
                    iteration_specific_name = f"{action_name}_{iteration_id}"
                    if iteration_specific_name in action_outputs:
                        action_output = action_outputs[iteration_specific_name]
                        logger.debug(f"Using backward-compatible iteration-specific name '{iteration_specific_name}'")
                    else:
                        # Try to find any iteration of this action
                        matching_keys = [k for k in action_outputs.keys() if k.startswith(f"{action_name}_")]
                        if matching_keys:
                            action_output = action_outputs[matching_keys[0]]
                            logger.debug(f"Using matching iteration '{matching_keys[0]}' for '{action_name}'")
                        else:
                            raise ValueError(f"Action '{action_name}' has not been executed yet or produced no outputs")
                else:
                    raise ValueError(f"Action '{action_name}' has not been executed yet or produced no outputs")
            else:
                raise ValueError(f"Action '{action_name}' has not been executed yet or produced no outputs")

        # Navigate nested dictionaries in the output
        output = action_output.outputs
        for key in output_key.split("."):
            if isinstance(output, dict) and key in output:
                output = output[key]
            else:
                raise ValueError(f"Cannot find '{output_key}' in action '{action_name}' outputs")

        return output


class LoopContextResolver(DataResolver):
    """Resolver for loop context references in the format {{loop.param}}"""

    def can_resolve(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith("{{loop.") and value.endswith("}}")

    def resolve(self, value: str, context: Dict[str, Any]) -> Any:
        # Parse reference like {{loop.revision}} or {{loop.model.revision}}
        parts = value[7:-2].split(".")  # Remove {{loop. and }}
        if len(parts) < 1:
            raise ValueError(f"Invalid loop context reference: {value}")

        loop_context = context.get("loop_context", {})
        if not loop_context:
            raise ValueError(f"No loop context available for reference: {value}")

        # Navigate nested dictionaries
        result = loop_context
        for key in parts:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                raise ValueError(f"Cannot find '{'.'.join(parts)}' in loop context. Available keys: {list(loop_context.keys())}")

        return result
