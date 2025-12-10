"""
Cache management functionality for LoopablePipeline.

This module handles cache checking and pre-loopable action loading for loopable pipelines.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

from .pipeline_action import PipelineAction
from .types import ActionOutput

logger = get_logger(__name__)


class LoopableCacheMixin:
    """
    Mixin class providing cache checking functionality for LoopablePipeline.

    This mixin handles:
    - Checking if iterations are cached
    - Loading pre-loopable actions from cache
    - Comparing configs to find matching cache entries
    """

    def _check_iteration_cache(self, iteration_idx: int, loop_context: Dict[str, Any]) -> bool:
        """
        Check if an iteration's loopable actions are cached.

        Args:
            iteration_idx: Index of the iteration
            loop_context: Loop context for this iteration

        Returns:
            True if all loopable actions for this iteration are cached, False otherwise
        """
        # Check force_rerun flag
        force_rerun = self.cfg.get('force_rerun', False)
        if force_rerun:
            logger.debug(f"Force rerun enabled, iteration {iteration_idx} is not cached")
            return False

        # Set current loop context for resolvers
        self.current_loop_context = loop_context

        # Generate iteration identifier
        iteration_id = self._get_iteration_id(loop_context)
        if not iteration_id:
            iteration_id = f"iteration_{iteration_idx + 1}"

        # Note: Pre-loopable actions are loaded ONCE before checking all iterations (in _submit_array_only_internal)
        # They should already be in self.action_outputs at this point

        # Check each loopable action
        for action_name in self.loopable_actions:
            # Find or create the action
            action = None
            for a in self.actions:
                if a.name == action_name:
                    action = a
                    break

            if action is None:
                # Create action from loopable_action_configs
                if action_name in self.loopable_action_configs:
                    stored_config = self.loopable_action_configs[action_name]
                    # Handle combined pipeline structure: {block_0: {...}, block_1: {...}}
                    # Extract the correct block's config based on loop context
                    if isinstance(stored_config, dict):
                        block_keys = [k for k in stored_config.keys() if k.startswith('block_')]
                        if block_keys:
                            # Determine which block we're in based on loop context
                            if loop_context and 'revision' in loop_context:
                                # First block uses revisions
                                block_key = 'block_0'
                            elif loop_context and 'model_name' in loop_context:
                                # Second block uses model names
                                block_key = 'block_1'
                            else:
                                # Default to first block
                                block_key = sorted(block_keys)[0]

                            if block_key in stored_config:
                                config_overrides = copy.deepcopy(stored_config[block_key])
                                logger.debug(f"🔧 Extracted config from {block_key} for action '{action_name}' in combined pipeline")
                            else:
                                # Fallback: use first available block
                                block_key = sorted(block_keys)[0]
                                config_overrides = copy.deepcopy(stored_config[block_key])
                                logger.warning(f"⚠️ Block key '{block_key}' not found, using {sorted(block_keys)[0]} for action '{action_name}'")
                        else:
                            # Standard pipeline structure - use stored_config directly
                            config_overrides = copy.deepcopy(stored_config)
                    else:
                        config_overrides = copy.deepcopy(stored_config)
                    action = PipelineAction(name=action_name, action_name=action_name, config_overrides=config_overrides, outputs_to_track=[])
                else:
                    logger.warning(f"Loopable action '{action_name}' not found, assuming not cached")
                    return False

            # Create iteration-aware action name
            iteration_action_name = f"{action_name}_{iteration_id}"

            # Check cache by creating a temporary action instance
            if not self._check_action_cache(action, iteration_action_name, loop_context):
                return False

        # All loopable actions are cached
        return True

    def _check_action_cache(self, action: PipelineAction, iteration_action_name: str, loop_context: Dict[str, Any]) -> bool:
        """
        Check if a specific action is cached for the given iteration.

        Args:
            action: The action to check
            iteration_action_name: Name of the iteration-specific action
            loop_context: Loop context for this iteration

        Returns:
            True if cached, False otherwise
        """
        try:
            # Resolve config with loop context (similar to what _run_action does)
            context = {"action_outputs": self.action_outputs, "loop_context": loop_context}
            resolved_config = self._resolve_value(action.config_overrides, context)

            # CRITICAL: Inject action outputs (like _run_action does)
            # This ensures dependencies like 'samples_path' are included in the cache key
            resolved_config = self._inject_action_outputs(resolved_config, action.name, loop_context=loop_context)

            # CRITICAL: Inject loop context BEFORE merging (same order as _run_action)
            # This ensures the cache key includes the revision value, not just the iteration index
            if loop_context:
                resolved_config = self._inject_loop_context(resolved_config, loop_context)
                logger.debug(f"🔄 Injected loop context into '{action.name}' for cache check: {loop_context}")

            # Merge with pipeline common configs AFTER injecting loop context (same order as _run_action)
            pipeline_common_configs = self._get_common_pipeline_configs()
            merged_config = OmegaConf.merge(
                OmegaConf.create(pipeline_common_configs),  # Base (pipeline defaults)
                OmegaConf.create(resolved_config),  # Override (action-specific, with injections and loop context)
            )

            # Add pipeline config hash (same as _run_action does)
            try:
                from urartu.common.pipeline.cache import PipelineCache

                pipeline_config_hash = PipelineCache.generate_config_hash(self.cfg)
                merged_config.pipeline_config_hash = pipeline_config_hash
            except Exception:
                pass

            # Create action_cfg structure (same as _run_action does)
            action_cfg = OmegaConf.create(self.cfg)
            action_cfg.action_name = action.action_name

            # Ensure run_dir is preserved
            if 'run_dir' not in action_cfg or not action_cfg.run_dir:
                if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                    action_cfg.run_dir = self.cfg.run_dir

            # Remove pipeline key so Action instances will use cfg.action instead
            if 'pipeline' in action_cfg:
                del action_cfg.pipeline

            # Set iteration_id if loop_context is provided
            if loop_context:
                iteration_id = self._get_iteration_id_from_context(loop_context)
                if iteration_id:
                    merged_config.iteration_id = iteration_id
                    logger.debug(f"🔑 Set iteration_id='{iteration_id}' in action config for '{action.name}'")

            # Validate merged_config before setting it (minimal logging in normal runs)
            merged_config_dict = OmegaConf.to_container(merged_config, resolve=True)
            logger.debug(
                f"🔍 Cache check config for '{action.name}': {len(merged_config_dict)} keys, "
                f"loop_context={loop_context}"
            )

            # Set the merged config as action config
            action_cfg.action = merged_config

            # Verify action_cfg.action is set correctly
            if not hasattr(action_cfg, 'action') or action_cfg.action is None:
                logger.error(f"❌ CRITICAL: action_cfg.action is not set for '{action.name}'!")
                raise ValueError(f"action_cfg.action is not set - cache check will fail")
            logger.debug(f"✅ Verified action_cfg.action is set for '{action.name}'")

            # Create action instance using helper method
            project_root, actions_dir = self._find_project_and_actions_dir()
            if not actions_dir or not actions_dir.exists():
                logger.warning(f"Actions directory not found, assuming not cached")
                return False

            self._setup_action_imports(project_root, actions_dir)
            action_class = self._import_action_class(action.action_name)

            if not action_class:
                logger.error(f"Could not find action class in {action.action_name}")
                raise ValueError(f"Action class not found in module {action.action_name}. The module must contain a class with a 'run' method.")

            # Verify action_cfg structure before creating instance
            logger.debug(f"🔍 About to create action instance for '{action.name}':")
            logger.debug(f"   action_cfg.action_name: {getattr(action_cfg, 'action_name', 'NOT SET')}")
            logger.debug(f"   action_cfg.action exists: {hasattr(action_cfg, 'action')}")
            if hasattr(action_cfg, 'action'):
                action_config_keys = list(action_cfg.action.keys()) if hasattr(action_cfg.action, 'keys') else []
                logger.debug(f"   action_cfg.action keys: {action_config_keys[:10]}...")

            # Create action instance
            action_instance = action_class(action_cfg, self.aim_run)

            # Verify action_instance.action_config is populated
            if not hasattr(action_instance, 'action_config') or not action_instance.action_config:
                logger.error(f"❌ CRITICAL: action_instance.action_config is empty for '{action.name}'!")
                logger.error(f"   This means cache key generation will use wrong config!")
                return False

            logger.debug(f"🔍 Action instance created for '{action.name}' for cache check")

            # Generate cache key
            cache_key = action_instance._generate_cache_key()
            cache_file_path = action_instance._get_cache_file_path(cache_key)
            logger.info(f"🔍 Checking cache for {iteration_action_name}: cache_key={cache_key}, file={cache_file_path}")
            logger.debug(f"   📂 Cache directory: {action_instance.cache_dir}")
            logger.debug(f"   📂 Expected cache path: {action_instance._get_cache_path(cache_key)}")

            # Check if cache exists
            if cache_file_path.exists():
                # Check if cache is valid
                action_instance._cache_key = cache_key  # Store for _load_from_cache()
                cached_outputs = action_instance._load_from_cache()
                if cached_outputs is None:
                    logger.info(f"⚠️  Cache file exists but is invalid for {iteration_action_name} (cache_key: {cache_key})")
                    return False
                logger.info(f"✅ {iteration_action_name} is cached (cache_key: {cache_key})")
                return True
            else:
                # Cache not found - try to find matching cache by comparing configs
                action_name_for_cache = getattr(action_instance.cfg, 'action_name', None) or action_class.__name__
                return self._find_matching_cache(action_instance, iteration_action_name, cache_key, action_name_for_cache, action_class)

        except Exception as e:
            logger.error(f"❌ Error checking cache for {iteration_action_name}: {e}")
            logger.error(f"   This iteration will be submitted even if it might be cached")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def _find_matching_cache(self, action_instance, iteration_action_name: str, cache_key: str, action_name: str, action_class) -> bool:
        """
        Try to find a matching cache entry by comparing configs.

        This handles backward compatibility with caches created before path normalization.
        """
        # Cache not found - log detailed debugging info
        logger.info(f"❌ {iteration_action_name} is not cached (cache_key: {cache_key}, file: {action_instance._get_cache_file_path(cache_key)})")

        # Debug: Check what cache directories exist
        if action_instance.cache_dir.exists():
            action_name_for_cache = getattr(action_instance.cfg, 'action_name', None) or action_class.__name__
            action_cache_dir = action_instance.cache_dir / action_name_for_cache
            if action_cache_dir.exists():
                existing_dirs = [d.name for d in action_cache_dir.iterdir() if d.is_dir()]
                cache_hash = cache_key.split('_')[-1] if '_' in cache_key else cache_key
                logger.info(f"   🔍 Cache directory exists: {action_cache_dir}")
                logger.info(f"   🔍 Generated cache hash: {cache_hash}")
                logger.info(
                    f"   🔍 Existing cache hashes ({len(existing_dirs)} total): {existing_dirs[:10]}{'...' if len(existing_dirs) > 10 else ''}"
                )

                if existing_dirs and cache_hash not in existing_dirs:
                    logger.warning(f"   ⚠️  Cache key mismatch! Generated: {cache_hash}, but this hash not found in existing directories")

                    # Try to find matching cache by comparing config content
                    logger.info(f"   🔄 Attempting to find matching cache by comparing configs (checking {len(existing_dirs)} cache entries)...")
                    try:
                        serializable_config = action_instance._get_serializable_config()
                        
                        # CRITICAL: Normalize paths in current config before comparison
                        # This ensures dependency paths (like samples_path) are normalized to match cache format
                        from urartu.utils.cache import make_serializable
                        normalized_current = make_serializable(serializable_config)

                        # Log current config for debugging
                        logger.info(f"   📋 CURRENT config being checked:")
                        logger.info(f"      Keys: {sorted(normalized_current.keys())}")
                        logger.info(f"      Full config JSON: {json.dumps(normalized_current, sort_keys=True, indent=2)}")

                        # Try each existing cache directory
                        for idx, existing_hash in enumerate(existing_dirs):
                            if idx % 10 == 0:  # Log progress every 10 entries
                                logger.debug(f"   🔍 Checking cache {idx+1}/{len(existing_dirs)}: {existing_hash}")

                            existing_cache_dir = action_cache_dir / existing_hash
                            metadata_path = existing_cache_dir / "metadata.yaml"
                            if metadata_path.exists():
                                import yaml

                                with open(metadata_path, 'r') as f:
                                    existing_metadata = yaml.safe_load(f)

                                existing_config = existing_metadata.get('full_config', {})
                                if existing_config:
                                    # Filter out ignored keys from existing config (recursively excludes 'hash' fields)
                                    from urartu.utils.cache import filter_config_for_cache

                                    # Filter existing config to remove keys that shouldn't affect cache
                                    # This also recursively excludes 'hash' fields from nested dicts (e.g., dataset.hash)
                                    filtered_existing = filter_config_for_cache(existing_config)

                                    # Normalize paths in existing config for comparison
                                    normalized_existing = make_serializable(filtered_existing)

                                    # Compare normalized configs
                                    current_config_str = json.dumps(normalized_current, sort_keys=True)
                                    existing_config_str = json.dumps(normalized_existing, sort_keys=True)

                                    # Check for key differences (use normalized_current, not serializable_config)
                                    current_keys = set(normalized_current.keys())
                                    existing_keys = set(normalized_existing.keys())
                                    missing_in_existing = current_keys - existing_keys
                                    extra_in_existing = existing_keys - current_keys

                                    # If current config is missing dependency paths that exist in cache,
                                    # try to reconstruct what the current config would be if dependencies were loaded
                                    # This ensures we compare configs that were generated the same way
                                    dependency_path_keys = {'samples_path', 'classification_path', 'data_path', 'output_path'}
                                    missing_dependency_paths = extra_in_existing & dependency_path_keys
                                    
                                    if missing_dependency_paths:
                                        # Try to reconstruct the missing dependency paths
                                        # This ensures the current config matches what would be generated if dependencies were loaded
                                        # We need to ensure the config generation process is identical to when the cache was created
                                        logger.debug(f"   🔄 Attempting to reconstruct missing dependency paths: {sorted(missing_dependency_paths)}")
                                        reconstructed_config = serializable_config.copy()
                                        
                                        for dep_key in missing_dependency_paths:
                                            # First, try to find this dependency in action_outputs
                                            found = False
                                            for action_name, action_output in self.action_outputs.items():
                                                if hasattr(action_output, 'outputs') and isinstance(action_output.outputs, dict):
                                                    if dep_key in action_output.outputs:
                                                        dep_value = action_output.outputs[dep_key]
                                                        # Normalize the path to match cache format
                                                        from urartu.utils.cache import make_serializable
                                                        normalized_dep_value = make_serializable(dep_value)
                                                        reconstructed_config[dep_key] = normalized_dep_value
                                                        logger.debug(f"      ✅ Reconstructed {dep_key} from {action_name}: {normalized_dep_value}")
                                                        found = True
                                                        break
                                            
                                            # If not found in action_outputs, try to load the dependency action from cache
                                            # For samples_path, it typically comes from _2_template_sample_constructor
                                            if not found:
                                                # Map dependency keys to likely action names
                                                dep_to_action = {
                                                    'samples_path': '_2_template_sample_constructor',
                                                    'classification_path': '_3_answer_rank_classifier',
                                                }
                                                
                                                source_action_name = dep_to_action.get(dep_key)
                                                if source_action_name:
                                                    # Find the action in the pipeline
                                                    source_action = None
                                                    for a in self.actions:
                                                        if a.name == source_action_name:
                                                            source_action = a
                                                            break
                                                    
                                                    if source_action:
                                                        logger.debug(f"      🔄 Attempting to load {source_action_name} from cache to get {dep_key}")
                                                        # Try to load the action from cache
                                                        if self._load_single_pre_loopable_action(source_action):
                                                            # Check if the dependency is now in action_outputs
                                                            if source_action_name in self.action_outputs:
                                                                action_output = self.action_outputs[source_action_name]
                                                                if hasattr(action_output, 'outputs') and isinstance(action_output.outputs, dict):
                                                                    if dep_key in action_output.outputs:
                                                                        dep_value = action_output.outputs[dep_key]
                                                                        from urartu.utils.cache import make_serializable
                                                                        normalized_dep_value = make_serializable(dep_value)
                                                                        reconstructed_config[dep_key] = normalized_dep_value
                                                                        logger.debug(f"      ✅ Reconstructed {dep_key} by loading {source_action_name}: {normalized_dep_value}")
                                                                        found = True
                                            
                                            if not found:
                                                logger.debug(f"      ⚠️  Could not reconstruct {dep_key} - config comparison may fail")
                                        
                                        # Use reconstructed config for comparison (normalize it too)
                                        normalized_reconstructed = make_serializable(reconstructed_config)
                                        if normalized_reconstructed != normalized_current:
                                            logger.debug(f"   🔄 Using reconstructed config with {len(normalized_reconstructed)} keys (was {len(normalized_current)})")
                                            normalized_current = normalized_reconstructed
                                            # Recalculate keys after reconstruction
                                            current_keys = set(normalized_current.keys())
                                            missing_in_existing = current_keys - existing_keys
                                            extra_in_existing = existing_keys - current_keys

                                    # Check for value differences in common keys
                                    common_keys = current_keys & existing_keys
                                    value_diffs = []
                                    for key in sorted(common_keys):
                                        current_val = normalized_current[key]
                                        existing_val = normalized_existing[key]
                                        if current_val != existing_val:
                                            value_diffs.append(key)

                                    if not missing_in_existing and not extra_in_existing and not value_diffs:
                                        logger.info(f"   ✅ Found matching cache by config comparison: {existing_hash}")
                                        # Use this cache key
                                        matching_cache_key = f"{action_name_for_cache}_{existing_hash}"
                                        matching_cache_file = action_instance._get_cache_file_path(matching_cache_key)
                                        if matching_cache_file.exists():
                                            action_instance._cache_key = matching_cache_key
                                            cached_outputs = action_instance._load_from_cache()
                                            if cached_outputs is not None:
                                                logger.info(f"✅ {iteration_action_name} is cached (matched cache_key: {matching_cache_key})")
                                                return True
                                            else:
                                                logger.warning(f"   ⚠️  Matched cache file exists but is invalid")
                                    else:
                                        # Log detailed differences for mismatches (only in debug mode)
                                        if idx < 3:  # Only log first 3 mismatches to avoid spam
                                            logger.debug(f"   ❌ Config mismatch for {existing_hash}:")
                                            logger.debug(f"      Missing in existing: {sorted(missing_in_existing)}")
                                            logger.debug(f"      Extra in existing: {sorted(extra_in_existing)}")
                                            logger.debug(f"      Value diffs: {value_diffs}")
                                            if value_diffs:
                                                for key in value_diffs[:3]:  # Show first 3 value diffs
                                                    logger.debug(
                                                        f"         {key}: current={normalized_current.get(key)}, existing={normalized_existing.get(key)}"
                                                    )
                                            logger.debug(
                                                f"      Existing config (filtered): {json.dumps(normalized_existing, sort_keys=True, indent=2)}"
                                            )
                    except Exception as e:
                        logger.warning(f"   ⚠️  Could not compare configs with existing caches: {e}")
                        import traceback

                        logger.debug(traceback.format_exc())

        return False

    def _load_pre_loopable_actions(self):
        """
        Load pre-loopable action outputs from cache ONCE before checking any iterations.

        This ensures that _inject_action_outputs has the dependencies it needs for all iterations.
        This method should be called before checking iteration caches.
        """
        logger.info("🔍 Loading pre-loopable action outputs from cache for cache checking...")
        logger.info(f"   Current action_outputs keys: {list(self.action_outputs.keys())}")
        logger.info(f"   Total actions in pipeline: {len(self.actions)}")

        try:
            # Find pre-loopable actions (actions before the loopable actions)
            # Handle both standard __loopable_actions__ and combined pipeline's __loopable_actions_block_*__ patterns
            loopable_idx = None
            for idx, a in enumerate(self.actions):
                if a.name == "__loopable_actions__" or (a.name.startswith("__loopable_actions_block_") and a.name.endswith("__")):
                    loopable_idx = idx
                    break
            logger.info(f"🔍 Found loopable_idx: {loopable_idx}, total actions: {len(self.actions)}")
            if loopable_idx is None:
                logger.warning(
                    "⚠️  Could not find '__loopable_actions__' or '__loopable_actions_block_*__' in actions list - pre-loopable loading skipped"
                )
                logger.warning(f"   Available action names: {[a.name for a in self.actions]}")
                return

            pre_loopable_actions = self.actions[:loopable_idx]
            logger.info(f"🔍 Found {len(pre_loopable_actions)} pre-loopable actions to load: {[a.name for a in pre_loopable_actions]}")

            for action in pre_loopable_actions:
                loaded = self._load_single_pre_loopable_action(action)
                if not loaded:
                    logger.warning(
                        f"⚠️  Pre-loopable action '{action.name}' could not be loaded from cache - cache checks for loopable actions may fail"
                    )
        except Exception as e:
            logger.warning(f"⚠️  Error loading pre-loopable outputs for cache check: {e}")
            import traceback

            logger.warning(f"   Traceback: {traceback.format_exc()}")

        # Log summary
        try:
            loopable_idx_check = next((idx for idx, a in enumerate(self.actions) if a.name == "__loopable_actions__"), None)
            if loopable_idx_check is not None:
                pre_loopable_action_names = [a.name for a in self.actions[:loopable_idx_check]]
                loaded_pre_loopable = [name for name in self.action_outputs.keys() if name in pre_loopable_action_names]
                logger.info(
                    f"📊 Pre-loopable loading summary: {len(loaded_pre_loopable)}/{len(pre_loopable_action_names)} actions loaded into action_outputs"
                )
                if loaded_pre_loopable:
                    logger.info(f"   Loaded actions: {loaded_pre_loopable}")
                else:
                    logger.warning(f"   ⚠️  No pre-loopable actions were loaded - cache checks may fail!")
        except Exception as e:
            logger.warning(f"   ⚠️  Error checking pre-loopable loading summary: {e}")

    def _load_single_pre_loopable_action(self, action: PipelineAction) -> bool:
        """
        Load a single pre-loopable action from cache.

        Args:
            action: The action to load

        Returns:
            True if loaded successfully, False otherwise
        """
        loaded = False
        try:
            # Create action config (similar to what _run_action does)
            context = {"action_outputs": self.action_outputs, "loop_context": {}}
            resolved_config = self._resolve_value(action.config_overrides, context)
            # Pre-loopable actions don't depend on loop context, so we can load them without it
            resolved_config = self._inject_action_outputs(resolved_config, action.name)

            # Merge with pipeline common configs
            pipeline_common_configs = self._get_common_pipeline_configs()
            merged_config = OmegaConf.merge(
                OmegaConf.create(pipeline_common_configs),
                OmegaConf.create(resolved_config),
            )
            
            # CRITICAL: Normalize paths in merged_config before creating action instance
            # This ensures cache keys are consistent regardless of absolute vs relative paths
            from urartu.utils.cache import make_serializable
            from urartu.common.action import Action
            merged_config_dict = OmegaConf.to_container(merged_config, resolve=True)
            
            # Normalize paths in the config (especially dataset.path and template_dir.path)
            # Also recompute dataset.hash after normalization since it depends on the path
            def normalize_paths_in_dict(d):
                if isinstance(d, dict):
                    result = {}
                    for k, v in d.items():
                        if isinstance(v, str) and ('path' in k.lower() or 'dir' in k.lower()):
                            # Normalize path strings
                            normalized = Action._make_path_portable(v)
                            result[k] = normalized
                        elif isinstance(v, (dict, list)):
                            result[k] = normalize_paths_in_dict(v)
                        else:
                            result[k] = v
                    return result
                elif isinstance(d, list):
                    return [normalize_paths_in_dict(item) for item in d]
                else:
                    return d
            
            merged_config_dict = normalize_paths_in_dict(merged_config_dict)
            
            # Remove dataset.hash if it exists - ActionDataset.__init__ will recompute it from normalized config
            # This ensures the hash is computed consistently from the normalized config (without the hash field itself)
            if 'dataset' in merged_config_dict and isinstance(merged_config_dict['dataset'], dict):
                dataset_config = merged_config_dict['dataset']
                if 'hash' in dataset_config:
                    del dataset_config['hash']
            
            merged_config = OmegaConf.create(merged_config_dict)

            # DO NOT add pipeline_config_hash here - it's pipeline-specific and would prevent
            # cross-pipeline cache sharing for pre-loopable actions. The cache key generation
            # in Action._generate_cache_key() will filter it out anyway, but we shouldn't add it
            # in the first place when loading for cache checking.

            # Create action_cfg structure
            action_cfg = OmegaConf.create(self.cfg)
            action_cfg.action_name = action.action_name

            # Ensure run_dir is preserved
            if 'run_dir' not in action_cfg or not action_cfg.run_dir:
                if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                    action_cfg.run_dir = self.cfg.run_dir

            # Remove pipeline key
            if 'pipeline' in action_cfg:
                del action_cfg.pipeline

            # Set the merged config as action config
            action_cfg.action = merged_config

            # Use helper methods to create action instance
            project_root, actions_dir = self._find_project_and_actions_dir()
            if not actions_dir or not actions_dir.exists():
                logger.warning(f"Actions directory not found, falling back to pipeline cache")
                return self._load_from_pipeline_cache_fallback(action, merged_config)

            self._setup_action_imports(project_root, actions_dir)
            action_class = self._import_action_class(action.action_name)

            if action_class:
                # Create action instance
                action_instance = action_class(action_cfg, self.aim_run)

                # Generate cache key using the action's own method
                cache_key = action_instance._generate_cache_key()
                logger.info(f"🔑 Generated cache key for '{action.name}' using action instance: {cache_key}")

                # Try to load from cache using the generated key first
                cache_file_path = action_instance._get_cache_file_path(cache_key)
                if cache_file_path.exists():
                    action_instance._cache_key = cache_key  # Store for _load_from_cache()
                    cached_outputs = action_instance._load_from_cache()
                    if cached_outputs is not None:
                        action_output = ActionOutput(
                            name=action.name,
                            action_name=action.action_name,
                            outputs=cached_outputs if isinstance(cached_outputs, dict) else {},
                            metadata={"from_cache": True},
                        )
                        self.action_outputs[action.name] = action_output
                        logger.info(f"📦 Loaded pre-loopable action '{action.name}' from cache for cache checking")
                        return True
            else:
                logger.warning(f"Could not find action class in {action.action_name}, falling back to pipeline cache")
                return self._load_from_pipeline_cache_fallback(action, merged_config)
        except Exception as e:
            logger.warning(f"Could not create action instance for cache check: {e}, falling back to pipeline cache")
            import traceback

            logger.debug(f"   Traceback: {traceback.format_exc()}")
            return self._load_from_pipeline_cache_fallback(action, merged_config)

        return False

    def _load_from_pipeline_cache_fallback(self, action: PipelineAction, merged_config) -> bool:
        """Fallback to pipeline cache method if action instance creation fails."""
        try:
            resolved_config_dict = OmegaConf.to_container(merged_config, resolve=True)
            cache_key = self._generate_cache_key(action, resolved_config_dict)
            logger.info(f"🔑 Generated cache key for '{action.name}' using pipeline method (fallback): {cache_key}")
            cache_entry = self._load_from_cache(cache_key)
            if cache_entry is not None and cache_entry.outputs:
                action_output = ActionOutput(
                    name=action.name,
                    action_name=action.action_name,
                    outputs=cache_entry.outputs if isinstance(cache_entry.outputs, dict) else {},
                    metadata={"from_cache": True},
                )
                self.action_outputs[action.name] = action_output
                logger.info(f"📦 Loaded pre-loopable action '{action.name}' from pipeline cache (fallback)")
                return True
        except Exception as e:
            logger.debug(f"   Pipeline cache fallback failed: {e}")
        return False
