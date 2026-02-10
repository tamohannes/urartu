"""
Unified plotting framework for Urartu actions.

Provides PlottingMixin for actions to inherit from, and Plotter utility class
for managing plot styles, color palettes, and saving plots.

When save_plot_data_csv is enabled (default: true), plot data is also written
as CSV alongside pickle/plots so there is always a human-readable log of plot data.
"""

import csv
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

logger = get_logger(__name__)


def _to_list(x: Any) -> List[Any]:
    """Convert array-like or scalar to a flat list for CSV writing."""
    if hasattr(x, "tolist"):
        return getattr(x, "tolist")()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def write_plot_data_csv(plot_data: Dict[str, Any], csv_path: Path) -> List[Path]:
    """
    Write plot data to one or more CSV files for a human-readable log.

    Supports:
    - Dict of 1D same-length arrays/lists -> one CSV with columns = keys.
    - Dict with key "data" or "table" as list of dicts -> one CSV (rows = dicts).
    - Dict of 2D arrays -> one CSV per key (rows = rows, columns = 0..n-1).
    - Otherwise: one CSV per key that is 1D (column with key name) or 2D.

    Args:
        plot_data: Dictionary of plot data (arrays, lists, list of dicts).
        csv_path: Target path (file or directory). If file, stem is used for multiple outputs.

    Returns:
        List of paths to written CSV files.
    """
    written: List[Path] = []
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv":
        base_dir = csv_path
        stem = "plot_data"
    else:
        base_dir = csv_path.parent
        stem = csv_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # Prefer "data" or "table" as list of dicts -> single CSV
    for key in ("data", "table"):
        if key in plot_data and isinstance(plot_data[key], (list, tuple)) and plot_data[key]:
            first = plot_data[key][0]
            if isinstance(first, dict):
                out = base_dir / f"{stem}.csv" if stem else base_dir / f"{key}.csv"
                fieldnames = list(first.keys())
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for row in plot_data[key]:
                        w.writerow({k: row[k] for k in fieldnames})
                written.append(out)
                return written

    # Dict of 1D same-length arrays -> one CSV
    lengths = {}
    rows_by_key: Dict[str, List[Any]] = {}
    for k, v in plot_data.items():
        lst = _to_list(v)
        if not lst:
            continue
        try:
            n = len(lst)
            lengths[k] = n
            rows_by_key[k] = lst
        except TypeError:
            continue
    if rows_by_key:
        lens = set(lengths.values())
        if len(lens) == 1:
            n = list(lens)[0]
            out = base_dir / f"{stem}.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(list(rows_by_key.keys()))
                for i in range(n):
                    w.writerow([rows_by_key[k][i] for k in rows_by_key])
            written.append(out)
            return written

    # Per-key: 1D as single column, 2D as rows x cols
    for k, v in plot_data.items():
        try:
            lst = _to_list(v)
        except Exception:
            continue
        if not lst:
            continue
        out = base_dir / f"{stem}_{k}.csv"
        if isinstance(lst[0], (list, tuple)) or (hasattr(v, "ndim") and getattr(v, "ndim") == 2):
            # 2D
            rows = [lst] if not isinstance(lst[0], (list, tuple)) else lst
            rows = [_to_list(r) for r in rows]
            ncols = len(rows[0]) if rows else 0
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["index"] + [f"col_{j}" for j in range(ncols)])
                for i, row in enumerate(rows):
                    w.writerow([i] + list(row))
            written.append(out)
        else:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([k])
                for x in lst:
                    w.writerow([x])
            written.append(out)
    return written


class Plotter:
    """
    Utility class for managing plot styles, color palettes, and matplotlib configuration.

    This class provides static methods for:
    - Applying matplotlib styles
    - Getting color palettes
    - Managing default plot configurations
    """

    # Color palette definitions
    COLOR_PALETTES = {
        "red_blue": {
            "colors": ["#8B0000", "#A52A2A", "#DC143C", "#FF6347", "#4682B4", "#4169E1", "#0000CD", "#00008B"],
            "gradient": ["#8B0000", "#B22222", "#DC143C", "#FF6347", "#87CEEB", "#4682B4", "#0000CD", "#00008B"],
            "description": "Deep red to deep blue gradient",
        },
        "red_orange": {
            "colors": ["#8B0000", "#A52A2A", "#DC143C", "#FF4500", "#FF6347", "#FF8C00", "#FFA500", "#FFD700"],
            "gradient": ["#8B0000", "#B22222", "#DC143C", "#FF6347", "#FF8C00", "#FFA500", "#FFD700"],
            "description": "Deep red to deep orange gradient",
        },
        "blue_green": {
            "colors": ["#00008B", "#0000CD", "#4169E1", "#4682B4", "#5F9EA0", "#20B2AA", "#008B8B", "#006400"],
            "gradient": ["#00008B", "#0000CD", "#4682B4", "#5F9EA0", "#20B2AA", "#008B8B", "#006400"],
            "description": "Deep blue to deep green gradient",
        },
        "viridis": {
            "colors": ["#440154", "#482777", "#3F4A8A", "#31678E", "#26838F", "#1F9D8A", "#6CCE5A", "#F6E746"],
            "gradient": ["#440154", "#482777", "#3F4A8A", "#31678E", "#26838F", "#1F9D8A", "#6CCE5A", "#F6E746"],
            "description": "Viridis color scheme",
        },
        "plasma": {
            "colors": ["#0D0887", "#46039F", "#7201A8", "#9C179E", "#BD3786", "#D8576B", "#ED7953", "#FD8F47"],
            "gradient": ["#0D0887", "#46039F", "#7201A8", "#9C179E", "#BD3786", "#D8576B", "#ED7953", "#FD8F47"],
            "description": "Plasma color scheme",
        },
    }

    @staticmethod
    def get_color_palette(palette_name: str, n_colors: Optional[int] = None) -> List[str]:
        """
        Get a color palette by name.

        Args:
            palette_name: Name of the palette (red_blue, red_orange, blue_green, viridis, plasma, or custom)
            n_colors: Number of colors to return (if None, returns all colors in palette)

        Returns:
            List of color hex codes
        """
        if palette_name == "custom":
            # Custom palettes should be provided via config
            logger.warning("Custom palette requested but not provided. Using red_blue as fallback.")
            palette_name = "red_blue"

        if palette_name not in Plotter.COLOR_PALETTES:
            logger.warning(f"Unknown palette '{palette_name}', using 'red_blue' as fallback.")
            palette_name = "red_blue"

        palette = Plotter.COLOR_PALETTES[palette_name]
        colors = palette["gradient"]  # Use gradient for smoother transitions

        if n_colors is None:
            return colors

        # Interpolate colors if needed
        if n_colors <= len(colors):
            return colors[:n_colors]

        # If more colors needed, interpolate
        import numpy as np

        indices = np.linspace(0, len(colors) - 1, n_colors).astype(int)
        return [colors[i] for i in indices]

    @staticmethod
    def apply_matplotlib_style(style_config: DictConfig) -> None:
        """
        Apply matplotlib style based on configuration.

        Args:
            style_config: DictConfig with style settings (font_family, fontsize, etc.)
        """
        # Try to use matplotlib style if specified
        matplotlib_style = style_config.get("matplotlib_style", "seaborn-v0_8-whitegrid")
        if matplotlib_style:
            try:
                plt.style.use(matplotlib_style)
            except OSError:
                try:
                    # Fallback to older seaborn style
                    plt.style.use("seaborn-whitegrid")
                except OSError:
                    # Use default if seaborn not available
                    logger.debug("Seaborn styles not available, using default matplotlib style")

        # Apply font settings
        font_family = style_config.get("font_family", "sans-serif")
        font_sans_serif = style_config.get("font_sans_serif", ["Helvetica", "Arial", "DejaVu Sans"])

        plt.rcParams['font.family'] = font_family
        if font_family == "sans-serif":
            plt.rcParams['font.sans-serif'] = font_sans_serif

        # Apply font sizes
        plt.rcParams['font.size'] = style_config.get("fontsize", 11)
        plt.rcParams['axes.labelsize'] = style_config.get("label_fontsize", 13)
        plt.rcParams['axes.titlesize'] = style_config.get("title_fontsize", 14)
        plt.rcParams['xtick.labelsize'] = style_config.get("tick_fontsize", 10)
        plt.rcParams['ytick.labelsize'] = style_config.get("tick_fontsize", 10)
        plt.rcParams['legend.fontsize'] = style_config.get("legend_fontsize", 10)
        plt.rcParams['figure.titlesize'] = style_config.get("title_fontsize", 14)

        # Apply DPI settings
        plt.rcParams['figure.dpi'] = style_config.get("figure_dpi", 150)
        plt.rcParams['savefig.dpi'] = style_config.get("save_dpi", 150)

        # Apply grid settings
        grid_alpha = style_config.get("grid_alpha", 0.3)
        grid_linewidth = style_config.get("grid_linewidth", 0.5)
        plt.rcParams['grid.alpha'] = grid_alpha
        plt.rcParams['grid.linewidth'] = grid_linewidth
        plt.rcParams['axes.grid'] = True

        # Apply other style settings
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05


class PlottingMixin:
    """
    Mixin class for actions that need plotting functionality.

    Provides methods for:
    - Getting plot configuration (merged defaults + action overrides)
    - Getting plots directory
    - Saving plots (PNG always, PDF optionally)
    - Getting color palettes
    - Applying plot styles

    Usage:
        class MyAction(Action, PlottingMixin):
            def create_plots(self):
                plots_dir = self.get_plots_dir()
                # ... create plots using self.save_plot(), etc.
    """

    _default_plot_config: Optional[DictConfig] = None

    @classmethod
    def _load_default_plot_config(cls) -> DictConfig:
        """
        Load default plot configuration from Urartu config directory.

        Returns:
            DictConfig with default plot settings
        """
        if cls._default_plot_config is not None:
            return cls._default_plot_config

        # Try to load from Urartu config directory
        import urartu
        from pathlib import Path

        urartu_path = Path(urartu.__file__).parent
        default_config_path = urartu_path / "config" / "action" / "default_plotting.yaml"

        if default_config_path.exists():
            cls._default_plot_config = OmegaConf.load(default_config_path)
            logger.debug(f"Loaded default plot config from {default_config_path}")
        else:
            # Create minimal default config if file doesn't exist
            logger.warning(f"Default plot config not found at {default_config_path}, using minimal defaults")
            cls._default_plot_config = OmegaConf.create(
                {
                    "plotting": {
                        "plots_subdirectory": "plots",
                        "save_png": True,
                        "save_pdf": False,
                        "style": {
                            "font_family": "sans-serif",
                            "font_sans_serif": ["Helvetica", "Arial", "DejaVu Sans"],
                            "title_fontsize": 14,
                            "label_fontsize": 13,
                            "legend_fontsize": 10,
                            "tick_fontsize": 10,
                            "figure_dpi": 150,
                            "save_dpi": 150,
                            "matplotlib_style": "seaborn-v0_8-whitegrid",
                            "grid_alpha": 0.3,
                            "grid_linewidth": 0.5,
                        },
                        "color_palette": "red_blue",
                    }
                }
            )

        return cls._default_plot_config

    def get_plot_config(self) -> DictConfig:
        """
        Get merged plot configuration (defaults + action overrides).

        Returns:
            DictConfig with plot settings (merged from defaults and action config)
        """
        # Load defaults
        default_config = self._load_default_plot_config()
        default_plotting = default_config.get("plotting", OmegaConf.create({}))

        # Get action-specific overrides
        action_plotting = OmegaConf.create({})
        if hasattr(self, 'action_config') and self.action_config:
            action_plotting = self.action_config.get("plotting", OmegaConf.create({}))

        # Merge: defaults first, then action overrides
        merged = OmegaConf.merge(default_plotting, action_plotting)

        return merged

    def get_plots_dir(self, subdirectory: Optional[str] = None) -> Path:
        """
        Get the plots directory for this action.

        All plots should be saved to this directory. The directory structure is:
        run_dir/plots/[subdirectory]/

        Args:
            subdirectory: Optional subdirectory within plots (e.g., "heatmaps", "scatter")

        Returns:
            Path to plots directory
        """
        plot_config = self.get_plot_config()
        plots_subdirectory = plot_config.get("plots_subdirectory", "plots")

        # Get base run directory
        if hasattr(self, 'get_run_dir'):
            base_plots_dir = self.get_run_dir(plots_subdirectory)
        else:
            # Fallback if get_run_dir not available
            run_dir_str = self.cfg.get('run_dir', '.runs')
            base_plots_dir = Path(run_dir_str) / plots_subdirectory

        if subdirectory:
            plots_dir = base_plots_dir / subdirectory
        else:
            plots_dir = base_plots_dir

        # Create directory if it doesn't exist
        plots_dir.mkdir(parents=True, exist_ok=True)

        return plots_dir

    def save_plot(
        self,
        fig,
        filename: str,
        subdirectory: Optional[str] = None,
        dpi: Optional[int] = None,
        output_dir: Optional[Path] = None,
        plot_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Save a plot figure to the plots directory.

        Always saves PNG format. Optionally saves PDF if configured.
        When plot_data is provided and save_plot_data_csv is enabled (default: true),
        also saves the plot data as CSV alongside the image for a human-readable log.

        Args:
            fig: Matplotlib figure object
            filename: Filename (without extension, will add .png/.pdf)
            subdirectory: Optional subdirectory within plots (ignored if output_dir is provided)
            dpi: Optional DPI override (uses config default if not provided)
            output_dir: Optional full path to output directory (overrides plots_dir + subdirectory)
            plot_data: Optional dict of data used to generate the plot; saved as CSV when save_plot_data_csv is true

        Returns:
            Tuple of (png_path, pdf_path) where pdf_path is None if PDF not enabled
        """
        if output_dir is not None:
            plots_dir = output_dir
        else:
            plots_dir = self.get_plots_dir(subdirectory)

        plot_config = self.get_plot_config()

        # Get DPI
        if dpi is None:
            style_config = plot_config.get("style", OmegaConf.create({}))
            dpi = style_config.get("save_dpi", 150)

        # Always save PNG
        png_path = plots_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        logger.debug(f"Saved plot: {png_path}")

        # Optionally save PDF
        pdf_path = None
        if plot_config.get("save_pdf", False):
            pdf_path = plots_dir / f"{filename}.pdf"
            fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')
            logger.debug(f"Saved plot: {pdf_path}")

        # Save plot data as CSV when enabled (log of plot data)
        if plot_data is not None and plot_config.get("save_plot_data_csv", True):
            csv_path = plots_dir / f"{filename}.csv"
            try:
                paths = write_plot_data_csv(plot_data, csv_path)
                for p in paths:
                    logger.debug(f"Saved plot data CSV: {p}")
            except Exception as e:
                logger.warning(f"Could not save plot data CSV for {filename}: {e}")

        return (png_path, pdf_path)

    def get_color_palette(self, n_colors: Optional[int] = None, palette_name: Optional[str] = None) -> List[str]:
        """
        Get color palette based on action configuration.

        Args:
            n_colors: Number of colors to return (if None, returns all colors)
            palette_name: Optional palette name override (uses config default if not provided)

        Returns:
            List of color hex codes
        """
        plot_config = self.get_plot_config()

        if palette_name is None:
            palette_name = plot_config.get("color_palette", "red_blue")

        # Check for custom palette
        if palette_name == "custom":
            custom_colors = plot_config.get("color_palette_custom", [])
            if custom_colors:
                if n_colors is None:
                    return custom_colors
                return custom_colors[:n_colors] if n_colors <= len(custom_colors) else custom_colors

        return Plotter.get_color_palette(palette_name, n_colors)

    def apply_plot_style(self) -> None:
        """
        Apply matplotlib style based on action configuration.

        This should be called at the beginning of plot creation methods.
        """
        plot_config = self.get_plot_config()
        style_config = plot_config.get("style", OmegaConf.create({}))
        Plotter.apply_matplotlib_style(style_config)

    def get_plot_data_cache_path(self, filename: str = "plot_data.pkl") -> Path:
        """
        Get the cache path for plot data.

        Plot data is stored in the same cache directory as action results,
        but in a separate 'plot_data' subdirectory to keep it organized.

        Args:
            filename: Name of the plot data file (default: "plot_data.pkl")

        Returns:
            Path to the plot data cache file

        Example:
            # Save plot data
            plot_data = {"accuracies": [...], "f1_scores": [...]}
            cache_path = self.get_plot_data_cache_path()
            with open(cache_path, "wb") as f:
                pickle.dump(plot_data, f)

            # Load plot data in create_plots()
            cache_path = self.get_plot_data_cache_path()
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    plot_data = pickle.load(f)
        """
        if not hasattr(self, 'get_cache_entry_dir'):
            raise AttributeError(
                "PlottingMixin requires Action base class with get_cache_entry_dir() method. "
                "Ensure your class inherits from both Action and PlottingMixin."
            )
        cache_dir = self.get_cache_entry_dir("plot_data")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / filename

    def save_plot_data(self, plot_data: Dict[str, Any], filename: str = "plot_data.pkl") -> Path:
        """
        Save plot data to cache.

        Plot data contains the processed data needed to regenerate plots,
        separate from the raw action results. This allows create_plots() to
        regenerate plots without re-running the action.

        Args:
            plot_data: Dictionary containing plot data (e.g., accuracies, F1 scores, etc.)
            filename: Name of the plot data file (default: "plot_data.pkl")

        Returns:
            Path to the saved plot data file

        Example:
            # In run() method, after processing data:
            plot_data = {
                "accuracies": self.accuracies,
                "f1_scores": self.f1_scores,
                "classifiers_by_entity_type": self.classifiers_by_entity_type,
            }
            self.save_plot_data(plot_data)
        """
        cache_path = self.get_plot_data_cache_path(filename)
        with open(cache_path, "wb") as f:
            pickle.dump(plot_data, f)
        logger.info(f"💾 Saved plot data to {cache_path}")

        # Also save as CSV when enabled (human-readable log of plot data)
        plot_config = self.get_plot_config()
        if plot_config.get("save_plot_data_csv", True):
            csv_stem = Path(filename).stem
            cache_dir = cache_path.parent
            csv_path = cache_dir / f"{csv_stem}.csv"
            try:
                paths = write_plot_data_csv(plot_data, csv_path)
                for p in paths:
                    logger.debug(f"Saved plot data CSV: {p}")
            except Exception as e:
                logger.warning(f"Could not save plot data CSV for {filename}: {e}")

        return cache_path

    def load_plot_data(self, filename: str = "plot_data.pkl") -> Optional[Dict[str, Any]]:
        """
        Load plot data from cache.

        Args:
            filename: Name of the plot data file (default: "plot_data.pkl")

        Returns:
            Dictionary containing plot data, or None if not found

        Example:
            # In create_plots() method:
            plot_data = self.load_plot_data()
            if plot_data:
                accuracies = plot_data["accuracies"]
                f1_scores = plot_data["f1_scores"]
                # ... regenerate plots from plot_data
        """
        cache_path = self.get_plot_data_cache_path(filename)
        if not cache_path.exists():
            return None
        with open(cache_path, "rb") as f:
            plot_data = pickle.load(f)
        logger.debug(f"📖 Loaded plot data from {cache_path}")
        return plot_data

    def has_plot_data(self, filename: str = "plot_data.pkl") -> bool:
        """
        Check if plot data exists in cache.

        Args:
            filename: Name of the plot data file (default: "plot_data.pkl")

        Returns:
            True if plot data exists, False otherwise
        """
        cache_path = self.get_plot_data_cache_path(filename)
        return cache_path.exists()

    def should_generate_plots(self) -> bool:
        """
        Check if plotting is enabled for this action.

        Returns:
            True if plotting is enabled, False otherwise
        """
        plot_config = self.get_plot_config()
        return plot_config.get("enabled", True)

    def _ensure_plot_data(self, filename: str = "plot_data.pkl") -> Optional[Dict[str, Any]]:
        """
        Ensure plot data exists, loading from cache.

        Args:
            filename: Name of the plot data file (default: "plot_data.pkl")

        Returns:
            Plot data dict or None if unavailable
        """
        return self.load_plot_data(filename)

    def create_plots_with_automation(self, filename: str = "plot_data.pkl"):
        """
        Automated plot generation wrapper.

        Handles:
        - Checking if plotting is enabled
        - Loading plot data
        - Calling _generate_plots() if data available
        - Error handling and logging

        Args:
            filename: Name of the plot data file (default: "plot_data.pkl")
        """
        if not self.should_generate_plots():
            logger.debug(f"⏭️  Plotting disabled for {self.__class__.__name__}")
            return

        plot_data = self._ensure_plot_data(filename)
        if plot_data and hasattr(self, '_generate_plots'):
            self.apply_plot_style()
            self._generate_plots(plot_data)
        elif not plot_data:
            # Check if action was cached (indicates old cache format without plot data)
            was_cached = getattr(self, '_cached_outputs', None) is not None
            if was_cached:
                logger.warning(
                    f"⚠️  No plot data available for {self.__class__.__name__} (action was loaded from cache). "
                    f"This cache was created before the new plot data system was implemented. "
                    f"To generate plots, clear the cache and re-run the action: "
                    f"`action_instance.clear_cache()` or set `force_rerun: true` in config."
                )
            else:
                raise ValueError(
                    f"No plot data available for {self.__class__.__name__}. "
                    f"Expected plot data file: {self.get_plot_data_cache_path(filename)}. "
                    "Run the action first to generate plot data."
                )
