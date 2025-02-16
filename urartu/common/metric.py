from typing import Any, Dict, List

import evaluate


class Metric:
    """
    A class designed to handle the loading of evaluation metrics using the `evaluate` library.
    This class enables the instantiation of different metrics based on a configuration,
    facilitating the integration of standardized metric evaluation in machine learning models.

    Methods:
        get_metric: Static method to load a metric based on the given configuration.
    """

    @staticmethod
    def get_metric(cfg: List[Dict[str, Any]]):
        """
        Loads and returns an evaluation metric based on the provided configuration.

        Args:
            cfg (List[Dict[str, Any]]): A list containing a dictionary that specifies the metric configuration.
                                        The dictionary must include a 'name' key that corresponds to the name
                                        of the metric to be loaded.

        Returns:
            evaluate.Metric: An instance of a metric as defined by the `evaluate` library.

        This method assumes the configuration list contains exactly one dictionary with at least a 'name' key,
        which is used to identify and load the corresponding metric from the `evaluate` library.
        """
        return evaluate.load(cfg[0]["name"])
