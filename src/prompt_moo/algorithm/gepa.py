"""Search tree for GEPA (Genetic-Pareto) Algorithm.

Each node holds a prompt variant, its per-instance per-objective predictions,
and ancestry links for building the genetic search tree. The tree enforces
Pareto-based domination: a new candidate is only admitted if its per-objective
metric vector is not strictly dominated by any existing node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from ..prompt_template import PromptTemplate

import numpy as np
from morphic import Typed, validate
from pydantic import Field, PrivateAttr


class GEPANode(Typed):
    """A single node in the GEPA search tree.

    Attributes:
        node_id: Unique identifier within the tree.
        prompt: The full prompt (skeleton + instruction) for this candidate.
        depth: Depth in the tree (root = 0).
        metadata: Optional arbitrary key-value data attached to this node.
    """

    node_id: int
    prompt_template: PromptTemplate
    depth: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    _predictions: np.ndarray = PrivateAttr()
    _children: List[GEPANode] = PrivateAttr(default_factory=list)
    _parent: Optional[GEPANode] = PrivateAttr(default=None)

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        predictions = data.pop("predictions", None)
        if predictions is not None:
            data["_predictions"] = predictions
            if not isinstance(predictions, np.ndarray):
                data["_predictions"] = np.asarray(predictions)

        parent = data.pop("parent", None)
        if parent is not None:
            data["_parent"] = parent
            data["depth"] = parent.depth + 1
        else:
            data["_parent"] = None
            data.setdefault("depth", 0)

    @property
    def predictions(self) -> np.ndarray:
        return self._predictions

    @property
    def children(self) -> List[GEPANode]:
        return list(self._children)

    @property
    def parent(self) -> Optional[GEPANode]:
        return self._parent

    @property
    def n_validation_samples(self) -> int:
        """Number of validation samples in the predictions matrix."""
        return self._predictions.shape[0]

    @property
    def n_objectives(self) -> int:
        """Number of objectives in the predictions matrix."""
        return self._predictions.shape[1]

    def add_child(self, node: GEPANode) -> None:
        """Append *node* to this node's children list."""
        self._children.append(node)

    def __repr__(self) -> str:
        return (
            f"GEPANode(id={self.node_id}, depth={self.depth}, "
            f"n_children={len(self._children)})"
        )


_METRIC_REGISTRY: Dict[str, Literal["mae", "f1"]] = {
    "mae": "mae",
    "mean_absolute_error": "mae",
    "f1": "f1",
    "f1_score": "f1",
}


class GEPASearchTree(Typed):
    """
    Search tree for GEPA that enforces Pareto domination.

    Parameters
    ----------
    metric_fn : str
        Which metric to use when converting predictions into objective scores.
        Supported values (case-insensitive, with aliases):
          - "mae" / "mean_absolute_error": lower is better
          - "f1" / "f1_score": higher is better

        The resolved metric name is stored internally (see `resolved_metric`).

    domination_direction : Any
        Direction of improvement for each objective.
        - +1 means "higher is better" for that objective (e.g., F1).
        - -1 means "lower is better" for that objective (e.g., MAE).

        You may pass:
          - a scalar (broadcast to all objectives), or
          - a 1-D array-like of shape (n_objectives,).

        Internally this is normalized to a 1-D numpy array `_dom_dir`.

    ground_truth : Optional[np.ndarray]
        Optional ground-truth matrix used to compute true metric values instead
        of falling back to raw averages.

        Expected shape:
          - (n_samples, n_objectives)

        Behavior:
          - If provided, candidate evaluation uses `metric_fn` comparing
            `predictions` vs `ground_truth` per objective.
          - If omitted, the tree falls back to using mean(predictions, axis=0)
            as a proxy metric vector (and per-instance "scores" are just the
            raw prediction values).

        Note: `ground_truth` is accepted during initialization via
        `pre_initialize` (it is popped from the init dict and stored as a
        private attribute `_ground_truth`).
    """

    metric_fn: str
    domination_direction: Any = None

    _ground_truth: Optional[np.ndarray] = PrivateAttr(default=None)
    _nodes: List[GEPANode] = PrivateAttr(default_factory=list)
    _root: Optional[GEPANode] = PrivateAttr(default=None)
    _next_id: int = PrivateAttr(default=0)
    _dom_dir: np.ndarray = PrivateAttr()
    _resolved_metric: str = PrivateAttr()

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        metric = str(data.get("metric_fn", "")).lower()
        resolved = _METRIC_REGISTRY.get(metric)
        if resolved is None:
            raise ValueError(
                f"Unknown metric_fn={metric!r}. Supported: {sorted(_METRIC_REGISTRY.keys())}"
            )
        data["_resolved_metric"] = resolved

        ground_truth = data.pop("ground_truth", None)
        if ground_truth is not None:
            if not isinstance(ground_truth, np.ndarray):
                ground_truth = np.asarray(ground_truth)
            data["_ground_truth"] = ground_truth

        dom = data.get("domination_direction")
        if dom is not None:
            if not isinstance(dom, np.ndarray):
                dom = np.asarray(dom, dtype=np.float64)
            data["_dom_dir"] = np.atleast_1d(dom)
        else:
            # Default: higher is better
            data["_dom_dir"] = np.array([1.0])

    @property
    def root(self) -> Optional[GEPANode]:
        return self._root

    @property
    def nodes(self) -> List[GEPANode]:
        return list(self._nodes)

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def dom_dir(self) -> np.ndarray:
        return self._dom_dir

    @property
    def resolved_metric(self) -> str:
        return self._resolved_metric

    def set_ground_truth(self, ground_truth: np.ndarray) -> None:
        """Set or update the ground-truth matrix."""
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.asarray(ground_truth)
        self._ground_truth = ground_truth

    def add_root(
        self,
        prompt_template: PromptTemplate,
        predictions: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GEPANode:
        """Create and return the root node.

        Raises:
            ValueError: If a root already exists.
        """
        if self._root is not None:
            raise ValueError("Root node already exists")

        node = GEPANode(
            node_id=self._next_id,
            prompt_template=prompt_template,
            predictions=predictions,
            parent=None,
            metadata=metadata,
        )
        self._next_id += 1
        self._root = node
        self._nodes.append(node)
        return node

    @validate
    def try_add_candidate(
        self,
        prompt_template: PromptTemplate,
        predictions: np.ndarray,
        parent: Optional[GEPANode] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GEPANode]:
        """Try to add a candidate node to the tree.

        If the candidate's per-objective metric vector is dominated by any
        existing node, it is **rejected** and ``None`` is returned.
        Otherwise the node is added (under *parent* if given) and returned.

        Args:
            prompt: The full prompt of the candidate.
            predictions: Per-instance per-objective predictions
                ``(n_samples, n_objectives)``.
            parent: The parent node. If ``None``, the node is added as an
                orphan (not linked to any parent).
            metadata: Optional extra data attached to the node.

        Returns:
            The newly created ``GEPANode``, or ``None`` if dominated.
        """
        if parent is not None and parent not in self._nodes:
            raise ValueError("Parent node is not in this tree")

        new_metrics = self._compute_metric_vector(predictions)
        if self._is_dominated_by_any(new_metrics):
            return None

        node = GEPANode(
            node_id=self._next_id,
            prompt_template=prompt_template,
            predictions=predictions,
            parent=parent,
            metadata=metadata,
        )
        self._next_id += 1
        self._nodes.append(node)

        if parent is not None:
            parent.add_child(node)

        return node

    @validate
    def metric_vector(self, node: GEPANode) -> np.ndarray:
        """Compute the per-objective metric vector for *node*."""
        return self._compute_metric_vector(node.predictions)

    @validate
    def all_metric_vectors(self) -> np.ndarray:
        """Return per-objective metric vectors for every node.

        Shape: ``(n_nodes, n_objectives)``.
        """
        return np.stack(
            [self._compute_metric_vector(n.predictions) for n in self._nodes]
        )

    @validate
    def is_dominated(
        self,
        metrics_a: np.ndarray,
        metrics_b: np.ndarray,
    ) -> bool:
        """Return ``True`` if candidate A is strictly dominated by candidate B.

        Domination means: for every objective, B is at least as good as A,
        and strictly better on at least one objective.

        Direction is controlled by ``domination_direction``:
        - ``+1``: higher is better → B dominates A iff ``B >= A``
          element-wise with at least one strict ``>``.
        - ``-1``: lower is better → B dominates A iff ``B <= A``
          element-wise with at least one strict ``<``.
        """
        scaled_a = metrics_a * self._dom_dir
        scaled_b = metrics_b * self._dom_dir
        return bool(np.all(scaled_b >= scaled_a) and np.any(scaled_b > scaled_a))

    @validate
    def get_pareto_frontier_nodes(self) -> List[Tuple[GEPANode, int]]:
        """Return non-dominated candidates and how many wins each has.

        For each validation sample (row of the predictions matrix),
        identifies which node achieves the best aggregate score. Collects
        all unique nodes that are best on at least one sample, then prunes
        strictly dominated ones.

        Returns:
            A list of ``(node, win_count)`` tuples for the non-dominated
            candidates. *win_count* is the number of validation samples for
            which this node is the best (after pruning dominated nodes).
        """
        if not self._nodes:
            return []

        metric_matrix = np.stack(
            [
                self._compute_instance_scores(n.predictions)
                for n in self._nodes
            ]
        )

        n_nodes, n_samples, n_objectives = metric_matrix.shape

        best_node_per_instance: List[int] = []
        for i in range(n_samples):
            scaled = metric_matrix[:, i, :] * self._dom_dir
            per_obj_scores = scaled.sum(axis=1)
            best_idx = int(np.argmax(per_obj_scores))
            best_node_per_instance.append(best_idx)

        candidate_set: List[GEPANode] = []
        seen: set[int] = set()
        for idx in best_node_per_instance:
            n = self._nodes[idx]
            if n.node_id not in seen:
                seen.add(n.node_id)
                candidate_set.append(n)

        dominated: set[int] = set()
        changed = True
        while changed:
            changed = False
            for a in candidate_set:
                if a.node_id in dominated:
                    continue
                for b in candidate_set:
                    if b.node_id in dominated or a is b:
                        continue
                    if self.is_dominated(
                        self._compute_metric_vector(a.predictions),
                        self._compute_metric_vector(b.predictions),
                    ):
                        dominated.add(a.node_id)
                        changed = True
                        break

        non_dominated = [n for n in candidate_set if n.node_id not in dominated]

        result: List[Tuple[GEPANode, int]] = []
        for n in non_dominated:
            count = sum(
                1 for idx in best_node_per_instance if self._nodes[idx] is n
            )
            result.append((n, count))

        return result

    @validate
    def select_candidate(self) -> Optional[GEPANode]:
        """Stochastically select a candidate from the Pareto frontier.

        Candidates are sampled proportional to their win count (number of
        validation samples they are best on).

        Returns:
            A ``GEPANode`` or ``None`` if the tree is empty.
        """
        frontier = self.get_pareto_frontier_nodes()
        if not frontier:
            return None
        nodes_list = [n for n, _ in frontier]
        weights = [w for _, w in frontier]
        total = sum(weights)
        if total == 0:
            return nodes_list[0]
        probs = [w / total for w in weights]
        rng = np.random.default_rng()
        idx = int(rng.choice(len(nodes_list), p=probs))
        return nodes_list[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metric_vector(self, predictions: np.ndarray) -> np.ndarray:
        """Compute per-objective metric values from *predictions*.

        If ground truth is available, computes the named metric
        (MAE, F1) per objective. Otherwise falls back to raw mean.
        """
        if self._ground_truth is None:
            if predictions.ndim == 1:
                return predictions
            return np.mean(predictions, axis=0)

        if predictions.ndim == 1:
            return self._compute_metric(
                predictions, self._ground_truth
            )

        n_objectives = predictions.shape[1]
        return np.array(
            [
                self._compute_metric(predictions[:, j], self._ground_truth[:, j])
                for j in range(n_objectives)
            ]
        )

    def _compute_instance_scores(self, predictions: np.ndarray) -> np.ndarray:
        """Return per-instance score vector for each sample."""
        if self._ground_truth is None:
            return predictions
        if predictions.ndim == 1:
            return predictions
        return np.array(
            [
                self._compute_metric(predictions[i], self._ground_truth[i])
                for i in range(predictions.shape[0])
            ]
        )

    def _compute_metric(self, preds: np.ndarray, truth: np.ndarray) -> float:
        """Compute the metric for a single sample (or 1-D vectors)."""
        if self._resolved_metric == "mae":
            return float(np.mean(np.abs(preds - truth)))

        if self._resolved_metric == "f1":
            from sklearn.metrics import f1_score

            preds_clean = np.round(preds).astype(int)
            truth_clean = np.round(truth).astype(int)
            return float(
                f1_score(truth_clean, preds_clean, average="macro", zero_division=0.0)
            )

        raise ValueError(f"Unknown resolved metric: {self._resolved_metric}")

    def _is_dominated_by_any(self, metrics: np.ndarray) -> bool:
        """Check whether *metrics* is dominated by any node in the tree."""
        for node in self._nodes:
            existing = self._compute_metric_vector(node.predictions)
            if self.is_dominated(metrics, existing):
                return True
        return False