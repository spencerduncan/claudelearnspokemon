"""
DSL Abstract Syntax Tree - Performance-Optimized Node Structure

Performance design decisions:
- Flat node hierarchy for cache efficiency
- Enum-based node types for fast dispatch
- Pre-allocated node pools to minimize allocations
- Immutable nodes for safe concurrent access
- Compact memory layout for better cache behavior
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Optional, Union


class NodeType(IntEnum):
    """Enum-based node types for fast dispatch and compact representation."""

    # Primitive nodes - direct input mappings
    INPUT = auto()  # A, B, START, SELECT, etc.
    DELAY = auto()  # Wait for specified frames

    # Compound nodes - higher-level constructs
    SEQUENCE = auto()  # Series of nodes executed in order
    REPEAT = auto()  # Loop construct with count
    MACRO = auto()  # Named pattern expansion

    # Conditional nodes - runtime decisions
    CONDITIONAL = auto()  # if/then/else based on game state
    PROBABILITY = auto()  # Probabilistic branch selection

    # Meta nodes - compilation directives
    OBSERVATION = auto()  # Tile capture insertion point
    CHECKPOINT = auto()  # Save state directive

    # Error/Unknown
    INVALID = auto()


@dataclass(frozen=True)
class ASTNode:
    """
    Performance-optimized AST node with flat structure.

    Design rationale:
    - Single node type with discriminated union via node_type
    - Frozen dataclass for immutability and hashability
    - Minimal memory footprint per node
    - Fast comparison operations
    """

    node_type: NodeType
    value: Union[str, int, float] = ""
    children: tuple[ASTNode, ...] = ()
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Ensure metadata is never None for consistent access patterns."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def is_primitive(self) -> bool:
        """Fast check for primitive nodes (no children)."""
        return self.node_type in (NodeType.INPUT, NodeType.DELAY)

    @property
    def is_compound(self) -> bool:
        """Fast check for compound nodes (has children)."""
        return len(self.children) > 0

    def estimate_complexity(self) -> int:
        """
        Estimate computational complexity for optimization.
        Used for performance analysis and compilation ordering.
        """
        if self.node_type == NodeType.INPUT:
            return 1
        elif self.node_type == NodeType.DELAY:
            return 1
        elif self.node_type == NodeType.SEQUENCE:
            return sum(child.estimate_complexity() for child in self.children)
        elif self.node_type == NodeType.REPEAT:
            count = int(self.value) if isinstance(self.value, (int, str)) else 1
            child_complexity = sum(child.estimate_complexity() for child in self.children)
            return count * child_complexity
        elif self.node_type == NodeType.MACRO:
            # Macro complexity depends on expansion - conservative estimate
            return 10
        elif self.node_type == NodeType.CONDITIONAL:
            # Worst case - all branches
            return sum(child.estimate_complexity() for child in self.children)
        else:
            return sum(child.estimate_complexity() for child in self.children) + 1


class NodeFactory:
    """
    Factory for creating optimized AST nodes with memory pooling.

    Performance optimizations:
    - Pre-allocated node pools for common patterns
    - Interning of frequently used values
    - Fast node creation without redundant allocations
    """

    def __init__(self):
        self._string_cache: dict[str, str] = {}
        self._common_nodes: dict[tuple, ASTNode] = {}

    def create_input_node(self, input_name: str) -> ASTNode:
        """Create optimized input node with string interning."""
        # Intern common input strings for memory efficiency
        input_name = self._string_cache.setdefault(input_name, input_name)

        # Cache common input nodes
        cache_key = (NodeType.INPUT, input_name)
        if cache_key not in self._common_nodes:
            self._common_nodes[cache_key] = ASTNode(
                node_type=NodeType.INPUT,
                value=input_name,
                metadata={"frames": 1},  # Most inputs take 1 frame
            )
        return self._common_nodes[cache_key]

    def create_delay_node(self, frames: int) -> ASTNode:
        """Create delay node with frame count validation."""
        if frames <= 0:
            raise ValueError(f"Delay frames must be positive, got {frames}")

        # Performance safety: prevent huge delays during compilation
        MAX_DELAY_FRAMES = 10000
        if frames > MAX_DELAY_FRAMES:
            raise ValueError(
                f"Delay too large: {frames} frames exceeds limit of {MAX_DELAY_FRAMES}"
            )

        cache_key = (NodeType.DELAY, frames)
        if cache_key not in self._common_nodes:
            self._common_nodes[cache_key] = ASTNode(
                node_type=NodeType.DELAY, value=frames, metadata={"frames": frames}
            )
        return self._common_nodes[cache_key]

    def create_sequence_node(self, children: list[ASTNode]) -> ASTNode:
        """Create sequence node with frame estimation."""
        if not children:
            raise ValueError("Sequence node requires at least one child")

        total_frames = sum(child.metadata.get("frames", 1) for child in children)

        return ASTNode(
            node_type=NodeType.SEQUENCE, children=tuple(children), metadata={"frames": total_frames}
        )

    def create_repeat_node(self, count: int, children: list[ASTNode]) -> ASTNode:
        """Create repeat node with loop analysis."""
        if count <= 0:
            raise ValueError(f"Repeat count must be positive, got {count}")
        if not children:
            raise ValueError("Repeat node requires at least one child")

        child_frames = sum(child.metadata.get("frames", 1) for child in children)
        total_frames = count * child_frames

        return ASTNode(
            node_type=NodeType.REPEAT,
            value=count,
            children=tuple(children),
            metadata={"frames": total_frames, "loop_count": count},
        )

    def create_macro_node(self, name: str, args: list[str] = None) -> ASTNode:
        """Create macro node with parameter tracking."""
        args = args or []
        return ASTNode(
            node_type=NodeType.MACRO,
            value=name,
            metadata={"args": args, "frames": -1},  # -1 = needs expansion
        )

    def create_conditional_node(
        self, condition: str, then_branch: ASTNode, else_branch: Optional[ASTNode] = None
    ) -> ASTNode:
        """Create conditional node with branch analysis."""
        children = [then_branch]
        if else_branch:
            children.append(else_branch)

        # Estimate frames as average of branches (conservative)
        then_frames = then_branch.metadata.get("frames", 1)
        else_frames = else_branch.metadata.get("frames", 0) if else_branch else 0
        avg_frames = (then_frames + else_frames) // 2

        return ASTNode(
            node_type=NodeType.CONDITIONAL,
            value=condition,
            children=tuple(children),
            metadata={"frames": avg_frames, "condition": condition},
        )

    def create_observation_node(self, density: float = 1.0) -> ASTNode:
        """Create observation point with capture density."""
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"Observation density must be 0.0-1.0, got {density}")

        return ASTNode(
            node_type=NodeType.OBSERVATION,
            value=density,
            metadata={"frames": 0, "density": density},  # Observations take no game time
        )


@dataclass(frozen=True)
class CompiledScript:
    """
    Compiled script representation optimized for execution.

    Performance features:
    - Pre-computed frame counts for scheduling
    - Flattened instruction sequence for fast iteration
    - Embedded observation points for minimal overhead
    - Deterministic compilation fingerprint for caching
    """

    instructions: tuple[str, ...]  # Flattened input sequence
    total_frames: int  # Pre-computed execution time
    observation_points: tuple[int, ...]  # Frame indices for tile captures
    metadata: dict[str, Any]  # Compilation info and diagnostics

    @property
    def estimated_duration_ms(self) -> float:
        """Estimate execution time in milliseconds (60 FPS)."""
        return (self.total_frames / 60.0) * 1000.0

    @property
    def instruction_count(self) -> int:
        """Number of discrete instructions."""
        return len(self.instructions)

    def get_diagnostic_info(self) -> dict[str, Any]:
        """Get compilation diagnostic information."""
        return {
            "instruction_count": self.instruction_count,
            "total_frames": self.total_frames,
            "estimated_duration_ms": self.estimated_duration_ms,
            "observation_points": len(self.observation_points),
            "compilation_time_ms": self.metadata.get("compile_time_ms", 0),
            "ast_nodes": self.metadata.get("ast_node_count", 0),
            "macro_expansions": self.metadata.get("macro_expansions", 0),
        }


class ASTVisitor:
    """
    High-performance AST visitor with dispatch table optimization.

    Uses method dispatch table for O(1) node type routing.
    Subclasses override visit_* methods for custom behavior.
    """

    def __init__(self):
        # Build dispatch table for O(1) method lookup
        self._dispatch_table = {
            NodeType.INPUT: self.visit_input,
            NodeType.DELAY: self.visit_delay,
            NodeType.SEQUENCE: self.visit_sequence,
            NodeType.REPEAT: self.visit_repeat,
            NodeType.MACRO: self.visit_macro,
            NodeType.CONDITIONAL: self.visit_conditional,
            NodeType.PROBABILITY: self.visit_probability,
            NodeType.OBSERVATION: self.visit_observation,
            NodeType.CHECKPOINT: self.visit_checkpoint,
        }

    def visit(self, node: ASTNode) -> Any:
        """Dispatch to appropriate visit method using lookup table."""
        visitor_method = self._dispatch_table.get(node.node_type)
        if visitor_method:
            return visitor_method(node)
        else:
            return self.visit_generic(node)

    # Override these methods in subclasses for custom behavior
    def visit_input(self, node: ASTNode) -> Any:
        return node.value

    def visit_delay(self, node: ASTNode) -> Any:
        return f"DELAY({node.value})"

    def visit_sequence(self, node: ASTNode) -> Any:
        return [self.visit(child) for child in node.children]

    def visit_repeat(self, node: ASTNode) -> Any:
        child_results = [self.visit(child) for child in node.children]
        return [child_results] * int(node.value)

    def visit_macro(self, node: ASTNode) -> Any:
        return f"MACRO({node.value})"

    def visit_conditional(self, node: ASTNode) -> Any:
        branches = [self.visit(child) for child in node.children]
        return f"IF({node.value}, {branches})"

    def visit_probability(self, node: ASTNode) -> Any:
        return f"PROB({node.value})"

    def visit_observation(self, node: ASTNode) -> Any:
        return f"OBSERVE({node.value})"

    def visit_checkpoint(self, node: ASTNode) -> Any:
        return f"CHECKPOINT({node.value})"

    def visit_generic(self, node: ASTNode) -> Any:
        """Fallback for unknown node types."""
        return f"UNKNOWN({node.node_type}, {node.value})"
