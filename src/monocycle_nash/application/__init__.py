"""Application layer services and configuration tree."""

from monocycle_nash.application.matrix_config_tree import (
    MatrixConfigTree,
    MatrixConfigTreeResolver,
    MatrixResolutionResult,
    ResolvedOutput,
)
from monocycle_nash.application.matrix_nodes import (
    ApproxDominantEigenpairNode,
    ApproxEquilibriumPreservingNode,
    ApproxMonocycleToGeneralNode,
    CharacterInlineSource,
    CharacterListFromFileNode,
    CharacterNode,
    CharacterSource,
    CharacterVectorGraphOutputNode,
    EquilibriumOutputNode,
    GeneralFromRawNode,
    GeneralFromTeamMatchupsNode,
    GeneralFromTeamsPayoffNode,
    MatrixNode,
    MonocycleFromCharactersNode,
    NodeResolutionContext,
    OutputNode,
    PayoffDirectedGraphOutputNode,
    RandomSkewSymmetricNode,
    TeamInlineSource,
    TeamListFromFileNode,
    TeamNode,
    TeamSource,
)

__all__ = [
    # matrix_config_tree
    "MatrixConfigTree",
    "MatrixConfigTreeResolver",
    "MatrixResolutionResult",
    "ResolvedOutput",
    # matrix_nodes — context
    "NodeResolutionContext",
    # matrix_nodes — matrix nodes
    "MatrixNode",
    "GeneralFromRawNode",
    "MonocycleFromCharactersNode",
    "GeneralFromTeamsPayoffNode",
    "GeneralFromTeamMatchupsNode",
    "RandomSkewSymmetricNode",
    "ApproxMonocycleToGeneralNode",
    "ApproxDominantEigenpairNode",
    "ApproxEquilibriumPreservingNode",
    # matrix_nodes — character nodes
    "CharacterSource",
    "CharacterNode",
    "CharacterInlineSource",
    "CharacterListFromFileNode",
    # matrix_nodes — team nodes
    "TeamSource",
    "TeamNode",
    "TeamInlineSource",
    "TeamListFromFileNode",
    # matrix_nodes — output nodes
    "OutputNode",
    "PayoffDirectedGraphOutputNode",
    "CharacterVectorGraphOutputNode",
    "EquilibriumOutputNode",
]
