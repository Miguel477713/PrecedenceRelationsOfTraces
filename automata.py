import os
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from graphviz import Digraph


if os.name == "nt":
    os.system("cls")
elif os.name == "posix":
    os.system("clear")


StartSentinelEvent = "-"
EndSentinelEvent = "-"


@dataclass(frozen=True)
class Trace:
    traceName: str
    eventsInOrder: Tuple[str, ...]


@dataclass(frozen=True)
class PrecedenceFact:
    earlierEvent: str
    laterEvent: str
    traceName: str

    def AsTuple(self) -> Tuple[str, str, str]:
        return (self.earlierEvent, self.laterEvent, self.traceName)


@dataclass
class AutomatonState:
    """
    IMPORTANT:
    transitionsByEvent stores target STATE IDS (strings), not AutomatonState objects.
    This avoids 'unhashable type: AutomatonState' when using sets.
    """
    stateId: str
    isAccepting: bool
    transitionsByEvent: Dict[str, Set[str]]

    def __init__(self, stateId: str):
        self.stateId = stateId
        self.isAccepting = False
        self.transitionsByEvent = {}

    def AddTransition(self, event: str, targetStateId: str) -> None:
        self.transitionsByEvent.setdefault(event, set()).add(targetStateId)


# ============================================================
# Parsing
# ============================================================

def ParseTracesFromTextFile(path: str) -> List[Trace]:
    """
    Input format:
        NTR: 3          # optional
        TR1: A B C ...
        TR2: ...
    "TRi:" can be omitted, in which case lines are treated as plain event sequences.
    Events are whitespace-separated tokens.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as inputFile:
        rawLines = inputFile.read().splitlines()

    cleanedLines: List[str] = []
    for rawLine in rawLines:
        line = rawLine.strip()
        if not line:
            continue

        commentIndex = line.find("#")
        if commentIndex >= 0:
            line = line[:commentIndex].strip()

        if line:
            cleanedLines.append(line)

    traces: List[Trace] = []
    automaticTraceIndex = 1

    for line in cleanedLines:
        if re.match(r"^\s*NTR\s*:", line, flags=re.IGNORECASE):
            continue

        match = re.match(r"^\s*(TR\s*\d+)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            traceName = match.group(1).replace(" ", "").upper()
            eventPart = match.group(2).strip()
        else:
            traceName = f"TR{automaticTraceIndex}"
            automaticTraceIndex += 1
            eventPart = line

        eventsInOrder = tuple(token.strip() for token in eventPart.split() if token.strip())
        if not eventsInOrder:
            continue

        traces.append(Trace(traceName=traceName, eventsInOrder=eventsInOrder))

    if not traces:
        raise ValueError("No traces were found in the input file.")

    return traces


def CollectEventAlphabet(traces: Sequence[Trace]) -> List[str]:
    alphabetSet: Set[str] = set()
    for trace in traces:
        for event in trace.eventsInOrder:
            alphabetSet.add(event)

    return sorted(alphabetSet)


# ============================================================
# BEF(delta): precedence facts per trace
# ============================================================

def BuildPrecedenceFacts(traces: Sequence[Trace]) -> List[PrecedenceFact]:
    precedenceFacts: List[PrecedenceFact] = []

    for trace in traces:
        eventsInOrder = trace.eventsInOrder
        traceName = trace.traceName

        precedenceFacts.append(
            PrecedenceFact(
                earlierEvent=StartSentinelEvent,
                laterEvent=eventsInOrder[0],
                traceName=traceName,
            )
        )

        for index in range(len(eventsInOrder) - 1):
            precedenceFacts.append(
                PrecedenceFact(
                    earlierEvent=eventsInOrder[index],
                    laterEvent=eventsInOrder[index + 1],
                    traceName=traceName,
                )
            )

        precedenceFacts.append(
            PrecedenceFact(
                earlierEvent=eventsInOrder[-1],
                laterEvent=EndSentinelEvent,
                traceName=traceName,
            )
        )

    return precedenceFacts


def BuildAggregatedPrecedenceEdges(precedenceFacts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    aggregatedEdges: Set[Tuple[str, str]] = set()

    for fact in precedenceFacts:
        if fact.earlierEvent == StartSentinelEvent:
            continue
        if fact.laterEvent == EndSentinelEvent:
            continue

        aggregatedEdges.add((fact.earlierEvent, fact.laterEvent))

    return aggregatedEdges


# ============================================================
# Conc and PO(delta)
# ============================================================

def ComputeConcurrentEventPairs(precedenceFacts: Sequence[PrecedenceFact]) -> FrozenSet[FrozenSet[str]]:
    tracesPerDirectedEdge: Dict[Tuple[str, str], Set[str]] = {}

    for fact in precedenceFacts:
        if fact.earlierEvent == StartSentinelEvent:
            continue
        if fact.laterEvent == EndSentinelEvent:
            continue

        directedEdge = (fact.earlierEvent, fact.laterEvent)
        tracesPerDirectedEdge.setdefault(directedEdge, set()).add(fact.traceName)

    concurrentPairs: Set[FrozenSet[str]] = set()

    for (eventA, eventB), tracesForAB in tracesPerDirectedEdge.items():
        reversedEdge = (eventB, eventA)
        tracesForBA = tracesPerDirectedEdge.get(reversedEdge, set())

        if tracesForAB and tracesForBA:
            if len(tracesForAB.union(tracesForBA)) >= 2:
                concurrentPairs.add(frozenset({eventA, eventB}))

    return frozenset(concurrentPairs)


def BuildPartialOrderEdges(precedenceFacts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    aggregatedEdges = BuildAggregatedPrecedenceEdges(precedenceFacts)
    concurrentPairs = ComputeConcurrentEventPairs(precedenceFacts)

    partialOrderEdges: Set[Tuple[str, str]] = set()

    for (earlierEvent, laterEvent) in aggregatedEdges:
        if frozenset({earlierEvent, laterEvent}) in concurrentPairs:
            continue
        partialOrderEdges.add((earlierEvent, laterEvent))

    return partialOrderEdges


def FindOneDirectedCycle(edges: Set[Tuple[str, str]]) -> Optional[List[str]]:
    """
    Iterative cycle detection (no recursion).
    Returns a cycle like [v0, v1, ..., v0] if found; otherwise None.
    """
    adjacency: Dict[str, List[str]] = {}
    allNodes: Set[str] = set()

    for (source, target) in edges:
        adjacency.setdefault(source, []).append(target)
        allNodes.add(source)
        allNodes.add(target)

    for node in adjacency.keys():
        adjacency[node].sort()

    visitState: Dict[str, int] = {node: 0 for node in allNodes}  # 0=unvisited, 1=visiting, 2=finished
    parent: Dict[str, Optional[str]] = {node: None for node in allNodes}

    for startNode in sorted(allNodes):
        if visitState[startNode] != 0:
            continue

        stack: List[Tuple[str, int]] = [(startNode, 0)]
        visitState[startNode] = 1
        parent[startNode] = None

        while stack:
            currentNode, nextNeighborIndex = stack[-1]
            neighbors = adjacency.get(currentNode, [])

            if nextNeighborIndex >= len(neighbors):
                visitState[currentNode] = 2
                stack.pop()
                continue

            neighbor = neighbors[nextNeighborIndex]
            stack[-1] = (currentNode, nextNeighborIndex + 1)

            if visitState.get(neighbor, 0) == 0:
                visitState[neighbor] = 1
                parent[neighbor] = currentNode
                stack.append((neighbor, 0))
            elif visitState.get(neighbor, 0) == 1:
                cycleNodesReversed: List[str] = [currentNode]
                walkNode = currentNode
                while walkNode != neighbor and parent[walkNode] is not None:
                    walkNode = parent[walkNode]
                    cycleNodesReversed.append(walkNode)

                cycleNodesReversed.reverse()
                cycleNodes = cycleNodesReversed + [neighbor]
                return cycleNodes

    return None


def CreateUniqueDuplicateNodeName(originalNodeName: str, usedNodeNames: Set[str]) -> str:
    duplicateIndex = 1
    while True:
        candidateName = f"{originalNodeName}Prime{duplicateIndex}"
        if candidateName not in usedNodeNames:
            return candidateName
        duplicateIndex += 1


def UnfoldOneCycleAutomatically(
    edges: Set[Tuple[str, str]],
    usedNodeNames: Set[str],
) -> Tuple[Set[Tuple[str, str]], Optional[str]]:
    """
    Unfold exactly one cycle by duplicating the cycle-entry node (cycle[0]),
    redirecting ONLY the closing edge into the duplicate node.

    IMPORTANT FIX:
    Do NOT copy outgoing edges from the original node into the duplicate node.
    Copying outgoing edges can reintroduce cycles and prevent termination.
    """
    cycle = FindOneDirectedCycle(edges)
    if cycle is None:
        return (set(edges), None)

    cycleEntryNode = cycle[0]
    duplicateNodeName = CreateUniqueDuplicateNodeName(cycleEntryNode, usedNodeNames)

    cycleEdges: List[Tuple[str, str]] = []
    for index in range(len(cycle) - 1):
        cycleEdges.append((cycle[index], cycle[index + 1]))

    closingEdge = cycleEdges[-1]  # ( ... -> cycleEntryNode )

    newEdges: Set[Tuple[str, str]] = set(edges)

    if closingEdge in newEdges:
        newEdges.remove(closingEdge)

    newEdges.add((closingEdge[0], duplicateNodeName))

    usedNodeNames.add(duplicateNodeName)
    return (newEdges, duplicateNodeName)


def UnfoldAllCyclesAutomatically(edges: Set[Tuple[str, str]]) -> Tuple[Set[Tuple[str, str]], List[str]]:
    createdDuplicateNodeNames: List[str] = []
    currentEdges: Set[Tuple[str, str]] = set(edges)

    usedNodeNames: Set[str] = set()
    for (source, target) in currentEdges:
        usedNodeNames.add(source)
        usedNodeNames.add(target)

    maximumUnfoldSteps = 1000
    unfoldStepIndex = 0

    while True:
        unfoldStepIndex += 1
        if unfoldStepIndex > maximumUnfoldSteps:
            raise RuntimeError("Cycle unfolding did not converge (exceeded maximumUnfoldSteps).")

        cycle = FindOneDirectedCycle(currentEdges)
        if cycle is None:
            break

        currentEdges, createdDuplicateNodeName = UnfoldOneCycleAutomatically(currentEdges, usedNodeNames)
        if createdDuplicateNodeName is None:
            break

        createdDuplicateNodeNames.append(createdDuplicateNodeName)

    return (currentEdges, createdDuplicateNodeNames)


# ============================================================
# Graphviz drawing
# ============================================================

def DrawDirectedEventGraph(
    events: Sequence[str],
    edges: Set[Tuple[str, str]],
    outputFilePrefix: str,
    graphTitle: str,
) -> None:
    dot = Digraph(name=graphTitle, format="png")
    dot.attr(rankdir="LR", splines="true", nodesep="0.5", ranksep="0.7")
    dot.attr("node", shape="circle")
    dot.attr("edge")

    for event in events:
        dot.node(event, label=event)

    for (source, target) in sorted(edges):
        dot.edge(source, target)

    dot.render(outputFilePrefix, cleanup=True)


# ============================================================
# NFA: Trie construction (safe, no unhashable states)
# ============================================================

def BuildTrieNfa(traces: Sequence[Trace]) -> Tuple[str, Dict[str, AutomatonState]]:
    """
    Returns:
      initialStateId,
      statesById dictionary
    """
    initialStateId = "q0"
    statesById: Dict[str, AutomatonState] = {initialStateId: AutomatonState(initialStateId)}
    nextStateIndex = 1

    for trace in traces:
        currentStateId = initialStateId

        for event in trace.eventsInOrder:
            currentState = statesById[currentStateId]
            existingTargetStateIds = currentState.transitionsByEvent.get(event, set())

            if existingTargetStateIds:
                nextStateId = sorted(existingTargetStateIds)[0]
            else:
                nextStateId = f"q{nextStateIndex}"
                nextStateIndex += 1
                statesById[nextStateId] = AutomatonState(nextStateId)
                currentState.AddTransition(event, nextStateId)

            currentStateId = nextStateId

        statesById[currentStateId].isAccepting = True

    return (initialStateId, statesById)


def ListAllReachableStateIds(initialStateId: str, statesById: Dict[str, AutomatonState]) -> List[str]:
    visitedStateIds: Set[str] = set()
    stack: List[str] = [initialStateId]

    while stack:
        stateId = stack.pop()
        if stateId in visitedStateIds:
            continue

        visitedStateIds.add(stateId)

        state = statesById[stateId]
        for event in state.transitionsByEvent:
            for targetStateId in state.transitionsByEvent[event]:
                if targetStateId not in visitedStateIds:
                    stack.append(targetStateId)

    return sorted(visitedStateIds)


def DrawNfaWithGraphviz(
    initialStateId: str,
    statesById: Dict[str, AutomatonState],
    outputFilePrefix: str,
    graphTitle: str
) -> None:
    dot = Digraph(name=graphTitle, format="png")
    dot.attr(rankdir="LR", splines="true", nodesep="0.5", ranksep="0.7")

    dot.node("__start__", label="", shape="none", width="0", height="0")

    reachableStateIds = ListAllReachableStateIds(initialStateId, statesById)

    for stateId in reachableStateIds:
        state = statesById[stateId]
        shape = "doublecircle" if state.isAccepting else "circle"
        dot.node(state.stateId, label=state.stateId, shape=shape)

    dot.edge("__start__", initialStateId, label="")

    for stateId in reachableStateIds:
        state = statesById[stateId]
        for event in sorted(state.transitionsByEvent.keys()):
            for targetStateId in sorted(state.transitionsByEvent[event]):
                dot.edge(stateId, targetStateId, label=event)

    dot.render(outputFilePrefix, cleanup=True)

def WritePrecedenceFactsToTextFile(precedenceFacts: Sequence[PrecedenceFact], outputPath: str) -> None:
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for fact in precedenceFacts:
            outputFile.write(f"({fact.earlierEvent}, {fact.laterEvent}, {fact.traceName})\n")

def Main() -> None:
    print("Proyecto: Procesamiento de secuencias de eventos (Graphviz)")

    inputPath = input("Enter input trace file path (e.g.: traces.txt): ").strip()
    if not inputPath:
        print("Error: input path is required.")
        return

    outputPrefix = input("Enter output file prefix (e.g.: output/run1): ").strip()
    if not outputPrefix:
        outputPrefix = "output/eventSequences"

    outputDirectory = os.path.dirname(outputPrefix)
    if outputDirectory and not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory, exist_ok=True)

    traces = ParseTracesFromTextFile(inputPath)
    alphabetEvents = CollectEventAlphabet(traces)

    print(f"\nLoaded traces: {len(traces)}")
    for trace in traces:
        print(f"  {trace.traceName}: {' '.join(trace.eventsInOrder)}")

    precedenceFacts = BuildPrecedenceFacts(traces)
    precedenceFactsOutputPath = f"{outputPrefix}_BefFacts.txt"
    WritePrecedenceFactsToTextFile(precedenceFacts, precedenceFactsOutputPath)
    print(f"\nWrote BEF facts to: {precedenceFactsOutputPath}")

    aggregatedPrecedenceEdges = BuildAggregatedPrecedenceEdges(precedenceFacts)
    precedenceGraphOutputPrefix = f"{outputPrefix}_PrecedenceGraph"
    DrawDirectedEventGraph(
        events=alphabetEvents,
        edges=aggregatedPrecedenceEdges,
        outputFilePrefix=precedenceGraphOutputPrefix,
        graphTitle="PrecedenceGraph",
    )
    print(f"Rendered precedence graph to: {precedenceGraphOutputPrefix}.png")

    partialOrderEdges = BuildPartialOrderEdges(precedenceFacts)
    partialOrderGraphOutputPrefix = f"{outputPrefix}_PartialOrderGraph"
    DrawDirectedEventGraph(
        events=alphabetEvents,
        edges=partialOrderEdges,
        outputFilePrefix=partialOrderGraphOutputPrefix,
        graphTitle="PartialOrderGraph",
    )
    print(f"Rendered partial order graph to: {partialOrderGraphOutputPrefix}.png")

    cycle = FindOneDirectedCycle(partialOrderEdges)
    if cycle is None:
        print("\nNo directed cycle detected in PO.")
    else:
        print("\nNote: A directed cycle exists in PO.")
        print("Cycle found:", " -> ".join(cycle))

        unfoldedEdges, createdDuplicateNodeNames = UnfoldAllCyclesAutomatically(partialOrderEdges)
        unfoldedEvents = sorted(set(alphabetEvents).union(set(createdDuplicateNodeNames)))

        unfoldedGraphOutputPrefix = f"{outputPrefix}_PartialOrderGraphUnfolded"
        DrawDirectedEventGraph(
            events=unfoldedEvents,
            edges=unfoldedEdges,
            outputFilePrefix=unfoldedGraphOutputPrefix,
            graphTitle="PartialOrderGraphUnfolded",
        )

        print("Automatically unfolded cycles.")
        if createdDuplicateNodeNames:
            print("Created duplicate nodes:", ", ".join(createdDuplicateNodeNames))
        print(f"Rendered unfolded PO graph to: {unfoldedGraphOutputPrefix}.png")

        partialOrderEdges = unfoldedEdges
        alphabetEvents = unfoldedEvents

    trieNfaInitialStateId, trieNfaStatesById = BuildTrieNfa(traces)
    trieNfaOutputPrefix = f"{outputPrefix}_NfaTrie"
    DrawNfaWithGraphviz(
        initialStateId=trieNfaInitialStateId,
        statesById=trieNfaStatesById,
        outputFilePrefix=trieNfaOutputPrefix,
        graphTitle="NfaTrie",
    )
    print(f"\nRendered trie NFA to: {trieNfaOutputPrefix}.png")

    print("Note: Compact NFA merging is not included in this fixed version (ID-based transitions).")

    print("\nDone.")


if __name__ == "__main__":
    Main()
