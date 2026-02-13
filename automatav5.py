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
    transitionsByEvent stores target STATE IDS (strings).
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as inputFile:
        rawLines = inputFile.read().splitlines()

    traces: List[Trace] = []
    automaticTraceIndex = 1

    for rawLine in rawLines:
        line = rawLine.strip()
        if not line or line.startswith("#") or "NTR:" in line or line.startswith("["):
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
    """
    Returns events in Discovery Order (order of appearance in file).
    This ensures that if P appears before N in the traces, P is listed first.
    """
    alphabetMap: Dict[str, None] = {}
    for trace in traces:
        for event in trace.eventsInOrder:
            alphabetMap[event] = None
    return list(alphabetMap.keys())


# ============================================================
# Precedence / BEF / PO Logic
# ============================================================

def BuildPrecedenceFacts(traces: Sequence[Trace]) -> List[PrecedenceFact]:
    precedenceFacts: List[PrecedenceFact] = []
    for trace in traces:
        events = trace.eventsInOrder
        traceName = trace.traceName

        precedenceFacts.append(PrecedenceFact(StartSentinelEvent, events[0], traceName))
        for i in range(len(events) - 1):
            precedenceFacts.append(PrecedenceFact(events[i], events[i + 1], traceName))
        precedenceFacts.append(PrecedenceFact(events[-1], EndSentinelEvent, traceName))

    return precedenceFacts


def BuildAggregatedPrecedenceEdges(precedenceFacts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    aggregatedEdges: Set[Tuple[str, str]] = set()
    for fact in precedenceFacts:
        if fact.earlierEvent != StartSentinelEvent and fact.laterEvent != EndSentinelEvent:
            aggregatedEdges.add((fact.earlierEvent, fact.laterEvent))
    return aggregatedEdges


def ComputeConcurrentEventPairs(precedenceFacts: Sequence[PrecedenceFact]) -> FrozenSet[FrozenSet[str]]:
    tracesPerDirectedEdge: Dict[Tuple[str, str], Set[str]] = {}
    for fact in precedenceFacts:
        if fact.earlierEvent == StartSentinelEvent or fact.laterEvent == EndSentinelEvent:
            continue
        tracesPerDirectedEdge.setdefault((fact.earlierEvent, fact.laterEvent), set()).add(fact.traceName)

    concurrentPairs: Set[FrozenSet[str]] = set()
    for (A, B), tracesAB in tracesPerDirectedEdge.items():
        tracesBA = tracesPerDirectedEdge.get((B, A), set())
        if tracesAB and tracesBA:
            if len(tracesAB.union(tracesBA)) >= 2:
                concurrentPairs.add(frozenset({A, B}))
    return frozenset(concurrentPairs)


def BuildPartialOrderEdges(precedenceFacts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    aggregated = BuildAggregatedPrecedenceEdges(precedenceFacts)
    concurrent = ComputeConcurrentEventPairs(precedenceFacts)
    poEdges = set()
    for (u, v) in aggregated:
        if frozenset({u, v}) not in concurrent:
            poEdges.add((u, v))
    return poEdges


def FindOneDirectedCycle(edges: Set[Tuple[str, str]], preferredOrder: List[str] = None) -> Optional[List[str]]:
    """
    Detects a cycle.
    If preferredOrder is provided, it iterates start nodes in that order.
    This allows us to prioritize unfolding 'P' over 'N' if 'P' appeared first in traces.
    """
    adj = {}
    nodes = set()
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        nodes.add(u);
        nodes.add(v)

    # Sort adjacency lists for deterministic behavior within neighbor selection
    for u in adj: adj[u].sort()

    # Determine iteration order for Start Nodes
    if preferredOrder:
        # Filter preferredOrder to only include nodes actually in the graph
        node_order = [n for n in preferredOrder if n in nodes]
        # Append any remaining nodes (like duplicates created later) sorted alphabetically
        remaining = sorted(list(nodes.difference(set(node_order))))
        node_order.extend(remaining)
    else:
        node_order = sorted(list(nodes))

    visit = {n: 0 for n in nodes}  # 0=new, 1=visiting, 2=done
    parent = {n: None for n in nodes}

    for start in node_order:
        if visit[start] != 0: continue
        stack = [(start, 0)]  # node, neighbor_idx
        visit[start] = 1

        while stack:
            u, idx = stack[-1]
            neighbors = adj.get(u, [])
            if idx >= len(neighbors):
                visit[u] = 2
                stack.pop()
                continue

            stack[-1] = (u, idx + 1)
            v = neighbors[idx]

            if visit.get(v, 0) == 1:  # Cycle detected
                path = [u]
                curr = u
                while curr != v and curr is not None:
                    curr = parent[curr]
                    if curr is not None: path.append(curr)
                path.reverse()
                return path + [v]
            elif visit.get(v, 0) == 0:
                visit[v] = 1
                parent[v] = u
                stack.append((v, 0))
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
        preferredOrder: List[str]
) -> Tuple[Set[Tuple[str, str]], Optional[str]]:
    cycle = FindOneDirectedCycle(edges, preferredOrder)
    if cycle is None:
        return (set(edges), None)

    # The entry node is the first node in the detected cycle list
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


def UnfoldAllCyclesAutomatically(edges: Set[Tuple[str, str]], preferredOrder: List[str]) -> Tuple[Set[Tuple[str, str]], List[str]]:
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

        cycle = FindOneDirectedCycle(currentEdges, preferredOrder)
        if cycle is None:
            break

        currentEdges, createdDuplicateNodeName = UnfoldOneCycleAutomatically(currentEdges, usedNodeNames, preferredOrder)
        if createdDuplicateNodeName is None:
            break

        createdDuplicateNodeNames.append(createdDuplicateNodeName)

    return (currentEdges, createdDuplicateNodeNames)


# ============================================================
# Graphviz
# ============================================================

def DrawDirectedEventGraph(events: Sequence[str], edges: Set[Tuple[str, str]], outputFilePrefix: str, graphTitle: str):
    dot = Digraph(name=graphTitle, format="png")
    dot.attr(rankdir="LR", splines="true", nodesep="0.5", ranksep="0.7")
    dot.attr("node", shape="circle")
    dot.attr("edge")

    for event in events:
        dot.node(event, label=event)
    for (source, target) in sorted(edges):
        dot.edge(source, target)

    dot.render(outputFilePrefix, cleanup=True)


def DrawNfaWithGraphviz(initialStateId: str, statesById: Dict[str, AutomatonState], outputFilePrefix: str,
                        graphTitle: str):
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


# ============================================================
# NFA Construction & Operations
# ============================================================

def MergeStates(keepStateId: str, removeStateId: str, statesById: Dict[str, AutomatonState],
                initialStateId: str) -> str:
    """
    Merges removeStateId into keepStateId.
    """
    if keepStateId == removeStateId:
        return keepStateId

    if removeStateId not in statesById or keepStateId not in statesById:
        return keepStateId

    removeState = statesById[removeStateId]
    keepState = statesById[keepStateId]

    # 1. Redirect incoming edges
    for sId, state in statesById.items():
        for evt, targets in state.transitionsByEvent.items():
            if removeStateId in targets:
                targets.remove(removeStateId)
                targets.add(keepStateId)

    # 2. Move outgoing edges
    for evt, targets in removeState.transitionsByEvent.items():
        for target in targets:
            targetToUse = keepStateId if target == removeStateId else target
            keepState.AddTransition(evt, targetToUse)

    # Preserve accepting status
    if removeState.isAccepting:
        keepState.isAccepting = True

    # 3. Remove state
    del statesById[removeStateId]

    print(f"  Merged state {removeStateId} into {keepStateId}")
    return keepStateId


def BuildTrieNfa(traces: Sequence[Trace]) -> Tuple[str, Dict[str, AutomatonState]]:
    initialStateId = "q0"
    statesById: Dict[str, AutomatonState] = {initialStateId: AutomatonState(initialStateId)}
    nextStateIndex = 1

    finalStateId = "q_end"
    statesById[finalStateId] = AutomatonState(finalStateId)
    statesById[finalStateId].isAccepting = True

    for trace in traces:
        currentStateId = initialStateId
        events = trace.eventsInOrder

        for index, event in enumerate(events):
            currentState = statesById[currentStateId]

            if index == len(events) - 1:
                currentState.AddTransition(event, finalStateId)
                currentStateId = finalStateId
                continue

            existingTargetStateIds = currentState.transitionsByEvent.get(event, set())
            if existingTargetStateIds:
                nextStateId = sorted(existingTargetStateIds)[0]
            else:
                nextStateId = f"q{nextStateIndex}"
                nextStateIndex += 1
                statesById[nextStateId] = AutomatonState(nextStateId)
                currentState.AddTransition(event, nextStateId)

            currentStateId = nextStateId

    return (initialStateId, statesById)


def FindAdjacentRepeats(events: Tuple[str, ...]) -> List[Tuple[int, int, int]]:
    repeats = []
    n = len(events)
    for length in range(n // 2, 0, -1):
        for i in range(n - 2 * length + 1):
            j = i + length
            if events[i: i + length] == events[j: j + length]:
                repeats.append((i, j, length))
                return [(i, j, length)]
    return repeats


def FoldNfaLoops(initialStateId: str, statesById: Dict[str, AutomatonState], traces: Sequence[Trace]) -> None:
    print("Detecting and folding repetitive loops...")

    for trace in traces:
        events = trace.eventsInOrder
        repeats = FindAdjacentRepeats(events)

        if not repeats:
            continue

        # Reconstruct path
        currentStateId = initialStateId
        statePath = [currentStateId]

        validTrace = True
        for event in events:
            if currentStateId not in statesById:
                validTrace = False
                break
            state = statesById[currentStateId]
            targets = state.transitionsByEvent.get(event, set())
            if not targets:
                validTrace = False
                break
            nextStateId = sorted(targets)[0]
            statePath.append(nextStateId)
            currentStateId = nextStateId

        if not validTrace:
            continue

        for (i, j, length) in repeats:
            for k in range(length):
                targetIdx1 = i + 1 + k
                targetIdx2 = j + 1 + k

                s1 = statePath[targetIdx1]
                s2 = statePath[targetIdx2]

                while s1 not in statesById and s1 != s2:
                    break

                if s1 != s2 and s1 in statesById and s2 in statesById:
                    MergeStates(s1, s2, statesById, initialStateId)

                    for idx in range(len(statePath)):
                        if statePath[idx] == s2:
                            statePath[idx] = s1


def MergeTerminalPredecessors(initialStateId: str, statesById: Dict[str, AutomatonState]):
    print("Merging terminal predecessors...")
    finalStates = {sid for sid, s in statesById.items() if s.isAccepting}

    groups: Dict[Tuple[str, str], List[str]] = {}

    for sid, s in statesById.items():
        if s.isAccepting: continue
        for evt, targets in s.transitionsByEvent.items():
            for tgt in targets:
                if tgt in finalStates:
                    groups.setdefault((evt, tgt), []).append(sid)

    for (evt, finalSid), sources in groups.items():
        if len(sources) > 1:
            sources = sorted(list(set(sources)))
            survivor = sources[0]
            for victim in sources[1:]:
                if victim != survivor:
                    print(f"  Tail Merge: Merging {victim} into {survivor} (Both go --{evt}--> {finalSid})")
                    MergeStates(survivor, victim, statesById, initialStateId)


def ListAllReachableStateIds(initialStateId: str, statesById: Dict[str, AutomatonState]) -> List[str]:
    visitedStateIds: Set[str] = set()
    stack: List[str] = [initialStateId]

    while stack:
        stateId = stack.pop()
        if stateId not in statesById:
            continue
        if stateId in visitedStateIds:
            continue

        visitedStateIds.add(stateId)

        state = statesById[stateId]
        for event in state.transitionsByEvent:
            for targetStateId in state.transitionsByEvent[event]:
                if targetStateId not in visitedStateIds:
                    stack.append(targetStateId)

    return sorted(visitedStateIds)


def BuildCompactedNfaBySuffixMerging(
        initialStateId: str,
        statesById: Dict[str, AutomatonState],
) -> Tuple[str, Dict[str, AutomatonState]]:
    print("Compacting NFA (Suffix Merge)...")
    reachable = ListAllReachableStateIds(initialStateId, statesById)

    groups: Dict[str, int] = {}
    for sId in reachable:
        groups[sId] = 1 if statesById[sId].isAccepting else 0

    while True:
        newGroups: Dict[str, int] = {}
        signatures: Dict[Tuple[int, Tuple[Tuple[str, Tuple[int, ...]], ...]], int] = {}
        nextGroupId = 0

        for sId in reachable:
            state = statesById[sId]
            transList = []
            for event in sorted(state.transitionsByEvent.keys()):
                targetGroups = tuple(sorted(groups.get(t, -1) for t in state.transitionsByEvent[event]))
                transList.append((event, targetGroups))

            signature = (groups[sId], tuple(transList))

            if signature not in signatures:
                signatures[signature] = nextGroupId
                nextGroupId += 1

            newGroups[sId] = signatures[signature]

        if len(set(newGroups.values())) == len(set(groups.values())):
            groups = newGroups
            break
        groups = newGroups

    compactedStatesById: Dict[str, AutomatonState] = {}
    groupToStateId: Dict[int, str] = {}

    for sId in reachable:
        gId = groups[sId]
        if gId not in groupToStateId:
            groupToStateId[gId] = sId
            compactedStatesById[sId] = AutomatonState(sId)
            compactedStatesById[sId].isAccepting = statesById[sId].isAccepting

    for sId in reachable:
        gId = groups[sId]
        repId = groupToStateId[gId]
        repState = compactedStatesById[repId]

        oldState = statesById[sId]
        for event, targets in oldState.transitionsByEvent.items():
            for t in targets:
                targetGId = groups.get(t)
                if targetGId is not None:
                    targetRepId = groupToStateId[targetGId]
                    repState.AddTransition(event, targetRepId)

    initialRepId = groupToStateId[groups[initialStateId]]
    return (initialRepId, compactedStatesById)


def WritePrecedenceFactsToTextFile(precedenceFacts: Sequence[PrecedenceFact], outputPath: str) -> None:
    seen: Set[Tuple[str, str, str]] = set()
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for fact in precedenceFacts:
            key = fact.AsTuple()
            if key in seen:
                continue
            seen.add(key)
            outputFile.write(f"({fact.earlierEvent}, {fact.laterEvent}, {fact.traceName})\n")


def Main() -> None:
    print("Proyecto: Procesamiento de secuencias de eventos (Graphviz)")

    inputPath = input("Enter input trace file path (e.g.: traces2.txt): ").strip()
    if not inputPath:
        inputPath = "traces2.txt"

    outputPrefix = input("Enter output file prefix (e.g.: output/run1): ").strip()
    if not outputPrefix:
        outputPrefix = "output/eventSequences"

    outputDirectory = os.path.dirname(outputPrefix)
    if outputDirectory and not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory, exist_ok=True)

    traces = ParseTracesFromTextFile(inputPath)
    # Collect Alphabet in Discovery Order (FIX for P vs N)
    alphabetEvents = CollectEventAlphabet(traces)

    print(f"\nLoaded traces: {len(traces)}")
    for trace in traces:
        print(f"  {trace.traceName}: {' '.join(trace.eventsInOrder)}")

    # 1. BEF Facts
    precedenceFacts = BuildPrecedenceFacts(traces)
    precedenceFactsOutputPath = f"{outputPrefix}_BefFacts.txt"
    WritePrecedenceFactsToTextFile(precedenceFacts, precedenceFactsOutputPath)
    print(f"\nWrote BEF facts to: {precedenceFactsOutputPath}")

    # 2. Precedence Graph
    aggregatedPrecedenceEdges = BuildAggregatedPrecedenceEdges(precedenceFacts)
    precedenceGraphOutputPrefix = f"{outputPrefix}_PrecedenceGraph"
    DrawDirectedEventGraph(
        events=alphabetEvents,
        edges=aggregatedPrecedenceEdges,
        outputFilePrefix=precedenceGraphOutputPrefix,
        graphTitle="PrecedenceGraph",
    )
    print(f"Rendered precedence graph to: {precedenceGraphOutputPrefix}.png")

    # 3. Partial Order Graph
    partialOrderEdges = BuildPartialOrderEdges(precedenceFacts)
    partialOrderGraphOutputPrefix = f"{outputPrefix}_PartialOrderGraph"
    DrawDirectedEventGraph(
        events=alphabetEvents,
        edges=partialOrderEdges,
        outputFilePrefix=partialOrderGraphOutputPrefix,
        graphTitle="PartialOrderGraph",
    )
    print(f"Rendered partial order graph to: {partialOrderGraphOutputPrefix}.png")

    # 4. Partial Order Graph Unfolded
    # Pass preferredOrder (alphabetEvents) to cycle detector to prioritize earlier events (P before N)
    cycle = FindOneDirectedCycle(partialOrderEdges, preferredOrder=alphabetEvents)
    if cycle is None:
        print("\nNo directed cycle detected in PO.")
    else:
        print("\nNote: A directed cycle exists in PO.")
        print("Cycle found:", " -> ".join(cycle))

        unfoldedEdges, createdDuplicateNodeNames = UnfoldAllCyclesAutomatically(partialOrderEdges, preferredOrder=alphabetEvents)
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

    # 5. NFA Generation Pipeline
    trieNfaInitialStateId, trieNfaStatesById = BuildTrieNfa(traces)
    trieNfaOutputPrefix = f"{outputPrefix}_NfaTrie"
    DrawNfaWithGraphviz(
        initialStateId=trieNfaInitialStateId,
        statesById=trieNfaStatesById,
        outputFilePrefix=trieNfaOutputPrefix,
        graphTitle="NfaTrie",
    )
    print(f"\nRendered trie NFA to: {trieNfaOutputPrefix}.png")

    # Fold Loops
    FoldNfaLoops(trieNfaInitialStateId, trieNfaStatesById, traces)
    foldedNfaOutputPrefix = f"{outputPrefix}_NfaFolded"
    DrawNfaWithGraphviz(
        initialStateId=trieNfaInitialStateId,
        statesById=trieNfaStatesById,
        outputFilePrefix=foldedNfaOutputPrefix,
        graphTitle="NfaFolded",
    )
    print(f"Rendered folded NFA to: {foldedNfaOutputPrefix}.png")

    # Merge Terminal Predecessors
    MergeTerminalPredecessors(trieNfaInitialStateId, trieNfaStatesById)

    # Compact
    compactedInitialStateId, compactedStatesById = BuildCompactedNfaBySuffixMerging(
        trieNfaInitialStateId, trieNfaStatesById
    )
    compactedNfaOutputPrefix = f"{outputPrefix}_NfaCompacted"
    DrawNfaWithGraphviz(
        initialStateId=compactedInitialStateId,
        statesById=compactedStatesById,
        outputFilePrefix=compactedNfaOutputPrefix,
        graphTitle="NfaCompacted",
    )
    print(f"Rendered compacted NFA to: {compactedNfaOutputPrefix}.png")

    print("\nDone.")


if __name__ == "__main__":
    Main()