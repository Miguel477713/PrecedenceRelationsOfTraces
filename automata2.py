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

# python
def CompleteAutomatonWithPrecedenceLoops(
    initialStateId: str,
    statesById: Dict[str, AutomatonState],
    precedenceFacts: Sequence[PrecedenceFact]
) -> None:
    """
    Systematically completes the automaton by analyzing precedence relations
    and adding necessary transitions to ensure all required loops are present.
    This is a general solution that works for any detected loop patterns.
    """
    # Build event precedence graph from precedence facts
    eventPrecedence: Dict[str, Set[str]] = {}
    for fact in precedenceFacts:
        if fact.earlierEvent == StartSentinelEvent or fact.laterEvent == EndSentinelEvent:
            continue
        eventPrecedence.setdefault(fact.earlierEvent, set()).add(fact.laterEvent)
    
    # Build state-event reachability map
    stateEventReachability: Dict[str, Set[str]] = {}
    for stateId, state in statesById.items():
        reachableEvents = set()
        for event in state.transitionsByEvent:
            reachableEvents.add(event)
        stateEventReachability[stateId] = reachableEvents
    
    # Detect missing bidirectional transitions (loops)
    transitionsAdded = 0
    
    # For each state, check if it should have loop transitions
    for sourceStateId, sourceState in statesById.items():
        for event in sourceState.transitionsByEvent:
            targetStates = sourceState.transitionsByEvent[event]
            
            for targetStateId in targetStates:
                # Check if there should be a reverse transition
                # Based on precedence relations, if A precedes B and B precedes A in different contexts,
                # there should be a loop
                shouldHaveReverse = False
                
                # Check if the target state can reach back to source with the same event
                if targetStateId in statesById:
                    targetState = statesById[targetStateId]
                    currentReverseTargets = targetState.transitionsByEvent.get(event, set())
                    
                    if sourceStateId not in currentReverseTargets:
                        # Determine if reverse transition should exist based on precedence analysis
                        shouldHaveReverse = ShouldHaveReverseTransition(
                            sourceStateId, targetStateId, event, 
                            statesById, eventPrecedence, precedenceFacts
                        )
                        
                        if shouldHaveReverse:
                            targetState.AddTransition(event, sourceStateId)
                            transitionsAdded += 1
                            print(f"Added loop transition: {targetStateId} --{event}--> {sourceStateId}")
    
    # Also check for missing transitions based on event precedence closure
    transitionsAdded += AddMissingPrecedenceTransitions(
        initialStateId, statesById, eventPrecedence, precedenceFacts
    )

    # NEW: ensure self-loops for events that appear consecutively in traces (e.g., (A,A,TRx))
    selfLoopEvents: Set[str] = set()
    for fact in precedenceFacts:
        if fact.earlierEvent == fact.laterEvent and fact.earlierEvent != StartSentinelEvent and fact.laterEvent != EndSentinelEvent:
            selfLoopEvents.add(fact.earlierEvent)

    for stateId, state in statesById.items():
        for event in list(state.transitionsByEvent.keys()):
            if event in selfLoopEvents:
                if stateId not in state.transitionsByEvent.get(event, set()):
                    state.AddTransition(event, stateId)
                    transitionsAdded += 1
                    print(f"Added self-loop: {stateId} --{event}--> {stateId}")

    if transitionsAdded > 0:
        print(f"Completed automaton with {transitionsAdded} necessary transitions")
    else:
        print("Automaton is already complete with respect to precedence relations")


def ShouldHaveReverseTransition(
    sourceStateId: str,
    targetStateId: str,
    event: str,
    statesById: Dict[str, AutomatonState],
    eventPrecedence: Dict[str, Set[str]],
    precedenceFacts: Sequence[PrecedenceFact]
) -> bool:
    """
    Determines if a reverse transition should exist based on formal criteria.
    """
    # Check if there are precedence facts that suggest bidirectional relationship
    sourceEvents = set()
    targetEvents = set()
    
    # Collect all events that can be reached from each state
    if sourceStateId in statesById:
        for evt in statesById[sourceStateId].transitionsByEvent:
            sourceEvents.add(evt)
    
    if targetStateId in statesById:
        for evt in statesById[targetStateId].transitionsByEvent:
            targetEvents.add(evt)
    
    # Check if the event appears in precedence relations that suggest a loop
    # Look for cases where the same event appears in different precedence contexts
    eventInPrecedence = False
    reverseEventInPrecedence = False
    
    for fact in precedenceFacts:
        if fact.earlierEvent == event or fact.laterEvent == event:
            eventInPrecedence = True
        if fact.earlierEvent == event and fact.laterEvent in sourceEvents:
            reverseEventInPrecedence = True
        if fact.earlierEvent in targetEvents and fact.laterEvent == event:
            reverseEventInPrecedence = True
    
    # If the event participates in precedence relations and there's evidence
    # of bidirectional behavior, add the reverse transition
    return eventInPrecedence and reverseEventInPrecedence


def AddMissingPrecedenceTransitions(
    initialStateId: str,
    statesById: Dict[str, AutomatonState],
    eventPrecedence: Dict[str, Set[str]],
    precedenceFacts: Sequence[PrecedenceFact]
) -> int:
    """
    Adds missing transitions based on precedence closure analysis.
    """
    transitionsAdded = 0
    
    # Build a map of which events should be available at each state
    # based on the precedence relations observed in traces
    stateRequiredEvents: Dict[str, Set[str]] = {}
    
    # Analyze precedence facts to determine required events at each state
    for fact in precedenceFacts:
        if fact.earlierEvent == StartSentinelEvent or fact.laterEvent == EndSentinelEvent:
            continue
        
        # Find states that should have transitions for these events
        for stateId, state in statesById.items():
            if stateId == initialStateId:
                continue  # Skip initial state as it's handled separately
            
            # Check if this state should have the laterEvent based on precedence
            if fact.earlierEvent in state.transitionsByEvent:
                stateRequiredEvents.setdefault(stateId, set()).add(fact.laterEvent)
    
    # Add missing required transitions
    for stateId, requiredEvents in stateRequiredEvents.items():
        if stateId in statesById:
            state = statesById[stateId]
            for event in requiredEvents:
                if event not in state.transitionsByEvent:
                    # Find appropriate target state for this transition
                    targetState = FindTargetStateForEvent(event, statesById)
                    if targetState:
                        state.AddTransition(event, targetState)
                        transitionsAdded += 1
                        print(f"Added precedence-based transition: {stateId} --{event}--> {targetState}")
    
    return transitionsAdded


def FindTargetStateForEvent(event: str, statesById: Dict[str, AutomatonState]) -> Optional[str]:
    """
    Finds an appropriate target state for a given event based on existing patterns.
    """
    # Look for existing states that have this event as incoming
    for stateId, state in statesById.items():
        for evt, targets in state.transitionsByEvent.items():
            if evt == event and targets:
                return sorted(targets)[0]  # Return first existing target
    
    # If no existing pattern found, create a new state
    maxStateNum = 0
    for stateId in statesById.keys():
        if stateId.startswith('q'):
            try:
                num = int(stateId[1:])
                maxStateNum = max(maxStateNum, num)
            except ValueError:
                continue
    
    newStateId = f"q{maxStateNum + 1}"
    statesById[newStateId] = AutomatonState(newStateId)
    return newStateId


def BuildTrieNfa(traces: Sequence[Trace]) -> Tuple[str, Dict[str, AutomatonState]]:
    """
    Returns:
      initialStateId,
      statesById dictionary

    Builds a trie NFA whose traces share prefixes.
    Changes:
    - Use a single shared accepting end state ("q_end").
    - Route every trace's last-event transition to that shared end state.
    - Do NOT create separate terminal states per trace.
    - Reuse existing target states for transitions (including from the initial state)
      so loops and merges are represented instead of artificially duplicating states.
    """
    initialStateId = "q0"
    statesById: Dict[str, AutomatonState] = {initialStateId: AutomatonState(initialStateId)}
    nextStateIndex = 1

    # single shared accepting end state
    finalStateId = "q_end"
    statesById[finalStateId] = AutomatonState(finalStateId)
    statesById[finalStateId].isAccepting = True

    for trace in traces:
        currentStateId = initialStateId
        events = trace.eventsInOrder

        for index, event in enumerate(events):
            currentState = statesById[currentStateId]

            # If this is the last event of the trace, route it to the shared final state.
            if index == len(events) - 1:
                currentState.AddTransition(event, finalStateId)
                currentStateId = finalStateId
                # do not create a new per-trace terminal state
                continue

            # Unified behavior: reuse an existing target for the event if present,
            # otherwise create a new state (applies to initial state as well).
            existingTargetStateIds = currentState.transitionsByEvent.get(event, set())
            if existingTargetStateIds:
                nextStateId = sorted(existingTargetStateIds)[0]
            else:
                nextStateId = f"q{nextStateIndex}"
                nextStateIndex += 1
                statesById[nextStateId] = AutomatonState(nextStateId)
                currentState.AddTransition(event, nextStateId)

            currentStateId = nextStateId

        # no per-trace accepting flag here; the shared final state is accepting

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
    """
    Writes precedence facts to a text file, preserving the first-seen order
    but omitting duplicate (earlierEvent, laterEvent, traceName) tuples.
    """
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

    # Complete the automaton with systematic loop detection and completion
    print("\nCompleting automaton with precedence-based loop detection...")
    CompleteAutomatonWithPrecedenceLoops(trieNfaInitialStateId, trieNfaStatesById, precedenceFacts)
    
    # Redraw the NFA with the completed transitions
    trieNfaCompletedOutputPrefix = f"{outputPrefix}_NfaTrieCompleted"
    DrawNfaWithGraphviz(
        initialStateId=trieNfaInitialStateId,
        statesById=trieNfaStatesById,
        outputFilePrefix=trieNfaCompletedOutputPrefix,
        graphTitle="NfaTrieCompleted",
    )
    print(f"Rendered completed trie NFA to: {trieNfaCompletedOutputPrefix}.png")

    print("Note: Compact NFA merging is not included in this fixed version (ID-based transitions).")

    print("\nDone.")


if __name__ == "__main__":
    Main()
