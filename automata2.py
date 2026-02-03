import os
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from graphviz import Digraph

# ============================================================
# Configuration & Constants
# ============================================================

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


@dataclass
class AutomatonState:
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
# 1. Parsing
# ============================================================

def ParseTracesFromTextFile(path: str) -> List[Trace]:
    """
    Parses input file as per PDF requirements (supports NTR, TR tags).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")

    with open(path, "r", encoding="utf-8") as inputFile:
        lines = inputFile.read().splitlines()

    traces: List[Trace] = []
    autoIdx = 1

    for line in lines:
        line = line.split("#")[0].strip()  # remove comments
        if not line:
            continue

        if re.match(r"^\s*NTR\s*:", line, flags=re.IGNORECASE):
            continue

        match = re.match(r"^\s*(TR\s*\d+)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            name = match.group(1).replace(" ", "").upper()
            content = match.group(2)
        else:
            name = f"TR{autoIdx}"
            autoIdx += 1
            content = line

        events = tuple(token.strip() for token in content.split() if token.strip())
        if events:
            traces.append(Trace(name, events))

    if not traces:
        raise ValueError("No traces found in file.")
    return traces


def CollectEventAlphabet(traces: Sequence[Trace]) -> List[str]:
    events = set()
    for t in traces:
        events.update(t.eventsInOrder)
    return sorted(events)


# ============================================================
# 2. BEF (Precedence) & PO (Partial Order) Logic
# ============================================================

def BuildPrecedenceFacts(traces: Sequence[Trace]) -> List[PrecedenceFact]:
    facts = []
    for t in traces:
        evs = t.eventsInOrder
        facts.append(PrecedenceFact(StartSentinelEvent, evs[0], t.traceName))
        for i in range(len(evs) - 1):
            facts.append(PrecedenceFact(evs[i], evs[i + 1], t.traceName))
        facts.append(PrecedenceFact(evs[-1], EndSentinelEvent, t.traceName))
    return facts


def BuildAggregatedEdges(facts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    edges = set()
    for f in facts:
        if f.earlierEvent != StartSentinelEvent and f.laterEvent != EndSentinelEvent:
            edges.add((f.earlierEvent, f.laterEvent))
    return edges


def ComputeConcurrentPairs(facts: Sequence[PrecedenceFact]) -> FrozenSet[FrozenSet[str]]:
    edge_traces: Dict[Tuple[str, str], Set[str]] = {}
    for f in facts:
        if f.earlierEvent == StartSentinelEvent or f.laterEvent == EndSentinelEvent:
            continue
        edge_traces.setdefault((f.earlierEvent, f.laterEvent), set()).add(f.traceName)

    concurrent = set()
    for (u, v), traces_uv in edge_traces.items():
        traces_vu = edge_traces.get((v, u), set())
        if traces_uv and traces_vu:
            if len(traces_uv.union(traces_vu)) >= 2:
                concurrent.add(frozenset({u, v}))
    return frozenset(concurrent)


def BuildPartialOrderEdges(facts: Sequence[PrecedenceFact]) -> Set[Tuple[str, str]]:
    all_edges = BuildAggregatedEdges(facts)
    conc_pairs = ComputeConcurrentPairs(facts)

    po_edges = set()
    for u, v in all_edges:
        if frozenset({u, v}) not in conc_pairs:
            po_edges.add((u, v))
    return po_edges


# ============================================================
# 3. Cycle Detection & Unfolding
# ============================================================

def FindOneCycle(edges: Set[Tuple[str, str]]) -> Optional[List[str]]:
    adj = {}
    nodes = set()
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        nodes.add(u);
        nodes.add(v)

    state = {n: 0 for n in nodes}
    parent = {n: None for n in nodes}

    for root in sorted(nodes):
        if state[root] != 0: continue
        stack = [(root, iter(sorted(adj.get(root, []))))]
        state[root] = 1

        while stack:
            u, children = stack[-1]
            try:
                v = next(children)
                if state.get(v, 0) == 1:  # Cycle detected
                    path = [v, u]
                    curr = u
                    while curr != v and parent[curr] is not None:
                        curr = parent[curr]
                        path.append(curr)
                    return path[::-1]
                elif state.get(v, 0) == 0:
                    state[v] = 1
                    parent[v] = u
                    stack.append((v, iter(sorted(adj.get(v, [])))))
            except StopIteration:
                stack.pop()
                state[u] = 2
    return None


def UnfoldCycles(edges: Set[Tuple[str, str]]) -> Tuple[Set[Tuple[str, str]], List[str]]:
    current_edges = set(edges)
    created_nodes = []

    for _ in range(100):
        cycle = FindOneCycle(current_edges)
        if not cycle:
            break

        pivot = cycle[0]
        source_of_back_edge = cycle[-2]

        idx = 1
        while True:
            new_name = f"{pivot}'{idx}"
            all_nodes = set(u for e in current_edges for u in e)
            if new_name not in all_nodes:
                break
            idx += 1

        if (source_of_back_edge, pivot) in current_edges:
            current_edges.remove((source_of_back_edge, pivot))
            current_edges.add((source_of_back_edge, new_name))
            created_nodes.append(new_name)
        else:
            break

    return current_edges, created_nodes


# ============================================================
# 4. Automata Construction (Trie & Compact)
# ============================================================

def BuildTrieNfa(traces: Sequence[Trace]) -> Tuple[str, Dict[str, AutomatonState]]:
    start_id = "q0"
    states = {start_id: AutomatonState(start_id)}
    next_idx = 1

    for t in traces:
        curr = start_id
        for ev in t.eventsInOrder:
            st = states[curr]
            if ev in st.transitionsByEvent:
                curr = list(st.transitionsByEvent[ev])[0]
            else:
                new_id = f"q{next_idx}"
                next_idx += 1
                states[new_id] = AutomatonState(new_id)
                st.AddTransition(ev, new_id)
                curr = new_id
        states[curr].isAccepting = True

    return start_id, states


def MinimizeNfa(start_id: str, states: Dict[str, AutomatonState]) -> Tuple[str, Dict[str, AutomatonState]]:
    """
    Groups equivalent states (same behavior/suffix) to create the Compact NFA.
    """
    memo_sig: Dict[str, Tuple] = {}

    def get_signature(uid: str) -> Tuple:
        if uid in memo_sig: return memo_sig[uid]
        st = states[uid]
        trans_sig = []
        for ev in sorted(st.transitionsByEvent.keys()):
            targets = st.transitionsByEvent[ev]
            target_sigs = tuple(sorted(get_signature(tid) for tid in targets))
            trans_sig.append((ev, target_sigs))
        sig = (st.isAccepting, tuple(trans_sig))
        memo_sig[uid] = sig
        return sig

    get_signature(start_id)

    sig_to_ids: Dict[Tuple, List[str]] = {}
    for uid in states:
        if uid in memo_sig:
            sig = memo_sig[uid]
            sig_to_ids.setdefault(sig, []).append(uid)

    old_to_new: Dict[str, str] = {}
    new_states: Dict[str, AutomatonState] = {}
    new_start_id = ""

    for idx, (sig, group) in enumerate(sig_to_ids.items()):
        new_id = f"S{idx}"
        if start_id in group:
            new_start_id = new_id

        new_st = AutomatonState(new_id)
        new_st.isAccepting = sig[0]
        new_states[new_id] = new_st

        for old_id in group:
            old_to_new[old_id] = new_id

    for sig, group in sig_to_ids.items():
        rep_id = group[0]
        new_src_id = old_to_new[rep_id]
        rep_st = states[rep_id]

        for ev, targets in rep_st.transitionsByEvent.items():
            for t in targets:
                new_tgt_id = old_to_new[t]
                new_states[new_src_id].AddTransition(ev, new_tgt_id)

    return new_start_id, new_states


# ============================================================
# 5. Graphviz Visualization
# ============================================================

def DrawGraph(events: List[str], edges: Set[Tuple[str, str]], name: str, output_prefix: str):
    dot = Digraph(name=name, format="png")
    dot.attr(rankdir="LR")
    for e in events:
        dot.node(e, shape="circle")
    for u, v in sorted(edges):
        dot.edge(u, v)
    try:
        print(f"Rendering graph '{name}' with {len(events)} events and {len(edges)} edges to {output_prefix}.png")
        dot.render(output_prefix, cleanup=True)
        print(f"Generated: {output_prefix}.png")
    except Exception as e:
        print(f"Graphviz error while rendering '{name}': {e}. (Ensure Graphviz is installed and in PATH)")


def DrawAutomaton(start_id: str, states: Dict[str, AutomatonState], name: str, output_prefix: str):
    dot = Digraph(name=name, format="png")
    dot.attr(rankdir="LR")
    dot.node("__start__", shape="none", label="")
    dot.edge("__start__", start_id)

    for uid, st in states.items():
        shape = "doublecircle" if st.isAccepting else "circle"
        dot.node(uid, shape=shape, label=uid)
        for ev, targets in st.transitionsByEvent.items():
            for t in targets:
                dot.edge(uid, t, label=ev)

    try:
        total_states = len(states)
        total_transitions = sum(len(tgts) for st in states.values() for tgts in st.transitionsByEvent.values())
        print(f"Rendering automaton '{name}' with {total_states} states and {total_transitions} transitions to {output_prefix}.png")
        dot.render(output_prefix, cleanup=True)
        print(f"Generated: {output_prefix}.png")
    except Exception as e:
        print(f"Graphviz error while rendering automaton '{name}': {e}")


# ============================================================
# Main
# ============================================================

def Main():
    import sys

    input_file = "traces.txt"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {input_file}...")
    try:
        traces = ParseTracesFromTextFile(input_file)
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    print(f"Parsed {len(traces)} traces:")
    for t in traces:
        print(f"  {t.traceName}: {t.eventsInOrder}")

    alphabet = CollectEventAlphabet(traces)
    print(f"Alphabet: {alphabet}")

    # 1. BEF
    bef_facts = BuildPrecedenceFacts(traces)
    bef_edges = BuildAggregatedEdges(bef_facts)
    DrawGraph(alphabet, bef_edges, "PrecedenceGraph", f"{output_dir}/01_PrecedenceGraph")

    # 2. PO
    po_edges = BuildPartialOrderEdges(bef_facts)
    DrawGraph(alphabet, po_edges, "PartialOrder", f"{output_dir}/02_PartialOrder")

    # 3. Cycle Unfolding
    cycle = FindOneCycle(po_edges)
    final_edges = po_edges
    final_alphabet = list(alphabet)

    if cycle:
        print(f"Cycle detected: {cycle}. Unfolding...")
        unfolded_edges, created_nodes = UnfoldCycles(po_edges)
        final_edges = unfolded_edges
        final_alphabet.extend(created_nodes)
        DrawGraph(final_alphabet, final_edges, "UnfoldedPO", f"{output_dir}/03_UnfoldedPO")
    else:
        print("No cycles detected.")

    # 4. Automata
    trie_start, trie_states = BuildTrieNfa(traces)
    DrawAutomaton(trie_start, trie_states, "TrieNFA", f"{output_dir}/04_TrieNFA")

    print("Building Compact NFA (Suffix Merging)...")
    compact_start, compact_states = MinimizeNfa(trie_start, trie_states)
    DrawAutomaton(compact_start, compact_states, "CompactNFA", f"{output_dir}/05_CompactNFA")

    print("\nDone! Check the 'output' folder for PNG images.")


if __name__ == "__main__":
    Main()