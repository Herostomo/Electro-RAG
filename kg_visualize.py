"""
Visualizer for the 'Knowledge graph' block.

kg_builder.py already builds and persists the real graph (GraphML,
loaded via load_knowledge_graph()) -- nodes are strings pulled straight
out of datasheet text (part numbers, pins, specs, standards), edges
carry a `relation` (e.g. "max_voltage", "supports") and a `source`
(which datasheet chunk it came from).

This module doesn't build a graph -- it just renders the one that
already exists as an interactive Plotly figure, so you get a picture
instead of only the GraphML file / the text dump from
query_knowledge_graph().
"""
import networkx as nx
import plotly.graph_objects as go

from kg_builder import load_knowledge_graph

# kg_builder's graph carries no node "type" -- nodes are just raw strings
# extracted by the LLM. These heuristics are ONLY for coloring the plot;
# they don't touch the graph's actual nodes/edges/relations.
_UNITS = ("v", "a", "hz", "w", "ohm", "\u00b0c", "ma", "mv", "khz", "mhz", "ns", "us", "ms")
_STANDARDS = {"i2c", "spi", "uart", "usb", "gpio", "pwm", "can", "rs232", "rs485", "usb-c", "jtag"}


def _guess_type(node: str) -> str:
    n = node.strip().lower()
    if n in _STANDARDS:
        return "standard"
    if "pin" in n or (n.startswith("p") and n[1:].isdigit()):
        return "pin"
    if any(c.isdigit() for c in n) and any(u in n for u in _UNITS):
        return "spec"
    return "component"


_TYPE_STYLE = {
    "component": {"color": "#10b981", "size": 26},
    "spec":      {"color": "#f59e0b", "size": 18},
    "standard":  {"color": "#38bdf8", "size": 18},
    "pin":       {"color": "#ec4899", "size": 16},
}


def plot_knowledge_graph(
    graph: "nx.MultiDiGraph | None" = None,
    max_nodes: int = 150,
    out_html: "str | None" = "kg_view.html",
    show: bool = False,
):
    """
    Renders the knowledge graph as an interactive Plotly figure.

    graph:      pass an in-memory nx.MultiDiGraph (e.g. straight out of
                build_knowledge_graph()); if omitted, loads it from disk
                via kg_builder.load_knowledge_graph().
    max_nodes:  a full datasheet set can produce a graph too dense to
                read. Above this size, only the highest-degree nodes
                (the ones with the most facts attached) are kept.
    out_html:   if set, saves a standalone interactive HTML file you can
                open in a browser or embed -- useful since your
                customtkinter GUI process has no plot window to pop up
                a fig.show() into.
    show:       call fig.show() as well (opens a browser tab). Off by
                default so this is safe to call from a headless
                pipeline run.

    Returns the Plotly Figure, or None if the graph is empty.
    """
    if graph is None:
        graph = load_knowledge_graph()

    if graph.number_of_nodes() == 0:
        print("Knowledge graph is empty -- nothing to plot. Run ingest_datasheets() first.")
        return None

    if graph.number_of_nodes() > max_nodes:
        top_nodes = [n for n, _ in sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]]
        graph = graph.subgraph(top_nodes).copy()
        print(f"Graph has more than {max_nodes} nodes -- showing the {max_nodes} most-connected only.")

    pos = nx.spring_layout(graph, k=0.6, iterations=50, seed=42)

    # Draw edges as plain lines, plus an invisible marker at each edge's
    # midpoint so hovering shows *which* relation that edge is (a
    # MultiDiGraph can have several parallel edges between two nodes,
    # e.g. both "max_voltage" and "min_voltage" between the same pair).
    edge_x, edge_y = [], []
    mid_x, mid_y, mid_text = [], [], []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        rel = data.get("relation", "related_to")
        src = data.get("source")
        mid_text.append(f"{u} --{rel}--> {v}" + (f"<br>source: {src}" if src else ""))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#888"), hoverinfo="none",
    )
    edge_hover_trace = go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        hoverinfo="text", hovertext=mid_text,
    )

    node_x, node_y, node_text, node_hover, node_color, node_size = [], [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_hover.append(f"{node}<br>{graph.degree(node)} connection(s)")
        style = _TYPE_STYLE[_guess_type(str(node))]
        node_color.append(style["color"])
        node_size.append(style["size"])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="bottom center",
        hoverinfo="text", hovertext=node_hover,
        marker=dict(color=node_color, size=node_size, line=dict(width=2, color="white")),
    )

    fig = go.Figure(
        data=[edge_trace, edge_hover_trace, node_trace],
        layout=go.Layout(
            title=f"Knowledge Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)",
            showlegend=False, hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if out_html:
        fig.write_html(out_html)
        print(f"Saved interactive graph view to {out_html}")
    if show:
        fig.show()

    return fig


if __name__ == "__main__":
    plot_knowledge_graph(show=True)
