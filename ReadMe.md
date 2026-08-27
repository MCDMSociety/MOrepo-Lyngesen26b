# Bi-objective Traveling Salesman problems with profits (MOrepo-Lyngesen26b)


This Repo contains a set of *bi-objecive traveling salesman problems with profits* (Bi-TSPP) instances used for the paper
"A Decomposition Framework for Bi-Objective Travelling Salesman Problems with Profits".

Importantly, in all instances the root vertex is a cut-vertex of the graph.

## How to cite

To cite the paper use:

Fourthcoming arXiv citation:
```
@article{Manzke2026bitspp,
    Title = {A Decomposition Framework for Bi-Objective Travelling Salesman Problems with Profits},
    Author = {Maren Manzke, Mark Lyngesen},
}
```

To cite this repository use 

```
@Electronic{MOrepo-Lyngesen26b,
  Title = {Bi-objective Traveling Salesman problems with profits (MOrepo-Lyngesen26b)},
  Author = {Lyngesen, M},
  Url = {https://github.com/MCDMSociety/MOrepo-Lyngesen26b},
  Year = {2026},
  Note = {Instance and result files at MOrepo.}
}
```

To cite the Multi-Objective Optimization Repository use

```
@Electronic{MOrepo,
  Title = {Multi-Objective Optimization Repository (MOrepo)},
  Author = {L. R. Nielsen},
  Url = {https://github.com/MCDMSociety/MOrepo},
  Year = {2017},
}
```

Always remember to cite the research paper and not the repository if you don't want to include all
citations.


## Test instances

Two testsets of instances are saved in `instances/presets/`

- `small-test` used for testing codebase.

| Parameter         | Value                        |
|:------------------|:-----------------------------|
| name              | small-test                   |
| N_VALUES          | [10, 14, 18]                 |
| S_VALUES          | [2, 4]                       |
| WEIGHT_TYPES      | ['random']                   |
| P_VALUES          | [0, 0.33, 0.66]              |
| SEED_VALUES       | [0]                          |
| root_in_subgraphs | True                         |
| timestamp         | 2026-08-13 21:39:26          |
| path              | instances/presets/small-test |

- `testbed` used for the computational study in the accompanying paper.


| Parameter         | Value                                                                                                                                                                                     |
|:------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name              | testbed                                                                                                                                                                                   |
| N_VALUES          | [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100] |
| S_VALUES          | [2, 4, 6, 8]                                                                                                                                                                              |
| WEIGHT_TYPES      | ['nondecreasing', 'nonincreasing', 'random']                                                                                                                                              |
| P_VALUES          | [0, 0.33, 0.66]                                                                                                                                                                           |
| SEED_VALUES       | [0, 1, 2, 3, 4]                                                                                                                                                                           |
| root_in_subgraphs | True                                                                                                                                                                                      |
| timestamp         | 2026-08-13 21:50:50                                                                                                                                                                       |
| path              | instances/presets/testbed                                                                                                                                                                 |


# Instance generation

Instances a generated based on the parameters: `n` (number of nodes), `S` (number of subproblems), `Profit correlation` (weight generation method), `SEED` and `gamma` (graph density paramater).

For each combination of generation parameters a Graph with a cut-vertex is generated using the method `generate_graph(n, S, gamma) -> G=(V,E,r)`, and weights are added using the `generate_weights(G, Profit correlation)` method. Both methods are stated below, and the full implementation details can be found in the repository [lyngesen/bi-TSPP](https://github.com/lyngesen/bi-TSPP).

```python
# function to create the bi-TSPP testbank
#     source: src.bitspp.utils.generators.py  (see Repo with implementation)

def generate_graph(n, S, type: float = 0.5):

    G = nx.Graph()
    G.add_node(0) # always add depot=0
    ...
    nodes = list(range(1, n))
    random.shuffle(nodes)

    # root children (exactly S)
    root_children = nodes[:S]
    rest = nodes[S:]

    if isinstance(type, (int, float)):
        parts = [[] for _ in range(S)]
        # create balanced distribution of remaining nodes to S subtrees
        for i, v in enumerate(rest):
            parts[i % S].append(v)
        for c, part in zip(root_children, parts):
            nodes_in_subtree = [c] + part

            # Step 1: create a random spanning tree
            nodes = nodes_in_subtree[:]
            random.shuffle(nodes)

            for i in range(1, len(nodes)):
                # connect each node to a random earlier node → ensures connectivity
                u = nodes[i]
                v = random.choice(nodes[:i])
                G.add_edge(u, v)

            # step 2: connect the root to the subtree
            _random_node = random.choice(nodes_in_subtree)
            G.add_edge(0, _random_node)

            # Step 3: add edges depending on gamma parameter
            gamma = type
            for u, v in combinations(nodes_in_subtree + [0], 2):
                if not G.has_edge(u, v) and random.random() < gamma:
                    G.add_edge(u, v)
    return G



def generate_weights(G: nx.Graph, type: str = "nondecreasing", M: int = 100, sigma=10):

    ...
    # Step 1: assign edge weights
    for u, v in G.edges():
        G[u][v]["w"] = random.randint(1, M)

    # Step 2: build a rooted tree structure
    T = nx.bfs_tree(G, root)
    parent = {root: None}
    for u, v in T.edges():
        parent[v] = u

    # Step 3: assign profits

    G.nodes[root]["p"] = 0

    if type in {"nondecreasing", "nonincreasing"}:
        # set a base ratio per branch
        depth = nx.single_source_shortest_path_length(T, root)
        max_depth = max(depth.values()) if len(T) > 1 else 1

        for v in nx.topological_sort(T):
            if v == root:
                continue
            u = parent[v]
            w_v = G[u][v]["w"]
            d = depth[v]
            if type == "nondecreasing":
                # mean increases linearly with depth: 0 → 100
                mu_v = 100.0 * d / max_depth

            elif type == "nonincreasing":
                # mean decreases linearly with depth: 100 → 0
                mu_v = 100.0 * (1 - d / max_depth)

            # truncated normal draw
            p_raw = random.gauss(mu_v, sigma)
            p_clamped = max(1, min(100, p_raw))

            G.nodes[v]["p"] = int(math.floor(p_clamped))
    # Random - assign profits at random, does not depend on costs 'w' 
    if type == "random":
        for v in G.nodes():
            G.nodes[v]["p"] = random.randint(1, M) if v != root else 0
```



# bi-TSPP instances format

File location `instances/presets/<preset-name>/`


Instances follow the naming pattern: `n-<N>_S-<S>_type-float-<gamma>_weights-<profit-correlation>_seed-<seed>.json`.

In the following is an example of a bi-TSPP instance  `n-10_S-2_type-float-0_weights-random_seed-0.json` from the `small-test`preset saved in `json` format.

```{python}
{"data":
    {"S":2,
    "n":10,
    "name":"n-10_S-2_type-float-0_weights-random_seed-0",
    "seed":0,
    "type":0,
    "weight_type":"random"},
    "edge_cost": [
        {"cost":23,"u":0,"v":6},
        {"cost":58,"u":0,"v":8},
        {"cost":3,"u":1,"v":5},
        {"cost":33,"u":2,"v":4},
        {"cost":46,"u":3,"v":7},
        {"cost":12,"u":3,"v":9},
        {"cost":72,"u":4,"v":5},
        {"cost":70,"u":4,"v":8},
        {"cost":20,"u":6,"v":9}
    ],
    "edges":[[0,6],[0,8],[1,5],[2,4],[3,7],[3,9],[4,5],[4,8],[6,9]],
    "nodes":[0,1,2,3,4,5,6,7,8,9],
    "profits":{"0":0,"1":43,"2":84,"3":65,"4":79,"5":73,"6":2,"7":56,"8":35,"9":88},
    "subgraphs":
        {
        "0":[0],
        "1":[1,2,4,5,8],
        "2":[3,6,7,9]
        }
    }

```

## Results

Results are presented in the `docs` folder based on the `csv` files in `results/data/`

A Marimo notebook file `results/docs/dataview.html` ([link](results/docs/dataview.html)) is provided which shows the numbers, tables and plots used in the paper, along with some robustness checks.

