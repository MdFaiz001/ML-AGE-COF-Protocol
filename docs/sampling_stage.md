


# make_cof_sampling_fixed_final.py  
## COF Design Sampler and Plan Generator

---

## What This Script Does

This script generates a large sampling plan of COF designs and writes it to a single CSV file:

**Output:**  
```

COF_generation_plan.csv

```

Each row in the CSV corresponds to one planned COF design (a blueprint), including:

- Topology name  
- Node type(s) and node linker(s) chosen consistently with topology coordination number (CN)  
- 2-connected edge linker (parent) and optional functionalized variant  
- Functionalization coverage level (%)  
- Parsed metadata (bridge type, base id, functionalization sites)  
- Deterministic output directory name for downstream COF builders  

This script **does not build COFs**.  
It prepares the COF generation plan used by downstream scripts to construct CIFs and run Zeo++, LAMMPS, etc.

---

## Inputs Required (CSV Files)

The script must be executed in a directory containing the following CSV files:

### 1. `topo_sorted.csv`

Must contain at least these columns:

- `Topology`
- `# Node types`
- `Node info`

Example CN pattern format:
```

type 0 (CN=3, ...)
type 1 (CN=4, ...)

```

This file defines the topological design space and provides CN ordering for mixed-node topologies.

---

### 2. `3c_linkers.csv`

Must contain:
```

name

```

All entries are treated as available 3-connected building units.

---

### 3. `4c_linkers.csv`

Must contain:
```

name

```

All entries are treated as available 4-connected building units.

---

### 4. `2c_linkers.csv`

Must contain:
```

name

```

All entries are treated as available unfunctionalized 2-connected edge linkers.

---

### 5. `functionalized_2c_linkers.csv`

Must contain:
```

name

```

Each entry encodes:

- Bridge type  
- Parent 2C name  
- Base identity  
- Functionalization sites (optional)

---

## Output Produced

### `COF_generation_plan.csv`

One row per sampled COF design.

### Identity and Topology Columns

- `cof_id` — unique ID (COF_000001, COF_000002, …)
- `topology_name`
- `case_id`
- `num_node_types`

---

### Node/Linker Assignment (CN-Consistent)

- `node1_type`, `node1_linker`
- `node2_type`, `node2_linker` (empty for single-node topologies)

Node types are strings like `"3C"` or `"4C"`.

---

### Edge Functionalization Plan

- `parent_2c`
- `coverage_pct` (0, 25, 50, 75, 100)
- `edge_fn_name`
- `bridge_type`
- `base_id`
- `fn_sites`

---

### Output Path Routing

- `output_dir`

Format:
```

<topology_name>__<cof_id>

```

Example:
```

dia__COF_000257

```

---

## Core Logic

### 1. Reproducible Sampling

Two seeds are set:

```

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

```

This guarantees identical output for identical inputs.

---

### 2. Topology Classification into Five Cases

CN information is parsed using regex and stored as:

```

{0: 3, 1: 4}

```

Topologies are classified into:

| Case | Meaning | CN Pattern |
|------|----------|------------|
| 1 | Single-node | 3 |
| 2 | Single-node | 4 |
| 3 | Two-node | 3–3 |
| 4 | Two-node | 4–4 |
| 5 | Two-node | 3–4 |

Invalid CN patterns are discarded.

---

### 3. Balanced Sampling Per Topology

Sampling quotas are defined using `CASE_TARGETS`.

Distribution method:

- Base quota = `target_total // n_topologies_in_case`
- Remainder distributed randomly

Console output example:
```

[info] Total planned COFs = ...

```

---

### 4. CN-Consistent Node Assignment

Case-specific logic:

**Case 1 (1 node, CN=3)**  
- node1_type = "3C"
- linker from 3C list

**Case 2 (1 node, CN=4)**  
- node1_type = "4C"
- linker from 4C list

**Case 3 (3–3)**  
- Both nodes assigned 3C linkers

**Case 4 (4–4)**  
- Both nodes assigned 4C linkers

**Case 5 (3–4 mixed)**  
- CN ordering taken from topology
- Linker selected according to each CN
- Prevents swapped node/linker assignment errors

---

### 5. Edge Functionalization Sampling

1. Select parent 2C linker  
2. Select coverage from:

```

[0, 25, 50, 75, 100]

```

If coverage = 0:
- No functionalization

If coverage > 0:
- Find matching functionalized variants
- Randomly select variant
- Parse:
  - bridge_type
  - base_id
  - fn_sites

If no variant exists:
- Coverage downgraded to 0 (safe fallback)

---

## Functionalized Linker Naming Convention

Expected format:

```

bridge_parent__base_id__site_labels

```

Examples:

```

dir_1_c_link__2_NMe2__site1
ch2_5_n_link__11_pyrazine__site2
ph_9_n_link__17__bsite1__lsite3

```

Parser extracts:

- Bridge type
- Parent 2C
- Base identity
- Functionalization site labels

---

## Expected Folder Structure

```

project_root/
│
├─ make_cof_sampling_fixed_final.py
├─ topo_sorted.csv
├─ 3c_linkers.csv
├─ 4c_linkers.csv
├─ 2c_linkers.csv
├─ functionalized_2c_linkers.csv
│
└─ COF_generation_plan.csv

```

Downstream scripts use `output_dir` to create:

```

generated_cofs/
└─ <topology_name>__COF_000001/
├─ COF_000001.cif
├─ metadata.json
└─ ...

```

---

## How to Run

From the directory containing required CSV files:

```

python make_cof_sampling_fixed_final.py

```

Expected console output:

```

[info] Total planned COFs = ...
[info] Wrote ... rows to COF_generation_plan.csv

```

---

## Parameters You May Modify

Inside the script:

- `RANDOM_SEED`
- `OUTPUT_CSV`
- `COVERAGES`
- `CASE_TARGETS`
```

