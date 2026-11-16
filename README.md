# Text-to-NoSQL Analysis: Experimentation

## Download the datasets first

Before running experiments, download **DocSpider** and **SM3** using the
dataset links in the respective papers. Follow the papers' instructions for
downloading and preparing the data.

- **DocSpider:** [paper and dataset information](https://doi.org/10.1017/nlp.2024.63)
- **SM3-Text-to-Query:** [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/a182a8e6ebc91728b6e6b6382c9f7b1e-Paper-Datasets_and_Benchmarks_Track.pdf)

Keep the original downloads under `data/Docspider/` and `data/SM3/` if useful,
and prepare CSV splits for the scripts. A three-row DocSpider sample
development file is provided at `data/dev.csv`, which is also the input used
in the example commands below. Use the dataset's instructions when loading
its MongoDB data.

## What this repository contains

This is an experimental collection of MongoDB natural-language-to-query
approaches:

- `systems/t1.py` and `systems/t1_ex.py`: schema-based generation; `t1_ex`
  includes representative field values.
- `systems/t2.py` and `systems/t2_expcot.py`: chain-of-thought and ExpCoT
  variants.
- `systems/t3.py`: retrieval-augmented few-shot generation using a local
  FAISS index.
- `systems/t6_reflect.py`: generate, execute, judge, and repair queries.
- `utils/rag_prep_ingestion/ingestion.py`: build the FAISS index used by
  `t3.py`.

The scripts generate MongoDB aggregation queries. The setup steps below
describe how to prepare the data, databases, and retrieval index for
experiments.

## CSV format expected by the experiment scripts

The `t1_ex.py`, `t3.py`, and `t6_reflect.py` commands expect a CSV with these
header columns:

| Column | Purpose |
| --- | --- |
| `question_id` | Identifier for the example |
| `db_id` | MongoDB database name for the example |
| `question` | Natural-language question |
| `query` | Reference/gold query, retained in the output for comparison |

`t3.py` uses training examples with at least `question_id`, `question`, and
`query` to build its retrieval index. Keep train and development/test examples
separate: do not include evaluation questions in the index.

## Prerequisites and setup

### Python environment

The project is intended to run with **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### MongoDB

The experiment scripts infer schemas from MongoDB collections. T3 also uses
MongoDB schema information, and T6 executes generated pipelines to assess and
repair them. Start a MongoDB server and load the benchmark databases before
running experiments. Each database name in the CSV's `db_id` column must exist
in MongoDB and contain the corresponding collections and documents.

The MongoDB client connects to `mongodb://localhost:27017/` by default. Update
the connection configuration in `mongo_client.py` and `systems/t6_reflect.py`
if your MongoDB server is elsewhere.

### Language model

The examples use Ollama through LiteLLM. Install and start Ollama separately,
then download the models you intend to run. For the models in the examples:

```bash
ollama pull llama3.1:8b
ollama pull gemma4:latest
```

The default Ollama endpoint is `http://localhost:11434`. `.env.example`
documents optional provider settings; copy it to `.env` and add credentials
only for the providers you use. Never commit API keys or other secrets.

### FAISS index for T3

T3 cannot run until its index directory contains `config.json`,
`metadata.json`, and `train.index`. Build an index from the training CSV with:

```bash
python utils/rag_prep_ingestion/ingestion.py \
  --input data/train.csv \
  --output-dir faiss_train_agg
```

The ingestion script downloads/loads
`sentence-transformers/all-MiniLM-L6-v2` to embed the training questions.
Use a training CSV that has the required `question_id`, `question`, and
`query` columns. For a different training set, build a separate index and
pass its directory using `--faiss-dir` (for example,
`faiss_train_sm3_agg`).

## Run the supplied approaches

Run commands from the repository root after setting up the CSV, MongoDB
databases, model provider, and (for T3) FAISS index.

The commands below are **representative examples only**, not an exhaustive
list of experiment combinations. They show three approaches, models, and
settings; they do not limit which combinations you can benchmark. Run any
implemented approach against either dataset split with the model and
parameters you want to compare, using a distinct `--run-name` for each run.
The repository also includes `systems/t1.py` (T1), `systems/t2.py` (T2), and
`systems/t2_expcot.py` (T2-ExpCoT). Use each script's `--help` to see its
available options. For T3, vary `--top-k` and the training-set FAISS index as
well as the model; for T1-ex and T6, options include `--max-values` and
`--max-attempts`, respectively.

### T3: retrieval-augmented few-shot

```bash
python systems/t3.py data/dev.csv \
  --run-name t3_llama318b \
  --model ollama/llama3.1:8b \
  --faiss-dir faiss_train_agg \
  --top-k 10
```

### T6: reflect and repair

```bash
python systems/t6_reflect.py data/dev.csv \
  --run-name t6_reflect_llama318b \
  --model ollama/llama3.1:8b
```

T6 defaults to a maximum of three generation/repair attempts per question.
Override this with `--max-attempts N` if needed.

### T1-ex: schema with representative values

```bash
python systems/t1_ex.py data/dev.csv \
  --run-name t1_ex_gemma4e4b \
  --model ollama/gemma4:latest \
  --max-values 5
```

All three scripts accept `--help` for their complete command-line options.

## Results

Each run writes under `output/<run-name>/`:

- `predictions.csv` contains the question, reference query, and generated
  query.
- `errors.csv` records examples that failed during processing.
- T3 also writes retrieval details and the prompts sent to the model.
- T6 writes per-question generation/repair traces under `traces/`.

Use distinct `--run-name` values to keep experiment outputs separate.

## Adding an approach such as `tx_method.py`

For consistency, implement the approach as `systems/tx_method.py` and preserve
the CSV and output conventions used by the existing systems.

1. **Define the method.** Specify the prompt, examples, schema context, or
   other novel mechanism that distinguishes the approach. Keep the model,
   dataset split, and any retrieval corpus fixed when comparing methods.
2. **Reuse shared helpers.** Call `litellm_call.call_llm` for model requests,
   use `mongo_client.get_database_schema` and `schema_to_text` if the method
   needs schema grounding, and use `llm_output_utils.clean_llm_output` to
   normalize model responses. Avoid duplicating provider and MongoDB setup.
3. **Keep the query contract stable.** Ask the model for valid JSON in the
   shape `{"collection": "...", "pipeline": [...]}` and validate that the
   collection and pipeline are present before formatting the result as a
   MongoDB aggregation query. Decide explicitly whether the approach only
   generates queries or also executes and evaluates them.
4. **Use a compatible CLI and output.** A useful interface would accept a
   positional input CSV plus required `--run-name` and optional `--model`,
   like the existing scripts. Read the required CSV columns above and write
   `output/<run-name>/predictions.csv` and `errors.csv` with compatible
   fields. This allows downstream comparisons without approach-specific
   preprocessing.

For example, an invocation with that interface looks like this:

```bash
python systems/tx_method.py data/dev.csv \
  --run-name tx_method_llama318b \
  --model ollama/llama3.1:8b
```

## References

```bibtex
@inproceedings{sivasubramaniam2024sm3,
  author    = {Sivasubramaniam, Sithursan and Osei-Akoto, Cedric and Zhang, Yi and Stockinger, Kurt and F\"{u}rst, Jonathan},
  title     = {SM3-Text-to-Query: Synthetic Multi-Model Medical Text-to-Query Benchmark},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {88627--88663},
  year      = {2024},
  doi       = {10.52202/079017-2812},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a182a8e6ebc91728b6e6b6382c9f7b1e-Paper-Datasets_and_Benchmarks_Track.pdf}
}

@article{ozer2025docspider,
  title     = {DocSpider: a dataset of cross-domain natural language querying for MongoDB},
  author    = {{\"O}zer, Arif G{\"o}rkem and Cekinel, Recep Firat and Toroslu, Ismail Hakki and Karagoz, Pinar},
  journal   = {Natural Language Processing},
  pages     = {1--32},
  publisher = {Cambridge University Press},
  doi       = {10.1017/nlp.2024.63}
}
```
