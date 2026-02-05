# Text-to-SQL Resource Server

## Description

This resource server evaluates text-to-SQL generation using an LLM as a judge. Given a natural language question and database schema, the model generates SQL queries that are compared against ground truth using semantic equivalence checking.

The server extracts SQL from model responses (supporting code blocks and raw SQL statements) and uses an LLM judge to determine if the generated query is functionally equivalent to the expected query.

## Supported SQL Dialects

- **MySQL**: backtick quoting, LIMIT syntax, MySQL-specific functions
- **PostgreSQL**: double-quote identifiers, `::` casting, array operations  
- **SQLite**: limited type system, `||` string concatenation

More dialects planned for future versions (TODO: Oracle, SQL Server, BigQuery, Snowflake).

## Verification Flow

1. **SQL Extraction**: Extract SQL from the model's response (code blocks or raw SQL)
2. **LLM Judge Evaluation**: Compare extracted SQL against ground truth using semantic equivalence
3. **Swap Check** (optional): Run second evaluation with swapped inputs to detect positional bias

## Input Format

Each data sample should include:
- `responses_create_params.input`: User message containing the prompt for the model
- `sql`: The ground truth SQL query
- `sql_dialect`: SQL dialect (`mysql`, `postgresql`, or `sqlite`)
- `sql_context`: Database schema and sample data (CREATE TABLE + INSERT statements)
- `sql_prompt`: Natural language question describing the desired query

### Example Input

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "system",
        "content": "You are a SQL assistant who expertly maps natural language commands and problems into clean, efficient, and well-structured SQL queries. You always carefully examine the database context to include only the relevant tables and fields. You return only SQL code (no explanations outside of SQL code block; only proper comments are permitted). Wrap your SQL query inside a ```sql``` code block."
      },
      {
        "role": "user",
        "content": "DIALECT: postgresql\n\nDATABASE CONTEXT:\nCREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));\nINSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');\n\nQUESTION:\nList all user names ordered alphabetically"
      }
    ]
  },
  "sql": "SELECT name FROM users ORDER BY name;",
  "sql_dialect": "postgresql",
  "sql_context": "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));\nINSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');",
  "sql_prompt": "List all user names ordered alphabetically"
}
```

### Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `sql` | Yes | Ground truth SQL query |
| `sql_dialect` | Yes | SQL dialect: `mysql`, `postgresql`, or `sqlite` |
| `sql_context` | Yes | Database schema (CREATE TABLE) and sample data (INSERT statements) |
| `sql_prompt` | Yes | Natural language question describing the query |
| `responses_create_params` | Yes | Model input containing the full prompt |
| `uuid` | Recommended | Unique identifier for tracking the example |

## Usage

### Running Servers

```bash
config_paths="resources_servers/text_to_sql/configs/text_to_sql.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[${config_paths}]" \
  +text_to_sql_resources_server.resources_servers.text_to_sql.judge_responses_create_params.max_output_tokens=512
```

### Collecting Rollouts

```bash
ng_collect_rollouts +agent_name=text_to_sql_simple_agent \
    +input_jsonl_fpath=resources_servers/text_to_sql/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/text_to_sql/data/example_rollouts.jsonl
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `judge_model_server` | - | Model server to use as the judge |
| `judge_system_message` | (see `prompts.py`) | System message for the judge LLM |
| `judge_equal_label` | `[[A=B]]` | Label indicating equivalent queries (A and B are equivalent) |
| `judge_not_equal_label` | `[[A!=B]]` | Label indicating non-equivalent queries |
| `check_twice_swap` | `true` | Run swap check to detect positional bias |
| `reward_if_swap_fails` | `0.0` | Reward when swap check fails |

The judge prompts are defined in `prompts.py`:
- `SQL_JUDGE_SYSTEM_MESSAGE`: SQL-specific system instructions with detailed equivalence rules, dialect considerations, and evaluation criteria
- `SQL_JUDGE_PROMPT_TEMPLATE`: User-level template with dialect, schema, question, and SQL queries labeled as A and B

## Equivalence Criteria

The LLM judge considers queries equivalent if they:
1. Produce the same result set (same rows and columns) **given the provided schema and data**
2. Have only syntactic differences (aliases, formatting, explicit vs implicit JOINs)
3. Use equivalent constructs (subquery vs CTE, etc.)
4. Are valid for the specified SQL dialect
5. Correctly use the table/column names defined in the schema

## Licensing Information

**Code**: Apache 2.0

**Data**: CC-BY-4.0 (synthetic examples)

## Dependencies

- nemo_gym: Apache 2.0
