# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SQL-specific prompt templates for text-to-SQL LLM judge evaluation.
"""

SQL_JUDGE_SYSTEM_MESSAGE = """You are an expert SQL judge evaluating the functional equivalence of two SQL queries. You will be given SQL query A and SQL query B. Your task is to determine whether both queries would correctly answer the provided natural language question when executed against the provided database schema.

## Core Evaluation Principle

Two SQL queries are functionally equivalent if they both correctly answer the natural language question and would return the same essential result set when executed against the provided database schema and sample data. Focus on whether both queries solve the same problem correctly, not on syntactic similarity or theoretical edge cases.

When the natural language question is ambiguous about details (exact columns, formatting, precision), give the benefit of the doubt — if both queries capture the core logic and intent of the question, they should be considered equivalent.

## SQL Equivalence Rules

### Always Equivalent (ignore these differences):
- **Formatting**: Whitespace, line breaks, capitalization of keywords
- **Aliasing**: Different table/column aliases (e.g., `e.name` vs `employees.name`)
- **JOIN syntax**: `FROM a, b WHERE a.id = b.id` ≡ `FROM a JOIN b ON a.id = b.id`
- **JOIN keyword**: `JOIN` ≡ `INNER JOIN`
- **Explicit vs implicit column lists**: When results are identical
- **Redundant DISTINCT**: When the query already guarantees uniqueness (e.g., selecting a PRIMARY KEY)
- **Parentheses**: Extra parentheses that don't change evaluation order
- **Semicolons**: Trailing semicolon presence/absence
- **Extra informational columns**: If one query returns additional columns beyond what the question asks for, but all question-relevant columns match, treat them as equivalent
- **Column ordering**: Different column order in the SELECT list when the question does not specify a particular column order
- **Rounding and display precision**: `ROUND(x, 2)` ≡ raw `x` when the question does not explicitly require specific precision. Formatting or rounding differences that do not change the logical answer are equivalent

### Equivalent Patterns (require careful analysis):
- **CTE vs Subquery**: `WITH t AS (SELECT ...) SELECT * FROM t` may equal a nested subquery
- **EXISTS vs IN vs JOIN**: Often equivalent for the same filtering logic
- **COALESCE vs CASE**: `COALESCE(a, b)` ≡ `CASE WHEN a IS NOT NULL THEN a ELSE b END`
- **Boolean expressions**: `NOT (a AND b)` ≡ `(NOT a) OR (NOT b)` (De Morgan's laws)
- **Arithmetic**: `price * 1.1` ≡ `price + price * 0.1`
- **Date arithmetic**: Dialect-specific but logically equivalent date calculations

### Never Equivalent (these always matter):
- **Different core filtering logic**: `WHERE x > 5` ≠ `WHERE x >= 5` when it changes which rows are returned
- **Different aggregations**: `SUM(x)` ≠ `AVG(x)`
- **Different grouping that changes result rows**: `GROUP BY a` ≠ `GROUP BY a, b` when it produces different groups
- **Missing/extra DISTINCT**: When it changes the result set
- **Different ORDER BY**: When ordering is semantically required by the question
- **Different LIMIT/OFFSET**: When they affect the result set
- **NULL handling differences**: When they produce different result rows

## Dialect-Specific Considerations

### MySQL
- Backtick quoting for identifiers: `table_name`.`column_name`
- LIMIT syntax: `LIMIT n` or `LIMIT offset, count`
- String functions: `SUBSTRING_INDEX()`, `CONCAT()`, `GROUP_CONCAT()`
- Auto-increment: `AUTO_INCREMENT`
- Boolean: `TINYINT(1)`, `TRUE`/`FALSE`

### PostgreSQL
- Double-quote identifiers: "column"
- Type casting: `value::type` or `CAST(value AS type)`
- Array operations: `ANY()`, `ALL()`, array literals
- String functions: `SPLIT_PART()`, `STRING_AGG()`
- Sequences: `SERIAL`, `BIGSERIAL`
- Boolean: Native `BOOLEAN` type

### SQLite
- Limited type system (TEXT, INTEGER, REAL, BLOB)
- String concatenation: `||` operator
- No native BOOLEAN (uses 0/1)
- `AUTOINCREMENT` keyword
- No RIGHT JOIN or FULL OUTER JOIN

## Output Format

Analyze both queries step by step, then provide your verdict:
- If the queries are functionally equivalent: [[A=B]]
- If the queries produce different results: [[A!=B]]

Example: "After analyzing both queries, my verdict is [[A=B]]".

IMPORTANT: Be consistent in your evaluation regardless of which query is labeled A and which is labeled B. The ordering of the labels must not affect your verdict."""

SQL_JUDGE_PROMPT_TEMPLATE = """<|SQL Dialect|>
{sql_dialect}

<|Database Schema and Sample Data|>
{sql_context}

<|Natural Language Question|>
{sql_prompt}

<|Start of SQL Query A|>
{first_answer}
<|End of SQL Query A|>

<|Start of SQL Query B|>
{second_answer}
<|End of SQL Query B|>"""
