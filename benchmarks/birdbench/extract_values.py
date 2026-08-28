"""
Code taken from OmniSQL repository which is released under Apache 2.0 License.

https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/process_dataset.py

License: https://github.com/RUCKBReasoning/OmniSQL/issues/25
"""


import json
import sqlite3
import os
from tqdm import tqdm
import re
import argparse
import random
from collections import OrderedDict
from pyserini.search.lucene import LuceneSearcher
from nltk.tokenize import word_tokenize
from nltk import ngrams
import ijson


SQL_RESERVED_WORDS = {'IDENTIFIED', 'FOREIGN', 'CONSTRAINT', 'USER', 'POSITION', 'DESCRIBE', 'CHECK', 'RECURSIVE', 'REAL', 'CONTINUE', 'GLOBAL', 'RLIKE', 'INSENSITIVE', 'BOOLEAN', 'CHAR', 'ROLE', 'CASE', 'SCHEMA', 'CLOB', 'RESIGNAL', 'ROW', 'DEC', 'TOP', 'EXCEPT', 'SENSITIVE', 'OUT', 'RENAME', 'READS', 'BLOB', 'INT', 'EXTERNAL', 'LOCALTIMESTAMP', 'DECLARE', 'DO', 'AS', 'OVER', 'CONDITION', 'SELECT', 'SAVEPOINT', 'WITHIN', 'ELSEIF', 'UNLOCK', 'DATABASE', 'TRIGGER', 'ACCESS', 'FALSE', 'BREAK', 'ITERATE', 'SMALLINT', 'ASC', 'YEAR', 'DELETE', 'ROLLBACK', 'ON', 'ESCAPE', 'CREATE', 'MONTH', 'SPECIFIC', 'SESSION', 'SQLSTATE', 'HOLD', 'SET', 'EXPLAIN', 'RETURN', 'ROWNUM', 'BINARY', 'SYSDATE', 'SQLWARNING', 'EXTEND', 'CAST', 'FOR', 'TERMINATED', 'VIEW', 'TRAILING', 'HOUR', 'VARYING', 'RESTRICT', 'RIGHT', 'DISTINCT', 'JOIN', 'UNKNOWN', 'VALUES', 'TABLE', 'OR', 'DOUBLE', 'DROP', 'COMMIT', 'PRECISION', 'LANGUAGE', 'START', 'INTERSECT', 'IGNORE', 'NULL', 'CURRENT_DATE', 'LOCK', 'INTO', 'NEW', 'DESC', 'STATIC', 'MODIFIES', 'GRANT', 'VALUE', 'LIMIT', 'MODULE', 'DATE', 'LOCALTIME', 'PERCENT', 'REPEAT', 'FULL', 'USAGE', 'ORDER', 'WHEN', 'PRIMARY', 'BETWEEN', 'CURSOR', 'DECIMAL', 'HAVING', 'IF', 'FILTER', 'INDEX', 'ILIKE', 'VARCHAR', 'EXEC', 'USING', 'ROWS', 'PLACING', 'WHILE', 'EXECUTE', 'EACH', 'LEFT', 'FLOAT', 'COLLATE', 'CURRENT_TIME', 'OPEN', 'RANGE', 'CROSS', 'FUNCTION', 'TIME', 'BOTH', 'NOT', 'CONVERT', 'NCHAR', 'KEY', 'DEFAULT', 'LIKE', 'ANALYZE', 'EXISTS', 'IN', 'BIT', 'INOUT', 'SUM', 'NUMERIC', 'AFTER', 'LEAVE', 'INSERT', 'TO', 'COUNT', 'THEN', 'BEFORE', 'OUTER', 'COLUMN', 'ONLY', 'END', 'PROCEDURE', 'OFFSET', 'ADD', 'INNER', 'RELEASE', 'FROM', 'DAY', 'NO', 'CALL', 'BY', 'LOCAL', 'ZONE', 'TRUE', 'EXIT', 'LEADING', 'INTEGER', 'MERGE', 'OLD', 'AVG', 'MIN', 'SQL', 'LOOP', 'SIGNAL', 'REFERENCES', 'MINUTE', 'UNIQUE', 'GENERATED', 'ALL', 'MATCH', 'CASCADE', 'UNION', 'COMMENT', 'FETCH', 'UNDO', 'UPDATE', 'WHERE', 'ELSE', 'PARTITION', 'BIGINT', 'CHARACTER', 'CURRENT_TIMESTAMP', 'ALTER', 'INTERVAL', 'REVOKE', 'CONNECT', 'WITH', 'TIMESTAMP', 'GROUP', 'BEGIN', 'CURRENT', 'REGEXP', 'NATURAL', 'SOME', 'SQLEXCEPTION', 'MAX', 'SUBSTRING', 'OF', 'AND', 'REPLACE', 'IS'}
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_]')


def load_json_file(file):
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for obj in tqdm(objects):
            dataset.append(obj)
    return dataset

def remove_sql_comments(sql):
    # Remove single-line comments
    sql = re.sub(r'--.*', '', sql)
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    return sql.strip()

def obtain_db_ddls(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    create_statements = []
    for table in tables:
        _, create_statement = table
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    # table_schemas = [remove_sql_comments(stat) for stat in create_statements]
    return create_statements

def needs_backticks(identifier):
    if identifier.upper() in SQL_RESERVED_WORDS:
        return True
    if SPECIAL_CHARS_PATTERN.search(identifier):
        return True
    return False

def format_identifier(identifier):
    if needs_backticks(identifier):
        return f'`{identifier}`'
    return identifier

def sample_table_values(db_file_dir, table_names, limit_num):
    db_values_dict = dict()
    
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        for column_name in column_names:
            # cursor.execute(f"SELECT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {limit_num};")
            query = f"""
            SELECT `{column_name}` 
            FROM (
                SELECT DISTINCT `{column_name}` 
                FROM `{table_name}` 
                WHERE `{column_name}` IS NOT NULL and `{column_name}` != ''
            ) AS unique_values
            LIMIT {limit_num};
            """
            cursor.execute(query)
            values = cursor.fetchall()
            values = [value[0] for value in values]

            # truncate too long strings
            for idx in range(len(values)):
                if isinstance(values[idx], str):
                    values[idx] = values[idx][:40]

            if len(values) > 0:
                db_values_dict[f"{table_name}.{column_name}".lower()] = values
    
    cursor.close()
    conn.close()

    return db_values_dict

def calculate_substring_match_percentage(query, target):
    query = query.lower()
    target = target.lower()
    
    substrings = []
    for i in range(len(query)):
        for j in range(i + 1, len(query) + 1):
            substrings.append(query[i:j])
    max_matched_substring_len = max([len(substring) for substring in substrings if substring in target])
    return max_matched_substring_len/len(query)

def retrieve_relevant_hits(searcher, queries):
    queries = list(dict.fromkeys(queries))
    # print("len(queries):", len(queries))
    q_ids = [f"{idx}" for idx in range(len(queries))]

    query2hits = dict()
    search_results = searcher.batch_search(queries, q_ids, k = 10, threads=60)
    for query, q_id in zip(queries, q_ids):
        hits = search_results[q_id]
        hits = list(dict.fromkeys(([hit.raw for hit in hits])))
        hits = [json.loads(hit) for hit in hits]
        query2hits[query] = hits
    
    return query2hits

def retrieve_question_related_db_values(hits, question):
    high_score_hits = []
    for idx, hit in enumerate(hits):
        table_name, column_name, c_id = hit["id"].split("-**-")
        score = calculate_substring_match_percentage(hit["contents"], question)
        if score > 0.85:
            high_score_hits.append(
                {
                    "table_dot_column_lower_case": f"{table_name}.{column_name}".lower(), 
                    "db_value": hit["contents"],
                    "score": score,
                    "index": idx,
                }
            )
    high_score_hits = sorted(high_score_hits, key=lambda x: (x["score"], len(x["db_value"]), x["index"]), reverse=True)
    high_score_hits = high_score_hits[:20] # remain top 20 db values
    
    relavant_db_values_dict = dict()
    for hit in high_score_hits:
        if hit["table_dot_column_lower_case"] in relavant_db_values_dict:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]].append(hit["db_value"])
        else:
            relavant_db_values_dict[hit["table_dot_column_lower_case"]] = [hit["db_value"]]

    return relavant_db_values_dict

def obtain_n_grams(sequence, max_n):
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_n_grams = []
    for n in range(1, max_n + 1):
        all_n_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_n_grams

input_prompt_template = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{db_engine}

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
'''

def obtain_pk_fk_column_idx(db_info):
    pk_fk_column_idx_list = []
    for primary_keys_idx in db_info["primary_keys"]:
        if isinstance(primary_keys_idx, int):
            pk_fk_column_idx_list.append(primary_keys_idx)
        elif isinstance(primary_keys_idx, list):
            pk_fk_column_idx_list.extend(primary_keys_idx)
    for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
        pk_fk_column_idx_list.append(source_column_idx)
        pk_fk_column_idx_list.append(target_column_idx)
    return pk_fk_column_idx_list

def prepare_schema_filter_data(question, db_info):
    data = dict()
    data["text"] = question
    data["schema"] = dict()
    data["schema"]["schema_items"] = []
    for outer_table_idx, table_name_original in enumerate(db_info["table_names_original"]):
        table_info = dict()
        table_info["table_name"] = table_name_original
        table_info["table_comment"] = ""
        table_info["column_names"] = []
        table_info["column_comments"] = []
        for (inner_table_idx, column_name_original), (_, column_comment) in zip(db_info["column_names_original"], db_info["column_names"]):
            if outer_table_idx == inner_table_idx:
                table_info["column_names"].append(column_name_original)
                table_info["column_comments"].append(column_comment)
        data["schema"]["schema_items"].append(table_info)
    return data

def obtain_db_details(db_info, data_source, sampled_db_values_dict, relavant_db_values_dict, output_seq, mode, question):
    db_details = []
    assert len(db_info["column_names_original"]) == len(db_info["column_names"]) == len(db_info["column_types"])
    
    if mode == "train":
        '''
        to increase training data's diversity, the input database schema includes: 
        [PK, FK, output sequence-used columns, random sampled unused columns]
        '''
        # remain primary and foreign key columns
        used_column_idx_list = obtain_pk_fk_column_idx(db_info)
        # remain SQL-used columns
        for column_idx, (inner_table_idx, column_name) in enumerate(db_info["column_names_original"]):
            if column_name.lower() in output_seq.lower():
                used_column_idx_list.append(column_idx)
        
        used_column_idx_list = list(set(used_column_idx_list))
        used_column_num = len(used_column_idx_list)
        all_column_idx_list = list(range(len(db_info["column_names_original"])))
        unused_column_idx_list = [idx for idx in all_column_idx_list if idx not in used_column_idx_list]
        
        # random select some unused columns to mimic noise in the input sequence
        if unused_column_idx_list:
            unused_column_prob = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            sample_size = int(unused_column_prob * len(unused_column_idx_list))

            max_column_num = 225
            if used_column_num > max_column_num:
                sample_size = 0
            elif used_column_num + sample_size > max_column_num:
                sample_size = max_column_num - used_column_num
            else:
                sample_size = sample_size

            used_column_idx_list.extend(random.sample(unused_column_idx_list, sample_size))
    else:
        # put all tables and columns in the prompt
        used_column_idx_list = list(range(len(db_info["column_names_original"])))

    # print(used_column_idx_list)
    for outer_table_idx, table_name in enumerate(db_info["table_names_original"]):
        column_info_list = []
        pk_columns = []
        fk_info = []
        
        column_comment_prob = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        for column_idx, ((inner_table_idx, column_name), (_, column_comment), column_type) in enumerate(zip(
            db_info["column_names_original"], db_info["column_names"], db_info["column_types"]
        )):
            if inner_table_idx == outer_table_idx:
                if column_idx not in used_column_idx_list:
                    continue

                column_values = []
                if f"{table_name}.{column_name}".lower() in relavant_db_values_dict:
                    column_values.extend(relavant_db_values_dict[f"{table_name}.{column_name}".lower()])
                if f"{table_name}.{column_name}".lower() in sampled_db_values_dict:
                    column_values.extend(sampled_db_values_dict[f"{table_name}.{column_name}".lower()])
                column_values = list(dict.fromkeys(column_values)) # dedup (reserve order)
                column_values = column_values[:6]

                if data_source == "synthetic":
                    if random.random() < column_comment_prob:
                        column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                        if len(column_values) > 0:
                            column_info += f", example: {column_values}"
                    else: # simulate some columns do not have comment
                        column_info = f'    {format_identifier(column_name)} {column_type},'
                        if len(column_values) > 0:
                            column_info += f" -- example: {column_values}"
                else:
                    if column_name.lower() in [column_comment.lower(), column_comment.lower().replace(" ", "_"), column_comment.lower().replace(" ", "")] \
                        or column_comment.strip() == "":
                        column_info = f'    {format_identifier(column_name)} {column_type},'
                        if len(column_values) > 0:
                            column_info += f" -- example: {column_values}"
                    else:
                        column_info = f'    {format_identifier(column_name)} {column_type}, -- {column_comment}'
                        if len(column_values) > 0:
                            column_info += f", example: {column_values}"
                
                column_info_list.append(column_info)
                
                for primary_keys_idx in db_info["primary_keys"]:
                    if isinstance(primary_keys_idx, int):
                        if column_idx == primary_keys_idx:
                            pk_columns.append(column_name) # f'    PRIMARY KEY ("{ }")'
                    elif isinstance(primary_keys_idx, list):
                        if column_idx in primary_keys_idx:
                            pk_columns.append(column_name)

                for (source_column_idx, target_column_idx) in db_info["foreign_keys"]:
                    if column_idx == source_column_idx:
                        source_table_idx = db_info["column_names_original"][source_column_idx][0]
                        source_table_name = db_info["table_names_original"][source_table_idx]
                        source_column_name = db_info["column_names_original"][source_column_idx][1]
                        target_table_idx = db_info["column_names_original"][target_column_idx][0]
                        target_table_name = db_info["table_names_original"][target_table_idx]
                        target_column_name = db_info["column_names_original"][target_column_idx][1]
                        fk_info.append(f'    CONSTRAINT fk_{source_table_name.lower().replace(" ", "_")}_{source_column_name.lower().replace(" ", "_")} FOREIGN KEY ({format_identifier(source_column_name)}) REFERENCES {format_identifier(target_table_name)} ({format_identifier(target_column_name)}),')
                
        if len(column_info_list) > 0:
            pk_columns = list(OrderedDict.fromkeys(pk_columns))
            if len(pk_columns) > 0:
                pk_info = ['    PRIMARY KEY (' + ', '.join([f'{format_identifier(column_name)}' for column_name in pk_columns]) + '),']
            else:
                pk_info = []
            fk_info = list(OrderedDict.fromkeys(fk_info))

            table_ddl = ""
            table_ddl += f'CREATE TABLE {format_identifier(table_name)} (\n'
            table_ddl += "\n".join(column_info_list + pk_info + fk_info)
            if table_ddl.endswith(","):
                table_ddl = table_ddl[:-1] # remove extra commas
            table_ddl += "\n);"

            db_details.append(table_ddl)

    if mode == "train":
        random.shuffle(db_details)

    db_details = "\n\n".join(db_details)

    # double check
    for column_idx, (_, column_name) in enumerate(db_info["column_names_original"]):
        if column_name == "*":
            continue
        if column_idx in used_column_idx_list:
            assert column_name.lower() in db_details.lower()

    return db_details

def deduplicate_dicts(dict_list):
    seen = set()
    unique_dicts = []
    
    for d in dict_list:
        dict_tuple = frozenset(d.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)
    
    return unique_dicts

def prepare_input_output_pairs(data, ek_key, db_id2relevant_hits, sampled_db_values_dict, db_info, source, output_key, mode):
    if data[ek_key].strip() == "":
        question = data["question"]
    else:
        question = data[ek_key] + "\n" + data["question"]

    relavant_db_values_dict = dict()
    if db_id2relevant_hits: # retrieve matched values from the databases
        queries = obtain_n_grams(question, 8) + [question]
        queries = list(dict.fromkeys(queries))
        hits = []
        for query in queries:
            hits.extend(db_id2relevant_hits[data["db_id"]][query])
        hits = deduplicate_dicts(hits)
        relavant_db_values_dict = retrieve_question_related_db_values(hits, question)

    db_details = obtain_db_details(
        db_info, source, sampled_db_values_dict, relavant_db_values_dict, 
        data[output_key], mode, question
    )
    
    input_seq = input_prompt_template.format(
        db_engine = "SQLite",
        db_details = db_details,
        question = question
    )

    return {"input_seq": input_seq, "output_seq": data[output_key]}

def process_data(args):
    data, ek_key, searcher, sampled_db_values, db_info, source, output_key, mode = args
    return prepare_input_output_pairs(data, ek_key, searcher, sampled_db_values, db_info, source, output_key, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type = str)
    parser.add_argument("--output_data_file", type = str)
    parser.add_argument("--db_path", type = str)
    parser.add_argument("--tables", type = str)
    parser.add_argument("--source", type = str)
    parser.add_argument("--mode", type = str)
    parser.add_argument("--value_limit_num", type = int)
    parser.add_argument("--db_content_index_path", type = str)

    opt = parser.parse_args()
    print(opt)

    random.seed(42)
    assert opt.mode in ["train", "dev", "test"]
    dataset = load_json_file(opt.input_data_file)

    output_key = "SQL"
    ek_key = "evidence"

    used_db_ids = list(set([data["db_id"] for data in dataset]))
    db_id2sampled_db_values = dict()
    db_id2db_info = dict()
    for db_info in tqdm(load_json_file(opt.tables)):
        db_id = db_info["db_id"]
        if db_id not in used_db_ids:
            continue
        db_file = os.path.join(opt.db_path, db_id, db_id + ".sqlite")
        sampled_db_values_dict = sample_table_values(db_file, db_info["table_names_original"], opt.value_limit_num)
        db_id2sampled_db_values[db_id] = sampled_db_values_dict
        db_id2db_info[db_id] = db_info

    batch_size = 20000
    sliced_datasets = [dataset[i: i+batch_size] for i in range(0, len(dataset), batch_size)]
    print(len(dataset))
    print([len(batch_dataset) for batch_dataset in sliced_datasets]) 
    assert len(dataset) == sum([len(batch_dataset) for batch_dataset in sliced_datasets])

    new_dataset = []
    for batch_idx, batch_dataset in enumerate(sliced_datasets):
        print(f"Process: {batch_idx+1}/{len(sliced_datasets)}")

        if opt.db_content_index_path:
            db_id2searcher = dict()
            batch_db_ids = list(set([data["db_id"] for data in batch_dataset]))
            # load db context index searchers
            for db_id in batch_db_ids:
                db_id2searcher[db_id] = LuceneSearcher(os.path.join(opt.db_content_index_path, db_id))
            
            db_id2queries = dict()
            for data in tqdm(batch_dataset):
                if data[ek_key].strip() == "":
                    question = data["question"]
                else:
                    question = data[ek_key] + "\n" + data["question"]

                queries = obtain_n_grams(question, 8) + [question]
                queries = list(set(queries))
                if data["db_id"] in db_id2queries:
                    db_id2queries[data["db_id"]].extend(queries)
                else:
                    db_id2queries[data["db_id"]] = queries
            
            # perform db content retrieval (in a large batch)
            db_id2relevant_hits = dict()
            for db_id in tqdm(batch_db_ids):
                db_id2relevant_hits[db_id] = retrieve_relevant_hits(db_id2searcher[db_id], db_id2queries[db_id])
        else:
            db_id2relevant_hits = None

        for data in tqdm(batch_dataset):
            new_dataset.append(
                prepare_input_output_pairs(data, ek_key, db_id2relevant_hits, db_id2sampled_db_values[data["db_id"]], 
                    db_id2db_info[data["db_id"]], opt.source, output_key, opt.mode)
            )
        del db_id2searcher, db_id2relevant_hits, 

    with open(opt.output_data_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(new_dataset, indent = 2, ensure_ascii = False))