-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Give the agent role, bird_agent, the objects of one task database, which the restore made the superuser's. The
-- Dockerfile runs it in each <db>_template. The role then owns the tables, views, sequences and types, and the public
-- schema, so a management task's SQL can change them, while the role cannot read the server's files, run programs or
-- change its settings. A sequence that a column owns follows its table, and an extension's objects stay its own.
DO $$
DECLARE
    item record;
BEGIN
    FOR item IN
        SELECT c.relkind, n.nspname, c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg\_%'
          AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
          AND NOT EXISTS (
              SELECT 1 FROM pg_depend d
              WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype IN ('a', 'i', 'e')
          )
    LOOP
        EXECUTE format(
            'ALTER %s %I.%I OWNER TO bird_agent',
            CASE item.relkind
                WHEN 'v' THEN 'VIEW'
                WHEN 'm' THEN 'MATERIALIZED VIEW'
                WHEN 'S' THEN 'SEQUENCE'
                WHEN 'f' THEN 'FOREIGN TABLE'
                ELSE 'TABLE'
            END,
            item.nspname,
            item.relname
        );
    END LOOP;
    FOR item IN
        SELECT n.nspname, t.typname
        FROM pg_type t
        JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg\_%'
          AND t.typtype IN ('e', 'd')
          AND NOT EXISTS (
              SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_type'::regclass AND d.objid = t.oid AND d.deptype = 'e'
          )
    LOOP
        EXECUTE format('ALTER TYPE %I.%I OWNER TO bird_agent', item.nspname, item.typname);
    END LOOP;
    FOR item IN
        SELECT nspname FROM pg_namespace WHERE nspname NOT IN ('pg_catalog', 'information_schema') AND nspname NOT LIKE 'pg\_%'
    LOOP
        EXECUTE format('ALTER SCHEMA %I OWNER TO bird_agent', item.nspname);
    END LOOP;
END
$$;
