CREATE TABLE entitlement (
    entitlement_id INTEGER PRIMARY KEY,
    account_name TEXT,
    coverage_hours TEXT,
    max_cases_per_month INTEGER
);

INSERT INTO entitlement (entitlement_id, account_name, coverage_hours, max_cases_per_month)
VALUES (73, 'Stark Industries', 'h8x5', 10);
