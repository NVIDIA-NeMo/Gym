-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- A synthetic bookshop database for the BIRD-Interact example tasks.
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER NOT NULL REFERENCES authors(author_id),
    genre TEXT NOT NULL,
    list_price REAL NOT NULL,
    published_year INTEGER NOT NULL
);
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(book_id),
    sale_date TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
INSERT INTO authors VALUES
    (1, 'Ana Ribeiro', 'Brazil'),
    (2, 'Kenji Mori', 'Japan'),
    (3, 'Lena Fischer', 'Germany'),
    (4, 'Omar Haddad', 'Jordan'),
    (5, 'Grace Okafor', 'Nigeria');
INSERT INTO books VALUES
    (1, 'River of Glass', 1, 'Fiction', 20.0, 2019),
    (2, 'The Quiet Harbor', 1, 'Fiction', 18.0, 2022),
    (3, 'Counting Stars', 1, 'Science', 30.0, 2023),
    (4, 'Paper Lanterns', 2, 'Fiction', 16.0, 2021),
    (5, 'Tidal Data', 2, 'Science', 40.0, 2024),
    (6, 'Bread and Salt', 3, 'Cooking', 25.0, 2020),
    (7, 'The Long Field', 3, 'History', 32.0, 2022),
    (8, 'Market Days', 4, 'History', 28.0, 2018),
    (9, 'Desert Light', 4, 'Fiction', 15.0, 2023),
    (10, 'Small Kitchens', 5, 'Cooking', 22.0, 2024);
INSERT INTO sales VALUES
    (1, 1, '2024-01-03', 6, 18.0),
    (2, 1, '2024-02-11', 5, 20.0),
    (3, 2, '2024-01-15', 4, 18.0),
    (4, 2, '2024-03-02', 7, 16.2),
    (5, 3, '2024-02-20', 3, 27.0),
    (6, 3, '2024-04-08', 8, 30.0),
    (7, 4, '2024-01-22', 6, 16.0),
    (8, 4, '2024-05-19', 5, 12.8),
    (9, 5, '2024-03-30', 6, 40.0),
    (10, 5, '2024-06-01', 6, 34.0),
    (11, 6, '2024-02-02', 9, 25.0),
    (12, 6, '2024-04-17', 4, 20.0),
    (13, 7, '2024-03-12', 2, 32.0),
    (14, 7, '2024-05-05', 1, 28.8),
    (15, 8, '2024-01-28', 3, 28.0),
    (16, 9, '2024-02-14', 5, 15.0),
    (17, 9, '2024-06-21', 2, 12.0),
    (18, 10, '2024-04-30', 4, 22.0),
    (19, 10, '2024-06-10', 3, 17.6);
