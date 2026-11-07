-- Census-Income CSV import
-- 1) Create the database once from a terminal:
--      createdb census
--    Or in psql connected to another database:
--      CREATE DATABASE census;
--
-- 2) Save this script and replace the CSV path in the \copy command below.
-- 3) Run:
--      psql -d census -f import_census_income.sql
--
-- Column names are preserved exactly as supplied (including spaces).
-- '?' is imported as SQL NULL. The target column 'income' is kept as TEXT.

CREATE TABLE IF NOT EXISTS income (
    "age" SMALLINT,
    "class of worker" TEXT,
    "detailed industry recode" SMALLINT,
    "detailed occupation recode" SMALLINT,
    "education" TEXT,
    "wage per hour" INTEGER,
    "enroll in edu inst last wk" TEXT,
    "marital stat" TEXT,
    "major industry code" TEXT,
    "major occupation code" TEXT,
    "race" TEXT,
    "hispanic origin" TEXT,
    "sex" TEXT,
    "member of a labor union" TEXT,
    "reason for unemployment" TEXT,
    "full or part time employment stat" TEXT,
    "capital gains" INTEGER,
    "capital losses" INTEGER,
    "dividends from stocks" INTEGER,
    "tax filer stat" TEXT,
    "region of previous residence" TEXT,
    "state of previous residence" TEXT,
    "detailed household and family stat" TEXT,
    "detailed household summary in household" TEXT,
    "instance weight" NUMERIC(14,2),
    "migration code-change in msa" TEXT,
    "migration code-change in reg" TEXT,
    "migration code-move within reg" TEXT,
    "live in this house 1 year ago" TEXT,
    "migration prev res in sunbelt" TEXT,
    "num persons worked for employer" SMALLINT,
    "family members under 18" TEXT,
    "country of birth father" TEXT,
    "country of birth mother" TEXT,
    "country of birth self" TEXT,
    "citizenship" TEXT,
    "own business or self employed" SMALLINT,
    "fill inc questionnaire for veteran's admin" TEXT,
    "veterans benefits" SMALLINT,
    "weeks worked in year" SMALLINT,
    "year" SMALLINT,
    "income" TEXT
);

\copy income (
    "age",
    "class of worker",
    "detailed industry recode",
    "detailed occupation recode",
    "education",
    "wage per hour",
    "enroll in edu inst last wk",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "tax filer stat",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "instance weight",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "num persons worked for employer",
    "family members under 18",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "weeks worked in year",
    "year",
    "income"
)
FROM 'census.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ',',
    QUOTE '"',
    ESCAPE '"',
    NULL '?'
);


-- Import Validation --

SELECT COUNT(*) AS imported_rows FROM income;
SELECT * FROM income LIMIT 5;