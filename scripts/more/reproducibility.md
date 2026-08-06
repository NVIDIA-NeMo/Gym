# Reproducing the published evaluation results

## What these are

## Prerequisites

## Running a recipe

## Benchmarks needing extra setup

Most recipes need nothing beyond the prerequisites above. These do:

### SciCode

Scoring needs `test_data.h5` (~1 GB) — the numeric ground-truth the test cases are
checked against. It is not downloaded automatically. Get it from the official SciCode
[Google Drive folder](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR),
save it anywhere, and point `TEST_DATA` at it:

```bash
sha256sum /path/to/test_data.h5
# 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890

TEST_DATA=/path/to/test_data.h5 ./scicode.sh
```

It must be readable by the user running the recipe — otherwise every test case fails
and the run scores 0 with no error.

## Results

## Comparing your numbers
