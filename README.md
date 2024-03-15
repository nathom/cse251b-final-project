# cse251b-final-project

Install `poetry` for dependency management.

Usage

```
poetry shell   # enter venv
poetry install # install dependencies
play --help    # our command
```


## Monte carlo tree search

We wrote this in C due to performance issues with the Python version (it would take 3 days to ru n). It is a single file and has no dependencies.

```
cd c # directory with c files
make
# Play 1 game with 200 branching factor with display
./2048 mc 200 1 true
```

The output is written to csv files in the current directory. You
can change the heuristic used by modifying the `METHOD` macro
at the top of the file.

