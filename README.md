# online k center prototype models.

## Instructions:

0. Install the required packages (you might already have them in your base environment):
```bash
pip install -r requirements.txt
```

1. Run the following command:
```bash
python run_experiments.py --help
```

2. Understand args and run again, e.g.:
```bash
python run_experiments.py --set_name test
python run_experiments.py --set_name test --gamma 2
python run_experiments.py --set_name test --gamma 2 --perm start
```

## Notes on final.csv schema:
For permuations: you can pass none, start, end, full to the permutation argument. When we store the results in final.csv, we store the permuation in the perm_order column, and the indices are the indices of the original ordering. All other columns that use indices of points, such as the facilities column or facilities file, are stored in the permuted ordering, so they are a function of the permutation.

As an example, consider an instance with T = 2, so points 0, 1, 2. First we run it with --perm null, so we use the original ordering on-disk. The optimal solution turns out to build facilities 0 and 1. Some columns in final.csv:

perm | perm_order | facilities |
none | 0-1-2 | 0-1 |

Next we run it with --perm full, and the permutation sampled randomly turns out to be 2-0-1. The optimal solution is now to build facility 2 which arrives at time period 0 in this permutation. Some columns in final.csv:

perm | perm_order | facilities |
full | 2-0-1 | 0 |

Notice that the facilities column is still 0, because it is a function of the permutation. The facilities file follows the same pattern.

