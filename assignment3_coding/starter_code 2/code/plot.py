from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

SCALARS = {'Max_Reward', 'Avg_Reward', 'Std_Reward', 'Eval_Reward'}

def early_exit(message):
    print(message)
    exit()

def load_single_file(path):
    results = {scalar: [] for scalar in SCALARS}
    for event in tf.train.summary_iterator(str(path)):
        for value in event.summary.value:
            if value.tag in SCALARS:
                results[value.tag].append((event.step, value.simple_value))
            else:
                print(f'WARNING! Unknown tag {value.tag} found in file {path}')
    return results
    
def group_by_scalar(results):
    return {scalar: np.array([run_results[scalar] for run_results in results]) for scalar in SCALARS}
    
def plot_combined(scalar_results):
    points = defaultdict(list)
    for run_results in scalar_results:
        for step, value in run_results:
            points[step].append(value)
    
    xs = sorted(points.keys())
    values = np.array([points[x] for x in xs])
    ys = np.mean(values, axis=1)
    yerrs = stats.sem(values, axis=1)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys)
    
def plot_individually(run_results):
    xs = [step for step, value in run_results]
    ys = [value for step, value in run_results]
    plt.plot(xs, ys)
        
def plot(results_list, names, title, combine, plots_dir):
    plt.figure()
    plt.title(title)
    plt.xlabel('Step')
    for results in results_list:
        if combine:
            plot_combined(results)
        else:
            plot_individually(results)
    suffix = '_combined' if combine else '_individual'
    save_path = plots_dir / (title + suffix + '.png')
    plt.legend(names)
    plt.savefig(str(save_path))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', required=True, help='Directory containing TensorFlow event files')
    parser.add_argument('--env', required=True, help='Name of environment')
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.is_dir():
        early_exit(f'Given path ({directory.resolve()}) is not a directory')
    
    baseline_results = []
    no_baseline_results = []
    all_results = []
    all_names = []
    for sub in directory.iterdir():
        if sub.is_dir() and sub.name.startswith(args.env):
            tfevents_files = list(sub.glob('*tfevents*'))
            if len(tfevents_files) == 0:
                early_exit(f'No TF events files found in {sub}')
            else:
                tfevents_file = tfevents_files[0]
                if len(tfevents_files) > 1:
                    print(f'WARNING: more than one TF events file found in {sub}. Arbitrarily picking {tfevents_file}')
                results = load_single_file(tfevents_file)
                all_results.append(results)
                all_names.append(sub.name[len(args.env)+1:])
                if 'no_baseline' in sub.name:
                    no_baseline_results.append(results)
                else:
                    assert 'baseline' in sub.name
                    baseline_results.append(results)

    plots_dir = directory / f'plots-{args.env}'
    plots_dir.mkdir(exist_ok=True)
    
    baseline_by_scalar = group_by_scalar(baseline_results)
    no_baseline_by_scalar = group_by_scalar(no_baseline_results)
    all_by_scalar = group_by_scalar(all_results)
    
    for scalar in SCALARS:
        plot([baseline_by_scalar[scalar], no_baseline_by_scalar[scalar]], ['Baseline', 'No baseline'], scalar, True, plots_dir)
        plot(all_by_scalar[scalar], all_names, scalar, False, plots_dir)