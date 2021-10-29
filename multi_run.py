import subprocess

gpu = 0

# Run Conv_2 with lr_scheduler.
subprocess.call(
    'python prune.py' + ' -m conv_2' + ' -o conv_2_with_lr'
    + f' -g {gpu}' + ' -lrs',
    shell=True
)

# Run Conv_2 without lr_scheduler.
subprocess.call(
    'python prune.py' + ' -m conv_2' + ' -o conv_2_with_lr'
    + f' -g {gpu}',
    shell=True
)

