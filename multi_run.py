import subprocess

gpu = 0
model = 'conv_4'

subprocess.call(
    'python varying_size.py' + f' -m {model}' + ' -o conv_4_50'
    + f' -g {gpu}' + ' -s 50',
    shell=True
)

# Run Conv_2 without lr_scheduler.
subprocess.call(
    'python prune.py' + ' -m conv_2' + ' -o conv_2_with_lr'
    + f' -g {gpu}',
    shell=True
)
