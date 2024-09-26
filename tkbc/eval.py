import subprocess
import argparse
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import defaultdict

PATH = './tkbc/output_240820/'

parser = argparse.ArgumentParser(
    description="BiTComplEx Evaluator"
)
parser.add_argument(
    '--skip_training', default=False, action="store_true"
)
parser.add_argument(
    '--model', default='BiTComplEx', type=str
)

args = parser.parse_args()

ranks = [400]
datasets = ['ICEWS14', 'ICEWS05-15', 'wikidata12k', 'yago11k', 'gdelt']
configs = {
    'wikidata12k': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'yago11k': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'gdelt': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.1'
    }],
    'ICEWS14': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.1'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.1'
    }],
    'ICEWS05-15': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'yago15k': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-3',
        'time_reg': '1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-1',
        'time_reg': '1',
        'lr': '0.01'
    }, {
        'emb_reg': '1e-2',
        'time_reg': '1',
        'lr': '0.01'
    }]
}


# Best configs for N3
configs = {
    'ICEWS14': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    }],
    'ICEWS05-15': [{
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'yago15k': [{
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    }],
    'wikidata12k': [{
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'yago11k': [{
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    }],
    'gdelt': [{
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }]
}

viz_configs = {
    'ICEWS14': {
        'emb_reg': '1e-2',
        'time_reg': '1e-1',
        'lr': '0.1'
    },
    'ICEWS05-15': {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    },
    'yago15k': {
        'emb_reg': '1e-1',
        'time_reg': '1e-1',
        'lr': '0.01'
    },
    'wikidata12k': {
        'emb_reg': '1e-3',
        'time_reg': '1e-2',
        'lr': '0.01'
    },
    'yago11k': {
        'emb_reg': '1e-1',
        'time_reg': '1e-2',
        'lr': '0.01'
    },
    'gdelt': {
        'emb_reg': '1e-2',
        'time_reg': '1e-2',
        'lr': '0.1'
    }
}

commands = []

for dataset in datasets:
    for rank in ranks:
        for emb_conf in configs[dataset]:
            commands.append([
                'python', 'tkbc/learner.py', 
                '--dataset', dataset, 
                '--model', 'BiTComplEx', 
                '--rank', str(rank), 
                '--emb_reg', emb_conf['emb_reg'], 
                '--time_reg', emb_conf['time_reg'],
                '--learning_rate', emb_conf['lr']
            ])

if (not args.skip_training):
    for idx, command in enumerate(commands):
        print(f"RUNNING {idx + 1}/{len(commands)}")
        subprocess.run(command)
else:
    print("Skipped training")

if not os.path.exists('./tkbc/figures'):
    os.makedirs('./tkbc/figures')

dataset_colors = {
    'yago15k': '#737aff',
    'ICEWS14': '#ff7a79',
    'ICEWS05-15': '#ffdd35',
}

# Plot 3D bar
data = {}

for dataset in datasets:
    data[dataset] = []
    for rank in ranks:
        emb_reg = float(viz_configs[dataset]['emb_reg'])
        time_reg = float(viz_configs[dataset]['time_reg'])
        lr = float(viz_configs[dataset]['lr'])
        with open(f'{PATH}{args.model}_{dataset}_{rank}_{lr}_{emb_reg}_{time_reg}_best.json', 'r') as f:
            best_record = json.load(f)
            data[dataset].append(best_record['test']['MRR'])

ypos, xpos = np.meshgrid(np.arange(0, len(ranks)) + 0.25, np.arange(0, len(ranks)) + 0.25, indexing="ij")
zpos = 0.5
dx = dy = 0.5 * np.ones_like(zpos)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for idx, dataset in enumerate(dataset_colors.keys()):
    ax.bar3d(xpos[idx], ypos[idx], zpos, dx, dy, np.array(data[dataset]) - zpos, color=dataset_colors[dataset], zsort='average', shade=True)

ax.set_xticks(np.arange((min(ranks) - 100) / 2, (max(ranks) - 100), 50) / 50, labels=ranks)
ax.set_yticks([])
ax.set_xlabel('Factorization rank')
ax.set_zlabel('MRR', labelpad=10)
ax.set_zlim(bottom=zpos)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=dataset_colors[dataset]) for dataset in dataset_colors.keys()], dataset_colors.keys(), 
          title="Dataset",
          loc="lower right",
          bbox_to_anchor=(1.1,-0.05),
          frameon=False)
plt.savefig('./tkbc/figures/mrr_by_rank_3d.svg', format='svg', bbox_inches='tight', pad_inches=0.4)

# Plot MRR/metrics line charts

best_mrr = defaultdict(lambda: 0)
best_mrr_rank = {}
for dataset in datasets:
    emb_reg = float(viz_configs[dataset]['emb_reg'])
    time_reg = float(viz_configs[dataset]['time_reg'])
    lr = float(viz_configs[dataset]['lr'])
    for rank in ranks:
        with open(f'{PATH}{args.model}_{dataset}_{rank}_{lr}_{emb_reg}_{time_reg}_best.json', 'r') as f:
            record = json.load(f)
            if (best_mrr[dataset] < record['test']['MRR']):
                best_mrr[dataset] = record['test']['MRR']
                best_mrr_rank[dataset] = rank

data = {
    'Avg': defaultdict(dict),
    'Head': defaultdict(dict),
    'Tail': defaultdict(dict)
}
for eval_type in data.keys():
    for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
        for dataset in datasets:
            emb_reg = float(viz_configs[dataset]['emb_reg'])
            time_reg = float(viz_configs[dataset]['time_reg'])
            lr = float(viz_configs[dataset]['lr'])
            for rank in ranks:
                if best_mrr_rank[dataset] == rank:
                    df = pd.read_csv(f'{PATH}{args.model}_{dataset}_{rank}_{lr}_{emb_reg}_{time_reg}.csv')
                    df = df[df['Split'] == 'test']
                    data[eval_type][metric][dataset] = {
                        "epoch": df['Epoch'] + 1,
                        "data": df[(eval_type + '_' if eval_type != 'Avg' else '') + metric]
                    }

for eval_type in data.keys():
    fig, axs = plt.subplots(2, 2)
    for k, v in data[eval_type]['MRR'].items():
        axs[0, 0].plot(v['epoch'], v['data'], color=dataset_colors[k], label=k)
    axs[0, 0].set_ylabel('MRR')
    axs[0, 0].legend(prop={'size': 7}, frameon=False)
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for k, v in data[eval_type]['Hits@1'].items():
        axs[0, 1].plot(v['epoch'], v['data'], color=dataset_colors[k], label=k)
    axs[0, 1].set_ylabel('Hits@1')
    axs[0, 1].legend(prop={'size': 7}, frameon=False)
    axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[0, 1].yaxis.tick_right()
    for k, v in data[eval_type]['Hits@3'].items():
        axs[1, 0].plot(v['epoch'], v['data'], color=dataset_colors[k], label=k)
    axs[1, 0].set_ylabel('Hits@3')
    axs[1, 0].legend(prop={'size': 7}, frameon=False)
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for k, v in data[eval_type]['Hits@10'].items():
        axs[1, 1].plot(v['epoch'], v['data'], color=dataset_colors[k], label=k)
    axs[1, 1].set_ylabel('Hits@10')
    axs[1, 1].legend(prop={'size': 7}, frameon=False)
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[1, 1].yaxis.tick_right()

    for ax in axs.flat:
        ax.set_xlabel('Epochs', labelpad=7)
    plt.savefig(f'./tkbc/figures/metrics_by_epoch_{eval_type.lower()}.svg', format='svg', bbox_inches='tight')

# Training time - Grouped bar chart
training_time = defaultdict(list)
for dataset in datasets:
    emb_reg = float(viz_configs[dataset]['emb_reg'])
    time_reg = float(viz_configs[dataset]['time_reg'])
    lr = float(viz_configs[dataset]['lr'])
    for rank in ranks:
        with open(f'{PATH}{args.model}_{dataset}_{rank}_{lr}_{emb_reg}_{time_reg}_best.json', 'r') as f:
            record = json.load(f)
            if ('avg_millis_per_epoch' in record):
                training_time[dataset].append(record['avg_millis_per_epoch'] / 1000)
            else:
                training_time[dataset].append(0)

x = np.arange(len(ranks))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for dataset, time_in_millis in training_time.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, time_in_millis, width, label=dataset, color=dataset_colors[dataset])
    ax.bar_label(rects, padding=3, fmt='%.1f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Avg. training time per epoch (in seconds)')
ax.set_xlabel('Factorization rank')
ax.set_xticks(x + width, ranks)
ax.legend(loc='upper left', frameon=False)

plt.savefig(f'./tkbc/figures/avg_training_time.svg', format='svg', bbox_inches='tight')