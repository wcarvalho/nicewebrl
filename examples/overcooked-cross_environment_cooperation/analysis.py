import msgpack
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import json
import asyncio
from nicewebrl.utils import read_all_records

async def load_and_analyze_files(extension=''):
    data_dir = f'data'
    all_filenames = os.listdir(data_dir)
    all_filenames = [f for f in all_filenames if (f.endswith('.json') and extension in f)]  # Change extension if needed
    all_filenames = [os.path.join(data_dir, f) for f in all_filenames]
    
    all_records = []
    for filename in all_filenames:
        records = await read_all_records(filename)
        all_records.append(records)
    
    return all_records

def analyze_single_user(user_data):
    success_dict = {}
    for datapoint in user_data:
        if 'name' not in datapoint:
            continue
        if "ik" in datapoint['name'] or "sk" in datapoint['name'] or 'fcp' in datapoint['name']:
            if datapoint['metadata']['type'] != 'EnvStage':
                continue
            prev_best_score = success_dict.get(datapoint['name'], 0)
            try:
                success_dict[datapoint['name']] = max(prev_best_score, datapoint['metadata']['nsuccesses'])
            except:
                pdb.set_trace()
    return success_dict

def analyze_all_users(all_records):
    all_success_dict = {}
    for user_data in all_records:
        success_dict = analyze_single_user(user_data)
        for key, value in success_dict.items():
            if key not in all_success_dict:
                all_success_dict[key] = [value]
            else:
                all_success_dict[key].append(value)
    return all_success_dict

def plot_success_dict(success_dict, extension=''):
    # Set the style
    import seaborn as sns
    sns.set_context('paper')
    
    # Calculate means and standard errors
    means = {k: np.mean(v) for k, v in success_dict.items()}
    sems = {k: np.std(v) / np.sqrt(len(v)) for k, v in success_dict.items()}
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    x_pos = np.arange(len(means))
    keys = list(means.keys())
    vals = [means[key] for key in keys]
    errs = [sems[key] for key in keys]
    plt.bar(x_pos, vals, 
            yerr=errs,
            capsize=5,
            color='skyblue',
            alpha=0.8)
    
    # Customize the plot
    # labels = ['IK' if 'IK' in key else 'SK' for key in keys]
    labels = [key for key in keys]
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Number of Successful Deliveries')
    plt.title('Success Rate by Task')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'hAI_success_{extension}.pdf')
    plt.close()

if __name__ == "__main__":
    layout = 'counter_circuit'
    # Run the async function
    records = asyncio.run(load_and_analyze_files(layout))
    success_dict = analyze_all_users(records)

    plot_success_dict(success_dict, layout)
    
    # Now you can analyze your records
    print(f"Loaded {len(records)} records")
   
