from experiments.run_experiment import run

run(
    storage_secret="a_very_secret_key_for_testing_only_12345",
    experiment_file="examples/craftax/experiment_structure.py",
    title="NiceWebRL Craftax Experiment",
    reload=False, 
    on_startup_fn=None, # e.g. restore cache
    on_termination_fn=None, # e.g. saving and uploading to cloud storage
)