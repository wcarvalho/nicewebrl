# Toy Cooperation Environment

Here, we show how to load trained RL agents to play with human participants in a multi-agent task in the dual-destination environment from [Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](https://arxiv.org/abs/2504.12714).

**Make sure you have CMake installed**
```bash
# Install dependencies
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[jaxmarl]"
# or 
pip install -e ".[jaxmarl]"

To get started, run

```bash
cd examples/dual_destination-CEC
python web_app.py
```

This will start a web app where you can play the game with the RL agents. To view how we loaded them in, check out the `experiment.py` file.


