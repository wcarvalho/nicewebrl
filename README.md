# nicewebrl
Python library for quickly making interaction RL Apps with [NiceGUI](https://nicegui.io/). It is particularly suited for hooking up [JAX](https://github.com/google/jax) based RL environments to web interfaces. JAX is useful for blazing fast iteration on AI algorithms. With this library, you can use the exact same environment for human subject experiments.

### Install

```
# pip install
pip install git+https://github.com/wcarvalho/nicewebrl

# more manauly
conda create -n nicewebrl python=3.10 pip wheel -y
conda activate nicewebrl
pip install -r requirements.tx
```
