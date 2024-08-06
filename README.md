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


### Jax environments

<table style="width:100%; border-collapse: collapse;">
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>Craftax</strong>
      </a><br>
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank">
        <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" alt="Craftax" style="width: 100%; max-width: 400px;">
      </a>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>Housemaze</strong>
      </a><br>
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank">
        <img src="https://github.com/wcarvalho/JaxHouseMaze/raw/main/example.png" alt="Housemaze" style="width: 100%; max-width: 400px;">
      </a>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>XLand-Minigrid</strong>
      </a><br>
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank">
        <img src="https://github.com/corl-team/xland-minigrid/blob/main/figures/ruleset-example.jpg?raw=true" alt="XLand-Minigrid" style="width: 100%; max-width: 400px;">
      </a>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/epignatelli/navix" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>Navix (Jax minigrid)</strong>
      </a><br>
      <a href="https://github.com/epignatelli/navix" target="_blank">
        <img src="https://minigrid.farama.org/_images/GoToObjectEnv.gif" alt="Navix" style="width: 100%; max-width: 400px;">
      </a>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>Overcooked (multi-agent)</strong>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" style="width: 100%; max-width: 400px;">
      </a>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <strong>STORM: Spatial-Temporal Matrix Games (multi-agent)</strong>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" style="width: 100%; max-width: 400px;">
      </a>
    </td>
  </tr>
</table>

