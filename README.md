# nicewebrl
Python library for quickly making interactive RL Apps with [NiceGUI](https://nicegui.io/). It is particularly suited for hooking up [JAX](https://github.com/google/jax) based RL environments to web interfaces. JAX is useful for blazing fast iteration on AI algorithms. With this library, you can use the exact same environment for human subject experiments.

### Install

```bash
# pip install
pip install git+https://github.com/wcarvalho/nicewebrl

# more manually (first clone then)
conda create -n nicewebrl python=3.10 pip wheel -y
conda activate nicewebrl
pip install -e .
```

### Jax environments

The following are all Jax environments which can be used with this framework to run human subject experiments. The power of using jax is that one can use the **exact** same environment for human subjects experiments as for developing modern machine learning algorithms (especially reinforcement learning algorithms).

When targetting normative solutions, one may want to study algorithms asymptotic behavior with a lot of data. Jax makes it cheap to do this. nicewebrl makes it easy to compare these algorithms to human subject behavior.
<table style="width:100%; border-collapse: collapse;">
  <tr style="max-height: 150px; overflow: hidden;">
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Craftax</strong></center>
      </a><br>
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank">
        <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" alt="Craftax" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a grid-world version of minecraft. </p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Housemaze</strong></center>
      </a><br>
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank">
        <img src="https://github.com/wcarvalho/JaxHouseMaze/raw/main/example.png" alt="Housemaze" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a maze environment where new mazes can be easily be described with a string.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>XLand-Minigrid</strong></center>
      </a><br>
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank">
        <img src="https://github.com/corl-team/xland-minigrid/blob/main/figures/ruleset-example.jpg?raw=true" alt="XLand-Minigrid" style="width: 100%; max-width: 400px;">
      </a>
      <p>This environment allows for complex, nested compositional tasks. XLand-Minigrid comes with 3 benchmarks which together defnine 3 million unique tasks.</p>
    </td>
  </tr>
  <tr style="max-height: 150px; overflow: hidden;">
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/epignatelli/navix" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Navix</strong></center>
      </a><br>
      <a href="https://github.com/epignatelli/navix" target="_blank">
        <img src="https://minigrid.farama.org/_images/GoToObjectEnv.gif" alt="Navix" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a jax implementation of the popular Minigrid environment.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Overcooked (multi-agent)</strong></center>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a popular multi-agent environment.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>STORM (multi-agent)</strong></center>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" style="width: 100%; max-width: 400px;">
      </a>
      <p>This environment specifies Matrix games represented as grid world scenarios.</p>
    </td>
  </tr>
</table>

