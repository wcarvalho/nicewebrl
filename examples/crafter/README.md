# Crafter Example

This example demonstrates how to create a web interface for the [Craftax environment](https://github.com/MichaelTMatthews/Craftax), a JAX implementation of the Crafter environment. Crafter is a procedurally generated survival game where the agent needs to collect resources, craft tools, and avoid dangers.

![Crafter Environment](https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif)

## Features
- Grid-world Minecraft-like environment
- Procedurally generated worlds
- Multiple tasks: gathering resources, crafting tools, avoiding dangers
- Day/night cycle

## Running the Example
```bash
# Install dependencies
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[craftax]"
# or 
pip install -e ".[craftax]"

# Run the web app
python web_app.py
``` 