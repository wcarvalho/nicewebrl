# Overcooked Example

This example demonstrates how to create a multi-agent web interface for the [JaxMARL Overcooked environment](https://github.com/FLAIROx/JaxMARL). Overcooked is a cooperative cooking game where two agents must work together to prepare and serve meals.

![Overcooked Environment](https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true)

## Features
- Multi-agent cooperative environment
- Cooking tasks requiring coordination
- Multiple layouts available
- Real-time interaction between two human players

## Running the Example
```bash
# Install dependencies
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[jaxmarl]"

# Run the web app
python web_app.py
```

Note: This example requires two players to connect to play together. 