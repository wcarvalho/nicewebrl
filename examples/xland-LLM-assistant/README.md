# XLand-Minigrid Example

## Running the Example
**From the root directory**
```bash
# Install dependencies
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[xland-assistant]"
# or 
pip install -e ".[xland-assistant]"
```
**From the this directory** `examples/xland-minigrid`:
```
# Run the web app
python web_app_assistant.py
```

## Launching online with fly.io

**From root directory**:
```
# setup configuration
flyctl launch \
--dockerfile Dockerfile \
--name xland-assistant \
--config xland-assistant.toml \
--vm-size 'performance-2x' --yes

# deploy to servers
flyctl deploy --config xland-assistant.toml

# scale to multiple regions (useful for decreasing latency)
flyctl scale count 10 --config xland-assistant.toml --region "iad,sea,lax,den"  --yes
```
