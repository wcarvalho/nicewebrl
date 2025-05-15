# XLand-Minigrid Example

## Running the Example
**From the root directory**
```bash
# Install dependencies
pip install "nicewebrl[xland-assistant] @ git+https://github.com/wcarvalho/nicewebrl.git"

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


## (Optional) Setting up google cloud for storing/loading data

While NiceWebRL saves all environment information, it provides no persistent mechanism for saving this data. One option for saving this data is to use Google cloud. We provide a file for saving data to google cloud (`google_cloud_utils.py`) and provide instructions for setting up a Google cloud account below:

Create a [Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts?) for accessing your Database. Select "CREATE SERVICE ACCOUNT", give it some name and the following permissions:
- Storage Object Creator (for uploading/saving data)
- Storage Object Viewer and Storage Object Admin (for viewing/downloading data)

- create a bucket using [this link](https://console.cloud.google.com/storage/). this will be used to store data.
- create a key using [this link](https://console.cloud.google.com/iam-admin/serviceaccounts/details/111959560397464491265/keys?project=human-web-rl). this will be used to authenticate uploading and downloading of data. store the key as `datastore-key.json` and make `GOOGLE_CREDENTIALS` point to it. e.g. `export GOOGLE_CREDENTIALS=datastore-key.json`. The Dockerfile assumes this.