# Consistent character

Create images of a given character in different poses

Run on Replicate:

https://replicate.com/fofr/consistent-character

There is a workflow that you can use directly in ComfyUI, but it will only produce one pose at a time:

https://github.com/fofr/cog-consistent-character/blob/main/workflow_ui.json

## Developing locally

Clone this repository:

```sh
git clone --recurse-submodules https://github.com/fofr/cog-consistent-character.git
```

Run the [following script](https://github.com/fofr/cog-consistent-character/blob/main/scripts/install_custom_nodes.py) to install all the custom nodes:

```sh
./scripts/install_custom_nodes.py
```

You can view the list of nodes in [custom_nodes.json](https://github.com/fofr/cog-consistent-character/blob/main/custom_nodes.json)

### Running the Web UI from your Cog container

1. **GPU Machine**: Start the Cog container and expose port 8188:
```sh
sudo cog run -p 8188 bash
```
Running this command starts up the Cog container and let's you access it

2. **Inside Cog Container**: Now that we have access to the Cog container, we start the server, binding to all network interfaces:
```sh
cd ComfyUI/
python main.py --listen 0.0.0.0
```

3. **Local Machine**: Access the server using the GPU machine's IP and the exposed port (8188):
`http://<gpu-machines-ip>:8188`

When you goto `http://<gpu-machines-ip>:8188` you'll see the classic ComfyUI web form!
