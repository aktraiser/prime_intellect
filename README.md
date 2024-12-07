# PrimeIntellect VM Setup Guide

This guide will help you set up and run the training environment on a PrimeIntellect VM.

## Connecting to Your VM

### 1. Download Private Key
After launching your PrimeIntellect instance, download the provided private key. This key is essential for secure VM access.

### 2. Set Key Permissions
Open your terminal and navigate to the directory containing the downloaded key. Set the correct permissions:
```bash
chmod 400 [your-key-name].pem
```

### 3. Connect to VM
Use SSH to connect to your VM:
```bash
ssh -i [your-key-name].pem ubuntu@[vm-ip-address]
```
> Replace `[your-key-name]` and `[vm-ip-address]` with your actual key filename and VM IP address.

## Setting Up the Environment

### 1. Clone Repository
Once connected to the VM, clone the repository:
```bash
git clone https://github.com/aktraiser/prime_intellect.git
```

### 2. Prepare Scripts
Make all scripts executable:
```bash
find prime-intellect-test -type f -name "*.sh" -exec chmod +x {} \;
```

### 3. Managing Directories
To remove directories (e.g., old checkpoints or failed runs):
```bash
rm -rf [directory_name]
```
> ⚠️ Warning: This command permanently deletes the directory and all its contents. Use with caution.

## Running the Training

Navigate to the project directory and start training:
```bash
cd prime_intellect
./training.sh
```

## Training Monitoring

The training script will display:
- Loss values at each step
- Gradient norm
- Learning rate
- Evaluation metrics every 100 steps
- GPU memory usage statistics

## Checkpoints

The training process automatically:
- Saves checkpoints in the `checkpoints` directory
- Keeps the 3 best models based on validation loss
- Creates a final merged model after training

## Support

For any issues or questions, please contact the PrimeIntellect support team or refer to the repository documentation.

# Hugging Face
## Install 
```bash
pip install huggingface_hub
```

## Config Login User 
```bash
huggingface-cli login
```

## Config User
```bash
git config --global user.name « Username »
git config --global user.email « Email_user »
```

##Install Git LFS 
```bash
apt-get update
apt-get install git-lfs
```

## Create Repository on Huggingface
```bash
huggingface-cli repo create nom-du-modele --type model
```

## Clone repository
```bash
git clone https://huggingface.co/nom-du-modele
cd nom-du-modele
```

## Config Git LFS
```bash
git lfs install
huggingface-cli lfs-enable-largefiles .
```

## Config tracking LFS
```bash
cat > .gitattributes << EOL
model.safetensors filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
EOL
```
```bash
cat > .gitattributes << EOL
model-*.safetensors filter=lfs diff=lfs merge=lfs -text
model.safetensors.index.json filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
EOL
```

## First add .gitattributes
```bash
git add .gitattributes
git commit -m "Add LFS configuration"
```

## Copy model files
```bash
cp ../llama_model_merged/* .
```

## Add normal configuration files
```bash
git add config.json
git add generation_config.json
git add special_tokens_map.json
git add tokenizer_config.json
git commit -m "Add configuration files"
```

## Adding large files with LFS
```bash
git add model.safetensors
git add tokenizer.json
git commit -m "Add model files with LFS"
```

```bash
git add model-*.safetensors
git add model.safetensors.index.json
git add tokenizer.json
git commit -m "Add sharded model files with LFS"
```

## Push vers Hugging Face
Before the push, give the right to the token.
```bash
git push origin main
```


