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
