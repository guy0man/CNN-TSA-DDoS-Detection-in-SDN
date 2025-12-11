# Getting Started

# System Requirements
**Mininet VM Specifications (Minimum)**
- RAM: 4GB (8GB recommended)
- CPU: 2 cores (4 cores recommended)
- Disk: 20GB
- OS: Ubuntu 20.04 LTS

**Core Components**
- Mininet 2.3.0+
- RYU Controller 4.34+
- Python 3.8+
- PyTorch 1.12+
- NumPy, Pandas

**Traffic Generation Tools**
- hping3 (TCP/UDP/ICMP attacks)
- iperf3 (benign traffic)
- scapy (custom packet crafting)

# Step 1 : Setting Up GitHub
Configure Access to Github
1. Generate SSH key on Mininet VM
```
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept default location (~/.ssh/id_ed25519)
Set passphrase or press Enter for no passphrase

**2. Display public key**
```
cat ~/.ssh/id_ed25519.pub
```
Copy the entire output

**3. Add to GitHub**
- Go to GitHub.com → Settings → SSH and GPG keys
- Click "New SSH key"
- Paste the public key
- Click "Add SSH key"

**4. Test connection**
```
ssh -T git@github.com
```
**Should see: "Hi username! You've successfully authenticated..."**
**5. Configure git**
```
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

# Step 2 : Install Dependencies
**Update system**
```
sudo apt update && sudo apt upgrade -y
```
**Install Mininet**
```
sudo apt install mininet -y
```
**Install RYU Controller**
```
sudo pip3 install ryu
```
**Install attack tools**
```
sudo apt install hping3 iperf3 -y
```
**Install Python ML libraries**
```
pip3 install torch torchvision numpy pandas scikit-learn
```
**Install additional utilities**
```
sudo apt install net-tools tcpdump -y
```
# Step 3: Clone Repository
**Navigate to home directory**
```
cd ~
```
**Clone repository using SSH**
```
git clone https://github.com/guy0man/CNN-TSA-DDoS-Detection-in-SDN
```
**Navigate to project directory**
```
cd ddos-detection
```
**Verify files**
```
ls -la
```
**Should see: models/, merged_outputs/, preprocessing_output/, etc.**
