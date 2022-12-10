# This win11 fix for WSL2 networking issue.
# Win11 WSL adapter hiddent, so we just adjust for all LROR
# Get-NetAdapterLso -Name * -IncludeHidden
# Get-NetAdapterChecksumOffload  -Name "*" -IncludeHidden
# Enable-NetAdapterLso -Name "*" -IPv4 -IPv6 * -IncludeHidden
# Enable-NetAdapterChecksumOffload  -IPv4 -IPv6 * -IncludeHidden

Enable-NetAdapterLso -Name "*" -IPv4 -IPv6 * -IncludeHidden
Enable-NetAdapterChecksumOffload  -IPv4 -IPv6 * -IncludeHidden
docker build -t meta_critic:v1 .
docker run --privileged --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/code meta_critic:v1 bash