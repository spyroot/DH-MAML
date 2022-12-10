# This win11 fix for WSL2 networking issue.
# Win11 WSL adapter hiddent, so we just adjust for all LROR
# Get-NetAdapterLso -Name * -IncludeHidden
# Enable-NetAdapterLso -Name "*" -IPv4 -IPv6 * -IncludeHidden
# Get-NetAdapterChecksumOffload  -Name "*" -IncludeHidden
docker build -t meta_critic:v1 .
