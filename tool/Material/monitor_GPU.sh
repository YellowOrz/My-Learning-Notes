# !/bin/bash
server=$1
max_memory=$2
username=xzf
echo 开始监测$server服务器
while true
do
    nvidia_info=$(ssh $username@192.168.1.$server -i ~/.ssh/$server "nvidia-smi"|grep Default)
    all_gpu_memory=$(echo $nvidia_info| tr "|" "\n"|grep MiB|cut -f 1 -d '/'|tr 'MiB' ' ')
    id=0
    for gpu_memory in $(echo $all_gpu_memory)
    do
        if [ $gpu_memory -lt $max_memory ]
        then
            send="GPU $id of $server is empty!"
            echo $send
            /mnt/c/Soft/wsl-notify-send_windows_amd64/wsl-notify-send.exe "$send"
            echo 等待30min
            sleep 30m
            break
        fi
        id=$(($id+1))
    done
    echo 等待5min
    sleep 5m
done
