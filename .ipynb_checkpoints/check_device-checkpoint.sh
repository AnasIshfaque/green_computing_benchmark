#!/bin/bash

CSV_FILE="system_metrics.csv"

# Function to get CPU usage
get_cpu_usage() {
    # Extract CPU usage from powertop_output.csv
    cpu_usage=$(awk -F';' '/Device Power Report/ {getline; getline; getline; print $1 }' powertop_output.csv)
    echo "$cpu_usage"
}

get_top_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | sed "s/., *\([0-9.]\)%* id.*/\1/" | awk '{print 100 - $1}'
}

# Function to get RAM usage
get_ram_usage() {
    free -m | awk '/Mem:/ {print $3}'
}

get_vcgencmd_ram() {
	ram_used=$(vcgencmd get_mem arm | awk -F'=' '{print $2}')
	echo "$ram_used"
}
# Function to get CPU temperature
get_cpu_temperature() {
    # Check if vcgencmd utility is available
    if command -v vcgencmd &> /dev/null; then
        # Get CPU temperature from vcgencmd
        temperature=$(vcgencmd measure_temp | awk -F '=' '{print $2}' | tr -d "'C")
        echo "$temperature"
    else
        echo "N/A (vcgencmd not available)"
    fi
}


get_power_draw() {
    # Extract power draw from powertop_output.csv
    power_draw=$(awk -F';' '/Device Power Report/ {getline; getline; getline; print $3 }' powertop_output.csv)
    echo "$power_draw"
}

# Function to remove temporary powertop_output.csv
remove_temp_file() {
    rm -f powertop_output.csv
}

# Append system metrics to CSV file
append_metrics_to_csv() {
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    cpu_usage=$(get_cpu_usage)
	top_cpu=$(get_top_cpu_usage)
    ram_usage=$(get_ram_usage)
	vcgencmd_ram_use=$(get_vcgencmd_ram)
    cpu_temperature=$(get_cpu_temperature)
    power_draw=$(get_power_draw)
    
    # Append the metrics to the CSV file
    echo "$timestamp,$cpu_usage,$ram_usage,$cpu_temperature,$power_draw" >> "$CSV_FILE"
}


while true; do

    # Run powertop to generate output and extract metrics
    sudo powertop --time=1 --csv=powertop_output.csv &> /dev/null

    # Check if powertop command was successful
    if [ $? -eq 0 ]; then
        # Append metrics to CSV file
        append_metrics_to_csv
        # Remove temporary powertop_output.csv
        remove_temp_file
    else
        echo "N/A (powertop failed to run)"
    fi

    # Adjust sleep duration as needed
    sleep 60

done



# Print system metrics to terminal
#echo "CPU Usage: $(get_cpu_usage) %"
#echo "RAM Usage (MB): $(get_ram_usage)"
#echo "CPU Temperature: $(get_cpu_temperature)"
#echo "Power Draw: $(get_power_draw) Watts"
