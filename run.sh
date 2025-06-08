#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Start the server if it's not already running
if ! nc -z localhost 5000 2>/dev/null; then
    python3 predict_reimbursement.py --server &
    # Wait for server to start
    sleep 2
fi

# Send space-delimited values to server and get prediction
echo "$1 $2 $3" | nc localhost 5000