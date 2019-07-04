#!/bin/bash

# Test camera stream with guvcview.

export GOOGLE_APPLICATION_CREDENTIALS=har-rgb-448107881c38.json
streamer -q -c /dev/video1 -f rgb24 -s 1280x1000 -r 26 -w 5 -t 00:00:05 -o outfile.avi
python3 full_system.py outfile.avi
