#!/bin/bash
PORT=8081
echo ""
echo "Starting viewer at http://localhost:$PORT"
echo ""
python3 -m http.server $PORT --quiet 2>/dev/null || python3 -m http.server $PORT
