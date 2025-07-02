#!/bin/bash

# Go one directory up to the project root (src/)
cd "$(dirname "$0")/.."

# Set PYTHONPATH to current dir (src/)
export PYTHONPATH="$(pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Run tests in tests/ folder
pytest tests  --maxfail=3 --tb=short

status=$?

if [ $status -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Exit code: $status"
fi

exit $status

