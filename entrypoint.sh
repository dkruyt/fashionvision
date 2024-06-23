#!/bin/sh

# Replace environment variables in the Python command
python server.py --hidden_neurons ${HIDDEN_NEURONS} --limit_per_class ${LIMIT_PER_CLASS}
