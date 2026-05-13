#!/bin/bash
# Run full test suite. Called before every merge to master.
python3 -m pytest tests/ -v
