#!/usr/bin/env python
from utils import sample
import sys

if __name__ == '__main__':
    args = ["--model-module", "wrapper.interface", "--model-class", "AliModel"] + sys.argv[1:]
    sample.main(args)
