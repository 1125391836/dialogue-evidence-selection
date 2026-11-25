"""Click Run to replay context compression and recovery on the development set."""
import sys

from run import main

if __name__ == "__main__":
    sys.argv.insert(1, "recover")
    main()
