import sys

outstr = ''


def main():
    print(outstr)

if __name__ == '__main__':
    # Nawawy's start
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the input and output directories and train/test (0/1) flag as arguments, e.g., python run.py data output 1')

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if sys.argv[3] == '1':
        train = True
    elif sys.argv[3] == '0':
        train = False
    else:
        raise ValueError("Train flag must be 0 or 1")


    # Nawawy's end
    main()