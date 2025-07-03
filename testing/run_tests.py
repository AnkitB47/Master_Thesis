import argparse
from testing.pipeline import full_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', required=True)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    full_pipeline(args.mechanism, args.out, steps=50)

if __name__ == '__main__':
    main()
