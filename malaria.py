import argparse

from src.malaria_main import malaria_main

def main():
    # Create commandline parser
    parser = argparse.ArgumentParser(description="Load a file from disk or internet.")
    parser.add_argument('--src', type=str, help="Path to a local file or a URL.",required=True)
    parser.add_argument('--silent', type=bool, action=argparse.BooleanOptionalAction)
    parser.set_defaults(silent=False)
    parser.set_defaults(src='')

    # Parse/extract arguments
    args = parser.parse_args()
    resource = args.src
    verbose = 0 if args.silent else 1

    try:
        malaria_main(resource=resource, verbose=verbose)
    except Exception as e:
        print(f'Error: {e}')
        exit(2)

if __name__ == "__main__":
    main()
