from dataclasses import dataclass
from typing import Optional
import argparse
import sys

@dataclass(frozen=True)
class AnalyzerArgs:
    load_file: str
    chess_file: str
    train_mode: bool
    predict_mode: bool
    save_file: Optional[str]

class CLIParser:
    @staticmethod
    def parse_arguments() -> AnalyzerArgs:
        parser = argparse.ArgumentParser(add_help=False)
        
        parser.add_argument("load_file", type=str, metavar="LOADFILE", help="File containing an artificial neural network")
        parser.add_argument("chess_file", type=str, metavar="CHESSFILE", help="File containing chessboards")
        
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("--train", action="store_true", help="Launch the neural network in training mode")
        mode_group.add_argument("--predict", action="store_true", help="Launch the neural network in prediction mode")
        
        parser.add_argument("--save", type=str, default=None, metavar="SAVEFILE", help="Save neural network into SAVEFILE")
        
        try:
            args = parser.parse_args()
            
            if not args.train and not args.predict:
                print("Error: Either --train or --predict must be specified.", file=sys.stderr)
                sys.exit(84)

            if args.save and not args.train:
                print("Error: --save can only be used with --train.", file=sys.stderr)
                sys.exit(84)

            return AnalyzerArgs(
                load_file=args.load_file,
                chess_file=args.chess_file,
                train_mode=args.train,
                predict_mode=args.predict,
                save_file=args.save
            )
            
        except SystemExit:
            sys.exit(84)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(84)
