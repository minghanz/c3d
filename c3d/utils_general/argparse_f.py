'''
This is to quickly initialize a argparser that could take a file as argument input, with comment enabled.
Modified from bts.
'''
import argparse

def convert_arg_line_to_args(arg_line):
    if len(arg_line) < 1:
        return
    if arg_line[0] == "#":
        return
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
        
def init_argparser_f(*args, **kwargs):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', *args, **kwargs)
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    return parser