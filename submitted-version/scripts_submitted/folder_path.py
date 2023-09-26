import sys, os

def gsm_path(args):

    argsd = vars(args)
    fpath = f"/gsm-frg/B{args.batch:02d}"

    if args.scale != 1:
        fpath = f"{fpath}-scale{args.scale:0.2f}"
    if args.covinit != 'identity':
        fpath = f"{fpath}-{args.covinit}cov"
    if args.modeinit:
        if args.modeinit == 2:
            fpath = f"{fpath}-modeinit2"
        else:
            fpath = f"{fpath}-modeinit"
    if args.warmup == 1:
        fpath = f"{fpath}-warmup"
    if args.suffix != '':
        fpath = f"{fpath}{args.suffix}"

    print("subfolder path : ", fpath)
    return fpath


def bbvi_path(args):

    argsd = vars(args)
    fpath = f"/bbvi-frg-{args.mode}/B{args.batch:02d}-lr{args.lr:0.1e}"

    if args.scale != 1:
        fpath = f"{fpath}-scale{args.scale:0.2f}"
    if args.covinit != 'identity':
        fpath = f"{fpath}-{args.covinit}cov"
    if args.modeinit:
        if args.modeinit == 2:
            fpath = f"{fpath}-modeinit2"
        else:
            fpath = f"{fpath}-modeinit"
    if args.suffix != '':
        fpath = f"{fpath}{args.suffix}"

    print("subfolder path : ", fpath)
    return fpath


def frg_path_gsm(args):

    argsd = vars(args)
    fpath = f"/B{args.batch:02d}-gsm"

    if args.scale != 1:
        fpath = f"{fpath}-scale{args.scale:0.2f}"

    print("subfolder path : ", fpath)
    return fpath


def frg_path_bbvi(args):

    argsd = vars(args)
    fpath = f"/B{args.batch:02d}-lr{args.lr:0.1e}"

    if args.scale != 1:
        fpath = f"{fpath}-scale{args.scale:0.2f}"

    print("subfolder path : ", fpath)
    return fpath

