def list_args(args):
    print(">>> training configs:")
    for key in vars(args).keys():
        print("%-20s %s" % (key, str(vars(args)[key])))
    print("")